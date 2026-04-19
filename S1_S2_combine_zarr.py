import h5py
import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from tqdm import tqdm


T       = 4
MAX_BA  = 3
MAX_GAP = 8
H, W    = 128, 128

REGIONS = {
    1: (2048, 6784,  4480, 11008),
    2: (1536, 14208, 4736, 20096),
    3: (6144, 16384, 10752, 19840),
}

H5_RE  = re.compile(r"r(\d+)_c(\d+)")
NPY_RE = re.compile(r"loc_r(\d+)_c(\d+)")

def normalise(r: str, c: str) -> str:
    return f"r{int(r)}_c{int(c)}"

def get_region_id(chip_id: str) -> int:
    m = H5_RE.fullmatch(chip_id)
    if not m:
        return -1
    row = int(m.group(1))
    col = int(m.group(2))
    for rid, (rmin, cmin, rmax, cmax) in REGIONS.items():
        if rmin <= row <= rmax and cmin <= col <= cmax:
            return rid
    return -1

def load_dates(h5file):
    return [datetime.strptime(d, "%Y-%m-%d") for d in h5file.attrs["dates"]]

def nearest_single(target, date_list):
    gaps = [(i, d, abs((d - target).days)) for i, d in enumerate(date_list)]
    return min(gaps, key=lambda x: x[2])

def within_window(target, date_list, max_gap):
    gaps     = [(i, d, abs((d - target).days)) for i, d in enumerate(date_list)]
    filtered = [x for x in gaps if x[2] <= max_gap]
    return sorted(filtered, key=lambda x: x[2])[:MAX_BA]

def build_block_index_with_csv(
    s2_h5, ca_h5, cd_h5, ba_h5, bd_h5,
    chip_id, year,
):
    """
    Returns:
        blocks   : list of block dicts for Zarr writing
        csv_rows : list of dicts → one row per S2 date for the year CSV
    """
    s2_dates     = load_dates(s2_h5)
    ca_dates     = load_dates(ca_h5)
    cd_dates     = load_dates(cd_h5)
    ba_dates     = load_dates(ba_h5)
    bd_dates     = load_dates(bd_h5)

    s2_tiles_all = list(s2_h5.attrs["tiles"])
    s2_cloud_all = list(s2_h5.attrs["cloud_coverages"])

    n_blocks = len(s2_dates) // T
    blocks   = []
    csv_rows = []

    for block_idx in range(n_blocks):
        s2_block  = s2_dates[block_idx * T : (block_idx + 1) * T]
        rows      = []
        skip      = False

        for t, s2_dt in enumerate(s2_block):
            s2_idx = block_idx * T + t

            s2_tile  = str(s2_tiles_all[s2_idx])
            s2_cloud = float(s2_cloud_all[s2_idx])

            ca_idx, ca_dt, ca_gap = nearest_single(s2_dt, ca_dates)

            cd_idx, cd_dt, cd_gap = nearest_single(s2_dt, cd_dates)

            ba_frames = within_window(s2_dt, ba_dates, MAX_GAP)

            bd_frames = within_window(s2_dt, bd_dates, MAX_GAP)

            if (ca_gap > MAX_GAP and cd_gap > MAX_GAP
                    and len(ba_frames) == 0 and len(bd_frames) == 0):
                skip = True
                break

            rows.append({
                "t"          : t,
                "s2_date"    : s2_dt.strftime("%Y-%m-%d"),
                "s2_tile"    : s2_tile,
                "s2_cloud"   : s2_cloud,
                "ca_idx"     : ca_idx,
                "ca_date"    : ca_dt.strftime("%Y-%m-%d"),
                "ca_gap_days": ca_gap,
                "cd_idx"     : cd_idx,
                "cd_date"    : cd_dt.strftime("%Y-%m-%d"),
                "cd_gap_days": cd_gap,
                "ba_indices" : [x[0] for x in ba_frames],
                "ba_dates"   : [x[1].strftime("%Y-%m-%d") for x in ba_frames],
                "ba_gaps"    : [x[2] for x in ba_frames],
                "bd_indices" : [x[0] for x in bd_frames],
                "bd_dates"   : [x[1].strftime("%Y-%m-%d") for x in bd_frames],
                "bd_gaps"    : [x[2] for x in bd_frames],
            })

            csv_rows.append({
                "chip"              : chip_id,
                "year"              : year,
                "block_idx"         : block_idx,
                "t"                 : t,
                "s2_date"           : s2_dt.strftime("%Y-%m-%d"),
                "s2_tile"           : s2_tile,
                "s2_cloud_coverage" : s2_cloud,
                "ca_idx"            : ca_idx,
                "ca_date"           : ca_dt.strftime("%Y-%m-%d"),
                "ca_gap_days"       : ca_gap,
                "ca_s2_tile"        : s2_tile,
                "cd_idx"            : cd_idx,
                "cd_date"           : cd_dt.strftime("%Y-%m-%d"),
                "cd_gap_days"       : cd_gap,
                "cd_s2_tile"        : s2_tile,
                "ba_n_frames"       : len(ba_frames),
                "ba_indices"        : str([x[0] for x in ba_frames]),
                "ba_dates"          : str([x[1].strftime("%Y-%m-%d") for x in ba_frames]),
                "ba_gaps"           : str([x[2] for x in ba_frames]),
                "ba_s2_tile"        : s2_tile,
                "bd_n_frames"       : len(bd_frames),
                "bd_indices"        : str([x[0] for x in bd_frames]),
                "bd_dates"          : str([x[1].strftime("%Y-%m-%d") for x in bd_frames]),
                "bd_gaps"           : str([x[2] for x in bd_frames]),
                "bd_s2_tile"        : s2_tile,
            })

        if not skip and len(rows) == T:
            blocks.append({"block_idx": block_idx, "rows": rows})
        else:
            csv_rows = csv_rows[:-len(rows)]

    return blocks, csv_rows


def write_block(
    store, block_idx, rows,
    s2_h5, ca_h5, cd_h5, ba_h5, bd_h5,
    y_data, chip_id, year, region_id,
):
    s2_block = np.zeros((T, 4, H, W), dtype=np.float32)
    s1_block = np.zeros((T, 4, H, W), dtype=np.float32)

    s2_dates  = []
    s2_clouds = []
    s2_tiles  = []

    ca_dates    = []
    ca_gaps     = []
    ca_s2_tiles = []

    cd_dates    = []
    cd_gaps     = []
    cd_s2_tiles = []

    ba_dates_buf = np.full((T, MAX_BA), "",  dtype=object)
    ba_gaps_buf  = np.full((T, MAX_BA), -1,  dtype=int)
    ba_n_frames  = []
    ba_max_gaps  = []
    ba_s2_tiles  = []

    bd_dates_buf = np.full((T, MAX_BA), "",  dtype=object)
    bd_gaps_buf  = np.full((T, MAX_BA), -1,  dtype=int)
    bd_n_frames  = []
    bd_max_gaps  = []
    bd_s2_tiles  = []

    for row in rows:
        t      = int(row["t"])
        s2_idx = block_idx * T + t

        s2_block[t] = s2_h5["X"][s2_idx]
        s2_dates.append(row["s2_date"])
        s2_clouds.append(float(row["s2_cloud"]))
        s2_tiles.append(str(row["s2_tile"]))

        s1_block[t, 0] = ca_h5["X"][int(row["ca_idx"])]
        ca_dates.append(row["ca_date"])
        ca_gaps.append(int(row["ca_gap_days"]))
        ca_s2_tiles.append(str(row["s2_tile"]))

        s1_block[t, 1] = cd_h5["X"][int(row["cd_idx"])]
        cd_dates.append(row["cd_date"])
        cd_gaps.append(int(row["cd_gap_days"]))
        cd_s2_tiles.append(str(row["s2_tile"]))

        ba_idx = row["ba_indices"]
        ba_dts = row["ba_dates"]
        ba_gps = row["ba_gaps"]
        n_ba   = len(ba_idx)
        ba_frames_data = np.stack([ba_h5["X"][i] for i in ba_idx], axis=0)
        s1_block[t, 2] = np.nanmean(ba_frames_data, axis=0)
        for f in range(n_ba):
            ba_dates_buf[t, f] = ba_dts[f]
            ba_gaps_buf[t, f]  = ba_gps[f]
        ba_n_frames.append(n_ba)
        ba_max_gaps.append(max(ba_gps))
        ba_s2_tiles.append(str(row["s2_tile"]))

        bd_idx = row["bd_indices"]
        bd_dts = row["bd_dates"]
        bd_gps = row["bd_gaps"]
        n_bd   = len(bd_idx)
        bd_frames_data = np.stack([bd_h5["X"][i] for i in bd_idx], axis=0)
        s1_block[t, 3] = np.nanmean(bd_frames_data, axis=0)
        for f in range(n_bd):
            bd_dates_buf[t, f] = bd_dts[f]
            bd_gaps_buf[t, f]  = bd_gps[f]
        bd_n_frames.append(n_bd)
        bd_max_gaps.append(max(bd_gps))
        bd_s2_tiles.append(str(row["s2_tile"]))

    meta = store["meta"]
    store["s2"].append(s2_block[np.newaxis])
    store["s1"].append(s1_block[np.newaxis])
    store["y"].append(y_data[np.newaxis])

    meta["chip_id"].append([chip_id])
    meta["year"].append([year])
    meta["block_idx"].append([block_idx])
    meta["region_id"].append([region_id])

    meta["s2_dates"].append([s2_dates])
    meta["s2_cloud_coverages"].append([s2_clouds])
    meta["s2_tiles"].append([s2_tiles])

    meta["ca_date"].append([ca_dates])
    meta["ca_gap_days"].append([ca_gaps])
    meta["ca_s2_tile"].append([ca_s2_tiles])

    meta["cd_date"].append([cd_dates])
    meta["cd_gap_days"].append([cd_gaps])
    meta["cd_s2_tile"].append([cd_s2_tiles])

    meta["ba_dates"].append(ba_dates_buf[np.newaxis])
    meta["ba_n_frames"].append([ba_n_frames])
    meta["ba_max_gap"].append([ba_max_gaps])
    meta["ba_gap_days"].append(ba_gaps_buf[np.newaxis])
    meta["ba_s2_tile"].append([ba_s2_tiles])

    meta["bd_dates"].append(bd_dates_buf[np.newaxis])
    meta["bd_n_frames"].append([bd_n_frames])
    meta["bd_max_gap"].append([bd_max_gaps])
    meta["bd_gap_days"].append(bd_gaps_buf[np.newaxis])
    meta["bd_s2_tile"].append([bd_s2_tiles])

    meta["label_path"].append([str("")])

def run_full_pipeline(
    year        : int,
    s2_root     : str,
    ca_root     : str,
    cd_root     : str,
    ba_root     : str,
    bd_root     : str,
    label_root  : str,
    zarr_path   : str,
    out_dir     : str,
):
    s2_root    = Path(s2_root)
    ca_root    = Path(ca_root)
    cd_root    = Path(cd_root)
    ba_root    = Path(ba_root)
    bd_root    = Path(bd_root)
    label_root = Path(label_root)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path     = out_dir / f"match_stats_{year}.csv"
    skipped_csv  = out_dir / f"skipped_{year}.csv"

    npy_files = {}
    for p in sorted(label_root.glob("loc_r*_c*.npy")):
        m = NPY_RE.search(p.stem)
        if m:
            npy_files[normalise(m.group(1), m.group(2))] = p

    h5_chips = {}
    for p in sorted(s2_root.glob("r*.h5")):
        m = H5_RE.fullmatch(p.stem)
        if m:
            h5_chips[normalise(m.group(1), m.group(2))] = p

    matched_chips = [k for k in h5_chips if k in npy_files]

    print(f"\n{'='*60}")
    print(f"Year             : {year}")
    print(f"S2 chips         : {len(h5_chips):,}")
    print(f"Label files      : {len(npy_files):,}")
    print(f"Matched chips    : {len(matched_chips):,}")
    print(f"Missing labels   : {len(h5_chips) - len(matched_chips):,}")
    print(f"{'='*60}\n")

    from initialize_zarr import initialize_zarr
    store = initialize_zarr(zarr_path, overwrite=True)

    all_csv_rows  = []
    skipped_rows  = []
    n_chips_done  = 0
    n_blocks_done = 0
    n_skipped     = 0

    for chip_id in tqdm(matched_chips, desc=f"Year {year}", unit="chip"):

        s2_path  = s2_root  / f"{chip_id}.h5"
        ca_path  = ca_root  / f"{chip_id}.h5"
        cd_path  = cd_root  / f"{chip_id}.h5"
        ba_path  = ba_root  / f"{chip_id}.h5"
        bd_path  = bd_root  / f"{chip_id}.h5"
        lbl_path = npy_files[chip_id]

        missing = [p for p in [ca_path, cd_path, ba_path, bd_path]
                   if not p.exists()]
        if missing:
            skipped_rows.append({
                "chip_id": chip_id,
                "reason" : f"Missing: {[p.name for p in missing]}",
            })
            n_skipped += 1
            continue

        try:
            y_data = np.load(lbl_path)[:3].astype(np.float32)
        except Exception as e:
            skipped_rows.append({"chip_id": chip_id, "reason": f"Label error: {e}"})
            n_skipped += 1
            continue

        region_id = get_region_id(chip_id)

        try:
            s2_h5 = h5py.File(s2_path, "r")
            ca_h5 = h5py.File(ca_path, "r")
            cd_h5 = h5py.File(cd_path, "r")
            ba_h5 = h5py.File(ba_path, "r")
            bd_h5 = h5py.File(bd_path, "r")
        except Exception as e:
            skipped_rows.append({"chip_id": chip_id, "reason": f"H5 open error: {e}"})
            n_skipped += 1
            continue

        try:
            blocks, csv_rows = build_block_index_with_csv(
                s2_h5, ca_h5, cd_h5, ba_h5, bd_h5,
                chip_id, year,
            )

            if not blocks:
                skipped_rows.append({"chip_id": chip_id, "reason": "No valid blocks"})
                n_skipped += 1
                continue

            for block in blocks:
                try:
                    write_block(
                        store,
                        block["block_idx"],
                        block["rows"],
                        s2_h5, ca_h5, cd_h5, ba_h5, bd_h5,
                        y_data, chip_id, year, region_id,
                    )
                    n_blocks_done += 1
                except Exception as e:
                    skipped_rows.append({
                        "chip_id": chip_id,
                        "reason" : f"Block {block['block_idx']} write error: {e}",
                    })

            all_csv_rows.extend(csv_rows)
            n_chips_done += 1

        finally:
            s2_h5.close()
            ca_h5.close()
            cd_h5.close()
            ba_h5.close()
            bd_h5.close()

    df_csv     = pd.DataFrame(all_csv_rows)
    df_skipped = pd.DataFrame(skipped_rows)

    df_csv.to_csv(csv_path, index=False)
    df_skipped.to_csv(skipped_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Year             : {year}")
    print(f"Chips processed  : {n_chips_done:,}")
    print(f"Chips skipped    : {n_skipped:,}")
    print(f"Blocks written   : {n_blocks_done:,}")
    print(f"CSV rows         : {len(df_csv):,}")
    print(f"Zarr s2 shape    : {store['s2'].shape}")
    print(f"Zarr s1 shape    : {store['s1'].shape}")
    print(f"Zarr y  shape    : {store['y'].shape}")
    print(f"Match CSV        : {csv_path}")
    print(f"Skipped CSV      : {skipped_csv}")
    print(f"{'='*60}")

    return store


if __name__ == "__main__":

    YEAR = 2018
    OUT  = "/export/students/aryal/combine_s1_s2"

    store = run_full_pipeline(
        year       = YEAR,
        s2_root    = f"/export/students/aryal/s2_h5/{YEAR}",
        ca_root    = f"/export/students/aryal/coherence_h5_ascending/{YEAR}",
        cd_root    = f"/export/students/aryal/coherence_h5_descending/{YEAR}",
        ba_root    = f"/export/students/aryal/backscattering_ascending_h5/{YEAR}",
        bd_root    = f"/export/students/aryal/backscattering_descending_h5/{YEAR}",
        label_root = f"/export/students/aryal/Label_Chips_npy_128_from_ref/{YEAR}",
        zarr_path  = f"{OUT}/{YEAR}.zarr",
        out_dir    = OUT,
    )
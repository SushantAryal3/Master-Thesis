from __future__ import annotations

import sys
import gc
import h5py
import json
import re
import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

H5_RE  = re.compile(r"r(\d+)_c(\d+)")
NPY_RE = re.compile(r"loc_r(\d+)_c(\d+)")
def normalise(r: str, c: str) -> str:
    return f"r{int(r)}_c{int(c)}"

def read_ref_meta(path: Path) -> dict:
    with rasterio.open(path) as ds:
        return {
            "crs"      : ds.crs,
            "transform": ds.transform,
            "width"    : ds.width,
            "height"   : ds.height,
        }

def chip_transform_from_offsets(
    ref_transform: Affine,
    row_off: int,
    col_off: int,
) -> Affine:
    return ref_transform * Affine.translation(col_off, row_off)


def chip_ul_xy(
    ref_transform: Affine,
    row_off: int,
    col_off: int,
) -> Tuple[float, float]:
    x, y = ref_transform * (col_off, row_off)
    return float(x), float(y)

def window_ul_xy(transform: Affine, win: Window) -> Tuple[float, float]:
    col, row = int(win.col_off), int(win.row_off)
    x, y = transform * (col, row)
    return float(x), float(y)

def parse_chip_id(chip_id: str) -> Tuple[int, int]:
    m = H5_RE.fullmatch(chip_id)
    if not m:
        raise ValueError(f"Bad chip_id: {chip_id!r}")
    return int(m.group(1)), int(m.group(2))

def load_norm_stats(norm_stats_json: Path) -> dict:
    """
    Loads cached backscatter normalization statistics from a JSON file.
    If the JSON file exists, reads and returns its contents.
    """
    if norm_stats_json.exists():
        with open(norm_stats_json, "r") as f:
            return json.load(f)
    return {"ascending": {}, "descending": {}}

def save_norm_stats(stats: dict, norm_stats_json: Path) -> None:
    """
    Saves backscatter normalization statistics to a JSON file on disk.
    """
    norm_stats_json.parent.mkdir(parents=True, exist_ok=True)
    with open(norm_stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Norm stats saved → {norm_stats_json}")

def get_cached_percentiles(
    orbit: str, year: int, norm_stats_json: Path
) -> Optional[Tuple[float, float]]:
    """
    Retrieves backscatter percentile values for a given orbit and year.
    """
    stats = load_norm_stats(norm_stats_json)
    entry = stats.get(orbit, {}).get(str(year))
    if entry is not None:
        return float(entry["vmin"]), float(entry["vmax"])
    return None

def cache_percentiles(
    orbit: str, year: int, vmin: float, vmax: float,
    norm_stats_json: Path,
    back_p_low: float,
    back_p_high: float,
) -> None:
    stats = load_norm_stats(norm_stats_json)
    stats.setdefault(orbit, {})[str(year)] = {
        "vmin"  : vmin,
        "vmax"  : vmax,
        "p_low" : back_p_low,
        "p_high": back_p_high,
    }
    save_norm_stats(stats, norm_stats_json)

def pass0_percentiles(
    back_root       : Path,
    year            : int,
    orbit           : str,
    norm_stats_json : Path,
    back_p_low      : float,
    back_p_high     : float,
) -> Tuple[float, float]:
    cached = get_cached_percentiles(orbit, year, norm_stats_json)
    if cached is not None:
        vmin, vmax = cached
        print(
            f"\n  [{year}] Pass 0 [{orbit}] — CACHED\n"
            f"    p{back_p_low}  = {vmin:.6f}\n"
            f"    p{back_p_high} = {vmax:.6f}"
        )
        return vmin, vmax

    all_chips = sorted(back_root.glob("r*.h5"))
    if not all_chips:
        raise FileNotFoundError(f"No backscatter H5 files in {back_root}")
    print(
        f"\n  [{year}] Pass 0 [{orbit}] — scanning {len(all_chips):,} chips ..."
    )
    rng       = np.random.default_rng(42)
    reservoir = []

    for chip_path in tqdm(
        all_chips,
        desc          = f"  [{year}] {orbit} scan",
        unit          = "chip",
        dynamic_ncols = True,
    ):
        try:
            with h5py.File(chip_path, "r") as f:
                data = f["X"][:]
        except Exception as e:
            tqdm.write(f"    ⚠ skip {chip_path.name}: {e}")
            continue

        flat = data.ravel().astype(np.float32)
        del data

        flat = flat[np.isfinite(flat)]

        if flat.size == 0:
            del flat
            continue

        reservoir.append(flat)
        del flat

    if not reservoir:
        raise RuntimeError(f"[{year}] No valid backscatter pixels found.")

    pool = np.concatenate(reservoir)
    del reservoir
    gc.collect()

    vmin = float(np.percentile(pool, back_p_low))
    vmax = float(np.percentile(pool, back_p_high))
    del pool
    gc.collect()

    print(
        f"  [{year}] {orbit}\n"
        f"    p{back_p_low}  = {vmin:.6f}\n"
        f"    p{back_p_high} = {vmax:.6f}"
    )
    cache_percentiles(orbit, year, vmin, vmax, norm_stats_json, back_p_low, back_p_high)
    return vmin, vmax

def run_pass0(
    year            : int,
    ba_root         : str,
    bd_root         : str,
    norm_stats_json : Path,
    back_p_low      : float,
    back_p_high     : float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    print(f"\n{'='*60}")
    print(f"Pass 0 — Backscatter Normalization  Year={year}")
    print(f"{'='*60}")
    ba_norm = pass0_percentiles(Path(ba_root), year, "ascending", norm_stats_json, back_p_low, back_p_high)
    bd_norm = pass0_percentiles(Path(bd_root), year, "descending", norm_stats_json, back_p_low, back_p_high)
    print(f"\n  Summary year={year}:")
    print(f"    ASC  vmin={ba_norm[0]:.6f}  vmax={ba_norm[1]:.6f}")
    print(f"    DESC vmin={bd_norm[0]:.6f}  vmax={bd_norm[1]:.6f}")
    return ba_norm, bd_norm


def normalize_back(
    arr  : np.ndarray,
    vmin : float,
    vmax : float,
) -> np.ndarray:
    arr = np.clip(arr.astype(np.float32), vmin, vmax)
    return ((arr - vmin) / (vmax - vmin + 1e-9)).astype(np.float32)


def load_dates(h5file) -> List[datetime]:
    return [datetime.strptime(d, "%Y-%m-%d") for d in h5file.attrs["dates"]]

def nearest_single(target, date_list):
    gaps = [(i, d, abs((d - target).days)) for i, d in enumerate(date_list)]
    return min(gaps, key=lambda x: x[2])

def within_window(target, date_list, max_gap, max_ba):
    gaps     = [(i, d, abs((d - target).days)) for i, d in enumerate(date_list)]
    filtered = [x for x in gaps if x[2] <= max_gap]
    return sorted(filtered, key=lambda x: x[2])[:max_ba]

def initialize_zarr(
    zarr_path    : str,
    year         : int,
    ba_vmin      : float,
    ba_vmax      : float,
    bd_vmin      : float,
    bd_vmax      : float,
    ref          : dict,
    T            : int,
    H            : int,
    W            : int,
    max_ba       : int,
    back_p_low   : float,
    back_p_high  : float,
    ref_tif      : Path,
    overwrite    : bool = True,
    sample_chunk : int  = 32,
) -> zarr.Group:
    T_      = T
    B_s2    = 4
    B_s1    = 4
    B_y     = 3
    MAX_BA_ = max_ba
    SC      = sample_chunk

    mode  = "w" if overwrite else "w-"
    store = zarr.open(zarr_path, mode=mode)

    store.create_dataset(
        "s2", shape=(0, T_, B_s2, H, W),
        chunks=(SC, T_, B_s2, H, W), dtype="float32", overwrite=True,
    )
    store.create_dataset(
        "s1", shape=(0, T_, B_s1, H, W),
        chunks=(SC, T_, B_s1, H, W), dtype="float32", overwrite=True,
    )
    store.create_dataset(
        "y", shape=(0, B_y, H, W),
        chunks=(SC, B_y, H, W), dtype="float32", overwrite=True,
    )

    meta = store.require_group("meta")

    meta.create_dataset("chip_id",   shape=(0,), chunks=(SC,), dtype=str, overwrite=True)
    meta.create_dataset("year",      shape=(0,), chunks=(SC,), dtype=int, overwrite=True)
    meta.create_dataset("block_idx", shape=(0,), chunks=(SC,), dtype=int, overwrite=True)
    meta.create_dataset("row_off",  shape=(0,), chunks=(SC,), dtype=np.int32,   overwrite=True)
    meta.create_dataset("col_off",  shape=(0,), chunks=(SC,), dtype=np.int32,   overwrite=True)
    meta.create_dataset("x0",       shape=(0,), chunks=(SC,), dtype=np.float64, overwrite=True)
    meta.create_dataset("y0",       shape=(0,), chunks=(SC,), dtype=np.float64, overwrite=True)
    meta.create_dataset("s2_dates",           shape=(0, T_), chunks=(SC, T_), dtype=str,       overwrite=True)
    meta.create_dataset("s2_cloud_coverages", shape=(0, T_), chunks=(SC, T_), dtype="float32", overwrite=True)
    meta.create_dataset("s2_tiles",           shape=(0, T_), chunks=(SC, T_), dtype=str,       overwrite=True)

    meta.create_dataset("ca_date",     shape=(0, T_), chunks=(SC, T_), dtype=str, overwrite=True)
    meta.create_dataset("ca_gap_days", shape=(0, T_), chunks=(SC, T_), dtype=int, overwrite=True)
    meta.create_dataset("ca_s2_tile",  shape=(0, T_), chunks=(SC, T_), dtype=str, overwrite=True)

    meta.create_dataset("cd_date",     shape=(0, T_), chunks=(SC, T_), dtype=str, overwrite=True)
    meta.create_dataset("cd_gap_days", shape=(0, T_), chunks=(SC, T_), dtype=int, overwrite=True)
    meta.create_dataset("cd_s2_tile",  shape=(0, T_), chunks=(SC, T_), dtype=str, overwrite=True)

    meta.create_dataset("ba_dates",    shape=(0, T_, MAX_BA_), chunks=(SC, T_, MAX_BA_), dtype=str, overwrite=True)
    meta.create_dataset("ba_n_frames", shape=(0, T_),          chunks=(SC, T_),          dtype=int, overwrite=True)
    meta.create_dataset("ba_max_gap",  shape=(0, T_),          chunks=(SC, T_),          dtype=int, overwrite=True)
    meta.create_dataset("ba_gap_days", shape=(0, T_, MAX_BA_), chunks=(SC, T_, MAX_BA_), dtype=int, overwrite=True)
    meta.create_dataset("ba_s2_tile",  shape=(0, T_),          chunks=(SC, T_),          dtype=str, overwrite=True)

    meta.create_dataset("bd_dates",    shape=(0, T_, MAX_BA_), chunks=(SC, T_, MAX_BA_), dtype=str, overwrite=True)
    meta.create_dataset("bd_n_frames", shape=(0, T_),          chunks=(SC, T_),          dtype=int, overwrite=True)
    meta.create_dataset("bd_max_gap",  shape=(0, T_),          chunks=(SC, T_),          dtype=int, overwrite=True)
    meta.create_dataset("bd_gap_days", shape=(0, T_, MAX_BA_), chunks=(SC, T_, MAX_BA_), dtype=int, overwrite=True)
    meta.create_dataset("bd_s2_tile",  shape=(0, T_),          chunks=(SC, T_),          dtype=str, overwrite=True)

    meta.create_dataset("label_path",  shape=(0,), chunks=(SC,), dtype=str, overwrite=True)

    store.attrs["s2_band_names"]    = ["B2", "B3", "B4", "B8"]
    store.attrs["s1_channel_names"] = ["cohe_asc", "cohe_desc", "back_asc", "back_desc"]
    store.attrs["y_channel_names"]  = ["extent", "boundary", "distance"]
    store.attrs["year"]             = year
    store.attrs["T"]                = T_
    store.attrs["H"]                = H
    store.attrs["W"]                = W
    store.attrs["max_ba_frames"]    = MAX_BA_
    store.attrs["sample_chunk"]     = SC
    store.attrs["back_p_low"]       = back_p_low
    store.attrs["back_p_high"]      = back_p_high
    store.attrs["ba_vmin"]          = ba_vmin
    store.attrs["ba_vmax"]          = ba_vmax
    store.attrs["bd_vmin"]          = bd_vmin
    store.attrs["bd_vmax"]          = bd_vmax
    store.attrs["back_norm"]        = "percentile_clip_minmax → [0,1]"
    store.attrs["cohe_norm"]        = "none (raw)"
    store.attrs["created_by"]       = "s1_s2_pipeline"
    store.attrs["crs_wkt"]   = ref["crs"].to_wkt()
    store.attrs["transform"] = tuple(map(float, ref["transform"]))
    store.attrs["width"]     = int(ref["width"])
    store.attrs["height"]    = int(ref["height"])
    store.attrs["chip"]      = H
    store.attrs["ref_tif"]   = str(ref_tif)

    print(f"\nZarr initialized → {zarr_path}")
    print(f"  s2     : {store['s2'].shape}  chunks={store['s2'].chunks}")
    print(f"  s1     : {store['s1'].shape}  chunks={store['s1'].chunks}")
    print(f"  y      : {store['y'].shape}   chunks={store['y'].chunks}")
    print(f"  sample_chunk = {SC}  (≈{2400//SC} files per array)")
    print(f"  BA  vmin={ba_vmin:.6f}  vmax={ba_vmax:.6f}")
    print(f"  BD  vmin={bd_vmin:.6f}  vmax={bd_vmax:.6f}")

    return store


def build_block_index_with_csv(
    s2_h5, ca_h5, cd_h5, ba_h5, bd_h5,
    chip_id: str,
    year   : int,
    T      : int,
    max_gap: int,
    max_ba : int,
):
    s2_dates     = load_dates(s2_h5)
    ca_dates     = load_dates(ca_h5)
    cd_dates     = load_dates(cd_h5)
    ba_dates     = load_dates(ba_h5)
    bd_dates     = load_dates(bd_h5)

    s2_tiles_all   = list(s2_h5.attrs["tiles"])
    s2_cloud_all   = list(s2_h5.attrs["cloud_coverages"])
    s2_folders_all = list(s2_h5.attrs["folders"])

    ca_raw_paths_all = list(ca_h5.attrs["cohe_paths"])
    cd_raw_paths_all = list(cd_h5.attrs["cohe_paths"])
    ba_raw_paths_all = list(ba_h5.attrs["angle_paths"])
    bd_raw_paths_all = list(bd_h5.attrs["angle_paths"])

    s2_h5_file = s2_h5.filename
    ca_h5_file = ca_h5.filename
    cd_h5_file = cd_h5.filename
    ba_h5_file = ba_h5.filename
    bd_h5_file = bd_h5.filename

    n_blocks      = len(s2_dates) // T
    blocks        = []
    csv_rows      = []
    skipped_blocks = []

    for block_idx in range(n_blocks):
        s2_block = s2_dates[block_idx * T : (block_idx + 1) * T]
        rows     = []
        temp_csv = []
        skip     = False

        for t, s2_dt in enumerate(s2_block):
            s2_idx    = block_idx * T + t
            s2_tile   = str(s2_tiles_all[s2_idx])
            s2_cloud  = float(s2_cloud_all[s2_idx])
            s2_folder = str(s2_folders_all[s2_idx])

            ca_idx, ca_dt, ca_gap = nearest_single(s2_dt, ca_dates)
            cd_idx, cd_dt, cd_gap = nearest_single(s2_dt, cd_dates)
            ba_frames             = within_window(s2_dt, ba_dates, max_gap, max_ba)
            bd_frames             = within_window(s2_dt, bd_dates, max_gap, max_ba)

            if (ca_gap > max_gap or cd_gap > max_gap
                    or len(ba_frames) == 0 or len(bd_frames) == 0):

                if ca_gap > max_gap:
                    reason = f"ca_gap too large ({ca_gap} days > MAX_GAP={max_gap})"
                elif cd_gap > max_gap:
                    reason = f"cd_gap too large ({cd_gap} days > MAX_GAP={max_gap})"
                elif len(ba_frames) == 0:
                    reason = f"no ba frames within {max_gap} days of s2_date"
                else:
                    reason = f"no bd frames within {max_gap} days of s2_date"

                skipped_blocks.append({
                    "chip_id"        : chip_id,
                    "year"           : year,
                    "block_idx"      : block_idx,
                    "t"              : t,

                    "s2_date"        : s2_dt.strftime("%Y-%m-%d"),
                    "s2_tile"        : s2_tile,
                    "s2_cloud"       : s2_cloud,
                    "s2_folder"      : s2_folder,
                    "s2_h5_file"     : s2_h5_file,

                    "ca_date"        : ca_dt.strftime("%Y-%m-%d"),
                    "ca_gap_days"    : ca_gap,
                    "ca_within_gap"  : ca_gap <= max_gap,
                    "ca_raw_path"    : str(ca_raw_paths_all[ca_idx]),
                    "ca_h5_file"     : ca_h5_file,

                    "cd_date"        : cd_dt.strftime("%Y-%m-%d"),
                    "cd_gap_days"    : cd_gap,
                    "cd_within_gap"  : cd_gap <= max_gap,
                    "cd_raw_path"    : str(cd_raw_paths_all[cd_idx]),
                    "cd_h5_file"     : cd_h5_file,

                    "ba_n_frames"    : len(ba_frames),
                    "ba_dates"       : str([x[1].strftime("%Y-%m-%d") for x in ba_frames]),
                    "ba_gaps"        : str([x[2] for x in ba_frames]),
                    "ba_h5_file"     : ba_h5_file,

                    "bd_n_frames"    : len(bd_frames),
                    "bd_dates"       : str([x[1].strftime("%Y-%m-%d") for x in bd_frames]),
                    "bd_gaps"        : str([x[2] for x in bd_frames]),
                    "bd_h5_file"     : bd_h5_file,

                    "reason"         : reason,
                })

                skip = True
                del ba_frames, bd_frames
                break

            ca_raw_path  = str(ca_raw_paths_all[ca_idx])
            cd_raw_path  = str(cd_raw_paths_all[cd_idx])
            ba_raw_paths = str([str(ba_raw_paths_all[x[0]]) for x in ba_frames])
            bd_raw_paths = str([str(bd_raw_paths_all[x[0]]) for x in bd_frames])

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

            temp_csv.append({
                "chip"              : chip_id,
                "year"              : year,
                "block_idx"         : block_idx,
                "t"                 : t,

                "s2_date"           : s2_dt.strftime("%Y-%m-%d"),
                "s2_tile"           : s2_tile,
                "s2_cloud_coverage" : s2_cloud,
                "s2_folder"         : s2_folder,
                "s2_h5_file"        : s2_h5_file,

                "ca_idx"            : ca_idx,
                "ca_date"           : ca_dt.strftime("%Y-%m-%d"),
                "ca_gap_days"       : ca_gap,
                "ca_s2_tile"        : s2_tile,
                "ca_raw_path"       : ca_raw_path,
                "ca_h5_file"        : ca_h5_file,

                "cd_idx"            : cd_idx,
                "cd_date"           : cd_dt.strftime("%Y-%m-%d"),
                "cd_gap_days"       : cd_gap,
                "cd_s2_tile"        : s2_tile,
                "cd_raw_path"       : cd_raw_path,
                "cd_h5_file"        : cd_h5_file,

                "ba_n_frames"       : len(ba_frames),
                "ba_indices"        : str([x[0] for x in ba_frames]),
                "ba_dates"          : str([x[1].strftime("%Y-%m-%d") for x in ba_frames]),
                "ba_gaps"           : str([x[2] for x in ba_frames]),
                "ba_s2_tile"        : s2_tile,
                "ba_raw_paths"      : ba_raw_paths,
                "ba_h5_file"        : ba_h5_file,

                "bd_n_frames"       : len(bd_frames),
                "bd_indices"        : str([x[0] for x in bd_frames]),
                "bd_dates"          : str([x[1].strftime("%Y-%m-%d") for x in bd_frames]),
                "bd_gaps"           : str([x[2] for x in bd_frames]),
                "bd_s2_tile"        : s2_tile,
                "bd_raw_paths"      : bd_raw_paths,
                "bd_h5_file"        : bd_h5_file,
            })

            del ba_frames, bd_frames
        if not skip and len(rows) == T:
            blocks.append({"block_idx": block_idx, "rows": rows})
            csv_rows.extend(temp_csv)
        del rows, temp_csv, s2_block
    del s2_dates, ca_dates, cd_dates, ba_dates, bd_dates
    del s2_tiles_all, s2_cloud_all, s2_folders_all
    del ca_raw_paths_all, cd_raw_paths_all
    del ba_raw_paths_all, bd_raw_paths_all
    return blocks, csv_rows, skipped_blocks


def write_block(
    store     : zarr.Group,
    block_idx : int,
    rows      : list,
    s2_h5, ca_h5, cd_h5, ba_h5, bd_h5,
    y_data    : np.ndarray,
    chip_id   : str,
    year      : int,
    ba_vmin   : float,
    ba_vmax   : float,
    bd_vmin   : float,
    bd_vmax   : float,
    ref_transform: Affine,
    T         : int,
    H         : int,
    W         : int,
    max_ba    : int,
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

    ba_dates_buf = np.full((T, max_ba), "",  dtype=object)
    ba_gaps_buf  = np.full((T, max_ba), -1,  dtype=int)
    ba_n_frames  = []
    ba_max_gaps  = []
    ba_s2_tiles  = []

    bd_dates_buf = np.full((T, max_ba), "",  dtype=object)
    bd_gaps_buf  = np.full((T, max_ba), -1,  dtype=int)
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
        ba_frames_data = np.stack(
            [ba_h5["X"][i] for i in ba_idx], axis=0
        ).astype(np.float32)
        ba_frames_data = normalize_back(ba_frames_data, ba_vmin, ba_vmax)
        s1_block[t, 2] = np.nanmean(ba_frames_data, axis=0)
        del ba_frames_data
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
        bd_frames_data = np.stack(
            [bd_h5["X"][i] for i in bd_idx], axis=0
        ).astype(np.float32)
        bd_frames_data = normalize_back(bd_frames_data, bd_vmin, bd_vmax)
        s1_block[t, 3] = np.nanmean(bd_frames_data, axis=0)
        del bd_frames_data
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
    row_off, col_off = parse_chip_id(chip_id)
    x0, y0 = chip_ul_xy(ref_transform, row_off, col_off)
    meta["row_off"].append([row_off])
    meta["col_off"].append([col_off])
    meta["x0"].append([x0])
    meta["y0"].append([y0])
    meta["label_path"].append([""])

    del s2_block, s1_block
    del ba_dates_buf, ba_gaps_buf
    del bd_dates_buf, bd_gaps_buf


class IncrementalCSV:
    def __init__(self, path: Path):
        self.path           = path
        self.header_written = False

    def write(self, rows: list) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        if not self.header_written:
            df.to_csv(self.path, index=False, mode="w")
            self.header_written = True
        else:
            df.to_csv(self.path, index=False, mode="a", header=False)
        del df


def process_chip(
    chip_id       : str,
    year          : int,
    h5_chips      : dict,
    ca_chips      : dict,
    cd_chips      : dict,
    ba_chips      : dict,
    bd_chips      : dict,
    npy_files     : dict,
    store         : zarr.Group,
    ba_vmin       : float,
    ba_vmax       : float,
    bd_vmin       : float,
    bd_vmax       : float,
    csv_writer    : IncrementalCSV,
    skipped_writer: IncrementalCSV,
    n_chips_done  : list,
    n_blocks_done : list,
    n_skipped     : list,
    pbar,
    ref_transform : Affine,
    T             : int,
    H             : int,
    W             : int,
    max_gap       : int,
    max_ba        : int,
):
    s2_path  = h5_chips.get(chip_id)
    ca_path  = ca_chips.get(chip_id)
    cd_path  = cd_chips.get(chip_id)
    ba_path  = ba_chips.get(chip_id)
    bd_path  = bd_chips.get(chip_id)
    lbl_path = npy_files.get(chip_id)

    def _skip(reason: str, status: str):
        skipped_writer.write([{"chip_id": chip_id, "reason": reason}])
        n_skipped[0] += 1
        pbar.set_postfix(done=n_chips_done[0], blocks=n_blocks_done[0],
                         skipped=n_skipped[0], status=status)
    missing = [
        name for name, path in [
            ("S2",  s2_path),
            ("CA",  ca_path),
            ("CD",  cd_path),
            ("BA",  ba_path),
            ("BD",  bd_path),
            ("LBL", lbl_path),
        ]
        if path is None or not path.exists()
    ]
    if missing:
        _skip(f"Missing: {missing}", f"SKIP(missing {missing})")
        pbar.update(1)
        return
    y_data = None
    try:
        y_data = np.load(lbl_path)[:3].astype(np.float32)
    except Exception as e:
        _skip(f"Label error: {e}", "SKIP(label err)")
        pbar.update(1)
        return

    s2_h5 = ca_h5 = cd_h5 = ba_h5 = bd_h5 = None
    blocks        = None
    csv_rows      = None
    skipped_rows  = None

    try:
        s2_h5 = h5py.File(s2_path, "r")
        ca_h5 = h5py.File(ca_path, "r")
        cd_h5 = h5py.File(cd_path, "r")
        ba_h5 = h5py.File(ba_path, "r")
        bd_h5 = h5py.File(bd_path, "r")

        blocks, csv_rows, skipped_rows = build_block_index_with_csv(
            s2_h5, ca_h5, cd_h5, ba_h5, bd_h5,
            chip_id, year, T=T, max_gap=max_gap, max_ba=max_ba,
        )

        for row in csv_rows:
            row["status"] = "selected"
            row["reason"] = ""

        for row in skipped_rows:
            row["status"] = "rejected"

        csv_writer.write(csv_rows + skipped_rows)

        if not blocks:
            _skip("No valid blocks", "SKIP(no blocks)")
            return

        for block in blocks:
            try:
                write_block(
                    store      = store,
                    block_idx  = block["block_idx"],
                    rows       = block["rows"],
                    s2_h5      = s2_h5,
                    ca_h5      = ca_h5,
                    cd_h5      = cd_h5,
                    ba_h5      = ba_h5,
                    bd_h5      = bd_h5,
                    y_data     = y_data,
                    chip_id    = chip_id,
                    year       = year,
                    ba_vmin    = ba_vmin,
                    ba_vmax    = ba_vmax,
                    bd_vmin    = bd_vmin,
                    bd_vmax    = bd_vmax,
                    ref_transform = ref_transform,
                    T = T, H = H, W = W, max_ba = max_ba,
                )
                n_blocks_done[0] += 1
            except Exception as e:
                skipped_writer.write([{
                    "chip_id": chip_id,
                    "reason" : f"Block {block['block_idx']} write error: {e}",
                }])
            finally:
                del block

        n_chips_done[0] += 1
        pbar.set_postfix(
            done    = n_chips_done[0],
            blocks  = n_blocks_done[0],
            skipped = n_skipped[0],
            status  = f"OK({len(blocks)}blk)",
        )

    except Exception as e:
        _skip(f"H5 error: {e}", "SKIP(H5 err)")

    finally:
        for f in [s2_h5, ca_h5, cd_h5, ba_h5, bd_h5]:
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

        if y_data        is not None: del y_data
        if blocks        is not None: del blocks
        if csv_rows      is not None: del csv_rows
        if skipped_rows  is not None: del skipped_rows
        gc.collect()

        pbar.update(1)


def run_full_pipeline(
    year             : int,
    s2_root          : str,
    ca_root          : str,
    cd_root          : str,
    ba_root          : str,
    bd_root          : str,
    label_root       : str,
    zarr_path        : str,
    out_dir          : str,
    ref_tif          : str,
    norm_stats_json  : str,
    T                : int,
    max_gap          : int,
    max_ba           : int,
    H                : int,
    W                : int,
    back_p_low       : float,
    back_p_high      : float,
    flush_every      : int = 100,
    sample_chunk     : int = 64,
):
    s2_root         = Path(s2_root)
    ca_root         = Path(ca_root)
    cd_root         = Path(cd_root)
    ba_root         = Path(ba_root)
    bd_root         = Path(bd_root)
    label_root      = Path(label_root)
    out_dir         = Path(out_dir)
    ref_tif         = Path(ref_tif)
    norm_stats_json = Path(norm_stats_json)
    out_dir.mkdir(parents=True, exist_ok=True)

    ba_norm, bd_norm = run_pass0(
        year, str(ba_root), str(bd_root),
        norm_stats_json = norm_stats_json,
        back_p_low      = back_p_low,
        back_p_high     = back_p_high,
    )
    ba_vmin, ba_vmax = ba_norm
    bd_vmin, bd_vmax = bd_norm
    print("\n[ref] reading reference TIF for CRS + transform ...")
    ref = read_ref_meta(ref_tif)
    print(f"  CRS        : {ref['crs'].to_string()}")
    print(f"  Pixel size : {ref['transform'].a} × {-ref['transform'].e} m")
    print(f"  Size       : {ref['width']} × {ref['height']}")

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

    ca_chips = {}
    for p in sorted(ca_root.glob("r*.h5")):
        m = H5_RE.fullmatch(p.stem)
        if m:
            ca_chips[normalise(m.group(1), m.group(2))] = p

    cd_chips = {}
    for p in sorted(cd_root.glob("r*.h5")):
        m = H5_RE.fullmatch(p.stem)
        if m:
            cd_chips[normalise(m.group(1), m.group(2))] = p

    ba_chips = {}
    for p in sorted(ba_root.glob("r*.h5")):
        m = H5_RE.fullmatch(p.stem)
        if m:
            ba_chips[normalise(m.group(1), m.group(2))] = p

    bd_chips = {}
    for p in sorted(bd_root.glob("r*.h5")):
        m = H5_RE.fullmatch(p.stem)
        if m:
            bd_chips[normalise(m.group(1), m.group(2))] = p
    matched_chips = [
        k for k in h5_chips
        if k in npy_files
    ]

    print(f"\n{'='*60}")
    print(f"Year                     : {year}")
    print(f"S2 chips total           : {len(h5_chips):,}")
    print(f"Label files              : {len(npy_files):,}")
    print(f"CA chips                 : {len(ca_chips):,}")
    print(f"CD chips                 : {len(cd_chips):,}")
    print(f"BA chips                 : {len(ba_chips):,}")
    print(f"BD chips                 : {len(bd_chips):,}")
    print(f"After region+label filter: {len(matched_chips):,}")
    print(f"BA norm : vmin={ba_vmin:.6f}  vmax={ba_vmax:.6f}")
    print(f"BD norm : vmin={bd_vmin:.6f}  vmax={bd_vmax:.6f}")
    print(f"{'='*60}\n")

    store = initialize_zarr(
        zarr_path    = zarr_path,
        year         = year,
        ba_vmin      = ba_vmin,
        ba_vmax      = ba_vmax,
        bd_vmin      = bd_vmin,
        bd_vmax      = bd_vmax,
        overwrite    = True,
        ref          = ref,
        T            = T,
        H            = H,
        W            = W,
        max_ba       = max_ba,
        back_p_low   = back_p_low,
        back_p_high  = back_p_high,
        ref_tif      = ref_tif,
        sample_chunk = sample_chunk,
    )

    csv_writer     = IncrementalCSV(out_dir / f"match_stats_{year}.csv")
    skipped_writer = IncrementalCSV(out_dir / f"skipped_{year}.csv")

    n_chips_done  = [0]
    n_blocks_done = [0]
    n_skipped     = [0]

    with tqdm(
        total         = len(matched_chips),
        desc          = f"Year {year}",
        unit          = "chip",
        dynamic_ncols = True,
    ) as pbar:

        for i, chip_id in enumerate(matched_chips):

            process_chip(
                chip_id        = chip_id,
                year           = year,
                h5_chips       = h5_chips,
                ca_chips       = ca_chips,
                cd_chips       = cd_chips,
                ba_chips       = ba_chips,
                bd_chips       = bd_chips,
                npy_files      = npy_files,
                store          = store,
                ba_vmin        = ba_vmin,
                ba_vmax        = ba_vmax,
                bd_vmin        = bd_vmin,
                bd_vmax        = bd_vmax,
                csv_writer     = csv_writer,
                skipped_writer = skipped_writer,
                n_chips_done   = n_chips_done,
                n_blocks_done  = n_blocks_done,
                n_skipped      = n_skipped,
                pbar           = pbar,
                ref_transform  = ref['transform'],
                T              = T,
                H              = H,
                W              = W,
                max_gap        = max_gap,
                max_ba         = max_ba,
            )

            if (i + 1) % flush_every == 0:
                gc.collect()
                tqdm.write(
                    f"  [flush] chip {i+1}/{len(matched_chips)}  "
                    f"blocks={n_blocks_done[0]:,}  "
                    f"s2={store['s2'].shape}  "
                    f"s1={store['s1'].shape}"
                )

    zarr.consolidate_metadata(zarr_path)

    disk_gb = sum(
        f.stat().st_size for f in Path(zarr_path).rglob("*") if f.is_file()
    ) / 1e9

    print(f"\n{'='*60}")
    print(f"Year             : {year}")
    print(f"Chips processed  : {n_chips_done[0]:,}")
    print(f"Chips skipped    : {n_skipped[0]:,}")
    print(f"Blocks written   : {n_blocks_done[0]:,}")
    print(f"Zarr s2 shape    : {store['s2'].shape}")
    print(f"Zarr s1 shape    : {store['s1'].shape}")
    print(f"Zarr y  shape    : {store['y'].shape}")
    print(f"Disk size        : {disk_gb:.2f} GB")
    print(f"Match CSV        : {csv_writer.path}")
    print(f"Skipped CSV      : {skipped_writer.path}")
    print(f"{'='*60}")

    print(f"\n  Verifying first sample ...")
    z = zarr.open(zarr_path, mode="r")
    if z["s2"].shape[0] == 0:
        print("  ⚠ WARNING: Zarr store is empty — 0 blocks written!")
        print(f"    Check: {skipped_writer.path}")
    else:
        s2_ = z["s2"][0]
        s1_ = z["s1"][0]
        y_  = z["y"][0]
        print(f"    s2[0] shape           : {s2_.shape}")
        print(f"    s1[0] cohe_asc  range : [{np.nanmin(s1_[:,0]):.4f}, {np.nanmax(s1_[:,0]):.4f}]  (raw)")
        print(f"    s1[0] cohe_desc range : [{np.nanmin(s1_[:,1]):.4f}, {np.nanmax(s1_[:,1]):.4f}]  (raw)")
        print(f"    s1[0] back_asc  range : [{np.nanmin(s1_[:,2]):.4f}, {np.nanmax(s1_[:,2]):.4f}]  (expect ≈[0,1])")
        print(f"    s1[0] back_desc range : [{np.nanmin(s1_[:,3]):.4f}, {np.nanmax(s1_[:,3]):.4f}]  (expect ≈[0,1])")
        print(f"    y[0] extent     range : [{np.nanmin(y_[0]):.4f}, {np.nanmax(y_[0]):.4f}]")
        print(f"    y[0] boundary   range : [{np.nanmin(y_[1]):.4f}, {np.nanmax(y_[1]):.4f}]")
        print(f"    y[0] distance   range : [{np.nanmin(y_[2]):.4f}, {np.nanmax(y_[2]):.4f}]")
        print(f"  ✓ Verification complete")
        return store


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python combineS1_S2.py YEAR [YEAR ...]")
        sys.exit(1)
    YEARS = [int(a) for a in sys.argv[1:]]
    OUT   = "/globalsc/ucl/elia/aryal/combine_zarr_time_4_s2_VV_test_2020"
    print(f"[main] Processing years: {YEARS}")
    for YEAR in YEARS:
        print(f"\n{'#'*60}")
        print(f"# Starting year {YEAR}")
        print(f"{'#'*60}\n")
        try:
            store = run_full_pipeline(
                year             = YEAR,
                s2_root          = f"/globalsc/ucl/elia/aryal/s2_h5_with_clouds/h5/{YEAR}",
                ca_root          = f"/globalsc/ucl/elia/aryal/S1_dataset/coherence/aescending/{YEAR}",
                cd_root          = f"/globalsc/ucl/elia/aryal/S1_dataset/coherence/descending/{YEAR}",
                ba_root          = f"/globalsc/ucl/elia/aryal/S1_dataset/backscattering_ascending_h5/{YEAR}",
                bd_root          = f"/globalsc/ucl/elia/aryal/S1_dataset/backscattering_descending_h5/{YEAR}",
                label_root       = f"/globalsc/ucl/elia/aryal/Label_Chips_npy_128_from_ref/{YEAR}",
                zarr_path        = f"{OUT}/{YEAR}.zarr",
                out_dir          = OUT,
                ref_tif          = "/globalsc/ucl/elia/aryal/WALLONIA_2018-07_8_median_trim.tif",
                norm_stats_json  = "/globalsc/ucl/elia/aryal/S1_S2_Combined/norm_stats.json",
                T                = 4,
                max_gap          = 15,
                max_ba           = 3,
                H                = 128,
                W                = 128,
                back_p_low       = 2.0,
                back_p_high      = 98.0,
                sample_chunk     = 64,
            )
            print(f"\n[main] ✓ Year {YEAR} complete")
        except Exception as e:
            print(f"\n[main] ✗ Year {YEAR} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
    print(f"\n[main] All years done: {YEARS}")
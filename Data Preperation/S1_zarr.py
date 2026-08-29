from __future__ import annotations

import h5py
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
import gc
import json
from pathlib import Path
from tqdm import tqdm
import re
from typing import Dict
import shutil
import numpy as np
import zarr
from numcodecs import Blosc
import sys
import rasterio
from rasterio.transform import Affine

T        = 10
MAX_GAP  = 6
SAMPLE_CHUNK = 64
BACK_P_LOW       = 2.0
BACK_P_HIGH      = 98.0
MAX_PIX_PER_CHIP = 100_000

def load_dates(h5file: h5py.File) -> List[datetime]:
    return [datetime.strptime(d, "%Y-%m-%d") for d in h5file.attrs["dates"]]


def build_date_index(date_list: List[datetime]) -> dict:
    return {dt.strftime("%Y-%m-%d"): i for i, dt in enumerate(date_list)}


def nearest_within_gap(
    target    : datetime,
    date_list : List[datetime],
    max_gap   : int,
) -> Optional[Tuple[int, datetime, int]]:
    best = None
    for i, dt in enumerate(date_list):
        gap = abs((dt - target).days)
        if gap <= max_gap:
            if best is None or gap < best[2]:
                best = (i, dt, gap)
    return best


def build_block_index(
    cd_h5  : h5py.File,
    ca_h5  : h5py.File,
    ba_h5  : h5py.File,
    bd_h5  : h5py.File,
    chip_id: str,
    year   : int,
    T      : int = T,
    max_gap: int = MAX_GAP,
) -> Tuple[List[dict], List[dict]]:
    cd_dates = load_dates(cd_h5)
    ca_dates = load_dates(ca_h5)
    ba_dates = load_dates(ba_h5)
    bd_dates = load_dates(bd_h5)
    ba_date_index = build_date_index(ba_dates)
    bd_date_index = build_date_index(bd_dates)
    n_blocks      = len(cd_dates) // T
    valid_blocks  : List[dict] = []
    skipped_blocks: List[dict] = []
    for block_idx in range(n_blocks):
        anchor_block = cd_dates[block_idx * T : (block_idx + 1) * T]
        rows: List[dict] = []
        skip     = False
        skip_rec : dict = {}
        for t, anchor_dt in enumerate(anchor_block):
            cd_idx      = block_idx * T + t
            cd_date_str = anchor_dt.strftime("%Y-%m-%d")

            bd_idx = bd_date_index.get(cd_date_str)
            if bd_idx is None:
                skip_rec = {
                    "chip_id"  : chip_id, "year": year,
                    "block_idx": block_idx, "t": t,
                    "cd_date"  : cd_date_str,
                    "reason"   : f"back_desc has no exact match for anchor date {cd_date_str}",
                }
                skip = True
                break
            ca_match = nearest_within_gap(anchor_dt, ca_dates, max_gap)
            if ca_match is None:
                skip_rec = {
                    "chip_id"  : chip_id, "year": year,
                    "block_idx": block_idx, "t": t,
                    "cd_date"  : cd_date_str,
                    "reason"   : f"cohe_asc has no date within {max_gap}d of {cd_date_str}",
                }
                skip = True
                break
            ca_idx, ca_dt, ca_gap = ca_match
            ca_date_str = ca_dt.strftime("%Y-%m-%d")
            ba_idx = ba_date_index.get(ca_date_str)
            if ba_idx is None:
                skip_rec = {
                    "chip_id"  : chip_id, "year": year,
                    "block_idx": block_idx, "t": t,
                    "cd_date"  : cd_date_str,
                    "reason"   : f"back_asc has no exact match for ca_date {ca_date_str}",
                }
                skip = True
                break
            rows.append({
                "t"          : t,
                "cd_idx"     : cd_idx,
                "cd_date"    : cd_date_str,
                "bd_idx"     : bd_idx,
                "bd_date"    : cd_date_str,
                "ca_idx"     : ca_idx,
                "ca_date"    : ca_date_str,
                "ca_gap_days": ca_gap,
                "ba_idx"     : ba_idx,
                "ba_date"    : ca_date_str,
            })
        if skip:
            skipped_blocks.append(skip_rec)
        elif len(rows) == T:
            valid_blocks.append({"block_idx": block_idx, "rows": rows})
    return valid_blocks, skipped_blocks


def load_norm_stats(norm_stats_json: Path) -> dict:
    if norm_stats_json.exists():
        with open(norm_stats_json, "r") as fh:
            return json.load(fh)
    return {"ascending": {}, "descending": {}}


def save_norm_stats(stats: dict, norm_stats_json: Path) -> None:
    norm_stats_json.parent.mkdir(parents=True, exist_ok=True)
    with open(norm_stats_json, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  ✓ norm_stats saved → {norm_stats_json}")


def get_cached_percentiles(
    orbit: str, year: int, norm_stats_json: Path
) -> Optional[Tuple[float, float]]:
    entry = load_norm_stats(norm_stats_json).get(orbit, {}).get(str(year))
    if entry is not None:
        return float(entry["vmin"]), float(entry["vmax"])
    return None


def cache_percentiles(
    orbit: str, year: int, vmin: float, vmax: float, norm_stats_json: Path,
    p_low: float = BACK_P_LOW, p_high: float = BACK_P_HIGH,
) -> None:
    stats = load_norm_stats(norm_stats_json)
    stats.setdefault(orbit, {})[str(year)] = {
        "vmin"  : vmin,
        "vmax"  : vmax,
        "p_low" : p_low,
        "p_high": p_high,
    }
    save_norm_stats(stats, norm_stats_json)


def pass0_percentiles(
    back_root: Path,
    year     : int,
    orbit    : str,
    norm_stats_json : Path,
    p_low    : float = BACK_P_LOW,
    p_high   : float = BACK_P_HIGH,
) -> Tuple[float, float]:
    cached = get_cached_percentiles(orbit, year, norm_stats_json)
    if cached is not None:
        vmin, vmax = cached
        print(
            f"  [{year}][{orbit}] Pass 0 — CACHED\n"
            f"    p{p_low}  = {vmin:.6f}\n"
            f"    p{p_high} = {vmax:.6f}"
        )
        return vmin, vmax
    all_chips = sorted(back_root.glob("r*.h5"))
    if not all_chips:
        raise FileNotFoundError(
            f"No backscatter H5 chips found in {back_root}"
        )
    print(
        f"\n  [{year}][{orbit}] Pass 0 — scanning {len(all_chips):,} chips "
        f"(no cache found) ..."
    )
    rng       = np.random.default_rng(42)
    reservoir = []
    for chip_path in tqdm(
        all_chips,
        desc          = f"  [{year}][{orbit}]",
        unit          = "chip",
        dynamic_ncols = True,
    ):
        try:
            with h5py.File(chip_path, "r") as fh:
                data = fh["X"][:]
        except Exception as exc:
            tqdm.write(f"    ⚠ skip {chip_path.name}: {exc}")
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
        raise RuntimeError(
            f"[{year}][{orbit}] No valid backscatter pixels found in {back_root}"
        )
    pool = np.concatenate(reservoir)
    del reservoir
    gc.collect()
    vmin = float(np.percentile(pool, p_low))
    vmax = float(np.percentile(pool, p_high))
    del pool
    gc.collect()
    print(
        f"  [{year}][{orbit}]\n"
        f"    p{p_low}  = {vmin:.6f}\n"
        f"    p{p_high} = {vmax:.6f}"
    )
    cache_percentiles(orbit, year, vmin, vmax, norm_stats_json, p_low, p_high)
    return vmin, vmax


def run_pass0(
    year    : int,
    ba_root : Path,
    bd_root : Path,
    norm_stats_json : Path,
    p_low   : float = BACK_P_LOW,
    p_high  : float = BACK_P_HIGH,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    print(f"\n{'='*60}")
    print(f"  Pass 0 — Backscatter normalisation  year={year}")
    print(f"{'='*60}")
    ba_norm = pass0_percentiles(ba_root, year, "ascending", norm_stats_json, p_low, p_high)
    bd_norm = pass0_percentiles(bd_root, year, "descending", norm_stats_json, p_low, p_high)
    print(f"\n  Summary year={year}:")
    print(f"    ASC   vmin={ba_norm[0]:.6f}  vmax={ba_norm[1]:.6f}")
    print(f"    DESC  vmin={bd_norm[0]:.6f}  vmax={bd_norm[1]:.6f}")
    return ba_norm, bd_norm


def normalize_back(
    arr : np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    arr = np.clip(arr.astype(np.float32), vmin, vmax)
    return ((arr - vmin) / (vmax - vmin + 1e-9)).astype(np.float32)


H5_RE  = re.compile(r"^r(\d+)_c(\d+)$")
NPY_RE = re.compile(r"loc_r(\d+)_c(\d+)")


def normalise_id(r: str, c: str) -> str:
    """Canonical chip id: r<int>_c<int> with no zero-padding."""
    return f"r{int(r)}_c{int(c)}"


def build_h5_index(root: Path) -> Dict[str, Path]:
    """
    Scan *root* for  r*.h5  files and return  chip_id → Path.
    Works for any of the four H5 directories (cd, ca, ba, bd).
    """
    index = {}
    for p in sorted(root.glob("r*.h5")):
        m = H5_RE.fullmatch(p.stem)
        if m:
            index[normalise_id(m.group(1), m.group(2))] = p
    return index


def build_npy_index(label_root: Path) -> Dict[str, Path]:
    """
    Scan *label_root* for  loc_r*_c*.npy  files and return  chip_id → Path.
    """
    index = {}
    for p in sorted(label_root.glob("loc_r*_c*.npy")):
        m = NPY_RE.search(p.stem)
        if m:
            index[normalise_id(m.group(1), m.group(2))] = p
    return index


def pass1_count_S1(
    cd_root    : Path,
    ca_root    : Path,
    ba_root    : Path,
    bd_root    : Path,
    label_root : Path,
    year       : int,
    T          : int = T,
    max_gap    : int = MAX_GAP,
) -> Tuple[int, List[dict]]:
    cd_index  = build_h5_index(cd_root)
    ca_index  = build_h5_index(ca_root)
    ba_index  = build_h5_index(ba_root)
    bd_index  = build_h5_index(bd_root)
    npy_index = build_npy_index(label_root)

    print(f"\n  [{year}] File counts:")
    print(f"    cohe_desc (anchor) : {len(cd_index):>6,}")
    print(f"    cohe_asc           : {len(ca_index):>6,}")
    print(f"    back_asc           : {len(ba_index):>6,}")
    print(f"    back_desc          : {len(bd_index):>6,}")
    print(f"    labels (.npy)      : {len(npy_index):>6,}")
    common = sorted(
        set(cd_index)
        & set(ca_index)
        & set(ba_index)
        & set(bd_index)
        & set(npy_index)
    )
    print(f"    common (all 5)     : {len(common):>6,}")
    N_total   = 0
    chip_info : List[dict] = []

    skipped_missing  = 0
    skipped_h5_error = 0
    skipped_no_block = 0
    for chip_id in tqdm(
        common,
        desc          = f"  [{year}] pass1 counting",
        unit          = "chip",
        dynamic_ncols = True,
    ):
        cd_path = cd_index[chip_id]
        ca_path = ca_index[chip_id]
        ba_path = ba_index[chip_id]
        bd_path = bd_index[chip_id]

        for p in (cd_path, ca_path, ba_path, bd_path):
            if not p.exists():
                skipped_missing += 1
                continue

        try:
            with (
                h5py.File(cd_path, "r") as cd_h5,
                h5py.File(ca_path, "r") as ca_h5,
                h5py.File(ba_path, "r") as ba_h5,
                h5py.File(bd_path, "r") as bd_h5,
            ):
                valid_blocks, _ = build_block_index(
                    cd_h5, ca_h5, ba_h5, bd_h5,
                    chip_id, year, T=T, max_gap=max_gap,
                )

        except Exception as exc:
            tqdm.write(f"    ⚠ H5 error {chip_id}: {exc}")
            skipped_h5_error += 1
            continue
        n_blocks = len(valid_blocks)
        del valid_blocks
        gc.collect()
        if n_blocks == 0:
            skipped_no_block += 1
            continue
        N_total += n_blocks
        chip_info.append({"chip_id": chip_id, "n_blocks": n_blocks})
    print(f"\n  [{year}] Pass 1 results:")
    print(f"    Chips with ≥1 valid block : {len(chip_info):>6,}")
    print(f"    Skipped — missing file    : {skipped_missing:>6,}")
    print(f"    Skipped — H5 error        : {skipped_h5_error:>6,}")
    print(f"    Skipped — no valid blocks : {skipped_no_block:>6,}")
    print(f"    N_total (blocks)          : {N_total:>6,}")

    return N_total, chip_info


H           = 128
W           = 128
C           = 4
L           = 3
CHANNEL_NAMES = ["cohe_asc", "cohe_desc", "back_asc", "back_desc"]
LABEL_NAMES   = ["extent", "boundary", "dist"]


def init_zarr_S1(
    out_path  : Path,
    N         : int,
    year      : int,
    ba_vmin   : float,
    ba_vmax   : float,
    bd_vmin   : float,
    bd_vmax   : float,
    crs_wkt   : str,
    transform : list,
    chip_size : int,
    T         : int = T,
    H         : int = H,
    W         : int = W,
    C         : int = C,
    L         : int = L,
    sample_chunk : int = SAMPLE_CHUNK,
    back_p_low   : float = BACK_P_LOW,
    back_p_high  : float = BACK_P_HIGH,
    channel_names: list = CHANNEL_NAMES,
    label_names  : list = LABEL_NAMES,
    overwrite : bool = True,
) -> zarr.Group:
    if overwrite and out_path.exists():
        shutil.rmtree(out_path)
        print(f"  Removed existing store: {out_path}")

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    root       = zarr.open_group(str(out_path), mode="w")

    root.create_dataset("X", shape=(N, T, C, H, W),
                        chunks=(sample_chunk, T, C, H, W),
                        dtype=np.float32, compressor=compressor)
    root.create_dataset("Y", shape=(N, L, H, W),
                        chunks=(sample_chunk, L, H, W),
                        dtype=np.float32, compressor=compressor)

    for name, dtype in [
        ("chip_id"  , str),
        ("year"     , np.int32),
        ("block_idx", np.int32),
        ("row_off"  , np.int32),
        ("col_off"  , np.int32),
    ]:
        root.create_dataset(name, shape=(N,), chunks=(4096,),
                            dtype=dtype, compressor=compressor)

    for name, dtype in [
        ("cd_dates"   , str),
        ("ca_dates"   , str),
        ("ba_dates"   , str),
        ("bd_dates"   , str),
        ("ca_gap_days", np.int32),
    ]:
        root.create_dataset(name, shape=(N, T), chunks=(sample_chunk, T),
                            dtype=dtype, compressor=compressor)

    root.attrs.update({
        "year": year, "N": N, "T": T, "C": C, "H": H, "W": W, "L": L,
        "channel_names" : channel_names,
        "label_names"   : label_names,
        "cohe_norm"     : "none (raw)",
        "back_norm"     : "percentile_clip_minmax → [0, 1]",
        "back_p_low"    : back_p_low,
        "back_p_high"   : back_p_high,
        "ba_vmin"       : ba_vmin,   "ba_vmax": ba_vmax,
        "bd_vmin"       : bd_vmin,   "bd_vmax": bd_vmax,
        "anchor"        : "cohe_desc",
        "max_gap_days"  : 8,
        "crs_wkt"       : crs_wkt,
        "transform"     : transform,
        "chip_size"     : chip_size,
        "description": (
            f"SAR combined asc+desc dataset, year={year}. "
            f"X shape=(N,T={T},C={C},H={H},W={W}). "
            f"Channels: {channel_names}. "
            f"row_off/col_off give chip origin in the reference raster. "
            f"CRS and affine transform stored; reconstruct GeoTIFF with rasterio."
        ),
    })

    print(f"\n  [{year}] Zarr initialised → {out_path}")
    print(f"    X      : {root['X'].shape}  chunks={root['X'].chunks}")
    print(f"    Y      : {root['Y'].shape}  chunks={root['Y'].chunks}")
    print(f"    crs    : {crs_wkt[:60]}...")
    print(f"    transform: {transform}")
    return root


def write_block_S1(
    root      : zarr.Group,
    idx       : int,
    rows      : list,
    block_idx : int,
    cd_h5     : h5py.File,
    ca_h5     : h5py.File,
    ba_h5     : h5py.File,
    bd_h5     : h5py.File,
    y_data    : np.ndarray,
    chip_id   : str,
    year      : int,
    ba_vmin   : float,
    ba_vmax   : float,
    bd_vmin   : float,
    bd_vmax   : float,
    row_off   : int,
    col_off   : int,
    T         : int = T,
    C         : int = C,
    H         : int = H,
    W         : int = W,
) -> None:
    X_buf = np.empty((T, C, H, W), dtype=np.float32)
    cd_dates_buf = []; ca_dates_buf = []
    ca_gap_days_buf = []; ba_dates_buf = []; bd_dates_buf = []

    for row in rows:
        t = row["t"]
        X_buf[t, 0] = np.asarray(ca_h5["X"][row["ca_idx"]], dtype=np.float32)
        X_buf[t, 1] = np.asarray(cd_h5["X"][row["cd_idx"]], dtype=np.float32)
        X_buf[t, 2] = normalize_back(np.asarray(ba_h5["X"][row["ba_idx"]], dtype=np.float32), ba_vmin, ba_vmax)
        X_buf[t, 3] = normalize_back(np.asarray(bd_h5["X"][row["bd_idx"]], dtype=np.float32), bd_vmin, bd_vmax)
        cd_dates_buf.append(row["cd_date"]); ca_dates_buf.append(row["ca_date"])
        ca_gap_days_buf.append(row["ca_gap_days"])
        ba_dates_buf.append(row["ba_date"]); bd_dates_buf.append(row["bd_date"])

    root["X"][idx]           = X_buf
    root["Y"][idx]           = y_data
    root["chip_id"][idx]     = chip_id
    root["year"][idx]        = year
    root["block_idx"][idx]   = block_idx
    root["row_off"][idx]     = row_off
    root["col_off"][idx]     = col_off
    root["cd_dates"][idx]    = cd_dates_buf
    root["ca_dates"][idx]    = ca_dates_buf
    root["ca_gap_days"][idx] = ca_gap_days_buf
    root["ba_dates"][idx]    = ba_dates_buf
    root["bd_dates"][idx]    = bd_dates_buf
    del X_buf


def pass2_write_S1(
    root       : zarr.Group,
    chip_info  : List[dict],
    cd_root    : Path,
    ca_root    : Path,
    ba_root    : Path,
    bd_root    : Path,
    label_root : Path,
    year       : int,
    ba_vmin    : float,
    ba_vmax    : float,
    bd_vmin    : float,
    bd_vmax    : float,
    T          : int = T,
    max_gap    : int = MAX_GAP,
) -> None:
    cd_index  = build_h5_index(cd_root)
    ca_index  = build_h5_index(ca_root)
    ba_index  = build_h5_index(ba_root)
    bd_index  = build_h5_index(bd_root)
    npy_index = build_npy_index(label_root)
    idx = 0; chips_written = 0; blocks_written = 0; chips_errored = 0
    pbar = tqdm(chip_info, desc=f"  [{year}] pass2 writing",
                unit="chip", dynamic_ncols=True)
    for entry in pbar:
        chip_id = entry["chip_id"]
        cd_path = cd_index.get(chip_id);  ca_path = ca_index.get(chip_id)
        ba_path = ba_index.get(chip_id);  bd_path = bd_index.get(chip_id)
        npy_path = npy_index.get(chip_id)
        try:
            y_data = np.load(npy_path)[:3].astype(np.float32)
        except Exception as exc:
            tqdm.write(f"    ⚠ label error {chip_id}: {exc}")
            chips_errored += 1; continue
        try:
            with (
                h5py.File(cd_path, "r") as cd_h5,
                h5py.File(ca_path, "r") as ca_h5,
                h5py.File(ba_path, "r") as ba_h5,
                h5py.File(bd_path, "r") as bd_h5,
            ):
                row_off = int(cd_h5.attrs["row_off"])
                col_off = int(cd_h5.attrs["col_off"])
                valid_blocks, _ = build_block_index(
                    cd_h5, ca_h5, ba_h5, bd_h5, chip_id, year, T=T, max_gap=max_gap)
                for block in valid_blocks:
                    write_block_S1(
                        root=root, idx=idx, rows=block["rows"],
                        block_idx=block["block_idx"],
                        cd_h5=cd_h5, ca_h5=ca_h5, ba_h5=ba_h5, bd_h5=bd_h5,
                        y_data=y_data, chip_id=chip_id, year=year,
                        ba_vmin=ba_vmin, ba_vmax=ba_vmax,
                        bd_vmin=bd_vmin, bd_vmax=bd_vmax,
                        row_off=row_off, col_off=col_off,
                        T=T,
                    )
                    idx += 1; blocks_written += 1

        except Exception as exc:
            tqdm.write(f"    ⚠ H5 error {chip_id}: {exc}")
            chips_errored += 1; continue
        finally:
            del y_data; gc.collect()
        chips_written += 1
        pbar.set_postfix(idx=idx, blocks=blocks_written, errors=chips_errored)

    expected_N = root["X"].shape[0]
    if idx != expected_N:
        print(f"\n  ⚠ WARNING: wrote {idx:,} but pre-allocated {expected_N:,}")
    print(f"\n  [{year}] Pass 2 complete: chips={chips_written:,}  "
          f"blocks={blocks_written:,}  errors={chips_errored:,}")


def disk_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def build_yearS1(
    year: int,
    ca_base: Path,
    cd_base: Path,
    ba_base: Path,
    bd_base: Path,
    label_train_base: Path,
    label_test_base: Path,
    out_base: Path,
    ref_tif: Path,
    norm_stats_json: Path,
    T: int = T,
    max_gap: int = MAX_GAP,
    sample_chunk: int = SAMPLE_CHUNK,
    chip_size: int = H,
    back_p_low: float = BACK_P_LOW,
    back_p_high: float = BACK_P_HIGH,
) -> None:
    print("\n" + "=" * 60)
    print(f"  Building year {year}")
    print("=" * 60)

    ca_root    = ca_base / str(year)
    cd_root    = cd_base / str(year)
    ba_root    = ba_base / str(year)
    bd_root    = bd_base / str(year)
    label_root = (label_train_base) / str(year)
    out_zarr   = out_base / f"{year}.zarr"

    for name, path in [
        ("CA", ca_root), ("CD", cd_root),
        ("BA", ba_root), ("BD", bd_root),
        ("labels", label_root),
    ]:
        if not path.exists():
            print(f"  ✗ Missing [{name}]: {path} — skipping year {year}")
            return

    with rasterio.open(ref_tif) as ds:
        crs_wkt   = ds.crs.to_wkt()
        crs_str   = str(ds.crs)
        tf        = ds.transform
        transform = [tf.a, tf.b, tf.c, tf.d, tf.e, tf.f]

    print(f"  [geo] CRS      : {crs_str}")
    print(f"  [geo] Transform: {transform}")

    (ba_vmin, ba_vmax), (bd_vmin, bd_vmax) = run_pass0(
        year    = year,
        ba_root = ba_root,
        bd_root = bd_root,
        norm_stats_json = norm_stats_json,
        p_low   = back_p_low,
        p_high  = back_p_high,
    )

    print(f"\n  [{year}] Pass 1 — counting valid samples ...")
    N_total, chip_info = pass1_count_S1(
        cd_root    = cd_root,
        ca_root    = ca_root,
        ba_root    = ba_root,
        bd_root    = bd_root,
        label_root = label_root,
        year       = year,
        T          = T,
        max_gap    = max_gap,
    )

    if N_total == 0:
        print(f"  [{year}] No valid samples found — skipping year.")
        return

    print(f"\n  [{year}] Initialising Zarr → {out_zarr}")
    root = init_zarr_S1(
        out_path     = out_zarr,
        N            = N_total,
        year         = year,
        ba_vmin      = ba_vmin,
        ba_vmax      = ba_vmax,
        bd_vmin      = bd_vmin,
        bd_vmax      = bd_vmax,
        crs_wkt      = crs_wkt,
        transform    = transform,
        chip_size    = chip_size,
        T            = T,
        sample_chunk = sample_chunk,
        back_p_low   = back_p_low,
        back_p_high  = back_p_high,
    )

    print(f"\n  [{year}] Pass 2 — writing data ...")
    pass2_write_S1(
        root       = root,
        chip_info  = chip_info,
        cd_root    = cd_root,
        ca_root    = ca_root,
        ba_root    = ba_root,
        bd_root    = bd_root,
        label_root = label_root,
        year       = year,
        ba_vmin    = ba_vmin,
        ba_vmax    = ba_vmax,
        bd_vmin    = bd_vmin,
        bd_vmax    = bd_vmax,
        T          = T,
        max_gap    = max_gap,
    )

    zarr.consolidate_metadata(str(out_zarr))
    print(f"  [{year}] Zarr metadata consolidated")

    print(f"\n  [{year}] Verifying first sample ...")
    z  = zarr.open_group(str(out_zarr), mode="r")
    X0 = z["X"][0]
    Y0 = z["Y"][0]
    print(f"    X[0] shape                : {X0.shape}")
    print(f"    X[0] cohe_asc  range      : [{np.nanmin(X0[:, 0]):.4f}, {np.nanmax(X0[:, 0]):.4f}]  (raw)")
    print(f"    X[0] cohe_desc range      : [{np.nanmin(X0[:, 1]):.4f}, {np.nanmax(X0[:, 1]):.4f}]  (raw)")
    print(f"    X[0] back_asc  range      : [{np.nanmin(X0[:, 2]):.4f}, {np.nanmax(X0[:, 2]):.4f}]  (expect ≈[0,1])")
    print(f"    X[0] back_desc range      : [{np.nanmin(X0[:, 3]):.4f}, {np.nanmax(X0[:, 3]):.4f}]  (expect ≈[0,1])")
    print(f"    Y[0] extent    range      : [{np.nanmin(Y0[0]):.4f},    {np.nanmax(Y0[0]):.4f}]")
    print(f"    Y[0] boundary  range      : [{np.nanmin(Y0[1]):.4f},    {np.nanmax(Y0[1]):.4f}]")
    print(f"    Y[0] dist      range      : [{np.nanmin(Y0[2]):.4f},    {np.nanmax(Y0[2]):.4f}]")
    print(f"    cd_dates[0]               : {list(z['cd_dates'][0])}")
    print(f"    ca_dates[0]               : {list(z['ca_dates'][0])}")
    print(f"    ca_gap_days[0]            : {list(z['ca_gap_days'][0])}")
    print(f"    chip_id[0]                : {z['chip_id'][0]}")
    print(f"    row_off[0]                : {z['row_off'][0]}")
    print(f"    col_off[0]                : {z['col_off'][0]}")
    print(f"    crs_wkt (first 80 chars)  : {z.attrs['crs_wkt'][:80]}...")
    print(f"    transform                 : {z.attrs['transform']}")
    print(f"    chip_size                 : {z.attrs['chip_size']}")
    gb = disk_size_gb(out_zarr)
    print(f"\n  [{year}] ✅ Done")
    print(f"           Zarr path    : {out_zarr}")
    print(f"           X shape      : {z['X'].shape}")
    print(f"           Y shape      : {z['Y'].shape}")
    print(f"           N samples    : {N_total:,}")
    print(f"           Disk size    : {gb:.2f} GB")
    print(f"           BA vmin/vmax : {ba_vmin:.6f} / {ba_vmax:.6f}")
    print(f"           BD vmin/vmax : {bd_vmin:.6f} / {bd_vmax:.6f}")

def buildS1(
    year             : int,
    ca_base          : Path | str = "/globalsc/ucl/elia/aryal/S1_dataset/coherence/aescending",
    cd_base          : Path | str = "/globalsc/ucl/elia/aryal/S1_dataset/coherence/descending",
    ba_base          : Path | str = "/globalsc/ucl/elia/aryal/S1_dataset/backscattering/ascending",
    bd_base          : Path | str = "/globalsc/ucl/elia/aryal/S1_dataset/backscattering/descending",
    label_train_base : Path | str = "/globalsc/ucl/elia/aryal/Label_Chips_npy_128_from_ref",
    label_test_base  : Path | str = "/globalsc/ucl/elia/aryal/Label_Chips_npy_128_from_ref",
    out_base         : Path | str = "/globalsc/ucl/elia/aryal/S1_zarr/combined_t_10",
    ref_tif          : Path | str = "/globalsc/ucl/elia/aryal/WALLONIA_2018-07_8_median_trim.tif",
    norm_stats_json  : Path | str = "/globalsc/ucl/elia/aryal/S1_zarr/combined/norm_stats.json",
    T                : int = T,
    max_gap          : int = MAX_GAP,
    sample_chunk     : int = SAMPLE_CHUNK,
    chip_size        : int = H,
    back_p_low       : float = BACK_P_LOW,
    back_p_high      : float = BACK_P_HIGH,
) -> None:
    ca_base   = Path(ca_base)
    cd_base   = Path(cd_base)
    ba_base   = Path(ba_base)
    bd_base   = Path(bd_base)
    label_train_base = Path(label_train_base)
    label_test_base  = Path(label_test_base)
    out_base  = Path(out_base)
    ref_tif   = Path(ref_tif)
    norm_stats_json = Path(norm_stats_json)
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SAR Combined Asc+Desc Zarr Builder")
    print(f"  Year        : {year}")
    print(f"  CA root     : {ca_base}")
    print(f"  CD root     : {cd_base}")
    print(f"  BA root     : {ba_base}")
    print(f"  BD root     : {bd_base}")
    print(f"  Label train : {label_train_base}")
    print(f"  Label test  : {label_test_base}")
    print(f"  Output      : {out_base}")
    print("=" * 60)

    build_yearS1(
        year              = year,
        ca_base           = ca_base,
        cd_base           = cd_base,
        ba_base           = ba_base,
        bd_base           = bd_base,
        label_train_base  = label_train_base,
        label_test_base   = label_test_base,
        out_base          = out_base,
        ref_tif           = ref_tif,
        norm_stats_json   = norm_stats_json,
        T                 = T,
        max_gap           = max_gap,
        sample_chunk      = sample_chunk,
        chip_size         = chip_size,
        back_p_low        = back_p_low,
        back_p_high       = back_p_high,
    )

    print("\n" + "=" * 60)
    print(f"  ✅ Year {year} complete")
    print("=" * 60)
    p = out_base / f"{year}.zarr"
    if p.exists():
        z  = zarr.open_group(str(p), mode="r")
        gb = disk_size_gb(p)
        print(f"  {year}  →  N={z['X'].shape[0]:,}  shape={z['X'].shape}  {gb:.2f} GB")
    else:
        print(f"  {year}  →  NOT BUILT")
    print("=" * 60)
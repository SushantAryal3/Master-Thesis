"""
S2 Optical Zarr Builder — 2018  (X + Y)
========================================
Builds a Zarr dataset from S2 HDF5 chips (4 bands: B2, B3, B4, B8)
combined with segmentation labels from .npy files.

Output Zarr structure:
    X        : (N, B=4, T=10, H=128, W=128)  float32
               channel 0 = B2
               channel 1 = B3
               channel 2 = B4
               channel 3 = B8
    Y        : (N, L=3, H=128, W=128)         float32
               channel 0 = extent
               channel 1 = boundary
               channel 2 = dist
    row_off  : (N,)   int32
    col_off  : (N,)   int32
    x0       : (N,)   float64
    y0       : (N,)   float64
    block_id : (N,)   int32   — which stride-10 block (0=first, 1=second...)
    year     : (N,)   int32
    dates    : (N, 10) U10    — the 10 actual acquisition dates

Rules:
    - Only chips that have a matching label .npy file are included
    - Chips with fewer than 10 timestamps are skipped
    - Y is the SAME for all blocks of the same chip
    - Stride = 10, non-overlapping blocks
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from rasterio.transform import Affine
from rasterio.windows import Window
import rasterio
from tqdm import tqdm

S2_ROOT     = Path("/export/students/aryal/s2_h5/2020")
LABEL_ROOT  = Path("/export/students/aryal/Label_Chips_npy_128_from_ref_2020_test/2020")
REF_TIF     = Path("/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif")
OUT_ZARR    = Path("/export/students/aryal/S2_dataset/s2_optical_2020_test.zarr")

YEAR        = 2020
CHIP        = 128
T_BLOCK     = 10          
B           = 4           
L           = 3           
BAND_NAMES  = ["B2", "B3", "B4", "B8"]
LABEL_NAMES = ["extent", "boundary", "dist"]
LABEL_CH    = [0, 1, 2]   

SAMPLE_CHUNK = 32
OVERWRITE    = True

NPY_RE = re.compile(r"loc_r(?P<r>\d+)_c(?P<c>\d+)\.npy$", re.IGNORECASE)

def loc_name(r0: int, c0: int) -> str:
    return f"r{r0:04d}_c{c0:05d}"


def window_ul_xy(transform: Affine, win: Window) -> Tuple[float, float]:
    col, row = int(win.col_off), int(win.row_off)
    x, y = transform * (col, row)
    return float(x), float(y)


def read_ref_meta(path: Path) -> dict:
    with rasterio.open(path) as ds:
        return {
            "crs"      : ds.crs,
            "transform": ds.transform,
            "width"    : ds.width,
            "height"   : ds.height,
        }


def load_label_index(label_root: Path) -> Dict[str, Path]:
    """
    Scan label_root for loc_r????_c?????.npy files.
    Returns dict: loc_name → npy_path
    """
    index = {}
    for p in sorted(label_root.glob("loc_r*_c*.npy")):
        m = NPY_RE.search(p.name)
        if not m:
            continue
        r0 = int(m.group("r"))
        c0 = int(m.group("c"))
        index[loc_name(r0, c0)] = p
    return index


def get_blocks(
    h5_path: Path,
    t_block: int,
) -> List[Tuple[List[str], List[int]]]:
    """
    Opens the H5 file (attrs only), reads dates,
    returns list of stride-t_block non-overlapping blocks.
    Each block: (dates_list, indices_list)
    Returns [] if fewer than t_block timestamps.
    """
    with h5py.File(h5_path, "r") as f:
        dates = list(f.attrs.get("dates", []))

    if len(dates) < t_block:
        return []

    blocks = []
    n_blocks = len(dates) // t_block
    for b in range(n_blocks):
        start = b * t_block
        end   = (b + 1) * t_block
        block_dates = dates[start:end]
        block_idxs  = list(range(start, end))
        blocks.append((block_dates, block_idxs))

    return blocks

def pass1_count(
    s2_root    : Path,
    label_index: Dict[str, Path],
    t_block    : int,
) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    Returns:
        N_total   : total number of samples
        chip_info : list of (r0, c0, n_blocks) for valid chips only
    """
    s2_chips = {p.stem: p for p in sorted(s2_root.glob("r*.h5"))}

    print(f"  S2 H5 chips         : {len(s2_chips):,}")
    print(f"  Label .npy files    : {len(label_index):,}")

    N_total            = 0
    chip_info          = []
    skipped_no_label   = 0
    skipped_low_t      = 0

    for name in tqdm(sorted(s2_chips.keys()), desc="[pass1] counting", unit="chip"):
        parts = name.split("_")
        r0    = int(parts[0][1:])
        c0    = int(parts[1][1:])

        if name not in label_index:
            skipped_no_label += 1
            continue

        blocks = get_blocks(s2_chips[name], t_block)
        if not blocks:
            skipped_low_t += 1
            continue

        n_blocks = len(blocks)
        N_total += n_blocks
        chip_info.append((r0, c0, n_blocks))

    print(f"\n  Skipped — no label file       : {skipped_no_label:,}")
    print(f"  Skipped — T < {t_block}              : {skipped_low_t:,}")
    print(f"  Valid chips                   : {len(chip_info):,}")
    print(f"  Total samples N               : {N_total:,}")
    return N_total, chip_info

def init_zarr(
    out_zarr    : Path,
    N           : int,
    B           : int,
    T           : int,
    L           : int,
    chip        : int,
    sample_chunk: int,
    compressor,
    ref         : dict,
    overwrite   : bool,
) -> zarr.Group:

    if overwrite and out_zarr.exists():
        shutil.rmtree(out_zarr)

    root = zarr.open_group(str(out_zarr), mode="w")

    root.create_dataset(
        "X",
        shape      = (N, B, T, chip, chip),
        chunks     = (sample_chunk, B, T, chip, chip),
        dtype      = np.float32,
        compressor = compressor,
    )

    root.create_dataset(
        "Y",
        shape      = (N, L, chip, chip),
        chunks     = (sample_chunk, L, chip, chip),
        dtype      = np.float32,
        compressor = compressor,
    )

    for name, dtype in [
        ("row_off",  np.int32),
        ("col_off",  np.int32),
        ("block_id", np.int32),
        ("year",     np.int32),
    ]:
        root.create_dataset(
            name,
            shape      = (N,),
            chunks     = (4096,),
            dtype      = dtype,
            compressor = compressor,
        )

    for name in ["x0", "y0"]:
        root.create_dataset(
            name,
            shape      = (N,),
            chunks     = (4096,),
            dtype      = np.float64,
            compressor = compressor,
        )

    root.create_dataset(
        "dates",
        shape      = (N, T),
        chunks     = (sample_chunk, T),
        dtype      = "U10",
        compressor = compressor,
    )

    root.attrs.update({
        "year"       : YEAR,
        "chip"       : chip,
        "T"          : T,
        "B"          : B,
        "L"          : L,
        "bands"      : BAND_NAMES,
        "labels"     : LABEL_NAMES,
        "crs_wkt"    : ref["crs"].to_wkt(),
        "transform"  : tuple(map(float, ref["transform"])),
        "width"      : int(ref["width"]),
        "height"     : int(ref["height"]),
        "s2_root"    : str(S2_ROOT),
        "label_root" : str(LABEL_ROOT),
        "stride"     : T,
        "description": (
            "S2 optical composite 2019. "
            f"X: bands = {BAND_NAMES}. "
            f"Y: channels = {LABEL_NAMES}. "
            f"Each sample = {T} consecutive timestamps "
            f"(stride={T}, non-overlapping blocks)."
        ),
    })

    return root

def pass2_write(
    root         : zarr.Group,
    s2_root      : Path,
    label_index  : Dict[str, Path],
    chip_info    : List[Tuple[int, int, int]],
    ref_transform: Affine,
    chip         : int,
    t_block      : int,
    year         : int,
) -> None:

    s2_chips = {p.stem: p for p in sorted(s2_root.glob("r*.h5"))}

    Xz      = root["X"]
    Yz      = root["Y"]
    row_z   = root["row_off"]
    col_z   = root["col_off"]
    blk_z   = root["block_id"]
    yr_z    = root["year"]
    x0_z    = root["x0"]
    y0_z    = root["y0"]
    dates_z = root["dates"]

    idx  = 0
    xbuf = np.empty((B, t_block, chip, chip), dtype=np.float32)
    ybuf = np.empty((L, chip, chip),           dtype=np.float32)

    pbar = tqdm(chip_info, desc="[pass2] writing", unit="chip")

    for r0, c0, n_blocks in pbar:
        name     = loc_name(r0, c0)
        h5_path  = s2_chips[name]
        npy_path = label_index[name]

        # load label once per chip — same Y for all blocks
        y_full   = np.load(npy_path, mmap_mode="r")         # (4, H, W)
        ybuf[:]  = np.asarray(y_full[LABEL_CH], dtype=np.float32)

        # get blocks
        blocks = get_blocks(h5_path, t_block)

        # open H5 pixel data once per chip
        with h5py.File(h5_path, "r") as f:
            s2_X = f["X"]   # shape: (T, 4, 128, 128)

            for b_id, (dates_10, idxs_10) in enumerate(blocks):

                for ti in range(t_block):
                    xbuf[:, ti, :, :] = s2_X[idxs_10[ti]]  # (4, 128, 128)

                win    = Window(c0, r0, chip, chip)
                x0, y0 = window_ul_xy(ref_transform, win)

                Xz[idx]      = xbuf
                Yz[idx]      = ybuf
                row_z[idx]   = r0
                col_z[idx]   = c0
                blk_z[idx]   = b_id
                yr_z[idx]    = year
                x0_z[idx]    = x0
                y0_z[idx]    = y0
                dates_z[idx] = np.array(dates_10, dtype="U10")

                idx += 1

        pbar.set_postfix({"samples": idx})

    print(f"\n  Written {idx:,} samples total")


def build():
    print("=" * 60)
    print(f"S2 Optical Zarr Builder — {YEAR}  (X + Y)")
    print(f"  S2 H5 root : {S2_ROOT}")
    print(f"  Labels     : {LABEL_ROOT}")
    print(f"  Output     : {OUT_ZARR}")
    print(f"  T={T_BLOCK}  B={B}  L={L}  chip={CHIP}")
    print("=" * 60)

    ref = read_ref_meta(REF_TIF)
    print(f"\n✓ REF TIF CRS  : {ref['crs'].to_string()}")
    print(f"  Pixel size   : {ref['transform'].a} × {-ref['transform'].e} m\n")

    print("[Labels] Scanning label files ...")
    label_index = load_label_index(LABEL_ROOT)
    print(f"  Found {len(label_index):,} label .npy files\n")

    print("[Pass 1] Counting valid samples ...")
    N_total, chip_info = pass1_count(S2_ROOT, label_index, T_BLOCK)

    if N_total == 0:
        raise RuntimeError(
            "No valid samples found. "
            "Check paths, year, and that label files exist."
        )

    x_bytes = N_total * B * T_BLOCK * CHIP * CHIP * 4
    y_bytes = N_total * L * CHIP * CHIP * 4
    raw_gb  = (x_bytes + y_bytes) / 1e9
    print(f"\n  Estimated uncompressed : {raw_gb:.1f} GB")
    print(f"  Estimated compressed   : {raw_gb * 0.5:.1f} GB\n")

    print("[Init] Creating Zarr store ...")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = init_zarr(
        out_zarr     = OUT_ZARR,
        N            = N_total,
        B            = B,
        T            = T_BLOCK,
        L            = L,
        chip         = CHIP,
        sample_chunk = SAMPLE_CHUNK,
        compressor   = compressor,
        ref          = ref,
        overwrite    = OVERWRITE,
    )
    print(f"  Created: {OUT_ZARR}\n")

    print("[Pass 2] Writing data ...")
    pass2_write(
        root          = root,
        s2_root       = S2_ROOT,
        label_index   = label_index,
        chip_info     = chip_info,
        ref_transform = ref["transform"],
        chip          = CHIP,
        t_block       = T_BLOCK,
        year          = YEAR,
    )

    zarr.consolidate_metadata(str(OUT_ZARR))

    total_bytes = sum(
        f.stat().st_size for f in OUT_ZARR.rglob("*") if f.is_file()
    )

    print("\n" + "=" * 60)
    print("✅ Done")
    print(f"   Output      : {OUT_ZARR}")
    print(f"   X shape     : {root['X'].shape}")
    print(f"   Y shape     : {root['Y'].shape}")
    print(f"   N samples   : {N_total:,}")
    print(f"   N chips     : {len(chip_info):,}")
    print(f"   Size on disk: {total_bytes / 1e9:.2f} GB")
    print("=" * 60)

    print("\n[Verify] Reading back first sample ...")
    z    = zarr.open_group(str(OUT_ZARR), mode="r")
    x0s  = z["X"][0]
    y0s  = z["Y"][0]
    d0   = z["dates"][0]

    print(f"  X[0] shape          : {x0s.shape}")
    print(f"  X[0] B2   range     : [{np.nanmin(x0s[0]):.4f}, {np.nanmax(x0s[0]):.4f}]")
    print(f"  X[0] B3   range     : [{np.nanmin(x0s[1]):.4f}, {np.nanmax(x0s[1]):.4f}]")
    print(f"  X[0] B4   range     : [{np.nanmin(x0s[2]):.4f}, {np.nanmax(x0s[2]):.4f}]")
    print(f"  X[0] B8   range     : [{np.nanmin(x0s[3]):.4f}, {np.nanmax(x0s[3]):.4f}]")
    print(f"  Y[0] shape          : {y0s.shape}")
    print(f"  Y[0] extent  range  : [{np.nanmin(y0s[0]):.4f}, {np.nanmax(y0s[0]):.4f}]")
    print(f"  Y[0] boundary range : [{np.nanmin(y0s[1]):.4f}, {np.nanmax(y0s[1]):.4f}]")
    print(f"  Y[0] dist range     : [{np.nanmin(y0s[2]):.4f}, {np.nanmax(y0s[2]):.4f}]")
    print(f"  dates[0]            : {list(d0)}")
    print(f"  row_off[0]          : {z['row_off'][0]}   col_off[0]: {z['col_off'][0]}")
    print(f"  x0[0]               : {z['x0'][0]:.4f}   y0[0]: {z['y0'][0]:.4f}")
    print("\n✓ Verification complete")


if __name__ == "__main__":
    build()
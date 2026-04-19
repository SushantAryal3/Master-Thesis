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

COHE_ROOT   = Path("/export/students/aryal/coherence_h5/2018")
BACK_ROOT   = Path("/export/students/aryal/backscattering_h5/2018")
LABEL_ROOT  = Path("/export/students/aryal/Label_Chips_npy_128_from_ref/2018")
REF_TIF     = Path("/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif")
OUT_ZARR    = Path("/export/students/aryal/S1_dataset/sar_composite_2018.zarr")

YEAR        = 2018
CHIP        = 128
T_BLOCK     = 16          
C           = 2         
L           = 3           
LABEL_NAMES = ["extent", "boundary", "dist"]
LABEL_CH    = [0, 1, 2]

SAMPLE_CHUNK = 32
OVERWRITE    = True

NPY_RE = re.compile(r"loc_r(?P<r>\d+)_c(?P<c>\d+)\.npy$", re.IGNORECASE)

# HELPERS
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

def get_overlap_blocks(
    cohe_path: Path,
    back_path: Path,
    t_block: int,
) -> List[Tuple[List[str], List[int], List[int]]]:
    """
    Opens both H5 files (attrs only), finds overlapping dates,
    returns list of stride-t_block non-overlapping blocks.
    Each block: (dates_16, cohe_indices_16, back_indices_16)
    Returns [] if fewer than t_block overlapping dates.
    """
    with h5py.File(cohe_path, "r") as fc, \
         h5py.File(back_path, "r") as fb:
        cohe_dates = list(fc.attrs.get("dates", []))
        back_dates = list(fb.attrs.get("dates", []))

    overlap = sorted(set(cohe_dates) & set(back_dates))

    if len(overlap) < t_block:
        return []

    cohe_idx = {d: i for i, d in enumerate(cohe_dates)}
    back_idx = {d: i for i, d in enumerate(back_dates)}

    blocks = []
    n_blocks = len(overlap) // t_block
    for b in range(n_blocks):
        dates_16     = overlap[b * t_block : (b + 1) * t_block]
        cohe_idxs_16 = [cohe_idx[d] for d in dates_16]
        back_idxs_16 = [back_idx[d] for d in dates_16]
        blocks.append((dates_16, cohe_idxs_16, back_idxs_16))

    return blocks

# PASS 1 — count valid samples
def pass1_count(
    cohe_root  : Path,
    back_root  : Path,
    label_index: Dict[str, Path],
    t_block    : int,
) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    Returns:
        N_total   : total number of samples
        chip_info : list of (r0, c0, n_blocks) for valid chips only
    """
    cohe_chips = {p.stem: p for p in sorted(cohe_root.glob("r*.h5"))}
    back_chips = {p.stem: p for p in sorted(back_root.glob("r*.h5"))}
    common     = sorted(set(cohe_chips) & set(back_chips))

    print(f"  Common H5 chips     : {len(common):,}")
    print(f"  Label .npy files    : {len(label_index):,}")

    N_total            = 0
    chip_info          = []
    skipped_no_label   = 0
    skipped_no_overlap = 0

    for name in tqdm(common, desc="[pass1] counting", unit="chip"):
        parts = name.split("_")
        r0    = int(parts[0][1:])
        c0    = int(parts[1][1:])
        if name not in label_index:
            skipped_no_label += 1
            continue
        blocks = get_overlap_blocks(
            cohe_chips[name], back_chips[name], t_block
        )
        if not blocks:
            skipped_no_overlap += 1
            continue

        n_blocks = len(blocks)
        N_total += n_blocks
        chip_info.append((r0, c0, n_blocks))

    print(f"\n  Skipped — no label file       : {skipped_no_label:,}")
    print(f"  Skipped — < {t_block} overlap dates : {skipped_no_overlap:,}")
    print(f"  Valid chips                   : {len(chip_info):,}")
    print(f"  Total samples N               : {N_total:,}")
    return N_total, chip_info

# ZARR INITIALISATION
def init_zarr(
    out_zarr    : Path,
    N           : int,
    C           : int,
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
        shape      = (N, C, T, chip, chip),
        chunks     = (sample_chunk, C, T, chip, chip),
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
        "C"          : C,
        "L"          : L,
        "channels"   : ["coherence", "backscatter"],
        "labels"     : LABEL_NAMES,
        "crs_wkt"    : ref["crs"].to_wkt(),
        "transform"  : tuple(map(float, ref["transform"])),
        "width"      : int(ref["width"]),
        "height"     : int(ref["height"]),
        "cohe_root"  : str(COHE_ROOT),
        "back_root"  : str(BACK_ROOT),
        "label_root" : str(LABEL_ROOT),
        "stride"     : T,
        "description": (
            "SAR composite 2018. "
            "X: channel 0=coherence, channel 1=backscatter. "
            f"Y: channels = {LABEL_NAMES}. "
            f"Each sample = {T} consecutive overlapping dates "
            f"(stride={T}, non-overlapping blocks)."
        ),
    })

    return root

# PASS 2 — write data
def pass2_write(
    root         : zarr.Group,
    cohe_root    : Path,
    back_root    : Path,
    label_index  : Dict[str, Path],
    chip_info    : List[Tuple[int, int, int]],
    ref_transform: Affine,
    chip         : int,
    t_block      : int,
    year         : int,
) -> None:

    cohe_chips = {p.stem: p for p in sorted(cohe_root.glob("r*.h5"))}
    back_chips = {p.stem: p for p in sorted(back_root.glob("r*.h5"))}

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
    xbuf = np.empty((C, t_block, chip, chip), dtype=np.float32)
    ybuf = np.empty((L, chip, chip),           dtype=np.float32)

    pbar = tqdm(chip_info, desc="[pass2] writing", unit="chip")

    for r0, c0, n_blocks in pbar:
        name      = loc_name(r0, c0)
        cohe_path = cohe_chips[name]
        back_path = back_chips[name]
        npy_path  = label_index[name]

        y_full   = np.load(npy_path, mmap_mode="r")
        ybuf[:]  = np.asarray(y_full[LABEL_CH], dtype=np.float32)

        blocks = get_overlap_blocks(cohe_path, back_path, t_block)

        with h5py.File(cohe_path, "r") as fc, \
             h5py.File(back_path, "r") as fb:

            cohe_X = fc["X"]
            back_X = fb["X"]

            for b_id, (dates_16, cohe_idxs, back_idxs) in enumerate(blocks):
                for ti in range(t_block):
                    xbuf[0, ti] = cohe_X[cohe_idxs[ti]]
                    xbuf[1, ti] = back_X[back_idxs[ti]]

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
                dates_z[idx] = np.array(dates_16, dtype="U10")

                idx += 1

        pbar.set_postfix({"samples": idx})

    print(f"\n  Written {idx:,} samples total")

# MAIN
def build():
    print("=" * 60)
    print(f"SAR Composite Zarr Builder — {YEAR}  (X + Y)")
    print(f"  Coherence  : {COHE_ROOT}")
    print(f"  Backscatter: {BACK_ROOT}")
    print(f"  Labels     : {LABEL_ROOT}")
    print(f"  Output     : {OUT_ZARR}")
    print(f"  T={T_BLOCK}  C={C}  L={L}  chip={CHIP}")
    print("=" * 60)

    ref = read_ref_meta(REF_TIF)
    print(f"\n✓ REF TIF CRS  : {ref['crs'].to_string()}")
    print(f"  Pixel size   : {ref['transform'].a} × {-ref['transform'].e} m\n")

    print("[Labels] Scanning label files ...")
    label_index = load_label_index(LABEL_ROOT)
    print(f"  Found {len(label_index):,} label .npy files\n")

    print("[Pass 1] Counting valid samples ...")
    N_total, chip_info = pass1_count(
        COHE_ROOT, BACK_ROOT, label_index, T_BLOCK
    )

    if N_total == 0:
        raise RuntimeError(
            "No valid samples found. "
            "Check paths, year, and that label files exist."
        )

    x_bytes = N_total * C * T_BLOCK * CHIP * CHIP * 4
    y_bytes = N_total * L * CHIP * CHIP * 4
    raw_gb  = (x_bytes + y_bytes) / 1e9
    print(f"\n  Estimated uncompressed : {raw_gb:.1f} GB")
    print(f"  Estimated compressed   : {raw_gb * 0.5:.1f} GB\n")

    print("[Init] Creating Zarr store ...")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = init_zarr(
        out_zarr     = OUT_ZARR,
        N            = N_total,
        C            = C,
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
        cohe_root     = COHE_ROOT,
        back_root     = BACK_ROOT,
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
    print(f"  X[0] cohe  range    : [{np.nanmin(x0s[0]):.4f}, {np.nanmax(x0s[0]):.4f}]")
    print(f"  X[0] back  range    : [{np.nanmin(x0s[1]):.4f}, {np.nanmax(x0s[1]):.4f}]")
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
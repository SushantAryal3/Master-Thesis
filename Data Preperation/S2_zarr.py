from __future__ import annotations
import argparse
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
    index = {}
    for p in sorted(label_root.glob("loc_r*_c*.npy")):
        m = NPY_RE.search(p.name)
        if not m:
            continue
        r0 = int(m.group("r"))
        c0 = int(m.group("c"))
        index[loc_name(r0, c0)] = p
    return index


def get_blocks_S2_Zarr(
    h5_path: Path,
    t_block: int,
) -> List[Tuple[List[str], List[int]]]:
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


def pass1_count_S2_Zarr(
    s2_root    : Path,
    label_index: Dict[str, Path],
    t_block    : int,
) -> Tuple[int, List[Tuple[int, int, int]]]:
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

        blocks = get_blocks_S2_Zarr(s2_chips[name], t_block)
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


def init_zarr_S2_Zarr(
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
    year        : int,
    s2_root     : Path,
    label_root  : Path,
    band_names  : List[str],
    label_names : List[str],
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
        "year"       : year,
        "chip"       : chip,
        "T"          : T,
        "B"          : B,
        "L"          : L,
        "bands"      : band_names,
        "labels"     : label_names,
        "crs_wkt"    : ref["crs"].to_wkt(),
        "transform"  : tuple(map(float, ref["transform"])),
        "width"      : int(ref["width"]),
        "height"     : int(ref["height"]),
        "s2_root"    : str(s2_root),
        "label_root" : str(label_root),
        "stride"     : T,
        "description": (
            f"S2 optical composite {year}. "
            f"X: bands = {band_names}. "
            f"Y: channels = {label_names}. "
            f"Each sample = {T} consecutive timestamps "
            f"(stride={T}, non-overlapping blocks)."
        ),
    })

    return root


def pass2_write_S2_Zarr(
    root         : zarr.Group,
    s2_root      : Path,
    label_index  : Dict[str, Path],
    chip_info    : List[Tuple[int, int, int]],
    ref_transform: Affine,
    chip         : int,
    t_block      : int,
    year         : int,
    B            : int,
    L            : int,
    label_ch     : List[int],
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
        ybuf[:]  = np.asarray(y_full[label_ch], dtype=np.float32)

        # get blocks
        blocks = get_blocks_S2_Zarr(h5_path, t_block)

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


def buildS2Zarr(
    year         : int,
    s2_root      : Path,
    label_root   : Path,
    out_zarr     : Path,
    ref_tif      : Path,
    chip         : int,
    t_block      : int,
    B            : int,
    L            : int,
    sample_chunk : int,
    band_names   : List[str],
    label_names  : List[str],
    label_ch     : List[int],
    overwrite    : bool,
) -> None:
    print("=" * 60)
    print(f"S2 Optical Zarr Builder — {year}  (X + Y)")
    print(f"  S2 H5 root : {s2_root}")
    print(f"  Labels     : {label_root}")
    print(f"  Output     : {out_zarr}")
    print(f"  T={t_block}  B={B}  L={L}  chip={chip}")
    print("=" * 60)

    ref = read_ref_meta(ref_tif)
    print(f"\n✓ REF TIF CRS  : {ref['crs'].to_string()}")
    print(f"  Pixel size   : {ref['transform'].a} × {-ref['transform'].e} m\n")

    print("[Labels] Scanning label files ...")
    label_index = load_label_index(label_root)
    print(f"  Found {len(label_index):,} label .npy files\n")

    print("[Pass 1] Counting valid samples ...")
    N_total, chip_info = pass1_count_S2_Zarr(s2_root, label_index, t_block)

    if N_total == 0:
        raise RuntimeError(
            "No valid samples found. "
            "Check paths, year, and that label files exist."
        )

    x_bytes = N_total * B * t_block * chip * chip * 4
    y_bytes = N_total * L * chip * chip * 4
    raw_gb  = (x_bytes + y_bytes) / 1e9
    print(f"\n  Estimated uncompressed : {raw_gb:.1f} GB")
    print(f"  Estimated compressed   : {raw_gb * 0.5:.1f} GB\n")

    print("[Init] Creating Zarr store ...")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = init_zarr_S2_Zarr(
        out_zarr     = out_zarr,
        N            = N_total,
        B            = B,
        T            = t_block,
        L            = L,
        chip         = chip,
        sample_chunk = sample_chunk,
        compressor   = compressor,
        ref          = ref,
        overwrite    = overwrite,
        year         = year,
        s2_root      = s2_root,
        label_root   = label_root,
        band_names   = band_names,
        label_names  = label_names,
    )
    print(f"  Created: {out_zarr}\n")

    print("[Pass 2] Writing data ...")
    pass2_write_S2_Zarr(
        root          = root,
        s2_root       = s2_root,
        label_index   = label_index,
        chip_info     = chip_info,
        ref_transform = ref["transform"],
        chip          = chip,
        t_block       = t_block,
        year          = year,
        B             = B,
        L             = L,
        label_ch      = label_ch,
    )

    zarr.consolidate_metadata(str(out_zarr))

    total_bytes = sum(
        f.stat().st_size for f in out_zarr.rglob("*") if f.is_file()
    )

    print("\n" + "=" * 60)
    print("✅ Done")
    print(f"   Output      : {out_zarr}")
    print(f"   X shape     : {root['X'].shape}")
    print(f"   Y shape     : {root['Y'].shape}")
    print(f"   N samples   : {N_total:,}")
    print(f"   N chips     : {len(chip_info):,}")
    print(f"   Size on disk: {total_bytes / 1e9:.2f} GB")
    print("=" * 60)

    print("\n[Verify] Reading back first sample ...")
    z    = zarr.open_group(str(out_zarr), mode="r")
    x0s  = z["X"][0]
    y0s  = z["Y"][0]
    d0   = z["dates"][0]

    print(f"  X[0] shape          : {x0s.shape}")
    for bi, bname in enumerate(band_names):
        print(f"  X[0] {bname:<5} range     : [{np.nanmin(x0s[bi]):.4f}, {np.nanmax(x0s[bi]):.4f}]")
    print(f"  Y[0] shape          : {y0s.shape}")
    for li, lname in enumerate(label_names):
        print(f"  Y[0] {lname:<10} range  : [{np.nanmin(y0s[li]):.4f}, {np.nanmax(y0s[li]):.4f}]")
    print(f"  dates[0]            : {list(d0)}")
    print(f"  row_off[0]          : {z['row_off'][0]}   col_off[0]: {z['col_off'][0]}")
    print(f"  x0[0]               : {z['x0'][0]:.4f}   y0[0]: {z['y0'][0]:.4f}")
    print("\n✓ Verification complete")

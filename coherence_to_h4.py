from __future__ import annotations

import gc
import json
import os
import re
import resource
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm


# Memory monitor
import os, threading, time, resource

def memory_monitor(interval=10):
    """Prints current RSS every `interval` seconds in the background."""
    pid = os.getpid()
    while getattr(memory_monitor, "running", True):
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[mem] Peak RSS: {rss_kb / 1024:.0f} MB", flush=True)
        time.sleep(interval)

memory_monitor.running = True
t = threading.Thread(target=memory_monitor, args=(10,), daemon=True)
t.start()


# Configuration
MANIFEST_PATH = Path("/export/students/aryal/metadata_s1/manifest_cohe_wallonia_ascending_sorted_month.json")
REF_TIF = Path("/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif")
OUT_ROOT = Path("/export/students/aryal/coherence_h5_ascending")

CHIP = 128
STRIDE = 128
EDGE_BUFFER = 30.0
MIN_VALID_FRAC = 1.0
VRT_RESAMPLING = Resampling.nearest

H5_COMPRESSION = "lzf"
H5_CHUNK_SHAPE = (1, CHIP, CHIP)
MAX_OPEN_H5 = 128

# Helpers — manifest
def load_all_records(manifest_path: Path) -> List[dict]:
    month_re = re.compile(r"^\d{4}-\d{2}$")
    skip_keys = {
        "date_from", "date_to", "angle_index_date_to_used",
        "n_angle_indexed_keys", "n_orbit_conflicts_in_angle",
        "n_kept_descending_and_wallonia", "matching_policy", "stats",
    }

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_records: List[dict] = []
    for k, v in data.items():
        if k in skip_keys:
            continue
        if month_re.match(k) and isinstance(v, list):
            all_records.extend(v)

    all_records.sort(key=lambda r: r["t2_end_dt"])
    return all_records

def filter_records_by_month(
    all_records: List[dict], year: int, month: int,
) -> List[dict]:
    prefix = f"{year:04d}-{month:02d}"
    return [rec for rec in all_records if rec["t2_end_dt"][:7] == prefix]

REGIONS = [
    (2048, 6784,  4480, 11008),   
    (1536, 14208, 4736, 20096),   
    (6144, 16384, 10752, 19840),
]

def enumerate_chips(
    ref_tif: Path,
    chip: int,
    stride: int,
    regions: List[Tuple[int, int, int, int]] = REGIONS,
) -> Tuple[List[Tuple[int, int, Window]], rasterio.Affine, dict]:

    with rasterio.open(ref_tif) as ds:
        H, W = ds.height, ds.width
        ref_transform = ds.transform
        ref_meta = {
            "crs": ds.crs,
            "transform": ds.transform,
            "width": ds.width,
            "height": ds.height,
        }

    chips: List[Tuple[int, int, Window]] = []

    for i, (row_min, col_min, row_max, col_max) in enumerate(regions):
        r_min = max(row_min, 0)
        c_min = max(col_min, 0)
        r_max = min(row_max, H - chip)
        c_max = min(col_max, W - chip)

        region_chips = []
        for r0 in range(r_min, r_max + 1, stride):
            for c0 in range(c_min, c_max + 1, stride):
                region_chips.append((r0, c0, Window(c0, r0, chip, chip)))

        print(f"  Region {i+1}: {len(region_chips):,} chips  "
              f"rows [{r_min}–{r_max}]  cols [{c_min}–{c_max}]")
        chips.extend(region_chips)

    print(f"[Chips] {len(chips):,} total chips across {len(regions)} regions")
    return chips, ref_transform, ref_meta

def chip_geo_bounds(
    tf: rasterio.Affine, r0: int, c0: int, chip: int,
) -> Tuple[float, float, float, float]:
    left, top = tf * (c0, r0)
    right, bottom = tf * (c0 + chip, r0 + chip)
    return (
        min(left, right), min(top, bottom),
        max(left, right), max(top, bottom),
    )


def build_file_to_chips(
    records: List[dict],
    chips: List[Tuple[int, int, Window]],
    ref_transform: rasterio.Affine,
    chip: int,
    buffer_m: float,
) -> Dict[int, List[int]]:
    cboxes = [
        chip_geo_bounds(ref_transform, r, c, chip) for r, c, _ in chips
    ]

    f2c: Dict[int, List[int]] = {}
    total_pairs = 0
    for fi, rec in enumerate(records):
        cb = rec["bounds"]
        cx0, cy0 = cb[0] + buffer_m, cb[1] + buffer_m
        cx1, cy1 = cb[2] - buffer_m, cb[3] - buffer_m

        hits = [
            ci
            for ci, (px0, py0, px1, py1) in enumerate(cboxes)
            if cx0 <= px0 and px1 <= cx1 and cy0 <= py0 and py1 <= cy1
        ]
        if hits:
            f2c[fi] = hits
            total_pairs += len(hits)

    print(f"  {len(f2c)}/{len(records)} files → {total_pairs:,} pairs")
    return f2c


# HDF5 naming
def loc_name(r0: int, c0: int) -> str:
    return f"r{r0:04d}_c{c0:05d}"


# LRU cache + buffered metadata (same as before)
class _ChipHandle:
    __slots__ = ("h5f", "path", "dates", "satellites", "orbits", "cohe_paths")

    def __init__(
        self,
        h5f: h5py.File,
        path: Path,
        dates: List[str],
        satellites: List[str],
        orbits: List[str],
        cohe_paths: List[str],
    ):
        self.h5f = h5f
        self.path = path
        self.dates = dates
        self.satellites = satellites
        self.orbits = orbits
        self.cohe_paths = cohe_paths

    def append(
        self,
        arr: np.ndarray,
        date_str: str,
        sat: str,
        orbit: str,
        cohe_path: str,
    ) -> None:
        ds = self.h5f["X"]
        idx = ds.shape[0]
        ds.resize(idx + 1, axis=0)
        ds[idx, :, :] = arr

        self.dates.append(date_str)
        self.satellites.append(sat)
        self.orbits.append(orbit)
        self.cohe_paths.append(cohe_path)

    def flush_and_close(self) -> None:
        try:
            self.h5f.attrs["dates"] = self.dates
            self.h5f.attrs["satellites"] = self.satellites
            self.h5f.attrs["orbits"] = self.orbits
            self.h5f.attrs["cohe_paths"] = self.cohe_paths
        finally:
            self.h5f.close()


class LRUChipCache:
    def __init__(self, out_dir: Path, chip: int, max_open: int = MAX_OPEN_H5):
        self._cache: OrderedDict[int, _ChipHandle] = OrderedDict()
        self._out_dir = out_dir
        self._chip = chip
        self._max_open = max_open
        self.unique_chips_seen: int = 0

    def get(self, ci: int, r0: int, c0: int) -> _ChipHandle:
        if ci in self._cache:
            self._cache.move_to_end(ci)
            return self._cache[ci]
        if len(self._cache) >= self._max_open:
            _, old = self._cache.popitem(last=False)
            old.flush_and_close()

        handle = self._open_or_create(r0, c0)
        self._cache[ci] = handle
        self.unique_chips_seen += 1
        return handle

    def close_all(self) -> None:
        while self._cache:
            _, handle = self._cache.popitem()
            handle.flush_and_close()

    def _open_or_create(self, r0: int, c0: int) -> _ChipHandle:
        path = self._out_dir / f"{loc_name(r0, c0)}.h5"
        chip = self._chip

        if path.exists():
            h5f = h5py.File(path, "a")
            dates = list(h5f.attrs.get("dates", []))
            satellites = list(h5f.attrs.get("satellites", []))
            orbits = list(h5f.attrs.get("orbits", []))
            cohe_paths = list(h5f.attrs.get("cohe_paths", []))
        else:
            h5f = h5py.File(path, "w")
            h5f.create_dataset(
                "X",
                shape=(0, chip, chip),
                maxshape=(None, chip, chip),
                dtype=np.float32,
                chunks=H5_CHUNK_SHAPE,
                compression=H5_COMPRESSION,
            )
            h5f.attrs["row_off"] = r0
            h5f.attrs["col_off"] = c0
            dates, satellites, orbits, cohe_paths = [], [], [], []

        return _ChipHandle(h5f, path, dates, satellites, orbits, cohe_paths)

# Core — extract one month using LRU cache
def extract_month(
    records: List[dict],
    file_to_chips: Dict[int, List[int]],
    chips: List[Tuple[int, int, Window]],
    ref_meta: dict,
    out_dir: Path,
    chip: int,
    resampling: Resampling,
    min_valid_frac: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = LRUChipCache(out_dir, chip, max_open=MAX_OPEN_H5)

    n_appends = 0
    n_skipped_nan = 0
    n_file_errors = 0

    buf = np.empty((chip, chip), dtype=np.float32)

    try:
        for fi in tqdm(
            sorted(file_to_chips.keys()),
            desc="  Extracting",
            unit="file",
            leave=False,
        ):
            rec = records[fi]
            cohe_path = Path(rec["cohe_path"])
            date_str = rec["t2_end_dt"][:10]
            sat = rec.get("sat_from_cohe_name", "")
            orbit = rec.get("orbit", "")

            if not cohe_path.exists():
                tqdm.write(f"    [!] Missing: {cohe_path.name}")
                n_file_errors += 1
                continue

            try:
                with rasterio.open(cohe_path) as src:
                    with WarpedVRT(
                        src,
                        crs=ref_meta["crs"],
                        transform=ref_meta["transform"],
                        width=ref_meta["width"],
                        height=ref_meta["height"],
                        resampling=resampling,
                        nodata=np.nan,
                        add_alpha=False,
                    ) as vrt:
                        for ci in file_to_chips[fi]:
                            r0, c0, win = chips[ci]

                            buf[:] = np.nan
                            cs = max(c0, 0)
                            rs = max(r0, 0)
                            ce = min(c0 + chip, vrt.width)
                            re_val = min(r0 + chip, vrt.height)

                            if cs < ce and rs < re_val:
                                vwin = Window(cs, rs, ce - cs, re_val - rs)
                                data = vrt.read(
                                    1, window=vwin,
                                ).astype(np.float32)
                                dr, dc = rs - r0, cs - c0
                                buf[
                                    dr : dr + (re_val - rs),
                                    dc : dc + (ce - cs),
                                ] = data
                                del data

                            buf[buf == 0.0] = np.nan

                            if np.isfinite(buf).mean() < min_valid_frac:
                                n_skipped_nan += 1
                                continue

                            handle = cache.get(ci, r0, c0)
                            handle.append(
                                buf.copy(),
                                date_str,
                                sat,
                                orbit,
                                str(cohe_path),
                            )
                            n_appends += 1

            except Exception as exc:
                tqdm.write(f"    [!] Error: {cohe_path.name}: {exc}")
                n_file_errors += 1

    finally:
        cache.close_all()

    print(
        f"    Appends: {n_appends:,}  |  "
        f"Skipped(NaN): {n_skipped_nan:,}  |  "
        f"Errors: {n_file_errors}"
    )
    return n_appends


# Core — merge temp H5 into main H5 (slice by slice)
def merge_temp_into_main(
    main_dir: Path,
    temp_dir: Path,
    chip: int,
) -> None:
    temp_files = sorted(temp_dir.glob("r*.h5"))
    if not temp_files:
        print("    No temp files to merge.")
        return

    n_merged = 0
    for temp_path in tqdm(
        temp_files, desc="  Merging", unit="chip", leave=False,
    ):
        main_path = main_dir / temp_path.name

        if not main_path.exists():
            shutil.move(str(temp_path), str(main_path))
            n_merged += 1
            continue

        with h5py.File(main_path, "a") as main_h5, \
             h5py.File(temp_path, "r") as temp_h5:

            main_ds = main_h5["X"]
            temp_ds = temp_h5["X"]
            T_temp = temp_ds.shape[0]

            if T_temp == 0:
                continue

            t_start = main_ds.shape[0]
            main_ds.resize(t_start + T_temp, axis=0)
            for i in range(T_temp):
                main_ds[t_start + i, :, :] = temp_ds[i, :, :]

            main_h5.attrs["dates"] = (
                list(main_h5.attrs["dates"]) +
                list(temp_h5.attrs["dates"])
            )
            main_h5.attrs["satellites"] = (
                list(main_h5.attrs["satellites"]) +
                list(temp_h5.attrs["satellites"])
            )
            main_h5.attrs["orbits"] = (
                list(main_h5.attrs["orbits"]) +
                list(temp_h5.attrs["orbits"])
            )
            main_h5.attrs["cohe_paths"] = (
                list(main_h5.attrs["cohe_paths"]) +
                list(temp_h5.attrs["cohe_paths"])
            )

        temp_path.unlink()
        n_merged += 1

    print(f"    Merged {n_merged:,} chip files")

    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass


# Core — clear memory
def clear_memory():
    gc.collect()
    gc.collect()
    print("    [mem] Memory cleared")

# Core — finalize one year (sort by date)
def finalize_year(year_dir: Path, year: int) -> pd.DataFrame:
    h5_files = sorted(year_dir.glob("r*.h5"))
    rows: List[dict] = []

    for h5_path in tqdm(
        h5_files, desc=f"  Finalizing {year}", unit="chip", leave=False,
    ):
        with h5py.File(h5_path, "a") as f:
            r0 = int(f.attrs["row_off"])
            c0 = int(f.attrs["col_off"])
            T = f["X"].shape[0]

            dates = list(f.attrs["dates"])
            satellites = list(f.attrs["satellites"])
            orbits = list(f.attrs["orbits"])
            paths = list(f.attrs["cohe_paths"])

            order = sorted(range(T), key=lambda i: dates[i])

            if order != list(range(T)):
                chip_h = f["X"].shape[1]
                chip_w = f["X"].shape[2]
                tmp = np.empty((chip_h, chip_w), dtype=np.float32)

                visited = [False] * T
                for start in range(T):
                    if visited[start] or order[start] == start:
                        visited[start] = True
                        continue
                    tmp[:] = f["X"][start]
                    j = start
                    while not visited[j]:
                        visited[j] = True
                        src = order[j]
                        if src == start:
                            f["X"][j] = tmp
                        else:
                            f["X"][j] = f["X"][src]
                        j = src
                del tmp

                dates = [dates[i] for i in order]
                satellites = [satellites[i] for i in order]
                orbits = [orbits[i] for i in order]
                paths = [paths[i] for i in order]

                f.attrs["dates"] = dates
                f.attrs["satellites"] = satellites
                f.attrs["orbits"] = orbits
                f.attrs["cohe_paths"] = paths

            rows.append({
                "chip_file": h5_path.name,
                "row_off": r0,
                "col_off": c0,
                "T": T,
                "dates": ",".join(dates),
                "date_first": dates[0] if dates else "",
                "date_last": dates[-1] if dates else "",
            })

    summary = pd.DataFrame(rows)
    summary_csv = year_dir / "_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print()
    print(f"  Year {year} Summary:")
    print(f"    Chip files : {len(summary):,}")
    if len(summary) > 0:
        print(f"    T range    : {summary['T'].min()} – {summary['T'].max()}")
        print(f"    T mean     : {summary['T'].mean():.1f}")
    return summary

# Main pipeline
def run_pipeline(
    year_start: int = 2018,
    year_end: int = 2019,
    manifest_path: Path = MANIFEST_PATH,
    ref_tif: Path = REF_TIF,
    out_root: Path = OUT_ROOT,
    chip: int = CHIP,
    stride: int = STRIDE,
    edge_buffer: float = EDGE_BUFFER,
    min_valid_frac: float = MIN_VALID_FRAC,
    resampling: Resampling = VRT_RESAMPLING,
) -> None:
    print("=" * 60)
    print("HDF5 Time-Series Pipeline — Month-by-Month v2")
    print(f"  Years      : {year_start} – {year_end}")
    print(f"  Output     : {out_root}")
    print(f"  Max open H5: {MAX_OPEN_H5}")
    print("=" * 60)
    print()

    print("[Step 0] Loading manifest...")
    all_records = load_all_records(manifest_path)
    print(f"  Total records: {len(all_records):,}")
    print()

    print("[Step 1] Enumerating chips...")
    chips, ref_transform, ref_meta = enumerate_chips(ref_tif, chip, stride)
    print()
    print(f"  Expected chip files to generate : {len(chips):,}")
    print(f"  Breakdown by region:")
    regions_info = [
        ("Region 1", 2048, 6784,  4480, 11008),
        ("Region 2", 1536, 14208, 4736, 20096),
        ("Region 3", 6144, 16384, 10752, 19840),
    ]
    for name, row_min, col_min, row_max, col_max in regions_info:
        n_rows = (row_max - row_min) // stride + 1
        n_cols = (col_max - col_min) // stride + 1
        print(f"    {name}: {n_rows} x {n_cols} = {n_rows * n_cols:,} chips")
    print()

    for year in range(year_start, year_end + 1):
        year_dir = out_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = out_root / f"_temp_{year}"

        if temp_dir.exists():
            print(f"  [!] Leftover temp dir found — cleaning up...")
            shutil.rmtree(temp_dir)
            print(f"      Cleaned: {temp_dir}")

        print("=" * 60)
        print(f"YEAR {year}")
        print("=" * 60)

        main_initialized = any(year_dir.glob("r*.h5"))

        for month in range(1, 13):
            month_label = f"{year}-{month:02d}"
            print(f"\n--- {month_label} ---")

            month_records = filter_records_by_month(all_records, year, month)
            if not month_records:
                print("  No records — skipping.")
                continue
            print(f"  Records: {len(month_records)}")

            file_to_chips = build_file_to_chips(
                month_records, chips, ref_transform, chip, edge_buffer,
            )
            if not file_to_chips:
                print("  No chips covered — skipping.")
                del month_records
                clear_memory()
                continue

            if not main_initialized:
                print("  → Writing MAIN H5 files...")
                extract_month(
                    month_records, file_to_chips, chips, ref_meta,
                    year_dir, chip, resampling, min_valid_frac,
                )
                main_initialized = True  
            else:
                print("  → Writing TEMP H5 files...")
                temp_dir.mkdir(parents=True, exist_ok=True)
                extract_month(
                    month_records, file_to_chips, chips, ref_meta,
                    temp_dir, chip, resampling, min_valid_frac,
                )
                print("  → Merging into main H5 files...")
                merge_temp_into_main(year_dir, temp_dir, chip)

            del file_to_chips
            del month_records
            clear_memory()

        print(f"\n[Finalize] Sorting {year} by date...")
        finalize_year(year_dir, year)
        clear_memory()

    print()
    print("=" * 60)
    print("ALL DONE")
    print("=" * 60)

# Entry point
def main() -> None:
    memory_monitor.running = True
    monitor_thread = threading.Thread(
        target=memory_monitor, args=(15,), daemon=True,
    )
    monitor_thread.start()

    try:
        run_pipeline(year_start=2021, year_end=2021)
    finally:
        memory_monitor.running = False

if __name__ == "__main__":
    main()
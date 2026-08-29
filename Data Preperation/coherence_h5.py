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
from typing import Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm
import os, threading, time, resource
from scipy.ndimage import binary_fill_holes
import geopandas as gpd
from shapely.geometry import box
from collections import defaultdict
import datetime

MANIFEST_PATH: Path = Path("/export/students/aryal/VH/coherence_wallonia_descending_VH.json")
REF_TIF  = Path("/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif")
OUT_ROOT = Path("/export/students/aryal/VH/coherence_h5_descending_VH")

CHIP          = 128
STRIDE        = 96
EDGE_BUFFER   = 30.0
MIN_VALID_FRAC = 1.0
VRT_RESAMPLING = Resampling.nearest
H5_COMPRESSION = "lzf"
H5_CHUNK_SHAPE = (1, CHIP, CHIP)
MAX_OPEN_H5    = 128


def load_all_records_coherence(manifest_path: Path) -> Tuple[List[dict], Optional[str]]:
    """Load records from a single manifest JSON file.

    Returns (records, pol) where `pol` is the polarization string
    ("VV" or "VH") read from the manifest's top-level "pol" key,
    or None if it isn't present.
    """
    manifest_path = Path(manifest_path)

    month_re = re.compile(r"^\d{4}-\d{2}$")
    skip_keys = {
        "date_from", "date_to", "orbit", "pol", "n_records",
        "sorted_by", "angle_index_date_to_used",
        "n_angle_indexed_keys", "n_orbit_conflicts_in_angle",
        "n_kept_descending_and_wallonia", "matching_policy", "stats",
    }

    all_records: List[dict] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pol = data.get("pol")

    if "records" in data and isinstance(data["records"], list):
        records = data["records"]
        print(f"  [manifest] {manifest_path.name}: {len(records):,} records "
              f"(orbit={data.get('orbit','?')}  pol={pol or '?'})")
        all_records.extend(records)

    else:
        for k, v in data.items():
            if k in skip_keys:
                continue
            if month_re.match(k) and isinstance(v, list):
                all_records.extend(v)
        print(f"  [manifest] {manifest_path.name}: "
              f"{len(all_records):,} records (month-grouped format, pol={pol or '?'})")

    all_records.sort(key=lambda r: r["t2_end_dt"])
    print(f"  Total records loaded: {len(all_records):,}")
    return all_records, pol


def filter_records_by_month(
    all_records: List[dict], year: int, month: int,
) -> List[dict]:
    prefix = f"{year:04d}-{month:02d}"
    return [rec for rec in all_records if rec["t2_end_dt"][:7] == prefix]

def enumerate_chips_coherence(
    ref_tif: Path,
    chip: int,
    stride: int,
    regions: Optional[List[Tuple[str, int, int, int, int]]] = None,
) -> Tuple[List[Tuple[int, int, Window]], rasterio.Affine, dict]:

    ref_tif = Path(ref_tif)

    with rasterio.open(ref_tif) as ds:
        H, W          = ds.height, ds.width
        ref_transform = ds.transform
        ref_meta      = {
            "crs": ds.crs,
            "transform": ds.transform,
            "width": ds.width,
            "height": ds.height,
        }
        nodata = ds.nodata
        data   = ds.read(1, masked=False)
    if nodata is not None:
        valid_raw = data != nodata
    else:
        valid_raw = np.isfinite(data)
    del data
    aoi_mask = binary_fill_holes(valid_raw)
    del valid_raw
    print(f"[Mask] Filled valid pixels: {aoi_mask.sum():,}")
    if regions is None:
        regions = [("Full Raster", 0, 0, H, W)]
    chips   = []
    skipped = 0
    print(f"[Chips] Raster: {H} × {W} px")
    for name, row_min, col_min, row_max, col_max in regions:
        r_min = max(row_min, 0)
        c_min = max(col_min, 0)
        r_max = min(row_max, H) - chip
        c_max = min(col_max, W) - chip

        if r_max < r_min or c_max < c_min:
            print(f"  {name}: region too small for chip size — skipped entirely.")
            continue

        region_chips = []

        for r0 in range(r_min, r_max + 1, stride):
            for c0 in range(c_min, c_max + 1, stride):

                if not aoi_mask[r0:r0 + chip, c0:c0 + chip].all():
                    skipped += 1
                    continue

                region_chips.append(
                    (r0, c0, Window(c0, r0, chip, chip))
                )
        print(
            f"  {name}: {len(region_chips):,} chips "
            f"rows [{r_min}–{r_max}] cols [{c_min}–{c_max}]"
        )

        chips.extend(region_chips)
    del aoi_mask
    print(f"[Chips] Total valid : {len(chips):,}")
    print(f"[Chips] Skipped     : {skipped:,} (outside filled AOI)")
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


def loc_name(r0: int, c0: int) -> str:
    return f"r{r0:04d}_c{c0:05d}"


class _ChipHandleCoherence:
    __slots__ = ("h5f", "path", "dates", "satellites", "orbits", "cohe_paths")

    def __init__(self, h5f, path, dates, satellites, orbits, cohe_paths):
        self.h5f = h5f
        self.path = path
        self.dates = dates
        self.satellites = satellites
        self.orbits = orbits
        self.cohe_paths = cohe_paths

    def append(self, arr, date_str, sat, orbit, cohe_path):
        ds = self.h5f["X"]
        idx = ds.shape[0]
        ds.resize(idx + 1, axis=0)
        ds[idx, :, :] = arr
        self.dates.append(date_str)
        self.satellites.append(sat)
        self.orbits.append(orbit)
        self.cohe_paths.append(cohe_path)

    def flush_and_close(self):
        try:
            self.h5f.attrs["dates"]      = self.dates
            self.h5f.attrs["satellites"] = self.satellites
            self.h5f.attrs["orbits"]     = self.orbits
            self.h5f.attrs["cohe_paths"] = self.cohe_paths
        finally:
            self.h5f.close()


class LRUChipCacheCoherence:
    def __init__(self, out_dir, chip, max_open=MAX_OPEN_H5, pol=None):
        self._cache: OrderedDict[int, _ChipHandleCoherence] = OrderedDict()
        self._out_dir = out_dir
        self._chip = chip
        self._max_open = max_open
        self._pol = pol
        self.unique_chips_seen = 0

    def get(self, ci, r0, c0):
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

    def close_all(self):
        while self._cache:
            _, handle = self._cache.popitem()
            handle.flush_and_close()

    def _open_or_create(self, r0, c0):
        path = self._out_dir / f"{loc_name(r0, c0)}.h5"
        chip = self._chip
        if path.exists():
            h5f = h5py.File(path, "a")
            dates      = list(h5f.attrs.get("dates",      []))
            satellites = list(h5f.attrs.get("satellites", []))
            orbits     = list(h5f.attrs.get("orbits",     []))
            cohe_paths = list(h5f.attrs.get("cohe_paths", []))
            # Backfill the pol attribute on pre-existing chip files
            if self._pol is not None and "pol" not in h5f.attrs:
                h5f.attrs["pol"] = self._pol
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
            if self._pol is not None:
                h5f.attrs["pol"] = self._pol
            dates, satellites, orbits, cohe_paths = [], [], [], []
        return _ChipHandleCoherence(h5f, path, dates, satellites, orbits, cohe_paths)


def extract_month_coherence(
    records, file_to_chips, chips, ref_meta,
    out_dir, chip, resampling, min_valid_frac, pol=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = LRUChipCacheCoherence(out_dir, chip, max_open=MAX_OPEN_H5, pol=pol)

    n_appends = n_skipped_nan = n_file_errors = 0
    buf = np.empty((chip, chip), dtype=np.float32)

    try:
        for fi in tqdm(sorted(file_to_chips.keys()),
                       desc="  Extracting", unit="file", leave=False):
            rec       = records[fi]
            cohe_path = Path(rec["cohe_path"])
            date_str  = rec["t2_end_dt"][:10]
            sat       = rec.get("sat_from_cohe_name", "")
            orbit     = rec.get("orbit", "")

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
                                data = vrt.read(1, window=vwin).astype(np.float32)
                                dr, dc = rs - r0, cs - c0
                                buf[dr:dr + (re_val - rs), dc:dc + (ce - cs)] = data
                                del data
                            buf[buf == 0.0] = np.nan
                            if np.isfinite(buf).mean() < min_valid_frac:
                                n_skipped_nan += 1
                                continue
                            handle = cache.get(ci, r0, c0)
                            handle.append(buf.copy(), date_str, sat, orbit, str(cohe_path))
                            n_appends += 1
            except Exception as exc:
                tqdm.write(f"    [!] Error: {cohe_path.name}: {exc}")
                n_file_errors += 1
    finally:
        cache.close_all()

    print(f"    Appends: {n_appends:,}  |  "
          f"Skipped(NaN): {n_skipped_nan:,}  |  "
          f"Errors: {n_file_errors}")
    return n_appends


def merge_temp_into_main_coherence(main_dir, temp_dir, chip):
    temp_files = sorted(temp_dir.glob("r*.h5"))
    if not temp_files:
        print("    No temp files to merge.")
        return

    n_merged = 0
    for temp_path in tqdm(temp_files, desc="  Merging", unit="chip", leave=False):
        main_path = main_dir / temp_path.name
        if not main_path.exists():
            shutil.move(str(temp_path), str(main_path))
            n_merged += 1
            continue

        with h5py.File(main_path, "a") as mh, h5py.File(temp_path, "r") as th:
            main_ds = mh["X"]
            T_temp  = th["X"].shape[0]
            if T_temp == 0:
                continue
            t_start = main_ds.shape[0]
            main_ds.resize(t_start + T_temp, axis=0)
            for i in range(T_temp):
                main_ds[t_start + i] = th["X"][i]
            for attr in ("dates", "satellites", "orbits", "cohe_paths"):
                mh.attrs[attr] = list(mh.attrs[attr]) + list(th.attrs[attr])
            if "pol" in th.attrs and "pol" not in mh.attrs:
                mh.attrs["pol"] = th.attrs["pol"]

        temp_path.unlink()
        n_merged += 1

    print(f"    Merged {n_merged:,} chip files")
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass


def clear_memory():
    gc.collect()
    gc.collect()
    print("    [mem] Memory cleared")


def finalize_year_coherence(year_dir, year):
    h5_files = sorted(year_dir.glob("r*.h5"))
    rows = []

    for h5_path in tqdm(h5_files, desc=f"  Finalizing {year}", unit="chip", leave=False):
        with h5py.File(h5_path, "a") as f:
            r0    = int(f.attrs["row_off"])
            c0    = int(f.attrs["col_off"])
            pol   = f.attrs.get("pol", "")
            T     = f["X"].shape[0]
            dates      = list(f.attrs["dates"])
            satellites = list(f.attrs["satellites"])
            orbits     = list(f.attrs["orbits"])
            paths      = list(f.attrs["cohe_paths"])
            order = sorted(range(T), key=lambda i: dates[i])

            if order != list(range(T)):
                chip_h, chip_w = f["X"].shape[1], f["X"].shape[2]
                tmp     = np.empty((chip_h, chip_w), dtype=np.float32)
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
                        f["X"][j] = tmp if src == start else f["X"][src]
                        j = src
                del tmp
                dates      = [dates[i]      for i in order]
                satellites = [satellites[i] for i in order]
                orbits     = [orbits[i]     for i in order]
                paths      = [paths[i]      for i in order]
                f.attrs["dates"]      = dates
                f.attrs["satellites"] = satellites
                f.attrs["orbits"]     = orbits
                f.attrs["cohe_paths"] = paths

            rows.append({
                "chip_file":  h5_path.name,
                "row_off":    r0,
                "col_off":    c0,
                "pol":        pol,
                "T":          T,
                "dates":      ",".join(dates),
                "date_first": dates[0] if dates else "",
                "date_last":  dates[-1] if dates else "",
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(year_dir / "_summary.csv", index=False)
    print(f"\n  Year {year} Summary:")
    print(f"    Chip files : {len(summary):,}")
    if len(summary) > 0:
        print(f"    T range    : {summary['T'].min()} – {summary['T'].max()}")
        print(f"    T mean     : {summary['T'].mean():.1f}")
    return summary

def _parse_dt(s: str) -> datetime.datetime:
    """Parse ISO datetime string safely."""
    return datetime.datetime.fromisoformat(s)
from collections import defaultdict
import datetime
import shutil


def _parse_dt(s: str) -> datetime.datetime:
    """Parse ISO datetime string safely."""
    return datetime.datetime.fromisoformat(s)


def run_pipeline_coherence(
    start_dt:       datetime.datetime,
    end_dt:         datetime.datetime,
    manifest_path:  Path,
    ref_tif:        Path,
    out_root:       Path,
    chip:           int,
    stride:         int,
    edge_buffer:    float,
    min_valid_frac: float,
    resampling:     Resampling,
    regions: Optional[List[Tuple[str, int, int, int, int]]] = None,
) -> None:

    print("=" * 60)
    print("HDF5 Time-Series Pipeline")
    print(f"  Start date : {start_dt:%Y-%m-%d}")
    print(f"  End date   : {end_dt:%Y-%m-%d}")
    print(f"  Manifest   : {manifest_path}")
    print(f"  Output     : {out_root}")
    print(f"  Max open H5: {MAX_OPEN_H5}")
    print("=" * 60)

    if start_dt > end_dt:
        raise ValueError("start_dt must be before end_dt")

    print("\n[Step 0] Loading manifest...")
    all_records, pol = load_all_records_coherence(manifest_path)

    print(f"  Polarization: {pol or 'unknown'}")
    print(f"  Total records loaded: {len(all_records)}")

    filtered_records = []

    for r in all_records:
        dt = _parse_dt(r["t2_end_dt"])

        if start_dt <= dt <= end_dt:
            r["_dt"] = dt
            filtered_records.append(r)

    all_records = filtered_records

    print(f"  Records in range: {len(all_records)}")

    if not all_records:
        print("No records found in requested date range.")
        return

    print("\n[Step 1] Enumerating chips...")
    chips, ref_transform, ref_meta = enumerate_chips_coherence(
        ref_tif,
        chip,
        stride,
        regions
    )

    print(f"  Expected chip files: {len(chips):,}")

    grouped = defaultdict(list)

    for r in all_records:
        dt = r["_dt"]
        grouped[(dt.year, dt.month)].append(r)

    current_year = None
    year_dir = None
    temp_dir = None
    main_initialized = False
    for (year, month) in sorted(grouped):

        if year != current_year:

            if current_year is not None:
                print(f"\n[Finalize] Sorting {current_year} by date...")
                finalize_year_coherence(year_dir, current_year)
                clear_memory()

            current_year = year

            year_dir = out_root / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)

            temp_dir = out_root / f"_temp_{year}"

            if temp_dir.exists():
                print("  [!] Cleaning leftover temp dir")
                shutil.rmtree(temp_dir)

            print("\n" + "=" * 60)
            print(f"YEAR {year}")
            print("=" * 60)

            main_initialized = any(year_dir.glob("r*.h5"))

        month_records = grouped[(year, month)]
        month_label = f"{year}-{month:02d}"

        print(f"\n--- {month_label} ---")
        print(f"  Records: {len(month_records)}")

        file_to_chips = build_file_to_chips(
            month_records,
            chips,
            ref_transform,
            chip,
            edge_buffer
        )

        if not file_to_chips:
            print("  No chips covered — skipping.")
            clear_memory()
            continue
        if not main_initialized:

            print("  → Writing MAIN H5 files...")

            extract_month_coherence(
                month_records,
                file_to_chips,
                chips,
                ref_meta,
                year_dir,
                chip,
                resampling,
                min_valid_frac,
                pol=pol
            )

            main_initialized = True

        else:

            print("  → Writing TEMP H5 files...")

            temp_dir.mkdir(parents=True, exist_ok=True)

            extract_month_coherence(
                month_records,
                file_to_chips,
                chips,
                ref_meta,
                temp_dir,
                chip,
                resampling,
                min_valid_frac,
                pol=pol
            )

            print("  → Merging into main H5 files...")

            merge_temp_into_main_coherence(
                year_dir,
                temp_dir,
                chip
            )

        del file_to_chips, month_records
        clear_memory()
    if current_year is not None:
        print(f"\n[Finalize] Sorting {current_year} by date...")
        finalize_year_coherence(year_dir, current_year)
        clear_memory()

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)

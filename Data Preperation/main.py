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
from typing import Dict, List, Optional, Tuple, Sequence, Any, Iterable, Union
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm
from numcodecs import Blosc
from rasterio.transform import Affine
import datetime
import glob
import gzip
from dataclasses import dataclass
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep
from collections import defaultdict
import h5py
from scipy.ndimage import binary_fill_holes
import geopandas as gpd
from shapely.geometry import box
from sentinel2_h5 import *
from coherence_h5 import *
from backscattering_h5 import *
from label_data_preperation import *

def compute_fmask_noise_percentage(fmask_path:str, noise_values: Sequence[int]=(2, 3, 4, 255)) -> Optional[float]:
    try:
        with rasterio.open(fmask_path) as src:
            arr = src.read(1)
    except Exception as e:
        print(f"[!] Could not read FMASK {fmask_path}: {e}")
        return None
    total_pixels = arr.size
    if total_pixels == 0:
        return None
    noise_mask = np.isin(arr, noise_values)
    noise_pixels = noise_mask.sum()
    return 100.0 * noise_pixels / float(total_pixels)

def find_sentinel_images(s2_folder: str, bands: Sequence[int]) -> List[str]:
    pat = re.compile(rf"_FRE_B0?({'|'.join(map(str, bands))})\.jp2$")
    return sorted(
        fp for fp in glob.glob(os.path.join(s2_folder, "*FRE_B*.jp2"))
        if pat.search(os.path.basename(fp))
    )

def find_mask(mask_path: str, tile: str, year: int, date_obj: datetime.datetime,
              rgbnir_files: Sequence[str], noise_values: Sequence[int],
              compute_cloud_pct: bool = True) -> Tuple[Optional[str], Optional[float]]:
    if not rgbnir_files:
        return None, None
    date_compact = date_obj.strftime("%Y%m%d")
    first_name = os.path.basename(rgbnir_files[0])
    satellite = first_name.split("_")[0].replace("SENTINEL2", "S2")
    fmask_pattern = os.path.join(
        mask_path, tile, str(year),
        f"{satellite}_MSIL1C_{date_compact}T*_T{tile}_{date_compact}T*_0pct*.tif"
    )
    candidates = glob.glob(fmask_pattern)
    if not candidates:
        return None, None
    fmask_used = candidates[0]
    cloud_cover_pct = (
        compute_fmask_noise_percentage(fmask_used, noise_values)
        if compute_cloud_pct else None
    )
    return fmask_used, cloud_cover_pct

def build_records(base_path: str, mask_path: str, tiles: Sequence[str], out_path: str,
                  start_date: Optional[str], end_date: Optional[str],
                  bands: Sequence[int], noise_values: Sequence[int],
                  compute_cloud_pct: bool = True):   # <-- new parameter
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    year_start = start_date.year
    year_end = end_date.year
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for year in range(year_start, year_end + 1):
            for tile in tiles:
                tile_path = os.path.join(base_path, tile, str(year))
                if not os.path.exists(tile_path):
                    print(f"[!] No data for {tile}/{year}, skipping...")
                    continue
                for folder in sorted(os.listdir(tile_path)):
                    parts = folder.split("_")
                    if len(parts) <= 1:
                        continue
                    try:
                        date_str = parts[1][:8]
                        date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
                    except ValueError:
                        continue
                    if not (start_date <= date_obj <= end_date):
                        continue
                    s2_path = os.path.join(tile_path, folder)
                    files = find_sentinel_images(s2_path, bands=bands)
                    if not files:
                        continue
                    fmask_used, cloud_cover_pct = find_mask(
                        mask_path=mask_path,
                        tile=tile, year=year,
                        date_obj=date_obj,
                        rgbnir_files=files,
                        noise_values=noise_values,
                        compute_cloud_pct=compute_cloud_pct,  # <-- pass it down
                    )
                    rec = {
                        "year": year,
                        "tile": tile,
                        "date": date_obj.strftime("%Y-%m-%d"),
                        "folder": folder,
                        "files": files,
                        "fmask": fmask_used,
                    }
                    if cloud_cover_pct is not None:
                        rec["cloud_cover_pct"] = cloud_cover_pct

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    
def convert_jsonlgz_to_json(input_path: str, output_path: str, indent: int = 2) -> None:
    """
    Convert a compressed JSONL (.jsonl.gz) file into a plain JSON array file.

    Args:
        input_path  : path to the .jsonl.gz file
        output_path : path to write the output .json file
        indent      : indentation for pretty-printing (default 2, set None for compact)
    """
    import gzip
    import json
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    records = []
    with gzip.open(input_path, "rt", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:  # skip empty lines
                records.append(json.loads(line))
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(records, fout, ensure_ascii=False, indent=indent)
    print(f"[i] Converted {len(records)} records")
    print(f"    from : {input_path}")
    print(f"    to   : {output_path}")

############################################# Sentinel- 1 ####################################################################
_TS_PAIR_RE = re.compile(r"(?P<t1>\d{8}T\d{6})_(?P<t2>\d{8}T\d{6})")
_SAT_RE = re.compile(
    r"^(?P<sat>S1[AB])_IW_SLC__1SDV_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_"
)
_ORBIT_RE   = re.compile(r"_(?P<orb>ASCENDING|DESCENDING)_", re.IGNORECASE)
_POL_END_RE = re.compile(r"_(?P<pol>VV|VH|HH|HV)(?:_comp)?\.tif$", re.IGNORECASE)

def _parse_dt(ts: str) -> datetime.datetime:
    return datetime.datetime.strptime(ts, "%Y%m%dT%H%M%S")

@dataclass(frozen=True)
class ParsedName:
    sat:      str
    start_dt: datetime.datetime
    end_dt:   datetime.datetime
    pol:      str
    orbit:    Optional[str]

def parse_s1_tif_name(p: Path) -> ParsedName:
    name  = p.name
    m_sat = _SAT_RE.search(name)
    if not m_sat:
        raise ValueError(f"Cannot parse satellite (S1A/S1B) from: {name}")
    sat      = m_sat.group("sat").upper()
    m_ts     = _TS_PAIR_RE.search(name)
    if not m_ts:
        raise ValueError(f"Cannot parse timestamp pair from: {name}")
    start_dt = _parse_dt(m_ts.group("t1"))
    end_dt   = _parse_dt(m_ts.group("t2"))
    m_pol    = _POL_END_RE.search(name)
    pol      = m_pol.group("pol").upper() if m_pol else "UNK"
    m_orb    = _ORBIT_RE.search(name)
    orbit    = m_orb.group("orb").upper() if m_orb else None
    return ParsedName(sat=sat, start_dt=start_dt, end_dt=end_dt, pol=pol, orbit=orbit)

def parse_date_like(
    x: Union[str, datetime.date, datetime.datetime], is_end: bool
) -> datetime.datetime:
    if isinstance(x, datetime.datetime):
        if is_end and x.time() == datetime.time.min:
            return x.replace(hour=23, minute=59, second=59, microsecond=999999)
        return x
    if isinstance(x, datetime.date):
        return datetime.datetime(x.year, x.month, x.day, 23, 59, 59) if is_end \
               else datetime.datetime(x.year, x.month, x.day, 0, 0, 0)
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        d = datetime.datetime.strptime(s, "%Y%m%d")
        return d.replace(hour=23, minute=59, second=59) if is_end else d
    s = s.replace("Z", "")
    try:
        return datetime.datetime.fromisoformat(s)
    except ValueError:
        pass
    d = datetime.date.fromisoformat(s)
    return datetime.datetime(d.year, d.month, d.day, 23, 59, 59) if is_end \
           else datetime.datetime(d.year, d.month, d.day, 0, 0, 0)

def iter_year_dirs(root: Path, year_min: int, year_max: int) -> Iterable[Path]:
    if not root.exists():
        return
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.isdigit():
            y = int(d.name)
            if year_min <= y <= year_max:
                yield d

def iter_tifs_in_range(
    root: Path, subdir: str,
    dt_min: datetime.datetime, dt_max: datetime.datetime,
) -> Iterable[Path]:
    for ydir in iter_year_dirs(root, dt_min.year, dt_max.year):
        base = ydir / subdir
        if not base.exists():
            continue
        for tif in base.rglob("*.tif"):
            try:
                parsed = parse_s1_tif_name(tif)
            except Exception:
                continue
            if dt_min <= parsed.end_dt <= dt_max:
                yield tif

def load_wallonia_union(wallonia_gpkg: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(wallonia_gpkg)
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf

def make_wallonia_prepared_cache(wall_gdf: gpd.GeoDataFrame):
    cache: Dict[str, Any] = {}
    def get_prepared_for(crs) -> Any:
        if crs is None:
            raise ValueError("Raster has no CRS.")
        key = crs.to_string()
        if key in cache:
            return cache[key]
        gg = wall_gdf if wall_gdf.crs == crs else wall_gdf.to_crs(crs)
        cache[key] = prep(unary_union(list(gg.geometry)))
        return cache[key]
    return get_prepared_for

BUCKET_FILENAMES: Dict[Tuple[str, str], str] = {
    ("ASCENDING",  "VV"): "coherence_wallonia_ascending_VV.json",
    ("ASCENDING",  "VH"): "coherence_wallonia_ascending_VH.json",
    ("DESCENDING", "VV"): "coherence_wallonia_descending_VV.json",
    ("DESCENDING", "VH"): "coherence_wallonia_descending_VH.json",
}

def select_coherence_for_wallonia(
    cohe_root:               Union[str, Path],
    angle_root:              Union[str, Path],
    wallonia_gpkg:           Union[str, Path],
    out_dir:                 Union[str, Path],
    date_from:               Union[str, datetime.date, datetime.datetime],
    date_to:                 Union[str, datetime.date, datetime.datetime],
    cohe_subdir:             str           = "COHE",
    angle_subdir:            str           = "PREPRO",
    buffer_days_for_t2:      int           = 14,
    pols:                    Sequence[str] = ("VV", "VH"),
    dedupe_t2_tolerance_sec: int           = 60,
) -> Dict[str, Any]:
    cohe_root     = Path(cohe_root)
    angle_root    = Path(angle_root)
    wallonia_gpkg = Path(wallonia_gpkg)
    out_dir       = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_min   = parse_date_like(date_from, is_end=False)
    dt_max   = parse_date_like(date_to,   is_end=True)
    pols_set = {p.upper() for p in pols}
    dt_max_angle = dt_max + datetime.timedelta(days=buffer_days_for_t2)
    angle_lookup:   Dict[Tuple[str, str], Dict[str, str]] = {}
    orbit_conflicts: List[Dict[str, Any]]                 = []
    for tif in iter_tifs_in_range(angle_root, angle_subdir, dt_min, dt_max_angle):
        try:
            p = parse_s1_tif_name(tif)
        except Exception:
            continue
        if p.pol.upper() not in pols_set or not p.orbit:
            continue
        t1   = p.start_dt.strftime("%Y%m%dT%H%M%S")
        k    = (t1, p.pol.upper())
        prev = angle_lookup.get(k)
        if prev is None:
            angle_lookup[k] = {"orbit": p.orbit, "path": str(tif), "sat": p.sat}
        else:
            if prev["orbit"].upper() != p.orbit.upper():
                orbit_conflicts.append({
                    "key":  k,
                    "prev": prev,
                    "new":  {"orbit": p.orbit, "path": str(tif), "sat": p.sat},
                })
    wall_gdf          = load_wallonia_union(wallonia_gpkg)
    get_wall_prepared = make_wallonia_prepared_cache(wall_gdf)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
        k: [] for k in BUCKET_FILENAMES
    }
    bad_files:       List[Dict[str, Any]]                         = []
    seen_near_dupes: Dict[Tuple[Any, ...], List[datetime.datetime]] = {}
    for cohe_tif in iter_tifs_in_range(cohe_root, cohe_subdir, dt_min, dt_max):
        try:
            cp = parse_s1_tif_name(cohe_tif)
            if cp.pol.upper() not in pols_set:
                continue
            t2     = cp.end_dt.strftime("%Y%m%dT%H%M%S")
            match2 = angle_lookup.get((t2, cp.pol.upper()))
            if match2 is None:
                continue
            orbit    = match2["orbit"].upper()
            pol      = cp.pol.upper()
            buck_key = (orbit, pol)
            if buck_key not in buckets:
                continue
            with rasterio.open(cohe_tif) as src:
                if src.crs is None:
                    raise ValueError("COHE tif has no CRS.")
                wall_prepared = get_wall_prepared(src.crs)
                if not wall_prepared.intersects(box(*src.bounds)):
                    continue
                bounds_key = (
                    round(float(src.bounds.left),   3),
                    round(float(src.bounds.bottom), 3),
                    round(float(src.bounds.right),  3),
                    round(float(src.bounds.top),    3),
                )
                dedupe_key = (
                    orbit, pol,
                    cp.start_dt.isoformat(),
                    src.crs.to_string(),
                    int(src.width), int(src.height),
                    bounds_key,
                )
                prev_t2s = seen_near_dupes.get(dedupe_key, [])
                if any(
                    abs((cp.end_dt - prev_t2).total_seconds()) <= dedupe_t2_tolerance_sec
                    for prev_t2 in prev_t2s
                ):
                    continue
                prev_t2s.append(cp.end_dt)
                seen_near_dupes[dedupe_key] = prev_t2s
                record: Dict[str, Any] = {
                    "cohe_path":          str(cohe_tif),
                    "cohe_filename":      cohe_tif.name,
                    "angle_path":         match2["path"],
                    "orbit":              orbit,
                    "sat_from_cohe_name": cp.sat,
                    "pol":                pol,
                    "t1_start_dt":        cp.start_dt.isoformat(),
                    "t2_end_dt":          cp.end_dt.isoformat(),
                    "crs":                src.crs.to_string(),
                    "width":              int(src.width),
                    "height":             int(src.height),
                    "transform": [
                        float(src.transform.a), float(src.transform.b),
                        float(src.transform.c), float(src.transform.d),
                        float(src.transform.e), float(src.transform.f),
                    ],
                    "bounds": [
                        float(src.bounds.left),  float(src.bounds.bottom),
                        float(src.bounds.right), float(src.bounds.top),
                    ],
                }
            buckets[buck_key].append(record)
        except Exception as e:
            bad_files.append({"path": str(cohe_tif), "error": str(e)})

    saved_paths: Dict[str, str] = {}
    for buck_key, records in buckets.items():
        records.sort(key=lambda r: r["t2_end_dt"])
        orbit, pol = buck_key
        filename   = BUCKET_FILENAMES[buck_key]
        out_path   = out_dir / filename
        payload = {
            "date_from": dt_min.isoformat(),
            "date_to":   dt_max.isoformat(),
            "orbit":     orbit,
            "pol":       pol,
            "n_records": len(records),
            "sorted_by": "t2_end_dt",
            "records":   records,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        saved_paths[filename] = str(out_path)
        print(f"  Saved {len(records):>5,} records → {out_path}")
    bad_path = out_dir / "bad_files.json"
    bad_path.write_text(json.dumps(bad_files, indent=2), encoding="utf-8")
    summary = {
        "saved_files":             saved_paths,
        "n_angle_indexed_keys":    len(angle_lookup),
        "n_orbit_conflicts_angle": len(orbit_conflicts),
        "n_bad":                   len(bad_files),
        "counts_per_bucket": {
            BUCKET_FILENAMES[k]: len(v) for k, v in buckets.items()
        },
    }
    return summary

############ Interactive Script ############################################
def ask_choice(
    prompt: str,
    options: List[str],
    default: str | None = None,
) -> str:
    options_lower = [o.lower() for o in options]
    if default is not None and default.lower() not in options_lower:
        raise ValueError(f"Default '{default}' is not in options")
    display = "/".join(options)
    while True:
        if default is not None:
            s = input(f"{prompt} [{display}] (default: {default}): ").strip().lower()
            if not s:
                return default.lower()
        else:
            s = input(f"{prompt} [{display}]: ").strip().lower()
        if s in options_lower:
            return s
        print(f"[!] Please enter one of: {display}")
        
def _expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def ask_str(prompt: str, default: Optional[str] = None) -> str:
    if default is not None:
        s = input(f"{prompt} [{default}]: ").strip()
        return s if s else default
    else:
        while True:
            s = input(f"{prompt}: ").strip()
            if s:
                return s
            
def ask_path(prompt: str, default: Optional[str] = None, must_exist: bool = True) -> str:
    while True:
        p = _expand(ask_str(prompt, default))
        if not must_exist or os.path.exists(p):
            return p
        print(f"[!] Path not found: {p}")

wallonia_tiles = {
    "31UDS", "31UES", "31UFS", "31UGS",
    "31UER", "31UFR", "31UGR", "31UFQ",
}
def ask_tiles(prompt: str, default: str, allowed: set[str]) -> list[str]:
    allowed_u = {t.upper() for t in allowed}
    all_tokens = {"ALL", "*"}

    default_u = (default or "").strip().upper()

    if default_u and default_u not in allowed_u and default_u not in all_tokens:
        raise ValueError(f"Default tile '{default}' is not in allowed tiles")

    while True:
        tiles = ask_csv_list(prompt, default) or []

        tiles_u = [str(t).strip().upper() for t in tiles if str(t).strip()]

        if not tiles_u:
            return sorted(allowed_u) if default_u in all_tokens else ([default_u] if default_u else [])

        if any(t in all_tokens for t in tiles_u):
            return sorted(allowed_u)

        invalid = [t for t in tiles_u if t not in allowed_u]
        if invalid:
            print(f"Invalid tile(s): {', '.join(invalid)}")
            print(f"Allowed tiles: {', '.join(sorted(allowed_u))}")
            continue

        seen = set()
        tiles_u = [t for t in tiles_u if not (t in seen or seen.add(t))]
        return tiles_u

def ask_csv_list(prompt: str, default_csv: str) -> List[str]:
    raw = ask_str(prompt, default_csv)
    return [x.strip() for x in raw.split(",") if x.strip()]

def ask_csv_ints(prompt: str, default_csv: str) -> List[int]:
    while True:
        try:
            return [int(x.strip()) for x in ask_str(prompt, default_csv).split(",") if x.strip()]
        except ValueError:
            print("[!] Enter comma-separated integers, e.g. 2,3,4,8")


def prompt_date_range() -> tuple[datetime.datetime, datetime.datetime]:
    while True:
        start_in = input("Enter start date (YYYY-MM-DD or YYYYMMDD): ").strip()
        end_in   = input("Enter end date   (YYYY-MM-DD or YYYYMMDD): ").strip()
        try:
            start_dt = parse_date_like(start_in, is_end=False)
            end_dt   = parse_date_like(end_in, is_end=True)
        except Exception as e:
            print(f"[!] Invalid date format: {e}")
            print("    Please try again.\n")
            continue
        if start_dt > end_dt:
            print("[!] Start date must be before (or equal to) end date. Please try again.\n")
            continue
        return start_dt, end_dt

def ask_bool(prompt: str, default: bool = True) -> bool:
    d = "y" if default else "n"
    s = ask_str(f"{prompt} [y/n]", d).strip().lower()
    return s in ("y", "yes", "1", "true", "t")

REGIONS = [
    ("Region 1", 1536,  12544,  4736,  22400),
    ("Region 2", 5888,  14976, 11264,  20608),
]

def interactive_s2():
    print("\n=== Sentinel-2 Data Preparation Pipeline ===\n")
    print("--- Step 1: Build Scene Index ---\n")
    base_path = ask_path(
        "Base path (contains tile/year folders)",
        default="/export/images/Sentinel2/Belgium/L2A_MAJA481",
    )
    mask_path = ask_path(
        "FMASK root path",
        default="/export/images/Sentinel2/Belgium/Fmask46",
    )
    index_out = Path(_expand(ask_str(
        "Output path for index file (.jsonl.gz)",
        default="/export/students/aryal/Final Submission Script/Data Preperation/s2info.jsonl.gz",
    )))
    tiles             = ask_tiles("Tiles (comma-separated)", "31UFR", wallonia_tiles)
    band_numbers      = ask_csv_ints("Bands (comma-separated)", "2,3,4,8")
    start_dt, end_dt  = prompt_date_range()
    noise_values      = ask_csv_ints("FMASK noise values to remove (comma-separated)", "2,3,4,255")
    compute_cloud_pct = ask_bool("Compute cloud coverage % per scene?", default=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)
    print("\n[i] Scanning Sentinel-2 tiles...")
    build_records(
        base_path=base_path,
        mask_path=mask_path,
        tiles=tiles,
        out_path=str(index_out),
        start_date=start_dt,
        end_date=end_dt,
        bands=band_numbers,
        noise_values=noise_values,
        compute_cloud_pct=compute_cloud_pct,
    )
    default_json = str(index_out).replace(".jsonl.gz", ".json")
    s2_info_json = Path(default_json)
    convert_jsonlgz_to_json(input_path=str(index_out), output_path=str(default_json))
    s2_info_json = default_json
    print("\n--- Step 2: H5 Chip Generation ---\n")
    band_names = [f"B{b}" for b in band_numbers]
    N_BANDS    = len(band_names)
    max_open_h5 = int(ask_str("Max open H5 file handles (LRU cache size)", default="128"))
    s2_info_path = Path(ask_path(
        "Path to s2info.json (scene index)",
        default=str(s2_info_json),
        must_exist=True,
    ))
    ref_tif = Path(ask_path(
        "Reference raster (.tif) path",
        default="/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif",
        must_exist=True,
    ))
    out_root = Path(_expand(ask_str(
        "Output root directory for H5 files",
        default="/export/students/aryal/s2_h5",
    )))
    chip_size   = int(ask_str("Chip size in pixels", default="128"))
    stride_size = int(ask_str("Stride in pixels (= chip for no overlap)", default=str(chip_size)))
    max_cloud   = float(ask_str("Max cloud % allowed per chip (0–100)", default="50.0"))
    min_valid   = float(ask_str("Min valid pixel fraction per chip (0.0–1.0)", default="1.0"))
    fmask_cloud_values = set(noise_values) - {255}    
    resamp_choice = ask_choice(
        "Resampling method for VRT warp",
        options=["nearest", "bilinear", "cubic"],
        default="nearest"
    )
    resamp_map = {
        "nearest":  Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic":    Resampling.cubic,
    }
    region_mode = ask_choice(
        "Chip generation area",
        options=[
            "Whole Wallonia",
            "Predefined regions",
        ],
        default = "Predefined regions"
    )
    regions = None if region_mode == "Whole Wallonia" else REGIONS
    run_pipeline_S2_h5(
        start_date    = start_dt.strftime("%Y-%m-%d"),
        end_date      = end_dt.strftime("%Y-%m-%d"),
        s2_info_path  = s2_info_path,
        ref_tif       = ref_tif,
        out_root      = out_root,
        chip          = chip_size,
        stride        = stride_size,
        min_valid_frac= min_valid,
        max_cloud_pct = max_cloud,
        resampling    = resamp_map[resamp_choice],
        band_names     = band_names,
        max_open_h5    = max_open_h5,       
        fmask_cloud_values = fmask_cloud_values,
        regions=regions,
    )
    
def interactive_s1():
    print("\n=== Sentinel-1 Coherence Index Builder ===\n")
    cohe_root = ask_path(
        "Coherence root path",
        default="/export/images/Sentinel1/Belgium/coherence/Belgium_all_6j",
        must_exist=True
    )
    angle_root = ask_path(
        "Angle root path",
        default="/export/images/Sentinel1/Belgium/amplitude_10m/Belgium_angle",
        must_exist=True
    )
    wallonia_gpkg = ask_path(
        "Wallonia boundary GPKG path",
        default="/export/students/aryal/AFBD_existing/Wallonia.gpkg",
        must_exist=True
    )
    out_dir = Path(_expand(ask_str(
        "Output directory for JSON files",
        default="/export/students/aryal/Final Submission Script/Data Preperation"
    )))
    start_dt, end_dt = prompt_date_range()
    pols_raw = ask_csv_list("Polarizations (comma-separated)", "VV,VH")
    pols     = [p.strip().upper() for p in pols_raw]
    print("\n[i] Scanning Sentinel-1 coherence files... this may take a while.")
    summary = select_coherence_for_wallonia(
        cohe_root     = cohe_root,
        angle_root    = angle_root,
        wallonia_gpkg = wallonia_gpkg,
        out_dir       = out_dir,
        date_from     = start_dt,
        date_to       = end_dt,
        pols          = pols,
    )
    print(f"\n[i] Done. Files saved to: {out_dir}")
    print("\n--- Step 2: H5 Chip Generation ---\n")
    chip_size   = int(ask_str("Chip size in pixels", default="128"))
    stride_size = int(ask_str("Stride in pixels (= chip for no overlap)", default=str(chip_size)))
    min_valid   = float(ask_str("Min valid pixel fraction per chip (0.0–1.0)", default="1.0"))
    h5_chunk_shape = (1, chip_size, chip_size)
    max_open_h5 = int(ask_str("Max open H5 file handles (LRU cache size)", default="128"))
    h5_compression = "lzf"
    resamp_choice = ask_choice(
        "Resampling method for VRT warp",
        options=["nearest", "bilinear", "cubic"],
        default="nearest"
    )
    resamp_map = {
        "nearest":  Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic":    Resampling.cubic,
    }
    ref_tif = Path(ask_path(
        "Reference raster (.tif) path",
        default="/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif",
        must_exist=True,
    ))
    region_mode = ask_choice(
        "Chip generation area",
        options=[
            "Whole Wallonia",
            "Predefined regions",
        ],
        default = "Predefined regions"
    )
    regions = None if region_mode == "Whole Wallonia" else REGIONS
    sensor = ask_choice(
        "Which S1 product do you want to process?",
        options=["coherence", "backscatter"]
    )
    if(sensor == "coherence"):
        manifest_path_coherence = ask_path(
            "Coherence metadata json path",
            default="/export/students/aryal/Final Submission Script/Data Preperation/coherence_wallonia_ascending_VV.json",
            must_exist=True
        )
        out_root_coherence = Path(_expand(ask_str(
            "Output root directory for H5 files",
            default="/export/students/aryal/Final Submission Script/Data Preperation/Coherence/Ascending/VV",
        )))
        start_dt, end_dt = prompt_date_range()
        run_pipeline_coherence(
            start_dt=start_dt,
            end_dt = end_dt,
            manifest_path = manifest_path_coherence,
            ref_tif=ref_tif,
            out_root = out_root_coherence,
            chip = chip_size,
            stride = stride_size,
            edge_buffer=30,
            min_valid_frac=min_valid,
            resampling=resamp_map[resamp_choice],
            regions=regions
        )
    elif (sensor == "backscatter"):
        manifest_path_backscatter = ask_path(
            "Coherence metadata json path",
            default="/export/students/aryal/Final Submission Script/Data Preperation/coherence_wallonia_ascending_VV.json",
            must_exist=True
        )
        out_root_coherence = Path(_expand(ask_str(
            "Output root directory for H5 files",
            default="/export/students/aryal/Final Submission Script/Data Preperation/Backscatter/Ascending/VV",
        )))
        start_dt, end_dt = prompt_date_range()
        run_pipeline_backscatter(
            start_dt=start_dt,
            end_dt = end_dt,
            manifest_path = manifest_path_backscatter,
            ref_tif=ref_tif,
            out_root = out_root_coherence,
            chip = chip_size,
            stride = stride_size,
            edge_buffer=30,
            min_valid_frac=min_valid,
            resampling=resamp_map[resamp_choice],
            regions=regions
        )

def label_dataset_Preperation():
    vector_label_path = ask_path("Boundary Vector Layer Path", default="/export/students/aryal/AFBD_existing", must_exist=True)
    store_label_data = ask_path("Label Data Storage Path", default="/export/students/aryal/label", must_exist=True)
    ref_tif = Path(ask_path(
        "Reference raster (.tif) path",
        default="/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif",
        must_exist=True,
    ))
    build_labels_for_all_years(
        root_dir = Path(vector_label_path),
        ref_raster =ref_tif,
        out_root = Path(store_label_data),
        overwrite = False,
        gpkg_glob = "*.gpkg",
    )
    out_chip_location = ask_path("Label Data Chip Storage Path", default="/export/students/aryal/label_chip", must_exist=True)
    out_gpkg_location = ask_path("Chip structure gpkg", default="/export/students/aryal/label_chip/label_chips_128.gpkg", must_exist=False)
    years = ask_csv_list("Years to process (comma-separated)", "2018,2019,2020,2021")
    chip_size   = int(ask_str("Chip size in pixels", default="128"))
    stride_size = int(ask_str("Stride in pixels (= chip for no overlap)", default=str(chip_size)))
    build_label_chips(
    ref_tif= ref_tif,
    label_dir= store_label_data,
    out_dir= out_chip_location,
    out_gpkg = out_gpkg_location,
    years=years,
    chip=chip_size,
    stride=stride_size)   
     
if __name__ == "__main__":
    stage = ask_choice(
        "Which Stage do you want to process?",
        options=["Label","HDF5", "Zarr"]
    )
    if (stage == "label"):
        print("\n========================================")
        print("   Label Data Preparation Pipeline   ")
        print("========================================\n")
        label_dataset_Preperation()
        
    elif(stage == "hdf5"):
        print("\n========================================")
        print("   Sentinel Data Preparation Pipeline   ")
        print("========================================\n")
        sensor = ask_choice(
            "Which sensor do you want to process?",
            options=["S1", "S2"]
        )
        if sensor == "s1":
            interactive_s1()
        elif sensor == "s2":
            interactive_s2()
            
    elif(stage == "zarr"):
        print("\n========================================")
        print("Zarr File Generation")
        print("========================================\n")
        sensorConfiguration = ask_choice(
            "Which configuration do you want to process?",
            options=["S2", "S1", "S1+S2"]
        )
        if(sensorConfiguration == "s2"):
            from S2_zarr import *
            year         = int(ask_str("Which year do you want to process ? ", default="2018"))
            s2_root      = Path(ask_path("S2 HDF5 Path", default=Path(f"/export/students/aryal/s2_h5/{year}"), must_exist=True))
            label_root   = Path(ask_path("Label Data location", default=Path(f"/export/students/aryal/label_chip/{year}"), must_exist=True))
            out_zarr     = Path(ask_path("Zarr Output Location:", default=Path(f"/export/students/aryal/Final Submission Script/Data Preperation/zarr/S2/s2_{year}.zarr"), must_exist=False))
            ref_tif = Path(ask_path(
                "Reference raster (.tif) path",
                default="/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif",
                must_exist=True,
            ))
            chip         = int(ask_str("Chip size in pixels", default="128"))
            t_block      = int(ask_str("Time Dimension in your DataCube", default="10"))
            B            = int(ask_str("Channel Dimension in your DataCube", default="4"))
            L            = int(ask_str("Label dataset Dimension", default="3"))
            sample_chunk = int(ask_str("Chunk Size (larger chunk less file count)", default="64"))
            band_names   = ["B2", "B3", "B4", "B8"]
            label_names  = ["extent", "boundary", "dist"]
            label_ch     = [0, 1, 2]
            overwrite    = False
            buildS2Zarr(
                year = year,
                s2_root  = s2_root,
                label_root  = label_root,
                out_zarr  = out_zarr,
                ref_tif   = ref_tif,
                chip      =  chip,
                t_block   = t_block,
                B         = B,
                L         = L,
                sample_chunk = sample_chunk,
                band_names   = band_names,
                label_names  = label_names,
                label_ch     = label_ch,
                overwrite    = overwrite,
            ) 
        elif(sensorConfiguration == "s1"):
            from S1_zarr import *
            year = int(ask_str("Which year do you want to process ? ", default="2020"))
            ca_base = Path(ask_path(
                "Path to your HDF5 Coherence orbit Ascending",
                default="/export/students/aryal/VH/coherence_h5_ascending_VH",
                must_exist=True,
            ))
            cd_base = Path(ask_path(
                "Path to your HDF5 Coherence orbit Descending",
                default="/export/students/aryal/VH/coherence_h5_descending_VH",
                must_exist=True,
            ))
            bd_base = Path(ask_path(
                "Path to your HDF5 Backscattering orbit Descending",
                default="/export/students/aryal/VH/backscattering_descending_h5",
                must_exist=True,
            ))
            ba_base= Path(ask_path(
                "Path to your HDF5 Backscattering orbit Ascending",
                default="/export/students/aryal/VH/backscattering_ascending_h5",
                must_exist=True,
            ))
            label_train_base = Path(ask_path(
                "Path to your HDF5 Backscattering orbit Ascending",
                default="/export/students/aryal/VH/backscattering_ascending_h5",
                must_exist=True,
            ))
            out_base = Path(ask_path(
                "Path to save your zarr file",
                default="/export/students/aryal/Final Submission Script/Data Preperation/zarr/S1",
                must_exist=True,
            ))
            ref_tif= Path(ask_path(
                "Reference raster (.tif) path",
                default="/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif",
                must_exist=True,
                ))
            T = int(ask_str("Time Dimension in the DataCube", default="10"))
            max_gap = int(ask_str("Max Gap between coherence aescending and descending", default="6"))
            sample_chunk = int(ask_str("Chunk Size (larger chunk less file count)", default="64"))
            chip         = int(ask_str("Chip size in pixels", default="128"))
            norm_stats= Path(ask_path(
                "Place to save the normalization statistics",
                default="/export/students/aryal/Final Submission Script/Data Preperation/zarr/S1/norm_stats.json",
                must_exist=False,
                ))
            back_p_high = float(ask_str("Highest percentile for the normalization calculation", default="98"))
            back_p_low = float(ask_str("Lowest percentile for the normalization calculation", default="2"))
            buildS1(
                year = year,
                ca_base=ca_base,
                cd_base= cd_base,
                ba_base=ba_base,
                bd_base=bd_base,
                label_train_base =  label_train_base,
                label_test_base = label_train_base,
                out_base=out_base,
                ref_tif = ref_tif,
                norm_stats_json = norm_stats,
                T=T,
                max_gap=max_gap,
                sample_chunk=sample_chunk,
                chip_size= chip,
                back_p_low=back_p_low,
                back_p_high=back_p_high,
            )
        elif(sensorConfiguration == "s1+s2"):
            year = int(ask_str("Which year do you want to process ? ", default="2020"))
            s2_root      = Path(ask_path("S2 HDF5 Path", default=Path(f"/export/students/aryal/s2_h5/{year}"), must_exist=True))
            ca_root      = Path(ask_path("S2 HDF5 Path", default=Path(f"/export/students/aryal/VH/coherence_h5_ascending_VH/{year}"), must_exist=True))
            cd_root      = Path(ask_path("S2 HDF5 Path", default=Path(f"/export/students/aryal/VH/coherence_h5_descending_VH/{year}"), must_exist=True))
            ba_root      = Path(ask_path("S2 HDF5 Path", default=Path(f"/export/students/aryal/VH/backscattering_ascending_h5/{year}"), must_exist=True))
            bd_root      = Path(ask_path("S2 HDF5 Path", default=Path(f"/export/students/aryal/VH/backscattering_descending_h5/{year}"), must_exist=True))
            label_root   = Path(ask_path("Label Data location", default=Path(f"/export/students/aryal/label_chip/{year}"), must_exist=True))
            OUT = Path(ask_path(
                "Path to save your zarr file",
                default="/export/students/aryal/Final Submission Script/Data Preperation/zarr/S1S2",
                must_exist=True,
            ))
            ref_tif= Path(ask_path(
                "Reference raster (.tif) path",
                default="/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif",
                must_exist=True,
                ))
            T = int(ask_str("Time Dimension in the DataCube", default="10"))
            max_gap = int(ask_str("Max Gap between coherence aescending and descending", default="6"))
            sample_chunk = int(ask_str("Chunk Size (larger chunk less file count)", default="64"))
            chip         = int(ask_str("Chip size in pixels", default="128"))
            norm_stats= Path(ask_path(
                "Place to save the normalization statistics",
                default="/export/students/aryal/Final Submission Script/Data Preperation/zarr/S1/norm_stats.json",
                must_exist=False,
                ))
            back_p_high = float(ask_str("Highest percentile for the normalization calculation", default="98"))
            back_p_low = float(ask_str("Lowest percentile for the normalization calculation", default="2"))
            max_ba = int(ask_str("How many backscatter chip to average? ", default="3"))
            store = run_full_pipeline(
                year = year,
                s2_root          = s2_root,
                ca_root          = ca_root,
                cd_root          = cd_root,
                ba_root          = ba_root,
                bd_root          = bd_root,
                label_root       = label_root,
                zarr_path        = f"{OUT}/{YEAR}.zarr",
                out_dir          = OUT,
                ref_tif          = ref_tif,
                norm_stats_json  = "/globalsc/ucl/elia/aryal/S1_S2_Combined/norm_stats.json",
                T                = T,
                max_gap          = max_gap,
                max_ba           = max_ba,
                H                = chip,
                W                = chip,
                back_p_low       = back_p_low,
                back_p_high      = back_p_high,
                sample_chunk     = sample_chunk,
            )

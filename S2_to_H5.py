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
from numcodecs import Blosc
from rasterio.transform import Affine
#==================================================================
## Part 1: MetaData Collection ##
#==================================================================
def compute_fmask_noise_percentage(fmask_path:str, noise_values: Sequence[int]=(2, 3, 4, 255)) -> Optional[float]:
    """
    Read an FMASK raster and compute the percentage of pixels belonging
    to noise/cloud classes.
    """
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
    """
    Find and sort Sentinel-2 band JP2 files in a scene folder for the
    requested band numbers.
    """
    pat = re.compile(rf"_FRE_B0?({'|'.join(map(str, bands))})\.jp2$")
    return sorted(
        fp for fp in glob.glob(os.path.join(s2_folder, "*FRE_B*.jp2"))
        if pat.search(os.path.basename(fp))
    )

# Locate the matching FMASK file for a tile/date and compute its noise/cloud percentage.
def find_mask(mask_path: str, tile:str, year:int, date_obj: datetime, rgbnir_files: sequence[str], noise_values: sequence[int]) -> Tuple[Optional[str], Optional[float]]:
    """
    Locate the FMASK file corresponding to a given tile and acquisition date,
    and compute its cloud/noise coverage percentage.
    """
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
    cloud_cover_pct = compute_fmask_noise_percentage(fmask_used, noise_values)
    return fmask_used, cloud_cover_pct


def build_records(base_path:str, mask_path:str, tiles:Sequence[str], out_path:str,
                         start_date:Optional[str], end_date:Optional[str], bands:Sequence[int], noise_values: Sequence[int]):
    """
    Scan Sentinel-2 tile/year directories, collect band file paths and FMASK
    metadata for each valid acquisition, and write the results to a
    compressed JSONL index file (.jsonl.gz).

    Each line in the output file is a JSON record representing one scene
    acquisition with the following fields:
        - year           : acquisition year
        - tile           : Sentinel-2 tile ID
        - date           : acquisition date (YYYY-MM-DD)
        - folder         : source scene folder name
        - files          : list of band JP2 file paths
        - fmask          : path to the corresponding FMASK file or None
        - cloud_cover_pct: noise/cloud percentage [0, 100] or None
    """
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
                    )

                    rec = {
                        "year": year,
                        "tile": tile,
                        "date": date_obj.strftime("%Y-%m-%d"),
                        "folder": folder,
                        "files": files,
                        "fmask": fmask_used,                 
                        "cloud_cover_pct": cloud_cover_pct,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

# #================================================================
## Part 2: H5 File generation
#================================================================
### Memory Monitoring #####
def memory_monitor(interval=10):
    pid = os.getpid()
    while getattr(memory_monitor, "running", True):
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[mem] Peak RSS: {rss_kb / 1024:.0f} MB", flush=True)
        time.sleep(interval)

memory_monitor.running = True
t = threading.Thread(target=memory_monitor, args=(10,), daemon=True)
t.start()

# ─── Paths ────────────────────────────────────────────────────────────────────
S2_INFO_PATH = Path("/export/students/aryal/S2_Satellite_Info/s2info.json")
REF_TIF      = Path("/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif")
OUT_ROOT     = Path("/export/students/aryal/s2_h5")

# ─── Chip Configuration ───────────────────────────────────────────────────────
CHIP   = 128   # Chip size in pixels (128 × 128)
STRIDE = 128   # Stride equals chip size → no overlap between chips

# ─── Quality Filters ──────────────────────────────────────────────────────────
EDGE_BUFFER    = 30.0  # Chips must lie at least 30 m inside the tile boundary
                       # to avoid edge artefacts from resampling or tiling seams

MIN_VALID_FRAC = 1.0   # All pixels in a chip must be valid (no MAJA nodata = −10000)
                       # 1.0 = 100% valid pixels required

MAX_CLOUD_PCT  = 0.0   # No cloud, cloud shadow, or ice pixels permitted (FMask)
                       # 0.0 = fully clear chips only

# ─── Resampling ───────────────────────────────────────────────────────────────
VRT_RESAMPLING = Resampling.nearest  # Used when warping S2 tiles to the reference
                                     # grid via VRT; nearest-neighbour preserves
                                     # ignore if both have same spatial information

# ─── Spectral Bands ───────────────────────────────────────────────────────────
BAND_NAMES = ["B2", "B3", "B4", "B8"]   # Blue, Green, Red, Near-Infrared (NIR)
N_BANDS    = len(BAND_NAMES)             # = 4

# ─── MAJA Atmospheric Correction ──────────────────────────────────────────────
MAJA_NODATA = -10000   # Sentinel-2 pixels flagged as invalid by the MAJA
                       # processor; replaced with NaN before writing to H5

# ─── FMask Cloud Classification ───────────────────────────────────────────────
FMASK_CLOUD_VALUES = {2, 3, 4}   # 2 = cloud shadow
                                  # 3 = cloud
                                  # 4 = ice / snow
                                  # Chips containing any of these are rejected

FMASK_NODATA_VALUE = 255         # FMask fill value (outside tile extent);
                                  # treated identically to cloud pixels

# ─── HDF5 Storage ─────────────────────────────────────────────────────────────
H5_COMPRESSION = "lzf"                    # Fast, lightweight compression;
                                          # good balance of speed vs. file size

H5_CHUNK_SHAPE = (1, N_BANDS, CHIP, CHIP) # One time-step per chunk → efficient
                                          # sequential reads along time axis
                                          # Shape: (T=1, C=4, H=128, W=128)

MAX_OPEN_H5 = 128                         # Maximum number of H5 file handles
                                          # kept open simultaneously (LRU cache);

def load_s2_records(info_path: Path) -> List[dict]:
    """
    Load all Sentinel-2 acquisition records from the s2info.json metadata file
    and return them sorted in chronological order.

    Each record is a dictionary containing metadata for one S2 acquisition:
        - "date"    : acquisition date (YYYY-MM-DD)
        - "tile"    : Sentinel-2 tile identifier (e.g. '31UFS')
        - "folder"  : path to the MAJA product folder
        - "files"   : list of band file paths [B2, B3, B4, B8]
        - "fmask"   : path to the corresponding FMask classification raster

    Records are sorted by date so that time-series data written to each
    H5 chip file follows chronological order from the start.

    Args:
        info_path (Path): Path to the metadata file.

    Returns:
        List[dict]: Chronologically sorted list of S2 acquisition records.
    """
    with open(info_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    records.sort(key=lambda r: r["date"])
    return records

def filter_records_by_month(
    all_records: List[dict], year: int, month: int,
) -> List[dict]:
    """
    Filter the full list of Sentinel-2 records to only those acquired
    in a specific year and month.
     For example, year=2021, month=7
    produces the prefix "2021-07", which matches "2021-07-03", "2021-07-15", etc.
    This allows the pipeline to process one month at a time, avoiding the
    need to load all records into memory simultaneously.
    Args:
        all_records (List[dict]) : Full chronologically sorted list of S2
                                   acquisition records (from load_s2_records).
        year        (int)        : Target year  (e.g. 2021).
        month       (int)        : Target month (e.g. 7 for July).

    Returns:
        List[dict]: Subset of records whose acquisition date falls within
                    the specified year and month. Returns an empty list if
                    no records exist for that period.
    """
    prefix = f"{year:04d}-{month:02d}"
    return [rec for rec in all_records if rec["date"][:7] == prefix]

# ─── Regions of Interest (pixel coordinates of the reference raster) ──────────
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
    '''
    Generate the complete list of chips to be extracted across all regions
    of interest by sliding a fixed-size window over the reference raster.

    The reference TIF is opened first to retrieve the raster dimensions and
    spatial metadata (CRS, affine transform, width, height). This metadata
    is later used to warp all Sentinel-2 tiles onto the same grid via VRT.

    For each region, a chip-sized window (128×128 px) is slid across the
    pixel extent with a step equal to the stride. When stride == chip size,
    chips are non-overlapping and perfectly tile the region. Region boundaries
    are clamped to the raster extent to prevent incomplete edge chips.
    
    Each chip is a (row, col, Window) tuple where:
    - row, col  : top-left pixel coordinate of the chip in the reference grid
    - Window    : rasterio Window object defining the exact pixel block to read

    '''
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
    """
    Convert a chip's pixel coordinates into a geographic bounding box
    using the affine transform of the reference raster.

    A rasterio Window only carries pixel offsets and has no knowledge of
    real-world coordinates. This function converts the chip's top-left
    corner (r0, c0) and its size into geographic (x, y) coordinates so
    that the chip's spatial extent can be compared against the geographic
    bounds of Sentinel-2 tiles during the containment check.

    Returns:
        Tuple[float, float, float, float]: Geographic bounding box of the chip
            as (xmin, ymin, xmax, ymax) in the CRS units of the reference raster.

    """
    left, top = tf * (c0, r0)
    right, bottom = tf * (c0 + chip, r0 + chip)
    return (
        min(left, right), min(top, bottom),
        max(left, right), max(top, bottom),
    )


def build_file_to_chips(
    records: List[dict],
    chips: List[Tuple[int, int, Window]],
    ref_meta: dict,
    chip: int,
) -> Dict[int, List[int]]:
    """
    Build a mapping from each Sentinel-2 file index to the list of chip indices
    that are fully contained within that file's geographic footprint.
    This mapping helps to know exactly which chips to extract from each S2 file.
    """
    cboxes = [
        chip_geo_bounds(ref_meta["transform"], r, c, chip)
        for r, c, _ in chips
    ]
    tile_bounds_cache: Dict[str, Optional[Tuple[float, float, float, float]]] = {}

    f2c: Dict[int, List[int]] = {}
    total_pairs = 0

    for fi, rec in enumerate(records):
        tile = rec["tile"]

        if tile not in tile_bounds_cache:
            band_path = Path(rec["files"][0])
            if band_path.exists():
                try:
                    with rasterio.open(band_path) as src:
                        b = src.bounds
                        tile_bounds_cache[tile] = (b.left, b.bottom, b.right, b.top)
                except Exception:
                    tile_bounds_cache[tile] = None
            else:
                tile_bounds_cache[tile] = None

        tb = tile_bounds_cache[tile]
        if tb is None:
            continue

        tx0 = tb[0] + EDGE_BUFFER
        ty0 = tb[1] + EDGE_BUFFER
        tx1 = tb[2] - EDGE_BUFFER
        ty1 = tb[3] - EDGE_BUFFER

        hits = [
            ci
            for ci, (px0, py0, px1, py1) in enumerate(cboxes)
            if tx0 <= px0 and px1 <= tx1 and ty0 <= py0 and py1 <= ty1
        ]
        if hits:
            f2c[fi] = hits
            total_pairs += len(hits)

    print(f"  {len(f2c)}/{len(records)} files → {total_pairs:,} pairs")
    return f2c

def loc_name(r0: int, c0: int) -> str:
    return f"r{r0:04d}_c{c0:05d}"

class _ChipHandle:
    __slots__ = (
        "h5f", "path",
        "dates", "tiles", "folders", "cloud_coverages",
        "_date_set",
    )
    def __init__(
        self,
        h5f: h5py.File,
        path: Path,
        dates: List[str],
        tiles: List[str],
        folders: List[str],
        cloud_coverages: List[float],
    ):
        self.h5f = h5f
        self.path = path
        self.dates = dates
        self.tiles = tiles
        self.folders = folders
        self.cloud_coverages = cloud_coverages
        self._date_set: set = set(dates)

    def has_date(self, date_str: str) -> bool:
        """
        Check whether this chip already has pixel data recorded for a given date.
        """
        return date_str in self._date_set

    def append(
        self,
        arr: np.ndarray,       
        date_str: str,
        tile: str,
        folder: str,
        cloud_pct: float,
    ) -> None:
        """
        Write one new time step (one valid chip acquisition) to the H5 file.
        Pixel data is written to disk immediately by resizing the H5 dataset
        along the time axis and inserting the array at the new position.
        Metadata (date, tile, folder, cloud coverage) is appended to the
        in-memory lists and will be flushed to disk as H5 attributes only
        when flush_and_close() is called. This avoids repeated slow attribute
        writes after every append.
        """
        ds = self.h5f["X"]
        idx = ds.shape[0]
        ds.resize(idx + 1, axis=0)
        ds[idx, :, :, :] = arr

        self.dates.append(date_str)
        self.tiles.append(tile)
        self.folders.append(folder)
        self.cloud_coverages.append(cloud_pct)
        self._date_set.add(date_str)

    def flush_and_close(self) -> None:
        """
        Write all accumulated metadata to disk as H5 attributes and close
        the file handle.
        Metadata lists (dates, tiles, folders, cloud_coverages) are held in
        memory during processing to avoid repeated slow H5 attribute writes.
        This method is called either when the file is evicted from the LRU
        cache or at the end of processing, at which point all metadata is
        written to disk in a single operation.
        """
        try:
            self.h5f.attrs["dates"] = self.dates
            self.h5f.attrs["tiles"] = self.tiles
            self.h5f.attrs["folders"] = self.folders
            self.h5f.attrs["cloud_coverages"] = self.cloud_coverages
        finally:
            self.h5f.close()


class LRUChipCache:
    """
    A Least Recently Used (LRU) cache for open HDF5 chip file handles.
    During satellite image processing, hundreds of geographic chips may be
    written to concurrently. Opening all of them simultaneously would exhaust
    the OS file descriptor limit. This cache keeps at most `max_open` HDF5
    files open at any time, automatically evicting and flushing the least
    recently used handle when the limit is reached.
    """
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
            tiles = list(h5f.attrs.get("tiles", []))
            folders = list(h5f.attrs.get("folders", []))
            cloud_coverages = list(h5f.attrs.get("cloud_coverages", []))
        else:
            h5f = h5py.File(path, "w")
            h5f.create_dataset(
                "X",
                shape=(0, N_BANDS, chip, chip),
                maxshape=(None, N_BANDS, chip, chip),
                dtype=np.float32,
                chunks=H5_CHUNK_SHAPE,
                compression=H5_COMPRESSION,
            )
            h5f.attrs["row_off"] = r0
            h5f.attrs["col_off"] = c0
            h5f.attrs["band_names"] = BAND_NAMES
            dates, tiles, folders, cloud_coverages = [], [], [], []
        return _ChipHandle(h5f, path, dates, tiles, folders, cloud_coverages)

def extract_month(
    records: List[dict],
    file_to_chips: Dict[int, List[int]],
    chips: List[Tuple[int, int, Window]],
    ref_meta: dict,
    out_dir: Path,
    chip: int,
    resampling: Resampling,
    min_valid_frac: float,
    max_cloud_pct: float,
    maja_nodata: float,
) -> int:
    """
    Process satellite scence and write valid chip acquisitions to HDF5 files.
    For each scene (file), every chip that spatially overlaps it is tested
    against a sequential filter pipeline. A chip must pass every gate before
    its pixel data is written to disk:
        1. Bounds gate     — chip must have a non-zero intersection with the raster
        2. Cloud gate      — cloud + nodata fraction must be <= max_cloud_pct
        3. Band read gate  — every spectral band must have a valid intersection
        4. Validity gate   — finite pixel fraction must be >= min_valid_frac
        5. Duplicate gate  — this (chip, date) pair must not already be written
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = LRUChipCache(out_dir, chip, max_open=MAX_OPEN_H5)
    n_appends = 0
    n_skipped_cloud = 0
    n_skipped_nodata = 0
    n_skipped_duplicate = 0
    n_file_errors = 0
    n_skipped_bounds = 0
    band_buf = np.empty((N_BANDS, chip, chip), dtype=np.float32)
    try:
        for fi in tqdm(
            sorted(file_to_chips.keys()),
            desc="  Extracting",
            unit="file",
            leave=False,
        ):
            rec = records[fi]
            date_str = rec["date"]
            tile = rec["tile"]
            folder = rec["folder"]
            band_paths = [Path(p) for p in rec["files"]]

            if not all(bp.exists() for bp in band_paths):
                missing = [bp.name for bp in band_paths if not bp.exists()]
                tqdm.write(f"    [!] Missing bands: {missing}")
                n_file_errors += 1
                continue

            fmask_raw = rec.get("fmask")
            if fmask_raw is None:
                tqdm.write(f"    [!] Fmask is None: {folder}")
                n_file_errors += 1
                continue
            fmask_path = Path(fmask_raw)

            if not fmask_path.exists():
                tqdm.write(f"    [!] Missing Fmask: {fmask_path.name}")
                n_file_errors += 1
                continue

            try:
                band_srcs = [rasterio.open(bp) for bp in band_paths]
                band_vrts = [
                    WarpedVRT(
                        src,
                        crs=ref_meta["crs"],
                        transform=ref_meta["transform"],
                        width=ref_meta["width"],
                        height=ref_meta["height"],
                        resampling=resampling,
                        src_nodata=maja_nodata, 
                        nodata=np.nan,
                        add_alpha=False,
                    )
                    for src in band_srcs
                ]

                fmask_src = rasterio.open(fmask_path)
                fmask_vrt = WarpedVRT(
                    fmask_src,
                    crs=ref_meta["crs"],
                    transform=ref_meta["transform"],
                    width=ref_meta["width"],
                    height=ref_meta["height"],
                    resampling=Resampling.nearest,
                    src_nodata=FMASK_NODATA_VALUE,
                    nodata=FMASK_NODATA_VALUE,
                    add_alpha=False,
                )

                for ci in file_to_chips[fi]:
                    r0, c0, win = chips[ci]
                    cs = max(c0, 0)
                    rs = max(r0, 0)
                    ce = min(c0 + chip, fmask_vrt.width)
                    re_val = min(r0 + chip, fmask_vrt.height)

                    if cs >= ce or rs >= re_val:
                        n_skipped_bounds += 1
                        continue

                    fwin = Window(cs, rs, ce - cs, re_val - rs)
                    fmask_data = fmask_vrt.read(1, window=fwin).astype(np.uint8)

                    if fmask_data.shape != (chip, chip):
                        fmask_buf = np.full(
                            (chip, chip), FMASK_NODATA_VALUE, dtype=np.uint8,
                        )
                        dr, dc = rs - r0, cs - c0
                        fmask_buf[
                            dr:dr + (re_val - rs),
                            dc:dc + (ce - cs),
                        ] = fmask_data
                        fmask_data = fmask_buf

                    bad_mask = np.isin(
                        fmask_data,
                        list(FMASK_CLOUD_VALUES | {FMASK_NODATA_VALUE}),
                    )
                    cloud_pct = float(bad_mask.sum()) / float(fmask_data.size) * 100.0
                    del fmask_data, bad_mask

                    if cloud_pct > max_cloud_pct:
                        n_skipped_cloud += 1
                        continue
                    band_buf[:] = np.nan
                    all_ok = True

                    for bi, bvrt in enumerate(band_vrts):
                        bcs = max(c0, 0)
                        brs = max(r0, 0)
                        bce = min(c0 + chip, bvrt.width)
                        bre = min(r0 + chip, bvrt.height)
                        if bcs >= bce or brs >= bre:
                            all_ok = False
                            break
                        bwin = Window(bcs, brs, bce - bcs, bre - brs)
                        data = bvrt.read(1, window=bwin).astype(np.float32)
                        dr, dc = brs - r0, bcs - c0
                        band_buf[bi, :, :] = np.nan
                        band_buf[
                            bi,
                            dr:dr + (bre - brs),
                            dc:dc + (bce - bcs),
                        ] = data
                        del data

                    if not all_ok:
                        n_skipped_bounds += 1
                        continue
                    band_buf[band_buf == maja_nodata] = np.nan

                    valid_frac = np.isfinite(band_buf).mean()
                    if valid_frac < min_valid_frac:
                        n_skipped_nodata += 1
                        continue
                    handle = cache.get(ci, r0, c0)
                    if handle.has_date(date_str):
                        n_skipped_duplicate += 1
                        continue
                    handle.append(
                        band_buf.copy(),
                        date_str,
                        tile,
                        folder,
                        cloud_pct,
                    )
                    n_appends += 1

                for bvrt in band_vrts:
                    bvrt.close()
                for bsrc in band_srcs:
                    bsrc.close()
                fmask_vrt.close()
                fmask_src.close()

            except Exception as exc:
                tqdm.write(f"    [!] Error: {rec['folder']}: {exc}")
                n_file_errors += 1

    finally:
        cache.close_all()

    print(
        f"Appends: {n_appends:,}  |  "
        f"Skipped(cloud): {n_skipped_cloud:,}  |  "
        f"Skipped(nodata): {n_skipped_nodata:,}  |  "
        f"Skipped(bounds): {n_skipped_bounds:,}  |  " 
        f"Skipped(duplicate): {n_skipped_duplicate:,}  |  "
        f"Errors: {n_file_errors}"
    )
    return n_appends

def merge_temp_into_main(
    main_dir: Path,
    temp_dir: Path,
    chip: int,
) -> None:
    """
    Merge temporary HDF5 chip files produced by a single month's processing
    run into the permanent main HDF5 chip files.

    During processing, new chip data is written to temp_dir rather than
    directly into main_dir. This keeps the main dataset safe if processing
    crashes mid-run. Once a month completes successfully, this function
    integrates the temp files into main_dir either by moving (new chips)
    or appending (existing chips).
    For each temp chip file:
        - If no corresponding main file exists: the temp file is moved
          directly to main_dir (fast path, no data copying needed).
        - If a main file already exists: pixel data is appended to the
          main X dataset along the time axis, and metadata attribute lists
          (dates, tiles, folders, cloud_coverages) are concatenated.

    After a successful merge, each temp file is deleted. The temp_dir
    itself is removed if it is empty after all merges complete.
    """
    temp_files = sorted(temp_dir.glob("r*.h5"))
    if not temp_files:
        print("No temp files to merge.")
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
                main_ds[t_start + i, :, :, :] = temp_ds[i, :, :, :]
            main_h5.attrs["dates"] = (
                list(main_h5.attrs["dates"]) +
                list(temp_h5.attrs["dates"])
            )
            main_h5.attrs["tiles"] = (
                list(main_h5.attrs["tiles"]) +
                list(temp_h5.attrs["tiles"])
            )
            main_h5.attrs["folders"] = (
                list(main_h5.attrs["folders"]) +
                list(temp_h5.attrs["folders"])
            )
            main_h5.attrs["cloud_coverages"] = (
                list(main_h5.attrs["cloud_coverages"]) +
                list(temp_h5.attrs["cloud_coverages"])
            )

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

def finalize_year(year_dir: Path, year: int) -> pd.DataFrame:
    """
     Sort all chip HDF5 files in a year directory into chronological order
    and produce a summary CSV describing the year's dataset.

    During monthly processing, chip time steps are written in file index
    order rather than date order. After all months are merged, time steps
    within each chip may be out of chronological sequence. This function
    corrects that by sorting the X dataset and all metadata attribute lists
    in-place by date, then building a per-chip summary.

    """
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
            tiles = list(f.attrs["tiles"])
            folders = list(f.attrs["folders"])
            cloud_coverages = list(f.attrs["cloud_coverages"])

            order = sorted(range(T), key=lambda i: dates[i])

            if order != list(range(T)):
                n_bands = f["X"].shape[1]
                chip_h = f["X"].shape[2]
                chip_w = f["X"].shape[3]
                tmp = np.empty((n_bands, chip_h, chip_w), dtype=np.float32)

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
                tiles = [tiles[i] for i in order]
                folders = [folders[i] for i in order]
                cloud_coverages = [cloud_coverages[i] for i in order]

                f.attrs["dates"] = dates
                f.attrs["tiles"] = tiles
                f.attrs["folders"] = folders
                f.attrs["cloud_coverages"] = cloud_coverages

            rows.append({
                "chip_file": h5_path.name,
                "row_off": r0,
                "col_off": c0,
                "T": T,
                "dates": ",".join(dates),
                "date_first": dates[0] if dates else "",
                "date_last": dates[-1] if dates else "",
                "mean_cloud_pct": float(np.mean(cloud_coverages)) if cloud_coverages else 0.0,
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
        print(f"    Mean cloud : {summary['mean_cloud_pct'].mean():.1f}%")
    return summary

def _generate_year_months(
    start_date: str, end_date: str,
) -> List[Tuple[int, int]]:
    """
    Given 'YYYY-MM-DD' start and end dates, return a list of (year, month)
    tuples covering every month from start to end (inclusive).
    """
    from datetime import date

    sd = date.fromisoformat(start_date)
    ed = date.fromisoformat(end_date)

    year_months: List[Tuple[int, int]] = []
    y, m = sd.year, sd.month
    while (y, m) <= (ed.year, ed.month):
        year_months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    return year_months


def run_pipeline(
    start_date: str = "2018-01-01",
    end_date: str = "2021-12-31",
    s2_info_path: Path = S2_INFO_PATH,
    ref_tif: Path = REF_TIF,
    out_root: Path = OUT_ROOT,
    chip: int = CHIP,
    stride: int = STRIDE,
    edge_buffer: float = EDGE_BUFFER,
    min_valid_frac: float = MIN_VALID_FRAC,
    max_cloud_pct: float = MAX_CLOUD_PCT,
    resampling: Resampling = VRT_RESAMPLING,
) -> None:
    year_months = _generate_year_months(start_date, end_date)
    years_in_range = sorted(set(y for y, m in year_months))

    print("S2 HDF5 Time-Series Pipeline — Month-by-Month")
    print(f"  Date range    : {start_date} → {end_date}")
    print(f"  Months        : {len(year_months)}")
    print(f"  Bands         : {BAND_NAMES}")
    print(f"  MAJA nodata   : {MAJA_NODATA}")
    print(f"  Max cloud %   : {max_cloud_pct}")
    print(f"  Min valid frac: {min_valid_frac}")
    print(f"  Output        : {out_root}")
    print(f"  Max open H5   : {MAX_OPEN_H5}")
    print("[Step 0] Loading S2 info...")
    all_records = load_s2_records(s2_info_path)
    print(f"  Total records: {len(all_records):,}")
    print()

    print("[Step 1] Enumerating chips...")
    chips, ref_transform, ref_meta = enumerate_chips(ref_tif, chip, stride)

    for year in years_in_range:
        year_dir = out_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = out_root / f"_temp_{year}"
        months_this_year = [m for y, m in year_months if y == year]

        print("=" * 60)
        print(f"YEAR {year}")
        print("=" * 60)

        is_first_month_of_year = True

        for month in months_this_year:
            month_label = f"{year}-{month:02d}"
            print(f"\n--- {month_label} ---")

            month_records = filter_records_by_month(all_records, year, month)
            if not month_records:
                print("  No records — skipping.")
                continue

            print(f"  Records: {len(month_records)}")

            file_to_chips = build_file_to_chips(
                month_records, chips, ref_meta, chip,
            )

            if not file_to_chips:
                print("  No chips covered — skipping.")
                clear_memory()
                continue

            if is_first_month_of_year:
                print("  → Writing MAIN H5 files...")
                extract_month(
                    month_records, file_to_chips, chips, ref_meta,
                    year_dir, chip, resampling, min_valid_frac,
                    max_cloud_pct, MAJA_NODATA,
                )
                is_first_month_of_year = False
            else:
                print("  → Writing TEMP H5 files...")
                temp_dir.mkdir(parents=True, exist_ok=True)
                extract_month(
                    month_records, file_to_chips, chips, ref_meta,
                    temp_dir, chip, resampling, min_valid_frac,
                    max_cloud_pct, MAJA_NODATA,
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

def main() -> None:
    memory_monitor.running = True
    monitor_thread = threading.Thread(
        target=memory_monitor, args=(15,), daemon=True,
    )
    monitor_thread.start()

    try:
        run_pipeline(
            start_date="2021-01-01",
            end_date="2021-12-31",
        )
    finally:
        memory_monitor.running = False

if __name__ == "__main__":
    main()

# #================================================================
## Part 2: Zarr File Generation
#==================================================================
S2_ROOT     = Path("/export/students/aryal/s2_h5/2020") # Root directory containing Sentinel-2 HDF5 chip files (*.h5) for year 2020
LABEL_ROOT  = Path("/export/students/aryal/Label_Chips_npy_128_from_ref_2020_test/2020") # Root directory containing segmentation label files (*.npy), 128×128 chips
REF_TIF     = Path("/export/students/aryal/WALLONIA_2018-07_8_median_trim.tif") # Reference GeoTIFF used to extract CRS and pixel transform (georeferencing metadata)
OUT_ZARR    = Path("/export/students/aryal/S2_dataset/s2_optical_2020_test.zarr") # Output Zarr store path where the final dataset (X + Y) will be written

YEAR        = 2020 # Acquisition year — stored as metadata per sample
CHIP        = 128  # Spatial size of each chip in pixels (128 × 128)
T_BLOCK     = 10   # Number of timestamps per block     
B           = 4    # Number of spectral bands: B2, B3, B4, B8
L           = 3    # Number of label channels:     
BAND_NAMES  = ["B2", "B3", "B4", "B8"]
# Sentinel-2 bands used:
#   B2 = Blue, B3 = Green, B4 = Red, B8 = Near-Infrared (NIR)

LABEL_NAMES = ["extent", "boundary", "dist"]
# Segmentation label channels:
#   extent   = binary field/parcel mask
#   boundary = parcel boundary mask
#   dist     = distance transform from parcel edges

LABEL_CH    = [0, 1, 2]   
# Channel indices to read from the .npy label array (selects all 3 channels)

SAMPLE_CHUNK = 32
# Number of samples per Zarr chunk along the N axis
# Balances read speed vs. memory during training
OVERWRITE    = True # If True, delete and recreate the Zarr store if it exists

NPY_RE = re.compile(r"loc_r(?P<r>\d+)_c(?P<c>\d+)\.npy$", re.IGNORECASE)

def loc_name(r0: int, c0: int) -> str:
    """
    Build a zero-padded location string from pixel row/column offsets.
    Used as a consistent dictionary key to match H5 chip files with
    their corresponding label .npy files by location.
    """
    return f"r{r0:04d}_c{c0:05d}"


def window_ul_xy(transform: Affine, win: Window) -> Tuple[float, float]:
    """
    Convert a rasterio Window's top-left pixel position to geographic coordinates.
    Applies the Affine transform of the reference raster to map the chip's
    upper-left corner from pixel space (col, row) into real-world space (x, y),
    """
    col, row = int(win.col_off), int(win.row_off)
    x, y = transform * (col, row)
    return float(x), float(y)


def read_ref_meta(path: Path) -> dict:
    """
    Read spatial metadata from the reference GeoTIFF.
    Opens the reference .tif with rasterio and extracts the four key
    pieces of georeferencing information needed to:
      - georeference each chip's (x0, y0) coordinates
      - write CRS and transform into the Zarr store's global attributes
        so the dataset remains spatially self-describing
    """
    with rasterio.open(path) as ds:
        return {
            "crs"      : ds.crs,
            "transform": ds.transform,
            "width"    : ds.width,
            "height"   : ds.height,
        }


def load_label_index(label_root: Path) -> Dict[str, Path]:
    """
    Scan the label directory and build a lookup dictionary mapping
    each chip's location name to its .npy label file path.

    This index is used in pass1 and pass2 to quickly check whether
    a given S2 H5 chip has a matching label file, and to retrieve
    that file's path without repeated directory scans.
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
    Read the acquisition dates from an H5 chip file and split them into
    non-overlapping fixed-size blocks of t_block timestamps each.

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
    First pass over all S2 H5 chips — counts valid samples WITHOUT writing any data.
    Purpose:
        Zarr arrays must be pre-allocated with a fixed shape before writing.
        This pass determines the exact total number of samples (N_total)
        needed to initialize the Zarr store in init_zarr(), before pass2
        does the actual data writing.

    Validation rules applied per chip:
        1. Chip must have a matching label .npy file in label_index
        2. Chip must have at least t_block timestamps (via get_blocks)
        Chips failing either rule are skipped and counted separately.
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
    """
    Initialize and pre-allocate the output Zarr store with all required
    arrays and metadata attributes — NO data is written here, only structure.

    This function is called once between pass1 and pass2:
        pass1 → determines N_total
        init_zarr → creates empty Zarr arrays of the correct shape
        pass2 → fills those arrays with actual data
    """

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
    """
    Second pass — reads actual pixel data and labels, writes every sample
    into the pre-allocated Zarr arrays created by init_zarr().

    Iterates over every valid chip (from chip_info), loads its H5 imagery
    and .npy label file, splits the timestamps into t_block-sized blocks,
    and writes each block as one sample at position [idx] in the Zarr store.
    """
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
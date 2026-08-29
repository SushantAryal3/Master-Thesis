from __future__ import annotations
import os
import glob
import datetime as dt
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.windows import transform as window_transform
from typing import Dict, List, Tuple
from tqdm import tqdm
import subprocess
from rasterio import windows
from rasterio.warp import transform_bounds
from scipy.ndimage import binary_dilation
import fiona
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from scipy.ndimage import binary_fill_holes
import geopandas as gpd
from shapely.geometry import box

def ensure_gdal_tools():
    for exe in ["gdal_rasterize", "gdal_proximity"]:
        if shutil.which(exe) is None:
            raise RuntimeError(f"Missing '{exe}' in PATH. Install GDAL command-line tools.")

def run_cmd(cmd: List[str]):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{proc.stderr}")
    return proc.stdout

def detect_first_layer_and_objectid(gpkg: Path, id_field: str = "OBJECTID") -> Tuple[str, str]:
    layers = fiona.listlayers(gpkg)
    if not layers:
        raise ValueError(f"No layers found in {gpkg}")
    lyr = layers[0]

    with fiona.open(gpkg, layer=lyr) as src:
        props = src.schema.get("properties", {}) or {}

    if id_field in props:
        return lyr, id_field

    for cand in ("objectid", "fid", "FID", "id", "ID", "ogc_fid"):
        if cand in props:
            return lyr, cand

    raise ValueError(
        f"'{id_field}' not found in first layer '{lyr}' of {gpkg}. "
        f"Available fields: {list(props.keys())}"
    )

def aligned_window_from_vector_bounds(fields_gpkg: Path, layer: str, ref_raster: Path) -> Window:
    with rasterio.open(ref_raster) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform

        with fiona.open(fields_gpkg, layer=layer) as vsrc:
            vb = vsrc.bounds
            vcrs = vsrc.crs_wkt or vsrc.crs

        if vcrs and ref_crs and str(vcrs) != ref_crs.to_string():
            vb = transform_bounds(vcrs, ref_crs, *vb, densify_pts=21)

        win = windows.from_bounds(*vb, transform=ref_transform)
        return win.round_offsets().round_lengths()

def create_template_raster_windowed(
    out_path: Path,
    ref_path: Path,
    win: Window,
    dtype: str,
    nodata,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            dtype=dtype,
            count=1,
            nodata=nodata,
            width=int(win.width),
            height=int(win.height),
            transform=window_transform(win, ref.transform),
            compress="DEFLATE",
            tiled=True,
            blockxsize=profile.get("blockxsize", 512),
            blockysize=profile.get("blockysize", 512),
        )

        fill = nodata if nodata is not None else 0
        dt = np.dtype(dtype)
        fill = np.array(fill, dtype=dt).item()

        with rasterio.open(out_path, "w", **profile) as dst:
            for _, bwin in dst.block_windows(1):
                arr = np.full((int(bwin.height), int(bwin.width)), fill, dtype=dt)
                dst.write(arr, 1, window=bwin)

def normalize_dist_per_parcel(
    dist_raw_tif: Path,
    parcel_id_tif: Path,
    extent_tif: Path,
    out_dist_norm_tif: Path,
    nodata: float = -9999.0,
    show_progress: bool = True,
):
    maxdist: Dict[int, float] = defaultdict(float)

    with rasterio.open(dist_raw_tif) as d_src, rasterio.open(parcel_id_tif) as pid_src, rasterio.open(extent_tif) as e_src:
        d_nodata = d_src.nodata if d_src.nodata is not None else nodata

        it = d_src.block_windows(1)
        it = tqdm(it, desc="dist pass1 (max/parcel)", unit="block", leave=False) if show_progress else it

        for _, win in it:
            w = Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
            dist = d_src.read(1, window=w).astype(np.float32)
            pid  = pid_src.read(1, window=w).astype(np.int32)
            ext  = e_src.read(1, window=w)

            mask = (ext == 1) & (pid > 0) & (dist != d_nodata)
            if not np.any(mask):
                continue

            ids = pid[mask]
            vals = dist[mask]
            order = np.argsort(ids, kind="mergesort")
            ids_s = ids[order]
            vals_s = vals[order]
            starts = np.r_[0, np.where(np.diff(ids_s) != 0)[0] + 1]
            block_max = np.maximum.reduceat(vals_s, starts)
            block_ids = ids_s[starts]

            for i, mx in zip(block_ids.tolist(), block_max.tolist()):
                if mx > maxdist[i]:
                    maxdist[i] = float(mx)

    with rasterio.open(dist_raw_tif) as d_src, rasterio.open(parcel_id_tif) as pid_src, rasterio.open(extent_tif) as e_src:
        d_nodata = d_src.nodata if d_src.nodata is not None else nodata

        profile = d_src.profile.copy()
        profile.update(dtype="float32", nodata=nodata, compress="DEFLATE", tiled=True)

        with rasterio.open(out_dist_norm_tif, "w", **profile) as out:
            it = d_src.block_windows(1)
            it = tqdm(it, desc="dist pass2 (normalize)", unit="block", leave=False) if show_progress else it

            for _, win in it:
                w = Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
                dist = d_src.read(1, window=w).astype(np.float32)
                pid  = pid_src.read(1, window=w).astype(np.int32)
                ext  = e_src.read(1, window=w)

                out_arr = np.zeros(dist.shape, dtype=np.float32)

                nod_mask = (dist == d_nodata) | (ext == 255)
                out_arr[nod_mask] = nodata

                inside = (ext == 1) & (pid > 0) & (~nod_mask)
                if np.any(inside):
                    ids_here = pid[inside]
                    d_here = dist[inside]

                    out_vals = np.zeros(d_here.shape, dtype=np.float32)
                    uids, inv = np.unique(ids_here, return_inverse=True)
                    mx = np.array([maxdist.get(int(u), 0.0) for u in uids], dtype=np.float32)
                    denom = mx[inv]
                    good = denom > 0
                    out_vals[good] = d_here[good] / denom[good]

                    out_arr[inside] = out_vals
                out.write(out_arr, 1, window=w)

def build_label_rasters_on_disk(
    fields_gpkg: Path,
    ref_raster: Path,
    labels_dir: Path,
    boundary_connectivity: int = 8,
    boundary_width_px: int = 1,
    overwrite: bool = False,
    rasterize_all_touched: bool = True,
) -> Dict[str, Path]:
    ensure_gdal_tools()
    labels_dir.mkdir(parents=True, exist_ok=True)
    parcel_id_tif      = labels_dir / "parcel_id.tif"
    extent_tif         = labels_dir / "extent.tif"
    boundary_tif       = labels_dir / "boundary.tif"
    dist_tif              = labels_dir / "dist.tif"
    boundary_for_dist_tif = labels_dir / "boundary_for_dist.tif"
    dist_raw_tif          = labels_dir / "dist_raw.tif"

    if (not overwrite and parcel_id_tif.exists() and extent_tif.exists()
        and boundary_tif.exists() and dist_tif.exists()):
        return {"parcel_id": parcel_id_tif, "extent": extent_tif, "boundary": boundary_tif, "dist": dist_tif}

    lyr, idf = detect_first_layer_and_objectid(fields_gpkg, "OBJECTID")
    with rasterio.open(ref_raster) as ref:
        ref_width = ref.width
        ref_height = ref.height
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_profile = ref.profile.copy()
    
    def create_template_exact_grid(out_path: Path, dtype: str, nodata):
        """Create template using EXACT reference grid - no windowing."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        profile = ref_profile.copy()
        profile.update(
            driver="GTiff",
            dtype=dtype,
            count=1,
            nodata=nodata,
            width=ref_width,      
            height=ref_height,
            transform=ref_transform,  
            crs=ref_crs,          
            compress="DEFLATE",
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )
        
        fill = nodata if nodata is not None else 0
        dt = np.dtype(dtype)
        fill = np.array(fill, dtype=dt).item()
        
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, bwin in dst.block_windows(1):
                arr = np.full((int(bwin.height), int(bwin.width)), fill, dtype=dt)
                dst.write(arr, 1, window=bwin)

    create_template_exact_grid(parcel_id_tif, dtype="int32", nodata=0)
    create_template_exact_grid(extent_tif, dtype="uint8", nodata=255)
    create_template_exact_grid(boundary_tif, dtype="uint8", nodata=255)
    
    cmd = ["gdal_rasterize", "-l", lyr, "-a", idf]
    if rasterize_all_touched:
        cmd += ["-at"]
    cmd += [str(fields_gpkg), str(parcel_id_tif)]
    run_cmd(cmd)

    halo = max(1, boundary_width_px)  
    dil_iters = max(0, boundary_width_px - 1)

    with rasterio.open(parcel_id_tif) as src, \
        rasterio.open(ref_raster) as ref, \
        rasterio.open(extent_tif, "r+") as dstE, \
        rasterio.open(boundary_tif, "r+") as dstB:

        for _, win in src.block_windows(1):
            w = Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
            exp = Window(w.col_off - halo, w.row_off - halo, w.width + 2*halo, w.height + 2*halo)
            arr = src.read(1, window=exp, boundless=True, fill_value=0)
            H = int(w.height)
            W = int(w.width)

            r0 = halo
            c0 = halo
            center = arr[r0:r0+H,     c0:c0+W]
            up     = arr[r0-1:r0-1+H, c0:c0+W]
            down   = arr[r0+1:r0+1+H, c0:c0+W]
            left   = arr[r0:r0+H,     c0-1:c0-1+W]
            right  = arr[r0:r0+H,     c0+1:c0+1+W]

            diff = (center != up) | (center != down) | (center != left) | (center != right)
            if boundary_connectivity == 8:
                ul = arr[r0-1:r0-1+H, c0-1:c0-1+W]
                ur = arr[r0-1:r0-1+H, c0+1:c0+1+W]
                dl = arr[r0+1:r0+1+H, c0-1:c0-1+W]
                dr = arr[r0+1:r0+1+H, c0+1:c0+1+W]
                diff |= (center != ul) | (center != ur) | (center != dl) | (center != dr)

            extent = (center > 0).astype(np.uint8)
            near_parcel = (center > 0) | (up > 0) | (down > 0) | (left > 0) | (right > 0)
            if boundary_connectivity == 8:
                near_parcel |= (ul > 0) | (ur > 0) | (dl > 0) | (dr > 0)

            boundary = (diff & near_parcel).astype(np.uint8)

            if dil_iters > 0:
                b_exp = np.zeros(arr.shape, dtype=bool)
                b_exp[r0:r0+H, c0:c0+W] = boundary.astype(bool)
                b_exp = binary_dilation(b_exp, iterations=dil_iters)
                boundary = b_exp[r0:r0+H, c0:c0+W].astype(np.uint8)


            dstE.write(extent, 1, window=w)
            dstB.write(boundary, 1, window=w)


    create_template_exact_grid(boundary_for_dist_tif, dtype="uint8", nodata=255)
    with rasterio.open(extent_tif) as e_src, rasterio.open(boundary_tif) as b_src, rasterio.open(boundary_for_dist_tif, "r+") as out:
        for _, win in e_src.block_windows(1):
            w = Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
            ext = e_src.read(1, window=w)
            bnd = b_src.read(1, window=w)

            out_arr = np.full(ext.shape, 255, dtype=np.uint8)
            inside = (ext == 1)
            out_arr[inside] = bnd[inside]
            out.write(out_arr, 1, window=w)

    run_cmd([
        "gdal_proximity",
        str(boundary_for_dist_tif),
        str(dist_raw_tif),
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-ot", "Float32",
        "-values", "1",
        "-distunits", "PIXEL", 
        "-nodata", "-9999",
        "-use_input_nodata", "YES",
    ])

    normalize_dist_per_parcel(
        dist_raw_tif=dist_raw_tif,
        parcel_id_tif=parcel_id_tif,
        extent_tif=extent_tif,
        out_dist_norm_tif=dist_tif,
        nodata=-9999.0,
        show_progress=True,
    ) 
    return {"extent": extent_tif, "boundary": boundary_tif, "dist": dist_tif, "parcel_id": parcel_id_tif}

def build_labels_for_all_years(
    root_dir: Path,
    ref_raster: Path,
    out_root: Path,
    gpkg_glob: str = "*.gpkg",
    overwrite: bool = False,
) -> Dict[str, Dict[str, Path]]:
    root_dir = Path(root_dir)
    out: Dict[str, Dict[str, Path]] = {}

    year_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.isdigit()])

    tasks: List[Tuple[Path, Path]] = []
    for ydir in year_dirs:
        for gpkg in sorted(ydir.glob(gpkg_glob)):
            tasks.append((ydir, gpkg))

    if not tasks:
        print(f"No gpkg files found under: {root_dir}")
        return out

    pbar = tqdm(tasks, desc="Generating labels", unit="file")
    for ydir, gpkg in pbar:
        pbar.set_postfix_str(f"{ydir.name}/{gpkg.name}", refresh=True)
        year = ydir.name
        labels_dir = out_root / year
        out[f"{ydir.name}/{gpkg.stem}"] = build_label_rasters_on_disk(
            fields_gpkg=gpkg,
            ref_raster=ref_raster,
            labels_dir=labels_dir,
            boundary_connectivity=8,
            boundary_width_px=1,
            overwrite=overwrite,
            rasterize_all_touched=True,
        )
    return out

LAYER_SPECS = [
    ("extent",    "extent.tif",    Resampling.nearest,  np.float32),
    ("boundary",  "boundary.tif",  Resampling.nearest,  np.float32),
    ("dist",      "dist.tif",      Resampling.bilinear, np.float32),
    ("parcel_id", "parcel_id.tif", Resampling.nearest,  np.float32),
]

def enumerate_chips(
    ref_tif: Path,
    chip: int,
    stride: int,
) -> Tuple[List[Tuple[int, int, Window]], rasterio.Affine, dict]:
    """
    Single source of truth for chip positions.
    Used by BOTH the H5 image pipeline and this label pipeline
    to guarantee identical chip sets.

    Validity logic (mirrors the H5 pipeline exactly):
      - nodata pixels → invalid
      - internal holes are filled → chips over interior gaps are kept
      - chips that straddle the outer AOI boundary → skipped
    Uses a summed-area table for O(1) per-chip validity checks.
    """
    ref_tif = Path(ref_tif)
    with rasterio.open(ref_tif) as ds:
        H, W          = ds.height, ds.width
        ref_transform = ds.transform
        ref_meta      = {
            "crs":       ds.crs,
            "transform": ds.transform,
            "width":     ds.width,
            "height":    ds.height,
        }
        nodata = ds.nodata
        data   = ds.read(1, masked=False)

    valid_raw = (data != nodata) if nodata is not None else np.isfinite(data)
    del data

    aoi_mask = binary_fill_holes(valid_raw).astype(np.uint8)
    del valid_raw
    ii = aoi_mask.cumsum(axis=0, dtype=np.uint64).cumsum(axis=1, dtype=np.uint64)
    del aoi_mask

    pix_total = chip * chip

    def window_sum(r0: int, c0: int) -> int:
        r1, c1 = r0 + chip - 1, c0 + chip - 1
        s = int(ii[r1, c1])
        if r0 > 0:             s -= int(ii[r0 - 1, c1])
        if c0 > 0:             s -= int(ii[r1,     c0 - 1])
        if r0 > 0 and c0 > 0:  s += int(ii[r0 - 1, c0 - 1])
        return s

    chips:   List[Tuple[int, int, Window]] = []
    skipped  = 0
    tile_id  = 0

    for r0 in range(0, H - chip + 1, stride):
        for c0 in range(0, W - chip + 1, stride):
            if window_sum(r0, c0) != pix_total:
                skipped += 1
                continue
            chips.append((tile_id, r0, c0, Window(c0, r0, chip, chip)))
            tile_id += 1

    del ii
    print(f"[Chips] Valid chips: {len(chips):,}  (holes included)")
    return chips, ref_transform, ref_meta


def open_snapped(path: str | Path, ref: dict, resampling: Resampling):
    src = rasterio.open(path)
    assert src.crs == ref["crs"], f"CRS mismatch: {path}"
    assert src.width == ref["width"] and src.height == ref["height"], (
        f"Size mismatch: {path} {src.width}x{src.height} vs "
        f"{ref['width']}x{ref['height']}"
    )
    return src, None


def window_to_geometry(win: Window, transform):
    minx, miny, maxx, maxy = array_bounds(
        win.height, win.width,
        rasterio.windows.transform(win, transform)
    )
    return box(minx, miny, maxx, maxy)

from typing import Union, Sequence

def build_label_chips(
    ref_tif: str | Path,
    label_dir: str | Path,
    out_dir: str | Path,
    out_gpkg: str | Path,
    years: Union[str, Sequence[str]],
    chip: int,
    stride: int,
    layer_specs = LAYER_SPECS
):
    if isinstance(years, (str, int)):
        years = [str(years)]
    else:
        years = [str(y) for y in years]
    chips, transform, ref_meta = enumerate_chips(ref_tif, chip=chip, stride=stride)
    crs = ref_meta["crs"]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for target_year in years:
        print(f"\n{'='*60}")
        print(f"[{target_year}] starting")
        print(f"{'='*60}")

        year_dir = Path(label_dir) / target_year
        if not year_dir.exists():
            print(f"  !! Year folder not found: {year_dir} -- skipping this year")
            continue

        year_out_dir = out_root / target_year
        year_out_dir.mkdir(parents=True, exist_ok=True)

        opened       = []
        gpkg_records = []

        try:
            print(f"\n[{target_year}] opening label rasters...")
            for layer_name, fname, resamp, out_dtype in layer_specs:
                p = year_dir / fname
                if not p.exists():
                    raise FileNotFoundError(f"Missing {layer_name}: {p}")
                src, vrt = open_snapped(p, ref_meta, resamp)
                ds = vrt if vrt is not None else src
                print(f"  - {layer_name:9s} | ds.nodata={ds.nodata}")
                opened.append((layer_name, src, vrt, ds, out_dtype))

            saved   = 0
            skipped = 0
            existed = 0

            for tile_id, r, c, win in tqdm(chips, desc=f"{target_year} chips", unit="chip"):
                name     = f"loc_r{r:04d}_c{c:05d}"
                npy_path = year_out_dir / f"{name}.npy"

                if npy_path.exists():
                    existed += 1
                    gpkg_records.append({
                        "tile_id" : tile_id,
                        "row_off" : r,
                        "col_off" : c,
                        "chip"    : chip,
                        "name"    : name,
                        "geometry": window_to_geometry(win, transform),
                    })
                    continue

                chans = []
                ok    = True

                for (layer_name, _src, _vrt, ds, out_dtype) in opened:
                    if layer_name == "parcel_id":
                        arr = ds.read(1, window=win)
                        chans.append(arr.astype(out_dtype, copy=False))
                        continue

                    if layer_name == "dist":
                        ma  = ds.read(1, window=win, masked=True)
                        arr = np.ma.filled(ma, 0.0)
                        chans.append(arr.astype(out_dtype, copy=False))
                        continue

                    ma = ds.read(1, window=win, masked=True)
                    if np.ma.is_masked(ma) and np.any(ma.mask):
                        ok = False
                        break
                    arr = np.asarray(ma)
                    if not np.isfinite(arr).all():
                        ok = False
                        break
                    chans.append(arr.astype(out_dtype, copy=False))

                if not ok:
                    skipped += 1
                    continue

                stack = np.stack(chans, axis=0).astype(np.float32, copy=False)
                np.save(npy_path, stack)
                saved += 1

                gpkg_records.append({
                    "tile_id" : tile_id,
                    "row_off" : r,
                    "col_off" : c,
                    "chip"    : chip,
                    "name"    : name,
                    "geometry": window_to_geometry(win, transform),
                })
            if gpkg_records:
                gdf = gpd.GeoDataFrame(gpkg_records, crs=crs)
                gdf = gdf[["tile_id", "row_off", "col_off", "chip", "name", "geometry"]]
                gdf.to_file(out_gpkg, driver="GPKG", layer=f"label_chips_{target_year}")
                print(f"\n[gpkg] {len(gdf):,} chips written -> {out_gpkg} "
                      f"(layer=label_chips_{target_year})")
            else:
                print("\n[gpkg] no chips to write.")
            print(f"\n[{target_year}] done")
            print(f"  saved   = {saved:,}")
            print(f"  skipped = {skipped:,}  (label data masked/non-finite)")
            print(f"  existed = {existed:,}")
            print(f"  out     = {year_out_dir}")
        finally:
            for (_layer, src, vrt, _ds, _dtype) in opened:
                if vrt is not None: vrt.close()
                src.close()
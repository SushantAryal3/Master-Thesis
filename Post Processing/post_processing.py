import sys
sys.path.insert(0, "/home/ucl/elia/aryal/Single Input Model")
sys.path.insert(0, "/home/ucl/elia/aryal/Single Input Model/Script")

import gc
import math
import re
import warnings
from pathlib import Path
import shutil 

import cv2
import fiona
import fiona.crs
import geopandas as gpd
import multiprocessing as mp
import numpy as np
import pandas as pd
import rasterio
import torch
import zarr

from affine import Affine
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import (
    MultiPolygon, GeometryCollection,
    Polygon, box, mapping, shape,
)
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid
from torch.utils.data import DataLoader
from tqdm import tqdm
 
from Script.dataset import ZarrChipDataset
from Script.model import ptavit3d_dn
 
warnings.filterwarnings("ignore")

GT_PATH        = Path("/home/ucl/elia/aryal/AFBD_existing/2020/PARC_AGRI_ANON_2020.gpkg")
CHIP_GRID_PATH = Path("/globalsc/ucl/elia/aryal/chip_grid_filled.gpkg")
ZARR_PATH      = Path("/globalsc/ucl/elia/aryal/S1_zarr/combined/2020.zarr")
CKPT_PATH      = Path("/home/ucl/elia/aryal/Single Input Model/4.S1/best_model.pth")
OUT_ROOT = Path("/globalsc/ucl/elia/aryal/Prediction/S2_S1_VV_4t_final_large")

IN_CHANNELS = 2
TIME_DIM    = 10
NF          = 64
BATCH_SIZE     = 4
NUM_WORKERS    = 0
T_B            = 0.40     
T_E            = 0.40     
MIN_AREA_M2    = 800.0    
SIMPLIFY_TOL_M = 10.0    
TILE_SIZE_PX   = 8192  
OVERLAP_PX     = 256  
POLY_BATCH     = 50_000

IOU_MATCH_THRESHOLD = 1e-3   
IOU_CORRECT_MIN     = 0.70   
MIN_FRAGMENT_M2     = 1.0  
TILE_SIZE_M         = 50_000.0  
N_WORKERS           = 4
CHUNK_S3            = 4
CHUNK_S4            = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHIP_DIR     = OUT_ROOT / "chips"
AVERAGED_DIR = OUT_ROOT / "averaged"
MOSAIC_DIR   = Path("/globalsc/ucl/elia/aryal/Prediction/S2_S1_VV_4t_final_large/mosaic")
MOSAIC_PATH  = Path("/globalsc/ucl/elia/aryal/Prediction/S2_S1_VV_4t_final_large/mosaic/mosaic_extent_boundary.tif")
COUNT_PATH   = MOSAIC_DIR / "mosaic_count.tif"
POLY_DIR  = OUT_ROOT / "polygons"
POLY_PATH = POLY_DIR / f"field_polygons_tb{T_B}_te{T_E}.gpkg"
EVAL_DIR          = OUT_ROOT / "evaluation"
PRED_CLEAN_PATH   = EVAL_DIR / "pred_clean.gpkg"
GT_CLEAN_PATH     = EVAL_DIR / "gt_clean.gpkg"
STEP3_DIR      = EVAL_DIR / "step3"
CORRECT_PATH   = STEP3_DIR / "correct_detections.gpkg"
INCORRECT_PATH = STEP3_DIR / "incorrect_detections.gpkg"
OMISSION_PATH  = STEP3_DIR / "omissions.gpkg"
PAIRS_CSV      = STEP3_DIR / "matched_pairs.csv"
STEP4_DIR          = EVAL_DIR / "step4"
CORRECT_AREA_PATH  = STEP4_DIR / "correct_area.gpkg"
EXCESS_AREA_PATH   = STEP4_DIR / "excess_area.gpkg"
MISSING_AREA_PATH  = STEP4_DIR / "missing_area.gpkg"
FALSE_DET_PATH     = STEP4_DIR / "false_detections.gpkg"
MISSED_FIELDS_PATH = STEP4_DIR / "missed_fields.gpkg"
STAGE_A_DIR = EVAL_DIR / "stageA_field_level"
STAGE_C_DIR = EVAL_DIR / "stageC_polis"
LOG_PATH    = EVAL_DIR / "pipeline_log.txt"

_LOG: list = []
 
def log(msg: str) -> None:
    print(msg, flush=True)
    _LOG.append(msg)
 
def save_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_LOG))
    print(f"[log] saved -> {LOG_PATH}", flush=True)
 
def _to_str(d) -> str:
    try:
        return str(d.decode()) if isinstance(d, bytes) else str(d)
    except Exception:
        return str(d)

def validate_and_repair(gdf, name: str) -> gpd.GeoDataFrame:
    """Fix invalid geometries, explode multi-parts, keep only Polygons."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"[{name}] Expected GeoDataFrame, got {type(gdf)}")
    geom_col = gdf.geometry.name if hasattr(gdf, '_geometry_column_name') else "geometry"
    gdf = gpd.GeoDataFrame(gdf, geometry=geom_col, crs=gdf.crs)
    n0  = len(gdf)
    bad = ~gdf.geometry.is_valid
    if bad.any():
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].apply(make_valid)
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty].copy()
    multi = gdf.geometry.geom_type.isin(["MultiPolygon", "GeometryCollection"])
    if multi.any():
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"].copy()    
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)
    log(f"  [{name}] {n0:,} → {len(gdf):,} polygons after repair/explode")
    return gdf.reset_index(drop=True)

def align_crs(pred: gpd.GeoDataFrame,
              gt:   gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log(f"  [CRS] GT   : {gt.crs}")
    log(f"  [CRS] PRED : {pred.crs}")
    if pred.crs is None:
        raise ValueError("Predicted layer has no CRS.")
    if gt.crs is None:
        raise ValueError("GT layer has no CRS.")
    if pred.crs.to_epsg() != gt.crs.to_epsg():
        log(f"  [CRS] reprojecting PRED → EPSG:{gt.crs.to_epsg()} ...")
        pred = pred.to_crs(gt.crs)
        log("  [CRS] done.")
    else:
        log(f"  [CRS] both EPSG:{gt.crs.to_epsg()} — OK")
    return pred

def fill_holes(geom: Polygon) -> Polygon:
    """Remove all interior holes from a Polygon."""
    if geom.geom_type != "Polygon":
        return geom
    return Polygon(geom.exterior)

def safe_iou(a, b) -> float:
    try:
        if not a.is_valid: a = make_valid(a)
        if not b.is_valid: b = make_valid(b)
        inter = a.intersection(b).area
        if inter == 0.0:
            return 0.0
        union = a.union(b).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0

def _build_tile_grid(total_bounds, tile_size_m: float) -> list:
    """
    Divide the bounding box of the study area into a regular grid of
    rectangular tiles and return them as a list of Shapely box geometries.
    
    Tiling is used for computational efficiency: instead of comparing
    every predicted polygon against every GT polygon globally, each worker
    process only matches polygons whose centroid falls inside the same tile.
    """
    minx, miny, maxx, maxy = total_bounds
    xs = np.arange(minx, maxx, tile_size_m)
    ys = np.arange(miny, maxy, tile_size_m)
    return [box(x0, y0,
                min(x0 + tile_size_m, maxx),
                min(y0 + tile_size_m, maxy))
            for x0 in xs for y0 in ys]

def _enforce_one_to_one(all_pairs: list) -> pd.DataFrame:
    """
    Convert a list of raw candidate (pred_id, gt_id, iou) pairs into a
    strict one-to-one matching where each predicted polygon is assigned
    to at most one GT polygon and vice versa.
    After tiled matching, the same predicted polygon may have been matched
    to multiple GT polygons (or the same GT polygon claimed by multiple
    predictions). This function resolves all conflicts greedily by IoU.
    """
    if not all_pairs:
        return pd.DataFrame(columns=["pred_id", "gt_id", "iou"])
    df = (pd.DataFrame(all_pairs)
            .sort_values("iou", ascending=False)
            .reset_index(drop=True))
    seen_pred, seen_gt, rows = set(), set(), []
    for row in df.itertuples(index=False):
        pid, gid = int(row.pred_id), int(row.gt_id)
        if pid in seen_pred or gid in seen_gt:
            continue
        rows.append({"pred_id": pid, "gt_id": gid, "iou": float(row.iou)})
        seen_pred.add(pid)
        seen_gt.add(gid)
    return pd.DataFrame(rows)


def load_chip_grid(target_crs):
    """
    Load chip_grid_filled.gpkg, dissolve into one polygon, return:
      chip_union    — the full valid-area polygon
      chip_boundary — its outer boundary
    """
    log(f"\n[chip_grid] loading {CHIP_GRID_PATH}")
    grid = gpd.read_file(CHIP_GRID_PATH)
    if grid.crs is None:
        raise ValueError("chip_grid_filled.gpkg has no CRS.")
    if grid.crs.to_epsg() != target_crs.to_epsg():
        log(f"  [chip_grid] reprojecting → EPSG:{target_crs.to_epsg()} ...")
        grid = grid.to_crs(target_crs)
    chip_union    = unary_union(grid.geometry.values)
    chip_boundary = chip_union.boundary
    log(f"  [chip_grid] valid area : {chip_union.area/1e6:.2f} km²")
    return chip_union, chip_boundary

def filter_gt_by_chip_grid(gt: gpd.GeoDataFrame,
                            chip_union) -> gpd.GeoDataFrame:
    """Keep only GT polygons FULLY CONTAINED inside the chip grid."""
    log("\n[filter_gt] keeping only GT polygons fully inside chip grid ...")
    n0 = len(gt)
    tree = STRtree(gt.geometry.values)
    candidate_idx = tree.query(chip_union) 
    candidate_geoms = gt.geometry.iloc[candidate_idx]
    inside = candidate_geoms.apply(lambda g: chip_union.contains(g))
    final_idx = candidate_idx[inside.values]
    
    gt_out = gt.iloc[final_idx].copy().reset_index(drop=True)
    gt_out = gpd.GeoDataFrame(gt_out, geometry="geometry", crs=gt.crs)
    
    log(f"  candidates after spatial index : {len(candidate_idx):,}")
    log(f"  {n0:,} → {len(gt_out):,}  "
        f"({n0 - len(gt_out):,} removed — outside / touching edge)")
    return gt_out

def filter_pred_by_chip_grid(pred: gpd.GeoDataFrame,
                              chip_union) -> gpd.GeoDataFrame:
    """Remove predicted polygons NOT FULLY CONTAINED inside the chip grid
    — mirrors the same strict filter applied to GT."""
    log("\n[filter_pred] keeping only PRED polygons fully inside chip grid ...")
    n0 = len(pred)
    if n0 == 0:
        return pred

    tree = STRtree(pred.geometry.values)
    candidate_idx = tree.query(chip_union)
    candidate_geoms = pred.geometry.iloc[candidate_idx]
    inside = candidate_geoms.apply(lambda g: chip_union.contains(g))
    final_idx = candidate_idx[inside.values]

    pred_out = pred.iloc[final_idx].copy().reset_index(drop=True)
    pred_out = gpd.GeoDataFrame(pred_out, geometry="geometry", crs=pred.crs)

    log(f"  candidates after spatial index : {len(candidate_idx):,}")
    log(f"  {n0:,} → {len(pred_out):,}  "
        f"({n0 - len(pred_out):,} removed — outside / touching chip grid edge)")
    return pred_out

def build_model() -> torch.nn.Module:
    cfg = dict(
        in_channels=IN_CHANNELS, spatial_size_init=(128, 128),
        depths=[2, 2, 5, 2], nfilters_init=NF, nheads_start=NF // 4,
        NClasses=1, verbose=False, segm_act="sigmoid",
        TimeDim=TIME_DIM, nfilters_embed=NF,
    )
    model = ptavit3d_dn(**cfg).to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log(f"[model] loaded epoch {ckpt.get('epoch','?')} from {CKPT_PATH}")
    return model

def get_spatial_meta(chip_indices: np.ndarray) -> dict:
    root       = zarr.open_group(str(ZARR_PATH), mode="r")
    attrs      = dict(root.attrs)
    transform  = Affine(*attrs["transform"])
    pixel_size = abs(float(transform.a))
    chip_size  = int(attrs["chip"])
    row_offs   = root["row_off"][:][chip_indices]
    col_offs   = root["col_off"][:][chip_indices]
    row_min, col_min = int(row_offs.min()), int(col_offs.min())
    row_max  = int(row_offs.max()) + chip_size
    col_max  = int(col_offs.max()) + chip_size
    x_origin = float(transform.c) + col_min * pixel_size
    y_origin = float(transform.f) - row_min * pixel_size
    dates_sel = root["dates"][:][chip_indices]
    all_dates = [_to_str(d) for d in np.unique(dates_sel.flatten())]
    log(f"[meta] pixel={pixel_size} m  "
        f"canvas={row_max-row_min}x{col_max-col_min}  "
        f"dates {all_dates[0]}→{all_dates[-1]}")
    meta = dict(
        crs_wkt=attrs["crs_wkt"], pixel_size=pixel_size,
        chip_size=chip_size,
        canvas_h=row_max - row_min, canvas_w=col_max - col_min,
        x_origin=x_origin, y_origin=y_origin,
        row_min=row_min, col_min=col_min,
        year=attrs.get("year", 2020),
    )
    if "channels" in attrs:
        meta["channels"] = attrs["channels"]
    return meta

def _write_chip_tif(out_path, ext, bnd, dst, transform, crs_wkt,
                    dates_str, year):
    h, w = ext.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", dtype="float32", width=w, height=h, count=3,
        crs=crs_wkt, transform=transform, nodata=float("nan"),
        compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(out_path, "w", **profile) as f:
        f.write(ext.astype(np.float32), 1)
        f.write(bnd.astype(np.float32), 2)
        f.write(dst.astype(np.float32), 3)
        f.update_tags(dates_used=dates_str, year=str(year))

@torch.no_grad()
def run_inference(model, loader, meta) -> None:
    log("\n" + "=" * 60)
    log("INFERENCE")
    log("=" * 60)
    n_saved = 0
    for x, y, bm in tqdm(loader, desc="[inference]"):
        x = x.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            preds = model(x)
        ext = preds[:, 0].float().cpu().numpy()
        bnd = preds[:, 1].float().cpu().numpy()
        dst = preds[:, 2].float().cpu().numpy()
        for b in range(ext.shape[0]):
            r, c   = int(bm["row_off"][b]), int(bm["col_off"][b])
            blk    = int(bm["block_id"][b])
            x0, y0 = float(bm["x0"][b]), float(bm["y0"][b])
            dates_str = ", ".join(
                [_to_str(bm["dates"][t][b]) for t in range(len(bm["dates"]))])
            chip_tf = from_origin(west=x0, north=y0,
                                  xsize=meta["pixel_size"],
                                  ysize=meta["pixel_size"])
            _write_chip_tif(
                CHIP_DIR / f"r{r:04d}_c{c:05d}" / f"block_{blk:02d}.tif",
                ext[b], bnd[b], dst[b], chip_tf,
                meta["crs_wkt"], dates_str, meta["year"],
            )
            n_saved += 1
    log(f"[inference] {n_saved:,} chip files → {CHIP_DIR}")
    
def average_blocks(meta) -> None:
    log("\n[average] averaging temporal blocks ...")
    AVERAGED_DIR.mkdir(parents=True, exist_ok=True)
    for rc in tqdm(sorted(CHIP_DIR.glob("r*_c*")), desc="[average]"):
        blocks = sorted(rc.glob("block_*.tif"))
        if not blocks:
            continue
        stacks = {"ext": [], "bnd": [], "dst": []}
        for bp in blocks:
            with rasterio.open(bp) as src:
                stacks["ext"].append(src.read(1))
                stacks["bnd"].append(src.read(2))
                stacks["dst"].append(src.read(3))
                tf = src.transform
        _write_chip_tif(
            AVERAGED_DIR / f"{rc.name}.tif",
            np.nanmean(np.stack(stacks["ext"]), 0).astype(np.float32),
            np.nanmean(np.stack(stacks["bnd"]), 0).astype(np.float32),
            np.nanmean(np.stack(stacks["dst"]), 0).astype(np.float32),
            tf, meta["crs_wkt"], "averaged", meta["year"],
        )
    log(f"[average] done → {AVERAGED_DIR}")
    if CHIP_DIR.exists():
        shutil.rmtree(CHIP_DIR)
        log(f"[average] deleted chips → {CHIP_DIR}")
        
def _init_geotiff(path, h, w, count, transform, crs_wkt):
    profile = dict(
        driver="GTiff", dtype="float32", width=w, height=h,
        count=count, crs=crs_wkt, transform=transform,
        nodata=float("nan"), compress="lzw", tiled=True,
        blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        for r0 in range(0, h, 256):
            rh = min(256, h - r0)
            dst.write(np.full((count, rh, w), np.nan, dtype=np.float32),
                      window=Window(0, r0, w, rh))
            
def _init_geotiff_count(path, h, w, transform, crs_wkt):
    """Separate initializer for count raster — uses 0 not NaN."""
    profile = dict(
        driver="GTiff", dtype="float32", width=w, height=h,
        count=1, crs=crs_wkt, transform=transform,
        nodata=0, compress="lzw", tiled=True,
        blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        for r0 in range(0, h, 256):
            rh = min(256, h - r0)
            dst.write(
                np.zeros((1, rh, w), dtype=np.float32),
                window=Window(0, r0, w, rh)
            )

def mosaic_chips(meta) -> None:
    log("\n[mosaic] stitching chips ...")
    MOSAIC_DIR.mkdir(parents=True, exist_ok=True)
    ch, cw   = meta["canvas_h"], meta["canvas_w"]
    row_min  = meta["row_min"];  col_min = meta["col_min"]
    tf       = Affine(meta["pixel_size"], 0.0, meta["x_origin"],
                      0.0, -meta["pixel_size"], meta["y_origin"])
    _init_geotiff(MOSAIC_PATH, ch, cw, 3, tf, meta["crs_wkt"])
    _init_geotiff_count(COUNT_PATH, ch, cw, tf, meta["crs_wkt"])  # ← fixed
    pat = re.compile(r"r(\d+)_c(\d+)\.tif$")
    n_overlap = 0
    with (
        rasterio.open(MOSAIC_PATH, "r+") as dm,
        rasterio.open(COUNT_PATH,  "r+") as dc,
    ):
        for cp in tqdm(sorted(AVERAGED_DIR.glob("r*_c*.tif")), desc="[mosaic]"):
            m = pat.search(cp.name)
            if not m:
                continue
            r0 = int(m.group(1)) - row_min
            c0 = int(m.group(2)) - col_min
            with rasterio.open(cp) as src:
                ce, cb, cd = src.read(1), src.read(2), src.read(3)
            h, w   = ce.shape
            rh, rw = min(h, ch - r0), min(w, cw - c0)
            if rh <= 0 or rw <= 0:
                continue
            ce, cb, cd = ce[:rh, :rw], cb[:rh, :rw], cd[:rh, :rw]
            valid = ~np.isnan(ce)
            if not valid.any():
                continue
            win = Window(c0, r0, rw, rh)
            ee  = dm.read(1, window=win)
            eb  = dm.read(2, window=win)
            ed  = dm.read(3, window=win)
            ec  = dc.read(1, window=win)
            first, ov = valid & (ec == 0), valid & (ec > 0)
            if first.any():
                ee[first], eb[first], ed[first], ec[first] = (
                    ce[first], cb[first], cd[first], 1)
            if ov.any():
                n_overlap += int(ov.sum())
                n = ec[ov].astype(np.float32)
                ee[ov] = (ee[ov]*n + ce[ov]) / (n+1)
                eb[ov] = (eb[ov]*n + cb[ov]) / (n+1)
                ed[ov] = (ed[ov]*n + cd[ov]) / (n+1)
                ec[ov] += 1
            dm.write(ee, 1, window=win)
            dm.write(eb, 2, window=win)
            dm.write(ed, 3, window=win)
            dc.write(ec, 1, window=win)
    log(f"[mosaic] {n_overlap:,} overlap pixels averaged → {MOSAIC_PATH}")
    if AVERAGED_DIR.exists():
        shutil.rmtree(AVERAGED_DIR)
        log(f"[mosaic] deleted averaged → {AVERAGED_DIR}")

def _process_tile(args):
    (ext_t, bnd_t, tile_tf, core_bounds,
     r0c, c0c, t_b, t_e, min_area, simp) = args
    results = []
    if ext_t.max() < t_e:
        return results
    tb = (bnd_t > t_b).astype(np.uint8) * 255
    tb = cv2.ximgproc.thinning(tb, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    ref = ext_t * (1.0 - tb.astype(np.float32) / 255.0)
    thr = (ref > t_e).astype(np.uint8) * 255
    del ref, tb
    mask = thr == 255
    if not mask.any():
        return results
    xn, yn, xx, yx = core_bounds
    for gd, val in rio_shapes(thr, mask=mask, transform=tile_tf):
        if val != 255:
            continue
        g = shape(gd)
        cx, cy = g.centroid.x, g.centroid.y
        if not (xn <= cx <= xx and yn <= cy <= yx):
            continue
        if g.area < min_area:
            continue
        g = g.simplify(simp, preserve_topology=True)
        if not g.is_valid:
            g = make_valid(g)
        if g.is_empty:
            continue
        parts = list(g.geoms) if g.geom_type in (
            "MultiPolygon", "GeometryCollection") else [g]
        for p in parts:
            if p.geom_type == "Polygon" and p.area >= min_area:
                results.append((mapping(p), round(p.area, 2), r0c, c0c))
    return results

def polygonise() -> None:
    log("\n" + "=" * 60)
    log("POLYGONISATION")
    log("=" * 60)
    POLY_DIR.mkdir(parents=True, exist_ok=True)
    if POLY_PATH.exists():
        POLY_PATH.unlink()
    with rasterio.open(MOSAIC_PATH) as src:
        H, W      = src.height, src.width
        transform = src.transform
        crs       = src.crs
        ps        = abs(float(transform.a))
        log(f"[poly] {H:,} x {W:,} px  pixel={ps} m")
        ext_full = src.read(1).astype(np.float32)
        bnd_full = src.read(2).astype(np.float32)
    np.nan_to_num(ext_full, copy=False, nan=0.0)
    np.nan_to_num(bnd_full, copy=False, nan=0.0)
    all_args = []
    for r0 in range(0, H, TILE_SIZE_PX):
        for c0 in range(0, W, TILE_SIZE_PX):
            r1  = min(r0 + TILE_SIZE_PX, H)
            c1  = min(c0 + TILE_SIZE_PX, W)
            re0 = max(r0 - OVERLAP_PX, 0); ce0 = max(c0 - OVERLAP_PX, 0)
            re1 = min(r1 + OVERLAP_PX, H); ce1 = min(c1 + OVERLAP_PX, W)
            tile_tf = Affine(ps, 0.0, transform.c + ce0 * ps,
                             0.0, -ps, transform.f - re0 * ps)
            core_bounds = (
                transform.c + c0*ps, transform.f - r1*ps,
                transform.c + c1*ps, transform.f - r0*ps,
            )
            all_args.append((
                ext_full[re0:re1, ce0:ce1].copy(),
                bnd_full[re0:re1, ce0:ce1].copy(),
                tile_tf, core_bounds, r0, c0,
                T_B, T_E, MIN_AREA_M2, SIMPLIFY_TOL_M,
            ))
    del ext_full, bnd_full; gc.collect()
    n_w = N_WORKERS or mp.cpu_count()
    log(f"[poly] {len(all_args)} tiles | {n_w} workers")
    schema    = {"geometry": "Polygon",
                 "properties": {"area_m2": "float",
                                "tile_row": "int", "tile_col": "int"}}
    epsg      = crs.to_epsg()
    fiona_crs = fiona.crs.from_epsg(epsg) if epsg else crs.to_wkt()
    n_written, batch = 0, []
    with (
        mp.Pool(processes=n_w) as pool,
        fiona.open(POLY_PATH, "w", driver="GPKG",
                   schema=schema, crs=fiona_crs,
                   layer="field_polygons") as sink,
    ):
        for tile_res in tqdm(
            pool.imap_unordered(_process_tile, all_args, chunksize=4),
            total=len(all_args), desc="[poly]",
        ):
            for gd, area, r0, c0 in tile_res:
                batch.append({"geometry": gd,
                               "properties": {"area_m2": area,
                                              "tile_row": r0,
                                              "tile_col": c0}})
                n_written += 1
            if len(batch) >= POLY_BATCH:
                for f in batch: sink.write(f)
                batch.clear()
        for f in batch: sink.write(f)
    log(f"[poly] {n_written:,} polygons → {POLY_PATH}")
    
def run_step1():
    log("\n" + "=" * 60)
    log("STEP 1 — Load · validate · align CRS · chip-grid filter")
    log("=" * 60)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pred = gpd.read_file(POLY_PATH)
    gt   = gpd.read_file(GT_PATH)
    log(f"  raw  pred={len(pred):,}  gt={len(gt):,}")
    pred = validate_and_repair(pred, "PRED")
    gt   = validate_and_repair(gt,   "GT")
    pred = align_crs(pred, gt)
    chip_union, chip_boundary = load_chip_grid(gt.crs)
    gt = filter_gt_by_chip_grid(gt, chip_union)
    pred = filter_pred_by_chip_grid(pred, chip_union) 
    pred = validate_and_repair(pred, "PRED-filtered")
    gt   = validate_and_repair(gt,   "GT-filtered")
    pred = pred.reset_index(drop=True)
    gt   = gt.reset_index(drop=True)
    pred["pred_id"] = pred.index.astype(int)
    gt["gt_id"]     = gt.index.astype(int)
    log(f"\n  Predicted (final) : {len(pred):,}")
    log(f"  GT        (final) : {len(gt):,}")
    log(f"  CRS               : {pred.crs}")
    pred.to_file(PRED_CLEAN_PATH, driver="GPKG", layer="pred_clean")
    gt.to_file(  GT_CLEAN_PATH,   driver="GPKG", layer="gt_clean")
    log("[step1] DONE")
    return pred, gt

def run_stage_A(pred: gpd.GeoDataFrame,
                gt:   gpd.GeoDataFrame) -> dict:
    log("\n" + "=" * 60)
    log("STAGE A — Field-level whole-area overlap")
    log("=" * 60)
    STAGE_A_DIR.mkdir(parents=True, exist_ok=True)

    if len(pred) == 0:
        log("  [SKIP] No predicted polygons — skipping Stage A.")
        stats = dict(
            area_gt_km2=0, area_pred_km2=0,
            area_overlap_km2=0, area_extra_km2=0,
            area_missed_km2=0,
            pct_overlap_of_gt=0, pct_overlap_of_pred=0,
            pct_extra_of_pred=0, pct_missed_of_gt=0,
            field_iou=0.0,
        )
        pd.DataFrame([stats]).to_csv(
            STAGE_A_DIR / "A_field_level_stats.csv", index=False)
        log("[stage A] DONE")
        return stats

    ap = pred.geometry.area.sum()
    ag = gt.geometry.area.sum()

    log("  computing overlaps via spatial index ...")
    gt_geoms  = list(gt.geometry)
    tree      = STRtree(gt_geoms)

    overlap_parts = []   
    excess_parts  = []  
    missing_parts = []  

    pred_matched = set()
    gt_matched   = set()

    for pi, pg in enumerate(pred.geometry):
        candidate_idxs = tree.query(pg)
        local_inter    = []

        for idx in candidate_idxs:
            inter = pg.intersection(gt_geoms[idx])
            if not inter.is_empty and inter.area >= MIN_FRAGMENT_M2:
                overlap_parts.append(inter)
                local_inter.append(inter)
                pred_matched.add(pi)
                gt_matched.add(idx)

        if local_inter:
            covered = unary_union(local_inter)
            exc     = pg.difference(covered)
        else:
            exc = pg  

        if not exc.is_empty and exc.area >= MIN_FRAGMENT_M2:
            excess_parts.append(exc)

    for gi, gg in enumerate(gt_geoms):
        candidate_idxs = tree.query(gg)  
        pass

    pred_geoms = list(pred.geometry)
    pred_tree  = STRtree(pred_geoms)

    for gi, gg in enumerate(gt_geoms):
        candidate_idxs = pred_tree.query(gg)
        local_inter    = []

        for idx in candidate_idxs:
            inter = gg.intersection(pred_geoms[idx])
            if not inter.is_empty and inter.area >= MIN_FRAGMENT_M2:
                local_inter.append(inter)

        if local_inter:
            covered = unary_union(local_inter)
            miss    = gg.difference(covered)
        else:
            miss = gg 

        if not miss.is_empty and miss.area >= MIN_FRAGMENT_M2:
            missing_parts.append(miss)

    ao = sum(g.area for g in overlap_parts)
    ae = sum(g.area for g in excess_parts)
    am = sum(g.area for g in missing_parts)
    iou_f = ao / (ap + ag - ao) if (ap + ag - ao) > 0 else 0.0

    log(f"  GT area      : {ag/1e6:.4f} km²")
    log(f"  PRED area    : {ap/1e6:.4f} km²")
    log(f"  Overlap      : {ao/1e6:.4f} km²  "
        f"({100*ao/ag:.2f}% of GT | " if ag > 0 else "(N/A% of GT | "
        f"{100*ao/ap:.2f}% of PRED)" if ap > 0 else "N/A% of PRED)")
    log(f"  Extra (FP)   : {ae/1e6:.4f} km²  "
        f"({100*ae/ap:.2f}% of PRED)" if ap > 0 else "(N/A% of PRED)")
    log(f"  Missed (FN)  : {am/1e6:.4f} km²  "
        f"({100*am/ag:.2f}% of GT)" if ag > 0 else "(N/A% of GT)")
    log(f"  Field IoU    : {iou_f:.4f}")

    crs = pred.crs

    def _save_parts(parts, path, layer):
        if not parts:
            log(f"  [skip] {layer} — no geometries")
            return
        all_geoms = []
        for g in parts:
            if g.geom_type in ("MultiPolygon", "GeometryCollection"):
                all_geoms.extend(
                    [p for p in g.geoms
                     if p.geom_type == "Polygon" and p.area >= MIN_FRAGMENT_M2])
            elif g.geom_type == "Polygon" and g.area >= MIN_FRAGMENT_M2:
                all_geoms.append(g)
        if all_geoms:
            gpd.GeoDataFrame(
                {"area_m2": [round(p.area, 2) for p in all_geoms]},
                geometry=all_geoms, crs=crs,
            ).to_file(path, driver="GPKG", layer=layer)
            log(f"  [save] {layer} ({len(all_geoms):,}) → {path}")

    _save_parts(overlap_parts, STAGE_A_DIR / "A_overlapped.gpkg", "overlapped")
    _save_parts(excess_parts,  STAGE_A_DIR / "A_extra.gpkg",      "extra")
    _save_parts(missing_parts, STAGE_A_DIR / "A_missed.gpkg",     "missed")

    stats = dict(
        area_gt_km2=round(ag/1e6, 6), area_pred_km2=round(ap/1e6, 6),
        area_overlap_km2=round(ao/1e6, 6), area_extra_km2=round(ae/1e6, 6),
        area_missed_km2=round(am/1e6, 6),
        pct_overlap_of_gt=round(100*ao/ag, 4)       if ag > 0 else 0.0,
        pct_overlap_of_pred=round(100*ao/ap, 4)     if ap > 0 else 0.0,
        pct_extra_of_pred=round(100*ae/ap, 4)       if ap > 0 else 0.0,
        pct_missed_of_gt=round(100*am/ag, 4)        if ag > 0 else 0.0,
        field_iou=round(iou_f, 6),
    )
    
    pd.DataFrame([stats]).to_csv(
        STAGE_A_DIR / "A_field_level_stats.csv", index=False)
    log("[stage A] DONE")
    return stats

_S3_PRED_GEOMS: list = []
_S3_PRED_IDS:   list = []
_S3_GT_GEOMS:   list = []
_S3_GT_IDS:     list = []
 
def _s3_init(pg, pi, gg, gi):
    global _S3_PRED_GEOMS, _S3_PRED_IDS, _S3_GT_GEOMS, _S3_GT_IDS
    _S3_PRED_GEOMS, _S3_PRED_IDS = pg, pi
    _S3_GT_GEOMS,   _S3_GT_IDS   = gg, gi
 
def _s3_tile(tile_geom) -> list:
    pred_in = [(pid, g) for pid, g in zip(_S3_PRED_IDS, _S3_PRED_GEOMS)
               if g.centroid.within(tile_geom)]
    gt_in   = [(gid, g) for gid, g in zip(_S3_GT_IDS, _S3_GT_GEOMS)
               if g.centroid.within(tile_geom)]
    if not pred_in or not gt_in:
        return []
    gt_g = [g for _, g in gt_in]
    gt_i = [i for i, _ in gt_in]
    gt_m = {i: g for i, g in gt_in}
    tree = STRtree(gt_g)
    pairs = []
    for pid, pg in pred_in:
        best_iou, best_gid = 0.0, None
        for idx in tree.query(pg):
            iou = safe_iou(pg, gt_m[gt_i[idx]])
            if iou > best_iou:
                best_iou, best_gid = iou, gt_i[idx]
        if best_gid is not None and best_iou >= IOU_MATCH_THRESHOLD:
            pairs.append({"pred_id": pid, "gt_id": best_gid, "iou": best_iou})
    return pairs

def run_step3(pred: gpd.GeoDataFrame, gt: gpd.GeoDataFrame):
    log("\n" + "=" * 60)
    log("STEP 3 — Per-parcel matching")
    log("=" * 60)
    STEP3_DIR.mkdir(parents=True, exist_ok=True)
    tiles = _build_tile_grid(gt.total_bounds, TILE_SIZE_M)
    n_w   = N_WORKERS or mp.cpu_count()
    log(f"  {len(tiles)} tiles | {n_w} workers")
 
    all_pairs: list = []
    with mp.Pool(n_w, initializer=_s3_init,
                 initargs=(list(pred.geometry),
                           list(pred["pred_id"].astype(int)),
                           list(gt.geometry),
                           list(gt["gt_id"].astype(int)))) as pool:
        for res in pool.imap_unordered(_s3_tile, tiles, chunksize=CHUNK_S3):
            all_pairs.extend(res)
 
    matched_df = _enforce_one_to_one(all_pairs)
    del all_pairs; gc.collect()
 
    m_pred = set(matched_df["pred_id"])
    m_gt   = set(matched_df["gt_id"])
    iou_map  = dict(zip(matched_df["pred_id"], matched_df["iou"]))
    gtid_map = dict(zip(matched_df["pred_id"], matched_df["gt_id"]))
 
    correct   = pred[ pred["pred_id"].isin(m_pred)].copy()
    incorrect = pred[~pred["pred_id"].isin(m_pred)].copy()
    omissions = gt  [~gt["gt_id"].isin(m_gt)].copy()
 
    correct["match_iou"]       = correct["pred_id"].map(iou_map)
    correct["matched_gt_id"]   = correct["pred_id"].map(gtid_map)
    incorrect["match_iou"]     = float("nan")
    incorrect["matched_gt_id"] = -1
 
    tp, fp, fn = len(correct), len(incorrect), len(omissions)
    prec   = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1     = 2*prec*recall / (prec+recall) if prec+recall > 0 else 0.0
 
    log(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}")
    log(f"  Precision={prec:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    if len(matched_df):
        log(f"  IoU mean={matched_df['iou'].mean():.4f}  "
            f"median={matched_df['iou'].median():.4f}  "
            f"min={matched_df['iou'].min():.4f}  "
            f"max={matched_df['iou'].max():.4f}")
 
    matched_df.to_csv(PAIRS_CSV, index=False)
    correct.to_file(  CORRECT_PATH,   driver="GPKG", layer="correct_detections")
    incorrect.to_file(INCORRECT_PATH, driver="GPKG", layer="incorrect_detections")
    omissions.to_file(OMISSION_PATH,  driver="GPKG", layer="omissions")
    log("[step3] DONE")
    return matched_df, correct, incorrect, omissions

_S4_PRED_MAP: dict = {}
_S4_GT_MAP:   dict = {}
 
def _s4_init(pm, gm):
    global _S4_PRED_MAP, _S4_GT_MAP
    _S4_PRED_MAP, _S4_GT_MAP = pm, gm
 
def _s4_decompose(row_tuple):
    pred_id, gt_id, iou = row_tuple
    p = _S4_PRED_MAP.get(pred_id)
    g = _S4_GT_MAP.get(gt_id)
    out = {"pred_id": pred_id, "gt_id": gt_id, "iou": iou,
           "correct_area": None, "excess_area": None, "missing_area": None}
    if p is None or g is None:
        return out
    try:
        if not p.is_valid: p = make_valid(p)
        if not g.is_valid: g = make_valid(g)
        c = p.intersection(g)
        e = p.difference(g)
        m = g.difference(p)
        if not c.is_empty and c.area >= MIN_FRAGMENT_M2: out["correct_area"] = c
        if not e.is_empty and e.area >= MIN_FRAGMENT_M2: out["excess_area"]  = e
        if not m.is_empty and m.area >= MIN_FRAGMENT_M2: out["missing_area"] = m
    except Exception:
        pass
    return out


def run_step4(matched_df: pd.DataFrame,
              pred:       gpd.GeoDataFrame,
              gt:         gpd.GeoDataFrame,
              incorrect:  gpd.GeoDataFrame,
              omissions:  gpd.GeoDataFrame) -> None:
    log("\n" + "=" * 60)
    log(f"STEP 4 — Decomposition  (IoU ≥ {IOU_CORRECT_MIN*100:.0f}%, hole-filled)")
    log("=" * 60)
    STEP4_DIR.mkdir(parents=True, exist_ok=True)
 
    high_iou = matched_df[matched_df["iou"] >= IOU_CORRECT_MIN].copy()
    high_ids = set(high_iou["pred_id"])
    log(f"  all matched pairs          : {len(matched_df):,}")
    log(f"  IoU ≥ {IOU_CORRECT_MIN*100:.0f}% pairs            : {len(high_iou):,}")
 
    correct_high = pred[pred["pred_id"].isin(high_ids)].copy()
    correct_high["geometry"] = correct_high["geometry"].apply(fill_holes)
    log(f"  holes filled               : {len(correct_high):,} parcels")
 
    pred_map = dict(zip(correct_high["pred_id"].astype(int),
                        correct_high.geometry))
    gt_map   = dict(zip(gt["gt_id"].astype(int), gt.geometry))
    tuples   = [(int(r.pred_id), int(r.gt_id), float(r.iou))
                for r in high_iou.itertuples(index=False)]
    log_step = max(1, len(tuples) // 20)
    n_w      = N_WORKERS or mp.cpu_count()
 
    correct_rows, excess_rows, missing_rows = [], [], []
    with mp.Pool(n_w, initializer=_s4_init,
                 initargs=(pred_map, gt_map)) as pool:
        for n_done, res in enumerate(
            pool.imap_unordered(_s4_decompose, tuples, chunksize=CHUNK_S4), 1
        ):
            pid, gid, iou = res["pred_id"], res["gt_id"], res["iou"]
            if res["correct_area"] is not None:
                correct_rows.append(
                    {"pred_id": pid, "gt_id": gid, "iou": iou,
                     "area_m2": round(res["correct_area"].area, 2),
                     "geometry": res["correct_area"]})
            if res["excess_area"] is not None:
                excess_rows.append(
                    {"pred_id": pid, "gt_id": gid, "iou": iou,
                     "area_m2": round(res["excess_area"].area, 2),
                     "geometry": res["excess_area"]})
            if res["missing_area"] is not None:
                missing_rows.append(
                    {"pred_id": pid, "gt_id": gid, "iou": iou,
                     "area_m2": round(res["missing_area"].area, 2),
                     "geometry": res["missing_area"]})
            if n_done % log_step == 0:
                log(f"  {n_done:,}/{len(tuples):,} "
                    f"({100*n_done/len(tuples):.0f}%)")
 
    crs = pred.crs
    for rows, name, path in [
        (correct_rows, "correct_area",  CORRECT_AREA_PATH),
        (excess_rows,  "excess_area",   EXCESS_AREA_PATH),
        (missing_rows, "missing_area",  MISSING_AREA_PATH),
    ]:
        if rows:
            gpd.GeoDataFrame(rows, crs=crs).to_file(
                path, driver="GPKG", layer=name)
            log(f"  [save] {name} ({len(rows):,}) → {path}")
 
    incorrect.to_file(FALSE_DET_PATH,     driver="GPKG", layer="false_detections")
    omissions.to_file(MISSED_FIELDS_PATH, driver="GPKG", layer="missed_fields")
    log(f"\n  correct  : {len(correct_rows):,}  "
        f"excess : {len(excess_rows):,}  "
        f"missing : {len(missing_rows):,}  "
        f"FP : {len(incorrect):,}  FN : {len(omissions):,}")
    log("[step4] DONE")

def _pt_seg_dist(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx-ax, by-ay
    l2 = dx*dx + dy*dy
    if l2 == 0.0:
        return math.hypot(px-ax, py-ay)
    t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / l2))
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy))
 
 
def _directed_polis(V_P: np.ndarray, V_Q: np.ndarray) -> float:
    if np.allclose(V_P[0], V_P[-1]): V_P = V_P[:-1]
    if np.allclose(V_Q[0], V_Q[-1]): V_Q = V_Q[:-1]
    Qn = np.roll(V_Q, -1, axis=0)
    ax, ay = V_Q[:, 0], V_Q[:, 1]
    bx, by = Qn[:, 0],  Qn[:, 1]
    total  = sum(
        min(_pt_seg_dist(px, py, ax[i], ay[i], bx[i], by[i])
            for i in range(len(V_Q)))
        for px, py in V_P
    )
    return total / len(V_P)
 
def polis_score(p_geom: Polygon, g_geom: Polygon) -> float:
    P = np.array(p_geom.exterior.coords)
    Q = np.array(g_geom.exterior.coords)
    return 0.5 * (_directed_polis(P, Q) + _directed_polis(Q, P))
 
_C_PRED_MAP: dict = {}
_C_GT_MAP:   dict = {}
 
def _c_init(pm, gm):
    global _C_PRED_MAP, _C_GT_MAP
    _C_PRED_MAP, _C_GT_MAP = pm, gm
 
def _c_worker(row_tuple):
    pred_id, gt_id, iou = row_tuple
    p = _C_PRED_MAP.get(pred_id)
    g = _C_GT_MAP.get(gt_id)
    score = float("nan")
    if p is not None and g is not None:
        try:
            if not p.is_valid: p = make_valid(p)
            if not g.is_valid: g = make_valid(g)
            if p.geom_type != "Polygon": p = p.convex_hull
            if g.geom_type != "Polygon": g = g.convex_hull
            score = polis_score(p, g)
        except Exception:
            pass
    return {"pred_id": pred_id, "gt_id": gt_id, "iou": iou, "polis_m": score}
 
 
def run_stage_C(matched_df: pd.DataFrame,
                pred:       gpd.GeoDataFrame,
                gt:         gpd.GeoDataFrame) -> pd.DataFrame:
    log("\n" + "=" * 60)
    log(f"STAGE C — PoLiS  (IoU ≥ {IOU_CORRECT_MIN*100:.0f}%, hole-filled)")
    log("=" * 60)
    STAGE_C_DIR.mkdir(parents=True, exist_ok=True)
 
    high_iou     = matched_df[matched_df["iou"] >= IOU_CORRECT_MIN].copy()
    high_ids     = set(high_iou["pred_id"])
    correct_high = pred[pred["pred_id"].isin(high_ids)].copy()
    correct_high["geometry"] = correct_high["geometry"].apply(fill_holes)
 
    pred_map = dict(zip(correct_high["pred_id"].astype(int),
                        correct_high.geometry))
    gt_map   = dict(zip(gt["gt_id"].astype(int), gt.geometry))
    tuples   = [(int(r.pred_id), int(r.gt_id), float(r.iou))
                for r in high_iou.itertuples(index=False)]
    n_w = N_WORKERS or mp.cpu_count()
 
    log(f"  {len(tuples):,} pairs | {n_w} workers")
    results = []
    with mp.Pool(n_w, initializer=_c_init,
                 initargs=(pred_map, gt_map)) as pool:
        for res in tqdm(
            pool.imap_unordered(_c_worker, tuples, chunksize=CHUNK_S4),
            total=len(tuples), desc="[PoLiS]",
        ):
            results.append(res)
 
    df = pd.DataFrame(results).dropna(subset=["polis_m"])
    log(f"\n  evaluated : {len(df):,} pairs")
    log(f"  mean      : {df['polis_m'].mean():.4f} m")
    log(f"  median    : {df['polis_m'].median():.4f} m")
    log(f"  std       : {df['polis_m'].std():.4f} m")
    log(f"  min/max   : {df['polis_m'].min():.4f} / {df['polis_m'].max():.4f} m")
 
    df.to_csv(STAGE_C_DIR / "C_polis_scores.csv", index=False)
 
    polis_map = dict(zip(df["pred_id"], df["polis_m"]))
    out_gdf   = correct_high.copy()
    out_gdf["polis_m"] = out_gdf["pred_id"].map(polis_map)
    out_gdf.to_file(
        STAGE_C_DIR / "C_correct_parcels_polis.gpkg",
        driver="GPKG", layer="correct_parcels_polis",
    )
    log(f"  [save] → {STAGE_C_DIR / 'C_correct_parcels_polis.gpkg'}")
    log("[stage C] DONE")
    return df
        
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        # ── 1. INFERENCE ────────────────────────────────────────────────────
        log("=" * 60 + "\nINFERENCE PIPELINE\n" + "=" * 60)
        root         = zarr.open_group(str(ZARR_PATH), mode="r")
        n_chips      = root["row_off"].shape[0]
        chip_indices = np.arange(n_chips)
        log(f"[zarr] {n_chips:,} chips")
        meta    = get_spatial_meta(chip_indices)
        model   = build_model()
        dataset = ZarrChipDataset(str(ZARR_PATH),
                                  indices=chip_indices, sensor="S1")
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=(DEVICE.type == "cuda"))
        run_inference(model, loader, meta)
        del model; gc.collect(); torch.cuda.empty_cache()
        average_blocks(meta)
        mosaic_chips(meta)
        # ── 2. POLYGONISATION ───────────────────────────────────────────────
        polygonise()
        # ── 3. EVALUATION ───────────────────────────────────────────────────
        pred, gt = run_step1()
        # Stage A: whole-area field-level overlap statistics
        pred = gpd.read_file("/globalsc/ucl/elia/aryal/Prediction/S2_cloud/evaluation/pred_clean.gpkg")
        gt = gpd.read_file("/globalsc/ucl/elia/aryal/Prediction/S2_cloud/evaluation/gt_clean.gpkg")
        run_stage_A(pred, gt)
        # # Step 3: per-parcel matching (any overlap)
        matched_df, correct, incorrect, omissions = run_step3(pred, gt)
        # # Step 4: geometry decomposition on IoU ≥ 85 % parcels (hole-filled)
        run_step4(matched_df, pred, gt, incorrect, omissions)
        # # Stage C: PoLiS on same IoU ≥ 90 % hole-filled parcels
        run_stage_C(matched_df, pred, gt)
 
    finally:
        save_log()
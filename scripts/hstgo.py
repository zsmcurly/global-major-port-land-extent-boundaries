#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-aligned H-STGO inference for encoded 12-band annual composites."""

from __future__ import annotations

import json
import math
import os
import re
import gc
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import maxflow  # pip install PyMaxflow
from pyproj import Geod, Transformer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from rasterio.windows import Window
from tqdm import tqdm


FILE_RE = re.compile(
    r"^CLUSTER_(?P<cluster_id>\d+)_cluster_(?P<cluster_uid>\d+)_(?P<year>(?:19|20)\d{2})_feat12_u16\.tif$",
    re.IGNORECASE,
)


#CONFIG
DATA_ROOT = r"new"
OUT_ROOT = "runs/spatiotemporal_hstgo_top100_u16"
CKPT_PATH = "best_fg.ckpt"
MAPPING_CSV = "results/cluster_port_mapping_rerun_failed2.csv"

YEAR_MIN, YEAR_MAX = 2017, 2025
REF_YEAR = 2022#基准年份
STRICT_YEARS = True
CLUSTERS: Optional[List[int]] = None

PATCH = 512
STRIDE = 256
BATCH_TILES = 6
USE_TTA = True
USE_EMA = True
HSTGO_BLOCK_SIZE = 256
HSTGO_PADDING = 64
HSTGO_NUM_WORKERS = max(1, (multiprocessing.cpu_count() if multiprocessing.cpu_count() else 2) - 2)

# Band order assumption (exported stack):
# 0 B2,1 B3,2 B4,3 B8,4 B11,5 B12,6 VV,7 VH,8 VV_VH,9 NDVI,10 MNDWI,11 NDBI,12 label
IDX_NDVI = 9
IDX_MNDWI = 10
IDX_NDBI = 11
IDX_VV = 6
IDX_VH = 7
N_FEATURES = 12

# Geo-physical spatial refinement
COAST_MAXDIST_M = 800.0#最大海岸距离
COAST_SIGMA_M = 120.0#高斯衰减标准差
LAMBDA_PRIOR = 0.55#先验融合权重

# Spatial mask extraction
#形态学提取，
MIN_COMP_AREA_M2 = 12_000.0#最小联通分量
MORPH_CLOSE_R = 3#开闭半径
MORPH_OPEN_R = 2
#各个先验的评分权重
# Prior weights
W0_PRIOR = -0.3
W_COAST = 2.0
W_NDBI = 1.0
W_NDVI_INV = 0.9

# Hysteresis thresholds
#联通区域提取
SPATIAL_T_HIGH = 0.65
SPATIAL_T_LOW = 0.40
SPATIAL_FALLBACK_Q = 0.995

# Water probability
#水体概率 0.95必然不是港口
WATER_SLOPE_MNDWI = 10.0
WATER_SLOPE_SAR = 10.0
WATER_MNDWI_QUANTILE = 0.85
WATER_HARD_FORBID = 0.95

# Index reliability gating (scheme #1)
#指数分母过小的问题
INDEX_DEN_TRUST_MIN = 0.02
INDEX_DEN_TRUST_MAX = 0.08
INDEX_CONF_POW = 1.0

# H-STGO topology priors (review-upgraded)
#沿海的权重
HSTGO_ANISO_BASE = 0.5
HSTGO_ANISO_GAIN = 1.5
HSTGO_ANISO_DECAY_M = 500.0
# Layer 1: water -> non-water is inexpensive; reverse transitions require water evidence.
HSTGO_WATER_TO_NON_WATER_COST = 2.0
HSTGO_NON_WATER_TO_WATER_BASE = 50.0
#时空一致性，向前向后约束
HSTGO_LC_DIST_DECAY_M = 200.0
HSTGO_LC_FWD_BASE = 20.0
HSTGO_LC_FWD_GAIN = 50.0
HSTGO_LC_BWD_BASE = 2.0
HSTGO_LC_BWD_GAIN = 40.0
HSTGO_LC_BWD_BONUS = 6.0
HSTGO_LC_BWD_MIN = 1.0

# H-STGO temporal consistency (balanced, avoids over-locking / over-expansion)
HSTGO_TEMPORAL_WEIGHTS = (1.0, 2.0, 3.0, 2.0, 1.0)  # 5-year weighted smoother
HSTGO_RETREAT_K = 2                                   # retreat needs >=K-year consistent evidence
HSTGO_RETREAT_RELAX = 0.60                            # relax anti-shrink penalty when persistent decline exists
HSTGO_FWD_MIN = 2.0
HSTGO_EXPAND_MID = 0.55
HSTGO_EXPAND_SLOPE = 6.0

# Directional decomposition relative to the first study year.
SEA_DOMAIN_WATER_THRESHOLD = 0.20
SEA_DOMAIN_DILATE_K = 5

GEOD = Geod(ellps="WGS84")

# Runtime safety for large ports
HSTGO_AUTO_DOWNSCALE_GB = 2.5
HSTGO_AUTO_MAX_WORKERS_LARGE = 2
HSTGO_MIN_BLOCK_SIZE = 192
HSTGO_MIN_PADDING = 48


@dataclass
class RunConfig:
    data_root: str = DATA_ROOT
    out_root: str = OUT_ROOT
    ckpt_path: str = CKPT_PATH
    mapping_csv: str = MAPPING_CSV
    year_min: int = YEAR_MIN
    year_max: int = YEAR_MAX
    patch: int = PATCH
    stride: int = STRIDE
    batch_tiles: int = BATCH_TILES
    use_tta: bool = USE_TTA
    use_ema: bool = USE_EMA
    strict_years: bool = STRICT_YEARS
    ref_year: int = REF_YEAR
    clusters: Optional[List[int]] = CLUSTERS
    exclude_clusters: Optional[List[int]] = None
    resume: bool = True
    graphcut_only: bool = False
    hstgo_block_size: int = HSTGO_BLOCK_SIZE
    hstgo_padding: int = HSTGO_PADDING
    hstgo_num_workers: int = HSTGO_NUM_WORKERS


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="H-STGO Temporal Inference Pipeline")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Input folder containing cluster/year tif files.")
    parser.add_argument("--out_root", type=str, default=OUT_ROOT, help="Output root directory.")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH, help="Checkpoint path.")
    parser.add_argument("--mapping_csv", type=str, default=MAPPING_CSV, help="Cluster-port mapping CSV.")
    parser.add_argument("--year-min", type=int, default=YEAR_MIN, help="First study year (paper default: 2017).")
    parser.add_argument("--year-max", type=int, default=YEAR_MAX, help="Last study year (paper default: 2025).")
    parser.add_argument("--ref-year", type=int, default=REF_YEAR, help="Reference year used only to read a raster profile.")
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=None,
        help="Whitelist: only process these cluster IDs.",
    )
    parser.add_argument(
        "--exclude",
        type=int,
        nargs="+",
        default=None,
        help="Blacklist: skip these cluster IDs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume (will rerun even if summary exists).",
    )
    parser.add_argument(
        "--graphcut-only",
        action="store_true",
        help="Reuse existing prob_refined/water_prob/dist_to_water_m and rerun only H-STGO + summary.",
    )
    args = parser.parse_args()
    if args.year_min > args.year_max:
        parser.error("--year-min must not exceed --year-max")
    return RunConfig(
        data_root=args.data_root,
        out_root=args.out_root,
        ckpt_path=args.ckpt_path,
        mapping_csv=args.mapping_csv,
        year_min=args.year_min,
        year_max=args.year_max,
        ref_year=args.ref_year,
        clusters=args.clusters,
        exclude_clusters=args.exclude,
        resume=not args.no_resume,
        graphcut_only=args.graphcut_only,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def check_runtime_deps() -> None:
    _ = smp.Unet
    _ = maxflow.Graph


def load_cluster_port_map(mapping_csv: str) -> Dict[int, str]:
    if not os.path.exists(mapping_csv):
        return {}
    df = pd.read_csv(mapping_csv, low_memory=False)
    if "cluster_id" not in df.columns:
        return {}
    port_col = "resolved_portids" if "resolved_portids" in df.columns else "contained_portids"
    if port_col not in df.columns:
        return {}
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            cid = int(row["cluster_id"])
        except Exception:
            continue
        out[cid] = str(row[port_col]) if pd.notna(row[port_col]) else ""
    return out


def scan_cluster_year_files(cfg: RunConfig) -> Dict[int, Dict[str, object]]:
    grouped: Dict[int, Dict[str, object]] = {}
    if not os.path.isdir(cfg.data_root):
        raise FileNotFoundError(f"Data root not found: {cfg.data_root}")

    for fn in sorted(os.listdir(cfg.data_root)):
        if not fn.lower().endswith(".tif"):
            continue
        m = FILE_RE.match(fn)
        if not m:
            continue
        cid = int(m.group("cluster_id"))
        uid = int(m.group("cluster_uid"))
        year = int(m.group("year"))
        if year < cfg.year_min or year > cfg.year_max:
            continue
        if cfg.exclude_clusters and cid in cfg.exclude_clusters:
            continue
        path = os.path.join(cfg.data_root, fn)
        if cid not in grouped:
            grouped[cid] = {"uid": uid, "years": {}}
        grouped[cid]["years"][year] = path

    if cfg.clusters:
        want = {int(c) for c in cfg.clusters}
        grouped = {cid: v for cid, v in grouped.items() if cid in want}

    if not grouped:
        raise RuntimeError(
            f"No cluster files matched naming pattern under {cfg.data_root}. "
            "(after applying whitelist/blacklist filters). "
            "Expected: CLUSTER_<id>_cluster_<uid>_<year>_feat12_u16.tif"
        )
    return grouped


def decode_feat12_u16(x_u16: np.ndarray) -> np.ndarray:
    x = x_u16.astype(np.float32, copy=False)
    y = np.empty_like(x, dtype=np.float32)
    y[0:6] = x[0:6] * 1e-4
    vv_db = x[6] * 1e-3 - 35.0
    vh_db = x[7] * 1e-3 - 35.0
    vv_vh_db = x[8] * 1e-3 - 20.0
    y[6] = np.clip((vv_db + 25.0) / 30.0, 0.0, 1.0)
    y[7] = np.clip((vh_db + 25.0) / 30.0, 0.0, 1.0)
    y[8] = np.clip(vv_vh_db / 15.0, 0.0, 1.0)
    y[9:12] = x[9:12] * 1e-4 - 1.0
    return y


def decode_single_band_u16(arr_u16: np.ndarray, band_idx0: int) -> np.ndarray:
    x = arr_u16.astype(np.float32, copy=False)
    if 0 <= band_idx0 <= 5:
        return x * 1e-4
    if band_idx0 == 6:
        vv_db = x * 1e-3 - 35.0
        return np.clip((vv_db + 25.0) / 30.0, 0.0, 1.0)
    if band_idx0 == 7:
        vh_db = x * 1e-3 - 35.0
        return np.clip((vh_db + 25.0) / 30.0, 0.0, 1.0)
    if band_idx0 == 8:
        vv_vh_db = x * 1e-3 - 20.0
        return np.clip(vv_vh_db / 15.0, 0.0, 1.0)
    if 9 <= band_idx0 <= 11:
        return x * 1e-4 - 1.0
    return x


def hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h) if h > 1 else np.ones((h,), np.float32)
    wx = np.hanning(w) if w > 1 else np.ones((w,), np.float32)
    win = np.outer(wy, wx).astype(np.float32)
    return np.clip(win, 1e-3, 1.0)


def pad_to_patch(x_c_hw: np.ndarray, patch: int) -> np.ndarray:
    c, h, w = x_c_hw.shape
    if h == patch and w == patch:
        return x_c_hw.astype(np.float32)
    pad_h = max(0, patch - h)
    pad_w = max(0, patch - w)
    mode = "reflect" if (h > 1 and w > 1) else "edge"
    x_pad = np.pad(x_c_hw, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)
    return x_pad[:, :patch, :patch].astype(np.float32)


def write_single_band_tif(path: str, ref_profile: dict, data: np.ndarray, dtype: str):
    profile = ref_profile.copy()
    profile.update(count=1, dtype=dtype, compress="deflate")
    ensure_dir(os.path.dirname(path))
    if "nodata" in profile:
        profile.pop("nodata")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


def estimate_pixel_metrics_m(ds: rasterio.DatasetReader) -> Tuple[float, float]:
    tr = ds.transform
    dx = abs(tr.a)
    dy = abs(tr.e)
    crs = ds.crs
    wkt = ""
    try:
        if crs is not None:
            wkt = crs.to_wkt() or ""
    except Exception:
        wkt = ""
    is_meter = False
    if crs is not None:
        try:
            if crs.is_projected:
                is_meter = True
        except Exception:
            pass
        if ("metre" in wkt.lower()) or ("meter" in wkt.lower()):
            is_meter = True
    if is_meter:
        px_x = dx
        px_y = dy
        return float((px_x + px_y) * 0.5), float(px_x * px_y)
    b = ds.bounds
    lat0 = float((b.top + b.bottom) * 0.5)
    lat_rad = math.radians(lat0)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    px_x = dx * m_per_deg_lon
    px_y = dy * m_per_deg_lat
    return float((px_x + px_y) * 0.5), float(px_x * px_y)


def effective_raster_crs(crs):
    if crs is None:
        raise ValueError("Raster has no CRS")
    wkt = crs.to_wkt() or ""
    if "LOCAL_CS" in wkt and "Pseudo-Mercator" in wkt:
        return "EPSG:3857"
    return crs


def geodesic_row_areas_m2(ds: rasterio.DatasetReader) -> np.ndarray:
    """Return the WGS84 ground area of one pixel in each raster row."""
    tr = ds.transform
    if not math.isclose(tr.b, 0.0, abs_tol=1e-12) or not math.isclose(tr.d, 0.0, abs_tol=1e-12):
        raise ValueError("Rotated rasters are not supported for row-wise area calculation")
    to_lonlat = Transformer.from_crs(effective_raster_crs(ds.crs), "EPSG:4326", always_xy=True)
    areas = np.empty(ds.height, dtype=np.float64)
    for row in range(ds.height):
        corners = [tr * (0, row), tr * (1, row), tr * (1, row + 1), tr * (0, row + 1)]
        lon, lat = to_lonlat.transform(
            [point[0] for point in corners],
            [point[1] for point in corners],
        )
        area, _ = GEOD.polygon_area_perimeter(lon, lat)
        areas[row] = abs(area)
    return areas


def weighted_mask_area_m2(mask: np.ndarray, row_areas_m2: np.ndarray) -> float:
    return float(np.dot(mask.astype(bool).sum(axis=1, dtype=np.int64), row_areas_m2))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def morph_close_open(bin_u8: np.ndarray, close_r: int, open_r: int) -> np.ndarray:
    m = bin_u8.astype(np.uint8)
    if close_r and close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r * 2 + 1, close_r * 2 + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if open_r and open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_r * 2 + 1, open_r * 2 + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def remove_small_components(bin_u8: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return bin_u8.astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bin_u8.astype(np.uint8), connectivity=8)
    out = np.zeros_like(bin_u8, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            out[lab == i] = 1
    return out


def dist_to_water_m(water_mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    water = (water_mask > 0).astype(np.uint8)
    land = (1 - water).astype(np.uint8)
    dist_px = cv2.distanceTransform(land, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    return dist_px * float(pixel_size_m)


def coast_score_from_dist(dist_m: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    d = dist_m.astype(np.float32)
    score = np.exp(-d / max(1e-6, COAST_SIGMA_M)).astype(np.float32)
    score = np.where(d <= COAST_MAXDIST_M, score, 0.0).astype(np.float32)
    score = np.where(water_mask > 0, 0.0, score).astype(np.float32)
    return score


def make_geo_prior(ndvi: np.ndarray, ndbi: np.ndarray, coast_score: np.ndarray) -> np.ndarray:
    ndvi = np.clip(ndvi.astype(np.float32), -1.0, 1.0)
    ndbi = np.clip(ndbi.astype(np.float32), -1.0, 1.0)
    ndvi01 = (ndvi + 1.0) * 0.5
    ndbi01 = (ndbi + 1.0) * 0.5
    prior_logit = (
        W0_PRIOR
        + W_COAST * coast_score
        + W_NDBI * (ndbi01 - 0.5)
        + W_NDVI_INV * ((1.0 - ndvi01) - 0.5)
    )
    prior = sigmoid_np(prior_logit).astype(np.float32)
    return np.clip(prior, 0.02, 0.98)


def hysteresis_mask(
    prob: np.ndarray,
    land_mask: np.ndarray,
    t_high: float,
    t_low: float,
    fallback_q: float,
) -> np.ndarray:
    prob = prob.astype(np.float32)
    seeds = ((prob >= t_high) & land_mask).astype(np.uint8)
    if seeds.sum() == 0:
        vals = prob[land_mask]
        if vals.size > 1000:
            thr = float(np.quantile(vals, fallback_q))
            seeds = ((prob >= thr) & land_mask).astype(np.uint8)
    low = ((prob >= t_low) & land_mask).astype(np.uint8)
    if seeds.sum() == 0:
        return np.zeros_like(seeds, dtype=np.uint8)
    num, lab = cv2.connectedComponents(low, connectivity=8)
    keep = np.zeros((num,), dtype=np.uint8)
    for i in range(1, num):
        comp = lab == i
        if np.any(comp & (seeds > 0)):
            keep[i] = 1
    out = np.zeros_like(low, dtype=np.uint8)
    for i in range(1, num):
        if keep[i]:
            out[lab == i] = 1
    return out


def infer_water_prob(mndwi: np.ndarray, vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    m = mndwi.astype(np.float32)
    thr_m = float(np.nanquantile(m, WATER_MNDWI_QUANTILE))
    if not np.isfinite(thr_m):
        thr_m = 0.0
    vv = vv.astype(np.float32)
    vh = vh.astype(np.float32)
    inten = np.sqrt(vv * vv + vh * vh + 1e-6)
    thr_i = float(np.nanpercentile(inten, 20.0))
    if np.isnan(thr_i):
        thr_i = 0.1
    w_m = sigmoid_np((m - thr_m) * WATER_SLOPE_MNDWI)
    w_s = sigmoid_np((thr_i - inten) * WATER_SLOPE_SAR)
    water_prob = (w_m * w_s).astype(np.float32)
    return np.nan_to_num(np.clip(water_prob, 0.0, 1.0), nan=0.0)


def confidence_from_denominator(den: np.ndarray) -> np.ndarray:
    den = den.astype(np.float32)
    conf = (den - float(INDEX_DEN_TRUST_MIN)) / (float(INDEX_DEN_TRUST_MAX) - float(INDEX_DEN_TRUST_MIN) + 1e-6)
    conf = np.clip(conf, 0.0, 1.0)
    if INDEX_CONF_POW != 1.0:
        conf = np.power(conf, float(INDEX_CONF_POW))
    return conf.astype(np.float32)


# ============================================================
# 2) Model (Infer) - MultiTask compatible
# ============================================================

class StripPooling(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        _, _, h, w = x.shape
        xh = self.conv_h(self.pool_h(x))
        xh = F.interpolate(xh, size=(h, w), mode="bilinear", align_corners=False)
        xw = self.conv_w(self.pool_w(x))
        xw = F.interpolate(xw, size=(h, w), mode="bilinear", align_corners=False)
        return x + xh + xw


class PAIM(nn.Module):
    def __init__(self, embed_dim: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.vv_idx = 6
        self.vh_idx = 7
        self.mndwi_idx = int(cfg.get("mndwi_idx", 10))
        self.micro = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.meso = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.macro = StripPooling(1) if bool(cfg.get("paim_use_strip_pool", True)) else nn.Sequential(
            nn.AvgPool2d(kernel_size=31, stride=1, padding=15),
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim + 3, embed_dim, 1),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )
        self.gamma = nn.Parameter(torch.tensor([float(cfg.get("paim_gamma_init", 0.0))], dtype=torch.float32))

    @staticmethod
    def _local_variance(img: torch.Tensor, k: int = 5):
        pad = k // 2
        mu = F.avg_pool2d(img, k, stride=1, padding=pad)
        var = F.avg_pool2d(img * img, k, stride=1, padding=pad) - mu * mu
        return F.relu(var)

    def forward(self, visual_feat: torch.Tensor, x_phys: torch.Tensor):
        with torch.no_grad():
            mndwi = x_phys[:, self.mndwi_idx : self.mndwi_idx + 1]
            if bool(self.cfg.get("paim_use_dynamic_thresh", True)):
                q = float(self.cfg.get("paim_dyn_q", 0.85))
                b = mndwi.shape[0]
                thr = torch.quantile(mndwi.view(b, -1), q=q, dim=1).view(b, 1, 1, 1)
            else:
                thr = torch.tensor(float(self.cfg.get("water_thresh", -0.05)), device=mndwi.device, dtype=mndwi.dtype).view(1, 1, 1, 1)
            slope = float(self.cfg.get("paim_water_slope", 10.0))
            water_prob = torch.sigmoid((mndwi - thr) * slope)
            vv = x_phys[:, self.vv_idx : self.vv_idx + 1]
            vh = x_phys[:, self.vh_idx : self.vh_idx + 1]
            sar_tex = self._local_variance(torch.sqrt(vv * vv + vh * vh + 1e-6), k=5)
        ts = visual_feat.shape[-2:]
        a_micro = self.micro(F.interpolate(sar_tex, size=ts, mode="bilinear", align_corners=False))
        w_small = F.interpolate(water_prob, size=ts, mode="bilinear", align_corners=False)
        a_meso = self.meso(w_small)
        a_macro = self.macro(w_small)
        fused = torch.cat([visual_feat, a_micro, a_meso, a_macro], dim=1)
        return visual_feat + self.gamma * self.fusion(fused)


class PhysicsAwareUnetMultiTaskInfer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        in_channels = int(cfg.get("input_channels", 12))
        self.unet = smp.Unet(
            encoder_name=cfg.get("encoder_name", "resnet34"),
            encoder_weights=None,
            in_channels=in_channels,
            classes=int(cfg.get("num_binary_classes", 2)),
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )
        pixel_mean = cfg.get("pixel_mean", None)
        pixel_std = cfg.get("pixel_std", None)
        assert pixel_mean is not None and pixel_std is not None, "ckpt cfg must contain pixel_mean/pixel_std"
        self.register_buffer("x_mean", torch.tensor(pixel_mean, dtype=torch.float32).view(1, in_channels, 1, 1), persistent=False)
        self.register_buffer("x_std", torch.tensor(pixel_std, dtype=torch.float32).view(1, in_channels, 1, 1), persistent=False)
        self.use_paim = bool(cfg.get("use_paim", True))
        if self.use_paim:
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, 64, 64)
                embed_dim = self.unet.encoder(dummy)[-1].shape[1]
            self.paim = PAIM(embed_dim=embed_dim, cfg=cfg)
        self.use_aux = bool(cfg.get("use_aux", False))
        if self.use_aux:
            self.aux_head = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, int(cfg.get("num_aux_classes", 9)), 1),
            )
        self.use_edge = bool(cfg.get("use_edge", False))
        if self.use_edge:
            self.edge_head = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.unet.encoder(x)
        if self.use_paim:
            x_phys = x * (self.x_std + 1e-6) + self.x_mean
            feats[-1] = self.paim(feats[-1], x_phys)
        dec = self.unet.decoder(*feats)
        return self.unet.segmentation_head(dec)


@torch.no_grad()
def tta_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits0 = model(x)
    logits1 = torch.flip(model(torch.flip(x, dims=[3])), dims=[3])
    logits2 = torch.flip(model(torch.flip(x, dims=[2])), dims=[2])
    logits3 = torch.flip(model(torch.flip(x, dims=[2, 3])), dims=[2, 3])
    return (logits0 + logits1 + logits2 + logits3) / 4.0


def load_model_from_ckpt(ckpt_path: str, device: str, use_ema: bool = True) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {}) or {}
    model = PhysicsAwareUnetMultiTaskInfer(cfg)
    if use_ema and ckpt.get("ema") is not None:
        state = ckpt["ema"]
        print("Loaded EMA weights.")
    else:
        state = ckpt["model"]
        print("Loaded model weights.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    mean = np.array(cfg["pixel_mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(cfg["pixel_std"], dtype=np.float32).reshape(-1, 1, 1)
    return model, mean, std


# ============================================================
# 3) Sliding-window inference (u16 decode on-the-fly)
# ============================================================

@torch.no_grad()
def run_batch(
    model: torch.nn.Module,
    batch_x: List[np.ndarray],
    device: str,
    use_tta: bool,
) -> np.ndarray:
    x = np.stack(batch_x, axis=0)
    xt = torch.from_numpy(x).float().to(device)
    if device.startswith("cuda"):
        with torch.cuda.amp.autocast(enabled=True):
            logits = tta_logits(model, xt) if use_tta else model(xt)
    else:
        logits = tta_logits(model, xt) if use_tta else model(xt)
    prob = torch.softmax(logits, dim=1)[:, 1]
    return prob.detach().cpu().numpy().astype(np.float32)


def accumulate(
    acc: np.ndarray,
    wsum: np.ndarray,
    prob_batch: np.ndarray,
    metas: List[Tuple[int, int, int, int]],
    win_full: np.ndarray,
) -> None:
    for i, (y0, x0, h, w) in enumerate(metas):
        p = prob_batch[i][:h, :w]
        win = win_full[:h, :w]
        acc[y0 : y0 + h, x0 : x0 + w] += p * win
        wsum[y0 : y0 + h, x0 : x0 + w] += win


@torch.no_grad()
def infer_prob_year_u16(
    ds_path: str,
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    cfg: RunConfig,
) -> Tuple[np.ndarray, dict]:
    with rasterio.open(ds_path) as ds:
        h_full, w_full = ds.height, ds.width
        profile = ds.profile.copy()

        acc = np.zeros((h_full, w_full), dtype=np.float32)
        wsum = np.zeros((h_full, w_full), dtype=np.float32)
        win_full = hann2d(cfg.patch, cfg.patch)
        bands = list(range(1, N_FEATURES + 1))

        ys = list(range(0, h_full, cfg.stride))
        xs = list(range(0, w_full, cfg.stride))
        tiles: List[Tuple[int, int, int, int]] = []
        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + cfg.patch, h_full)
                x1 = min(x0 + cfg.patch, w_full)
                tiles.append((y0, x0, y1 - y0, x1 - x0))

        batch_x: List[np.ndarray] = []
        batch_meta: List[Tuple[int, int, int, int]] = []
        desc = f"Infer tiles: {os.path.basename(ds_path)}"
        for (y0, x0, h, w) in tqdm(tiles, desc=desc, leave=False):
            win = Window(x0, y0, w, h)
            x_u16 = ds.read(bands, window=win, boundless=False)
            x = decode_feat12_u16(x_u16)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = pad_to_patch(x, cfg.patch)
            x = (x - mean) / (std + 1e-6)
            x = np.ascontiguousarray(x)
            batch_x.append(x)
            batch_meta.append((y0, x0, h, w))
            if len(batch_x) >= cfg.batch_tiles:
                prob_batch = run_batch(model, batch_x, device, cfg.use_tta)
                accumulate(acc, wsum, prob_batch, batch_meta, win_full)
                batch_x, batch_meta = [], []

        if batch_x:
            prob_batch = run_batch(model, batch_x, device, cfg.use_tta)
            accumulate(acc, wsum, prob_batch, batch_meta, win_full)

        prob = acc / (wsum + 1e-6)
        return np.clip(prob, 0.0, 1.0).astype(np.float32), profile


# ============================================================
# 4) Spatial refinement with robust index re-computation
# ============================================================

def refine_spatial_u16(
    ds_path: str,
    prob_raw: np.ndarray,
    profile: dict,
    out_dir: str,
    year: int,
) -> None:
    for sub in ["prob_raw", "prob_refined", "mask_spatial", "water_prob", "water_mask", "dist_to_water_m"]:
        ensure_dir(os.path.join(out_dir, sub))

    with rasterio.open(ds_path) as ds:
        # scheme #1: read source bands and recompute indices
        b3 = decode_single_band_u16(ds.read(2), 1)
        b4 = decode_single_band_u16(ds.read(3), 2)
        b8 = decode_single_band_u16(ds.read(4), 3)
        b11 = decode_single_band_u16(ds.read(5), 4)
        vv = decode_single_band_u16(ds.read(IDX_VV + 1), IDX_VV)
        vh = decode_single_band_u16(ds.read(IDX_VH + 1), IDX_VH)
        pixel_size_m, pixel_area_m2 = estimate_pixel_metrics_m(ds)
        row_areas_m2 = geodesic_row_areas_m2(ds)

    b3 = np.nan_to_num(b3, nan=0.0)
    b4 = np.nan_to_num(b4, nan=0.0)
    b8 = np.nan_to_num(b8, nan=0.0)
    b11 = np.nan_to_num(b11, nan=0.0)
    vv = np.nan_to_num(vv, nan=0.0)
    vh = np.nan_to_num(vh, nan=0.0)

    den_ndvi = b8 + b4
    den_mndwi = b3 + b11
    den_ndbi = b11 + b8
    ndvi_raw = np.clip((b8 - b4) / (den_ndvi + 1e-6), -1.0, 1.0)
    mndwi = np.clip((b3 - b11) / (den_mndwi + 1e-6), -1.0, 1.0)
    ndbi_raw = np.clip((b11 - b8) / (den_ndbi + 1e-6), -1.0, 1.0)

    conf_ndvi = confidence_from_denominator(den_ndvi)
    conf_ndbi = confidence_from_denominator(den_ndbi)
    ndvi = (ndvi_raw * conf_ndvi).astype(np.float32)
    ndbi = (ndbi_raw * conf_ndbi).astype(np.float32)

    water_prob = infer_water_prob(mndwi, vv, vh)
    water_mask = (water_prob >= 0.5).astype(np.uint8)
    dist_m = dist_to_water_m(water_mask, pixel_size_m=pixel_size_m)
    coast = coast_score_from_dist(dist_m, water_mask)
    prior = make_geo_prior(ndvi=ndvi, ndbi=ndbi, coast_score=coast)

    l_model = logit_np(prob_raw)
    l_prior = logit_np(prior)
    prob_ref = sigmoid_np(l_model + LAMBDA_PRIOR * l_prior).astype(np.float32)
    prob_ref = np.where(water_prob > WATER_HARD_FORBID, 0.0, prob_ref).astype(np.float32)

    land = water_prob < 0.5
    mask = hysteresis_mask(prob_ref, land_mask=land, t_high=SPATIAL_T_HIGH, t_low=SPATIAL_T_LOW, fallback_q=SPATIAL_FALLBACK_Q)
    mask = morph_close_open(mask, close_r=MORPH_CLOSE_R, open_r=MORPH_OPEN_R)
    min_area_px = int(np.ceil(MIN_COMP_AREA_M2 / max(1e-6, pixel_area_m2)))
    mask = remove_small_components(mask, min_area_px=min_area_px)

    write_single_band_tif(os.path.join(out_dir, "prob_raw", f"{year}.tif"), profile, prob_raw, "float32")
    write_single_band_tif(os.path.join(out_dir, "prob_refined", f"{year}.tif"), profile, prob_ref, "float32")
    write_single_band_tif(os.path.join(out_dir, "mask_spatial", f"{year}.tif"), profile, mask, "uint8")
    write_single_band_tif(os.path.join(out_dir, "water_prob", f"{year}.tif"), profile, water_prob, "float32")
    write_single_band_tif(os.path.join(out_dir, "water_mask", f"{year}.tif"), profile, water_mask, "uint8")
    write_single_band_tif(os.path.join(out_dir, "dist_to_water_m", f"{year}.tif"), profile, dist_m, "float32")

    info = {
        "year": int(year),
        "pixel_size_m": float(pixel_size_m),
        "projected_pixel_area_m2": float(pixel_area_m2),
        "geodesic_pixel_area_m2_min": float(row_areas_m2.min()),
        "geodesic_pixel_area_m2_max": float(row_areas_m2.max()),
        "water_frac": float((water_prob >= 0.5).mean()),
        "mask_px": int(mask.sum()),
        "mask_area_m2": weighted_mask_area_m2(mask, row_areas_m2),
        "lowconf_ndvi_frac": float((conf_ndvi < 0.1).mean()),
        "lowconf_ndbi_frac": float((conf_ndbi < 0.1).mean()),
    }
    with open(os.path.join(out_dir, "spatial_refine_diag.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")


# ============================================================
# 5) H-STGO (in-memory global optimization)
# ============================================================

class HSTGO_Optimizer:
    def __init__(self, shape_t_h_w):
        self.t, self.h, self.w = shape_t_h_w

    def _compute_contrast_weights(self, img_slice):
        gy, gx = np.gradient(img_slice)
        grad = np.sqrt(gy**2 + gx**2)
        grad = grad / (grad.max() + 1e-6)
        return 1.0 + 5.0 * np.exp(-grad / 0.1)

    def solve_layer_binary(
        self,
        data_cost_fg,
        data_cost_bg,
        guidance_img,
        fwd_penalty_map,
        bwd_penalty_map,
    ):
        n_nodes = self.t * self.h * self.w
        g = maxflow.Graph[float](n_nodes, n_nodes * 6)
        nodeids = g.add_grid_nodes((self.t, self.h, self.w))
        g.add_grid_tedges(nodeids, data_cost_fg, data_cost_bg)

        structure = np.array(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        )
        spatial_weights = np.zeros((self.t, self.h, self.w), dtype=np.float64)
        for tt in range(self.t):
            grad_weight = self._compute_contrast_weights(guidance_img[tt])
            # guidance_img is dist_to_water_m in current pipeline.
            geo_modulator = HSTGO_ANISO_BASE + HSTGO_ANISO_GAIN * np.exp(
                -guidance_img[tt] / max(1e-6, HSTGO_ANISO_DECAY_M)
            )
            spatial_weights[tt] = grad_weight * geo_modulator
        g.add_grid_edges(nodeids, weights=spatial_weights, structure=structure, symmetric=True)

        struct_fwd = np.zeros((3, 1, 1))
        struct_fwd[2, 0, 0] = 1
        g.add_grid_edges(nodeids, weights=fwd_penalty_map, structure=struct_fwd, symmetric=False)

        struct_bwd = np.zeros((3, 1, 1))
        struct_bwd[0, 0, 0] = 1
        w_bwd_shifted = np.roll(bwd_penalty_map, 1, axis=0)
        w_bwd_shifted[0] = 0
        g.add_grid_edges(nodeids, weights=w_bwd_shifted, structure=struct_bwd, symmetric=False)

        g.maxflow()
        return g.get_grid_segments(nodeids)


def _weighted_temporal_smooth(stack_t_h_w: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted moving average along time axis.
    Time length is small (typically <=10), so an explicit loop is fast and stable.
    """
    t = stack_t_h_w.shape[0]
    if t <= 1:
        return stack_t_h_w.astype(np.float32, copy=False)
    w = weights.astype(np.float32)
    half = len(w) // 2
    out = np.zeros_like(stack_t_h_w, dtype=np.float32)
    for ti in range(t):
        a = max(0, ti - half)
        b = min(t, ti + half + 1)
        wa = half - (ti - a)
        wb = wa + (b - a)
        ww = w[wa:wb]
        s = float(ww.sum())
        if s <= 1e-8:
            out[ti] = stack_t_h_w[ti]
        else:
            out[ti] = (stack_t_h_w[a:b] * ww[:, None, None]).sum(axis=0) / s
    return np.clip(out, 0.0, 1.0)


def _persistent_decline_score(prob_t_h_w: np.ndarray, k: int) -> np.ndarray:
    """
    Decline evidence in [0,1], requiring persistence across K years.
    Larger score means stronger evidence that shrinkage is real (not one-year noise).
    """
    prev = np.concatenate([prob_t_h_w[:1], prob_t_h_w[:-1]], axis=0)
    decline = np.clip(prev - prob_t_h_w, 0.0, 1.0)
    if k <= 1 or decline.shape[0] <= 1:
        return decline.astype(np.float32)
    out = decline.copy()
    for off in range(1, k):
        sh = np.roll(decline, -off, axis=0)
        sh[-off:] = decline[-1]
        out = np.minimum(out, sh)
    return out.astype(np.float32)


def read_all_years_to_ram(paths: List[str], ref_profile: dict) -> np.ndarray:
    h = ref_profile["height"]
    w = ref_profile["width"]
    t = len(paths)
    stack = np.zeros((t, h, w), dtype=np.float32)
    folder_name = os.path.basename(os.path.dirname(paths[0])) if paths else "unknown"
    print(f"   🔍 Diagnostics for {folder_name}:")
    for i, p in enumerate(paths):
        with rasterio.open(p) as src:
            data = src.read(1).astype(np.float32, copy=False)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            stack[i] = data
            if i == 0 or i == (t - 1):
                print(
                    f"      Year {i}: Min={data.min():.4f}, Max={data.max():.4f}, "
                    f"Mean={data.mean():.4f}, NonZero={int(np.count_nonzero(data))}"
                )
    if float(stack.max()) == 0.0:
        print("      ⚠️ WARNING: This data stack is ALL ZEROS! Graph Cut will output 0.")
    return stack


def process_single_block_task(args):
    (
        r_idx,
        c_idx,
        h_padded,
        w_padded,
        t_years,
        p_water_block,
        p_port_block,
        dist_block,
    ) = args

    p_water_block = p_water_block.astype(np.float32, copy=False)
    p_port_block = p_port_block.astype(np.float32, copy=False)
    dist_block = dist_block.astype(np.float32, copy=False)
    opt = HSTGO_Optimizer((t_years, h_padded, w_padded))

    cost_l1_bg = -np.log(np.clip(p_water_block, 1e-6, 1.0))
    cost_l1_fg = -np.log(np.clip(1.0 - p_water_block, 1e-6, 1.0))
    w_fwd_l1 = np.full_like(p_water_block, HSTGO_WATER_TO_NON_WATER_COST)
    w_bwd_l1 = HSTGO_NON_WATER_TO_WATER_BASE * (1.0 - p_water_block)
    mask_non_water = opt.solve_layer_binary(
        cost_l1_fg,
        cost_l1_bg,
        guidance_img=dist_block,
        fwd_penalty_map=w_fwd_l1,
        bwd_penalty_map=w_bwd_l1,
    )

    # 5-year weighted smoothing (more stable than 3-year while still interpretable).
    tw = np.array(HSTGO_TEMPORAL_WEIGHTS, dtype=np.float32)
    p_port_smoothed = _weighted_temporal_smooth(np.clip(p_port_block, 0.0, 1.0), tw)

    # Retreat is allowed only when decline signal persists for K years.
    retreat_evidence = _persistent_decline_score(p_port_smoothed, k=HSTGO_RETREAT_K)

    cost_l2_bg = -np.log(np.clip(1.0 - p_port_smoothed, 1e-6, 1.0))
    cost_l2_fg = -np.log(np.clip(p_port_smoothed, 1e-6, 1.0))

    # Expansion confidence uses a smooth sigmoid (avoids aggressive exponential trigger).
    expand_conf = 1.0 / (1.0 + np.exp(-(p_port_smoothed - HSTGO_EXPAND_MID) * HSTGO_EXPAND_SLOPE))
    dist_bonus = np.exp(-dist_block / max(1e-6, HSTGO_LC_DIST_DECAY_M))

    # Forward penalty (anti-shrink) is relaxed when decline evidence is persistent.
    w_fwd_base = HSTGO_LC_FWD_BASE + HSTGO_LC_FWD_GAIN * p_port_smoothed
    w_fwd_l2 = w_fwd_base * (1.0 - HSTGO_RETREAT_RELAX * retreat_evidence)
    w_fwd_l2 = np.clip(w_fwd_l2, HSTGO_FWD_MIN, None)

    # Backward penalty (anti-expansion) stays conservative; near-coast bonus is moderated.
    coast_bonus = HSTGO_LC_BWD_BONUS * (dist_bonus * expand_conf)
    w_bwd_l2 = HSTGO_LC_BWD_BASE + HSTGO_LC_BWD_GAIN * (1.0 - expand_conf) - coast_bonus
    w_bwd_l2 = np.clip(w_bwd_l2, HSTGO_LC_BWD_MIN, None)
    mask_port = opt.solve_layer_binary(
        cost_l2_fg,
        cost_l2_bg,
        guidance_img=dist_block,
        fwd_penalty_map=w_fwd_l2,
        bwd_penalty_map=w_bwd_l2,
    )

    final_mask = np.logical_and(mask_non_water, mask_port).astype(np.uint8)
    return r_idx, c_idx, final_mask


def run_hstgo_pipeline(port_out_dir: str, years: List[int], ref_profile: dict, cfg: RunConfig):
    print("\n🧠 Running H-STGO (Hybrid: In-Memory + Multi-Core Parallel)...")
    print(f"   Using {cfg.hstgo_num_workers} CPU workers.")
    prob_paths = [os.path.join(port_out_dir, "prob_refined", f"{y}.tif") for y in years]
    water_paths = [os.path.join(port_out_dir, "water_prob", f"{y}.tif") for y in years]
    dist_paths = [os.path.join(port_out_dir, "dist_to_water_m", f"{y}.tif") for y in years]

    print("   ⏳ Loading data into RAM to kill IO latency...")
    p_port_stack = read_all_years_to_ram(prob_paths, ref_profile)
    p_water_stack = read_all_years_to_ram(water_paths, ref_profile)
    dist_stack = read_all_years_to_ram(dist_paths, ref_profile)
    t_years, h, w = p_port_stack.shape
    est_stack_gb = (3.0 * t_years * h * w * 4.0) / (1024.0 ** 3)
    workers = cfg.hstgo_num_workers
    block_size = cfg.hstgo_block_size
    padding = cfg.hstgo_padding
    if est_stack_gb >= HSTGO_AUTO_DOWNSCALE_GB:
        workers = min(workers, HSTGO_AUTO_MAX_WORKERS_LARGE)
        block_size = max(HSTGO_MIN_BLOCK_SIZE, block_size // 2)
        padding = max(HSTGO_MIN_PADDING, padding // 2)
        print(
            f"   ⚙️ Large stack (~{est_stack_gb:.2f} GB): auto-tuning workers={workers}, "
            f"block={block_size}, padding={padding}"
        )
    print(f"   ✅ Data Loaded: ({t_years}, {h}, {w}). Splitting tasks...")

    tasks = []
    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            r_start = max(0, r - padding)
            r_end = min(h, r + block_size + padding)
            c_start = max(0, c - padding)
            c_end = min(w, c + block_size + padding)
            tasks.append(
                (
                    r,
                    c,
                    r_end - r_start,
                    c_end - c_start,
                    t_years,
                    p_water_stack[:, r_start:r_end, c_start:c_end],
                    p_port_stack[:, r_start:r_end, c_start:c_end],
                    dist_stack[:, r_start:r_end, c_start:c_end],
                )
            )

    final_result = np.zeros((t_years, h, w), dtype=np.uint8)
    print(f"   🚀 Starting {len(tasks)} parallel blocks...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_block_task, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="   Parallel Solving"):
            r_origin, c_origin, mask_padded = future.result()
            valid_r_start = 0 if r_origin == 0 else padding
            valid_c_start = 0 if c_origin == 0 else padding
            valid_h = min(block_size, h - r_origin)
            valid_w = min(block_size, w - c_origin)
            mask_valid = mask_padded[
                :,
                valid_r_start : valid_r_start + valid_h,
                valid_c_start : valid_c_start + valid_w,
            ]
            final_result[
                :,
                r_origin : r_origin + valid_h,
                c_origin : c_origin + valid_w,
            ] = mask_valid

    out_dir = os.path.join(port_out_dir, "mask_hstgo")
    ensure_dir(out_dir)
    prof_uint8 = ref_profile.copy()
    if "nodata" in prof_uint8:
        prof_uint8.pop("nodata")
    prof_uint8.update(dtype="uint8", count=1, compress="deflate")

    print("   💾 Saving results...")
    for i, y in enumerate(years):
        dst_path = os.path.join(out_dir, f"{y}.tif")
        with rasterio.open(dst_path, "w", **prof_uint8) as dst:
            dst.write(final_result[i], 1)
    print(f"✅ H-STGO Finished. Check {out_dir}")

    del p_port_stack, p_water_stack, dist_stack, final_result, tasks
    gc.collect()


def final_cleanup_hstgo_masks(port_out: str, years: List[int]):
    sample_year = years[len(years) // 2]
    with rasterio.open(os.path.join(port_out, "mask_hstgo", f"{sample_year}.tif")) as ds:
        _, pixel_area_m2 = estimate_pixel_metrics_m(ds)
    min_area_px = int(math.ceil(MIN_COMP_AREA_M2 / max(1e-6, pixel_area_m2)))
    for y in tqdm(years, desc="Cleanup"):
        mpath = os.path.join(port_out, "mask_hstgo", f"{y}.tif")
        with rasterio.open(mpath) as ds:
            mask = ds.read(1).astype(np.uint8)
            prof = ds.profile.copy()
        mask = remove_small_components(mask, min_area_px=min_area_px)
        write_single_band_tif(mpath, prof, mask, "uint8")


def summarize_area_hstgo(port_out: str, years: List[int]) -> Dict:
    base_year = int(years[0])
    baseline_years_for_water = list(years[:2])
    base_path = os.path.join(port_out, "mask_hstgo", f"{base_year}.tif")
    with rasterio.open(base_path) as ds:
        h, w = ds.height, ds.width
        row_areas_m2 = geodesic_row_areas_m2(ds)
        base_mask = ds.read(1).astype(bool)

    max_water_prob = np.zeros((h, w), dtype=np.float32)
    for by in baseline_years_for_water:
        water_path = os.path.join(port_out, "water_prob", f"{by}.tif")
        if os.path.exists(water_path):
            with rasterio.open(water_path) as ds:
                wp = ds.read(1).astype(np.float32)
                max_water_prob = np.maximum(max_water_prob, wp)

    sea_domain = (max_water_prob >= SEA_DOMAIN_WATER_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SEA_DOMAIN_DILATE_K, SEA_DOMAIN_DILATE_K))
    sea_domain = cv2.dilate(sea_domain, kernel, iterations=1).astype(bool)
    sea_domain &= ~base_mask
    base_area_m2 = weighted_mask_area_m2(base_mask, row_areas_m2)

    stats = {
        "years": [],
        "area_m2_hstgo": [],
        "seaward_net_change_m2_from_base": [],
        "landward_net_change_m2_from_base": [],
        "total_net_change_m2_from_base": [],
        "annual_seaward_net_change_m2": [],
        "annual_landward_net_change_m2": [],
        "annual_total_net_change_m2": [],
        "base_year": base_year,
        "area_method": "WGS84 geodesic pixel area by raster row",
        "sea_domain_water_threshold": float(SEA_DOMAIN_WATER_THRESHOLD),
        "sea_domain_dilate_k": int(SEA_DOMAIN_DILATE_K),
    }

    previous_mask = base_mask
    for y in years:
        with rasterio.open(os.path.join(port_out, "mask_hstgo", f"{y}.tif")) as ds:
            port_y = ds.read(1).astype(bool)

        total_area = weighted_mask_area_m2(port_y, row_areas_m2)
        gain_mask = port_y & ~base_mask
        loss_mask = ~port_y & base_mask
        seaward_net = (
            weighted_mask_area_m2(gain_mask & sea_domain, row_areas_m2)
            - weighted_mask_area_m2(loss_mask & sea_domain, row_areas_m2)
        )
        landward_net = (
            weighted_mask_area_m2(gain_mask & ~sea_domain, row_areas_m2)
            - weighted_mask_area_m2(loss_mask & ~sea_domain, row_areas_m2)
        )

        annual_gain = port_y & ~previous_mask
        annual_loss = ~port_y & previous_mask
        annual_seaward_net = (
            weighted_mask_area_m2(annual_gain & sea_domain, row_areas_m2)
            - weighted_mask_area_m2(annual_loss & sea_domain, row_areas_m2)
        )
        annual_landward_net = (
            weighted_mask_area_m2(annual_gain & ~sea_domain, row_areas_m2)
            - weighted_mask_area_m2(annual_loss & ~sea_domain, row_areas_m2)
        )
        if not math.isclose(seaward_net + landward_net, total_area - base_area_m2, abs_tol=1e-3):
            raise RuntimeError(f"Directional area identity failed for year {y}")
        previous_area = weighted_mask_area_m2(previous_mask, row_areas_m2)
        if not math.isclose(annual_seaward_net + annual_landward_net, total_area - previous_area, abs_tol=1e-3):
            raise RuntimeError(f"Annual area identity failed for year {y}")

        stats["years"].append(int(y))
        stats["area_m2_hstgo"].append(total_area)
        stats["seaward_net_change_m2_from_base"].append(seaward_net)
        stats["landward_net_change_m2_from_base"].append(landward_net)
        stats["total_net_change_m2_from_base"].append(seaward_net + landward_net)
        stats["annual_seaward_net_change_m2"].append(annual_seaward_net)
        stats["annual_landward_net_change_m2"].append(annual_landward_net)
        stats["annual_total_net_change_m2"].append(annual_seaward_net + annual_landward_net)
        previous_mask = port_y

    return stats


# ============================================================
# 6) Cluster orchestration + main
# ============================================================

def resolve_years_for_cluster(year_map: Dict[int, str], cfg: RunConfig) -> List[int]:
    years = [y for y in range(cfg.year_min, cfg.year_max + 1) if y in year_map]
    missing = sorted(list(set(range(cfg.year_min, cfg.year_max + 1)) - set(years)))
    if missing and cfg.strict_years:
        raise RuntimeError(f"Missing years: {missing}")
    return years


def _validate_graphcut_inputs(out_dir: str, years: List[int]) -> None:
    missing: List[str] = []
    need_subdirs = ("prob_refined", "water_prob", "dist_to_water_m")
    for y in years:
        for sub in need_subdirs:
            p = os.path.join(out_dir, sub, f"{y}.tif")
            if not os.path.exists(p):
                missing.append(p)
                if len(missing) >= 6:
                    break
        if len(missing) >= 6:
            break
    if missing:
        head = "\n  - ".join(missing)
        raise FileNotFoundError(
            "graphcut-only mode requires existing intermediates, missing e.g.:\n  - "
            + head
        )


def process_one_cluster(
    cluster_id: int,
    cluster_uid: int,
    year_map: Dict[int, str],
    model: Optional[torch.nn.Module],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    device: str,
    cfg: RunConfig,
    resolved_ports: str,
) -> Dict[str, object]:
    years = resolve_years_for_cluster(year_map, cfg)
    if not years:
        raise RuntimeError(f"cluster {cluster_id} has no years in requested range")
    out_name = f"cluster_{cluster_uid:03d}"
    out_dir = os.path.join(cfg.out_root, out_name)
    ensure_dir(out_dir)
    for sub in ["prob_raw", "prob_refined", "mask_spatial", "mask_hstgo", "water_prob", "water_mask", "dist_to_water_m"]:
        ensure_dir(os.path.join(out_dir, sub))

    ref_year = cfg.ref_year if cfg.ref_year in year_map else years[len(years) // 2]
    with rasterio.open(year_map[ref_year]) as dsref:
        ref_profile = dsref.profile.copy()

    if cfg.graphcut_only:
        print(f"\n[cluster={cluster_id} uid={cluster_uid:03d}] graphcut-only mode: reuse existing prob_refined/water/dist.")
        _validate_graphcut_inputs(out_dir, years)
    else:
        if model is None or mean is None or std is None:
            raise RuntimeError("model/mean/std must be available when graphcut-only is disabled.")
        for y in years:
            src = year_map[y]
            print(f"\n[cluster={cluster_id} uid={cluster_uid:03d} year={y}] inference + spatial refine")
            prob_raw, profile = infer_prob_year_u16(src, model, mean, std, device, cfg)
            refine_spatial_u16(src, prob_raw, profile, out_dir, y)

    run_hstgo_pipeline(out_dir, years, ref_profile, cfg)
    final_cleanup_hstgo_masks(out_dir, years)

    summary = summarize_area_hstgo(out_dir, years)
    summary["cluster_id"] = int(cluster_id)
    summary["cluster_uid"] = int(cluster_uid)
    summary["resolved_portids"] = resolved_ports
    summary["model"] = "H-STGO (Hybrid In-Memory + Multi-Core Block Parallel) + on-the-fly u16 decode + robust index recompute"
    with open(os.path.join(out_dir, "summary_hstgo.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    cfg = parse_args()
    check_runtime_deps()
    ensure_dir(cfg.out_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | data_root={cfg.data_root} | out_root={cfg.out_root}")
    print(
        f"whitelist={cfg.clusters} | blacklist={cfg.exclude_clusters} | "
        f"resume={cfg.resume} | graphcut_only={cfg.graphcut_only}"
    )

    model: Optional[torch.nn.Module] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    if not cfg.graphcut_only:
        model, mean, std = load_model_from_ckpt(cfg.ckpt_path, device=device, use_ema=cfg.use_ema)
    grouped = scan_cluster_year_files(cfg)
    port_map = load_cluster_port_map(cfg.mapping_csv)
    cluster_ids = sorted(grouped.keys())
    print(f"matched clusters={len(cluster_ids)} -> {cluster_ids}")
    out_csv = os.path.join(cfg.out_root, "cluster_year_area_hstgo_GLOBAL.csv")
    if (not cfg.resume) and os.path.exists(out_csv):
        os.remove(out_csv)
        print(f"overwrite mode: removed existing {out_csv}")

    for cid in cluster_ids:
        entry = grouped[cid]
        uid = int(entry["uid"])
        year_map = entry["years"]  # type: ignore[assignment]
        ports = port_map.get(cid, "")

        out_name = f"cluster_{uid:03d}"
        cluster_out_dir = os.path.join(cfg.out_root, out_name)
        summary_file = os.path.join(cluster_out_dir, "summary_hstgo.json")
        if cfg.resume and os.path.exists(summary_file):
            print(f"\n[SKIP] cluster={cid} uid={uid:03d} already done -> {summary_file}")
            continue

        try:
            summary = process_one_cluster(
                cluster_id=cid,
                cluster_uid=uid,
                year_map=year_map,  # type: ignore[arg-type]
                model=model,
                mean=mean,
                std=std,
                device=device,
                cfg=cfg,
                resolved_ports=ports,
            )
        except Exception as e:
            print(f"[ERROR] cluster={cid} failed: {e}")
            continue

        cluster_rows: List[Dict[str, object]] = []
        years = summary.get("years", [])
        areas = summary.get("area_m2_hstgo", [])
        seawards = summary.get("seaward_net_change_m2_from_base", [])
        landwards = summary.get("landward_net_change_m2_from_base", [])
        totals = summary.get("total_net_change_m2_from_base", [])
        annual_seawards = summary.get("annual_seaward_net_change_m2", [])
        annual_landwards = summary.get("annual_landward_net_change_m2", [])
        annual_totals = summary.get("annual_total_net_change_m2", [])

        for i in range(len(years)):
            cluster_rows.append(
                {
                    "cluster_id": cid,
                    "cluster_uid": uid,
                    "year": int(years[i]),
                    "base_year": int(summary.get("base_year", years[0])),
                    "area_method": str(summary.get("area_method", "")),
                    "area_m2_hstgo": float(areas[i]),
                    "seaward_net_change_m2_from_base": float(seawards[i]),
                    "landward_net_change_m2_from_base": float(landwards[i]),
                    "total_net_change_m2_from_base": float(totals[i]),
                    "annual_seaward_net_change_m2": float(annual_seawards[i]),
                    "annual_landward_net_change_m2": float(annual_landwards[i]),
                    "annual_total_net_change_m2": float(annual_totals[i]),
                    "resolved_portids": ports,
                }
            )

        if cluster_rows:
            df_new = pd.DataFrame(cluster_rows)
            if not os.path.exists(out_csv):
                df_new.to_csv(out_csv, index=False, encoding="utf-8-sig")
            else:
                df_new.to_csv(out_csv, mode="a", header=False, index=False, encoding="utf-8-sig")
            print(f"appended cluster={cid} rows={len(cluster_rows)} -> {out_csv}")

    print("pipeline completed")


if __name__ == "__main__":
    main()

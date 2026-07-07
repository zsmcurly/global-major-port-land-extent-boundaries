# -*- coding: utf-8 -*-
"""Paper-aligned trainer for the port-aware segmentation model in English V2."""

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import segmentation_models_pytorch as smp

#配置
#波段名
RAW_BAND_NAMES = (
    "B2", "B3", "B4", "B8", "B11", "B12", "VV", "VH", "VV_VH", "NDVI", "MNDWI", "NDBI"
)
#光学波段（蓝，绿，红，近红外，短波红外）
OPTICAL_RAW_IDX = (0, 1, 2, 3, 4, 5)
#微波波段
SAR_RAW_IDX = (6, 7, 8)
DEFAULT_CHANNEL_INDICES = tuple(range(12))
#默认波段切片。
#dataclass装饰器，管理超参数，不用导入，直接赋值
@dataclass
class CFG:
    data_root: str = "dataset_final_npy_png"#训练数据文件夹
    save_dir: str = "runs/expv1"#结果保存文件夹
    num_workers: int = 4#加载处理数据的子进程数量。

    #backbone
    #resnet34，去掉分类输出头，imagenet预训练，左侧的编码层
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"

    # data
    #原始数据的波段
    raw_input_channels: int = 12        # source npy channels (fixed export layout)
    #输入的波段切片，用于操作消融实验
    channel_indices: Tuple[int, ...] = DEFAULT_CHANNEL_INDICES  # raw-index subset for current experiment
    input_channels: int = 12            # effective model input channels after channel selection
    #模型输入维度。
    #图片后缀，12波段npy
    img_ext: str = ".npy"
    #真值的矢量png后缀
    mask_ext: str = ".png"
    #大小512
    crop_size: int = 512               # 512: full patch; <512: random crop
    #0.85偏好要前景大于0.02
    prefer_port_crop: float = 0.85     # 提高有效港口暴露
    train_fg_min_frac: float = 0.05    # paper sample criterion: at least 5% port foreground
    #寻找的耐心值，最多12次。
    tries_per_crop: int = 12

    # classes
    #基础两类，打开多分类辅助为9类
    num_binary_classes: int = 2
    num_aux_classes: int = 9
    use_aux: bool = True               # full model uses background + eight port-related classes
    #权重0.15，第5个epoch开启
    aux_weight: float = 0.15
    aux_warmup_epochs: int = 5

    # training
    #种子定死
    seed: int = 42
    #batchsize=10
    batch_size: int = 10
    epochs: int = 220
    lr: float = 8e-5
    #损失函数，l2正则化
    weight_decay: float = 1e-2
    #梯度裁剪，防止梯度爆炸
    grad_clip: float = 1.0
    accum_steps: int = 1
    #自动混合精度，省计算
    amp: bool = True
    #ema的权重，计算新模型？
    ema_decay: float = 0.999

    #学习率设定
    #前5epoch慢慢跳到设定
    warmup_epochs: int = 5
    #学习率最多降到0.05
    min_lr_ratio: float = 0.05

    # loss weights
    # 你的数据前景占比约 12.6%，bg_weight=0.08 过于偏置前景，容易 FP 多、IoU 卡住
    #ce分类loss，前景4倍权重
    ce_bg_weight: float = 0.25
    ce_fg_weight: float = 1.00
    #dice是一个pred与iou重叠的loss
    dice_weight: float = 1.0

    # Lovasz (IoU-friendly) —— 延后启动更稳
    #一个黑盒的loss
    lovasz_weight: float = 0.30
    lovasz_start_epoch: int = 20

    # boundary relaxation (保持，但可视情况关)
    #算celoss时，收缩2个像素
    boundary_ignore: bool = True
    boundary_radius: int = 2

    #Edge head (boundary supervision)
    #边界监督头
    use_edge: bool = True
    edge_weight: float = 0.12
    edge_dilate: int = 1
    edge_pos_weight: float = 4.0
    edge_start_epoch: int = 10

    # Coastal band reweight (港口近水：只对“陆地且靠近水体”加权)
    use_coastal_weight: bool = True
    coastal_start_epoch: int = 10      # 前几轮不启用，先学语义再引入先验
    mndwi_idx: Optional[int] = 10       # local index after channel selection
    vv_idx: Optional[int] = 6           # local index after channel selection
    vh_idx: Optional[int] = 7           # local index after channel selection
    optical_local_indices: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    sar_local_indices: Tuple[int, ...] = (6, 7, 8)
    #用mndwi+vv和vh算水

    # Reuse PAIM's sample-adaptive q=0.85 MNDWI threshold for coastal weighting.
    water_thresh_mode: str = "quantile"
    water_thresh: float = -0.05        # fixed 模式备用阈值（基于你的统计：bg均值约 -0.02）
    water_otsu_clip_p1: float = 1.0    # Otsu 前对 MNDWI 做稳健裁剪
    water_otsu_clip_p99: float = 99.0
    water_morph_open: int = 2          # 形态学开运算核半径
    water_morph_close: int = 3         # 形态学闭运算核半径
    water_min_area: int = 128          # 去掉极小水体噪声（像素数）

    # coastal 权重图参数（作用在陆地近水带）
    coastal_alpha: float = 1.2         # 最大权重 1+alpha（别太大，避免沿海 FP），靠水边的预测错误loss增大
    coastal_sigma: float = 10.0         #离水越近权重改变。
    coastal_maxdist: float = 50.0
    coastal_land_only: bool = True     # 关键：水体像素权重固定 1.0

    # PAIM
    #分位数抽样
    use_paim: bool = True
    paim_use_strip_pool: bool = True
    paim_gamma_init: float = 0.0
    paim_water_slope: float = 10.0     # sigmoid 斜率
    paim_use_dynamic_thresh: bool = True
    paim_dyn_q: float = 0.85           # 动态阈值：取 MNDWI 的 85% 分位作为水体阈值（no_grad）

    #tta+ema手段提高精度
    tta: bool = True

    # diagnostics
    #开始训练前快速抽检
    run_diagnostics: bool = True
    diagnostics_max_samples: int = 0   # 0 = all
    diagnostics_save_vis: int = 40     # 每个 split 保存多少张可视化
    diagnostics_rgb_idx: Tuple[int, int, int] = (2, 1, 0)  # 可视化用的伪 RGB 通道

    # threshold sweep (校准阈值，防止“0.5 固定阈值”卡住 IoU)
    do_thresh_sweep: bool = True
    thresh_eval_interval: int = 10
    thresh_sweep_max_batches: int = 80  # 控制开销：最多扫多少个 batch
    thresh_grid: Tuple[float, float, float] = (0.25, 0.75, 0.025)  # start, end, step

    # normalization (直接写入你 audit 的 train 统计，避免 fallback 固化)
    pixel_mean: Optional[list] = (
        0.07706034928560257,
        0.09551528841257095,
        0.09704498201608658,
        0.17003397643566132,
        0.15285293757915497,
        0.1184576228260994,
        0.4406967759132385,
        0.2066088616847992,
        0.513742983341217,
        0.15791882574558258,
        -0.050888244062662125,
        -0.10067463666200638,
    )
    pixel_std: Optional[list] = (
        0.061246443539857864,
        0.0696338340640068,
        0.08587417006492615,
        0.12404727190732956,
        0.12103671580553055,
        0.10358399897813797,
        0.2041836529970169,
        0.1791996955871582,
        0.14971023797988892,
        0.3698381185531616,
        0.4747348427772522,
        0.2225547730922699,
    )
    experiment_name: str = "all_eta"

#下为工具函数
#用种子保证可复现性
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False#关闭快速卷积
    torch.backends.cudnn.deterministic = True#只使用确定性的卷积算法
    torch.use_deterministic_algorithms(True, warn_only=True)#静止不确定算子

#确保目录存在
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
#子进程种子设置，数据增强多样，且种子一样可浮现
def make_seed_worker():
    def _seed_worker(worker_id: int):
        ws = torch.initial_seed() % 2**32
        random.seed(ws)
        np.random.seed(ws)
    return _seed_worker
#解析命令行输入的波段切片字符串
def parse_channel_indices(text: str) -> Tuple[int, ...]:
    vals: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("empty channel list")
    if len(set(vals)) != len(vals):
        raise ValueError(f"duplicated channel index in {vals}")
    for v in vals:
        if v < 0 or v >= len(RAW_BAND_NAMES):
            raise ValueError(f"channel index out of range: {v} (allowed 0..{len(RAW_BAND_NAMES)-1})")
    return tuple(vals)
#统一将数组转成高度宽度通道模式，hwc，检查0，2是否是通道
def _to_hwc(x: np.ndarray, expected_c: int) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"expect 3D npy, got shape={x.shape}")
    #检查0，2是否是通道
    if x.shape[-1] == expected_c:
        return x
    if x.shape[0] == expected_c:
        return x.transpose(1, 2, 0)
    raise ValueError(f"cannot infer CHW/HWC for shape={x.shape} with expected_c={expected_c}")
#数据加载，标准化数据格式和波段切片
def adapt_loaded_image_channels(x: np.ndarray, cfg: CFG) -> np.ndarray:
    """
    Standardize to HWC and apply channel subset if input is raw 12-ch.
    Backward compatible with already-subset arrays.
    """
    x = _to_hwc(x, cfg.raw_input_channels) if (cfg.raw_input_channels in x.shape) else _to_hwc(x, cfg.input_channels)
    if x.shape[-1] == cfg.raw_input_channels:
        x = x[..., list(cfg.channel_indices)]
    elif x.shape[-1] != cfg.input_channels:
        raise ValueError(
            f"unexpected channel count after load: {x.shape[-1]} (expected raw={cfg.raw_input_channels} or selected={cfg.input_channels})"
        )
    return x
#用哪些波段，就把该波段的归一化参数切出来
def slice_stats_to_channels(mean: np.ndarray, std: np.ndarray, cfg: CFG) -> Tuple[np.ndarray, np.ndarray]:
    mean = mean.reshape(-1).astype(np.float32)
    std = std.reshape(-1).astype(np.float32)
    if mean.size == cfg.raw_input_channels and std.size == cfg.raw_input_channels:
        idx = np.array(cfg.channel_indices, dtype=np.int64)
        return mean[idx], std[idx]
    if mean.size == cfg.input_channels and std.size == cfg.input_channels:
        return mean, std
    raise ValueError(
        f"stats dim mismatch: mean={mean.size}, std={std.size}, raw={cfg.raw_input_channels}, input={cfg.input_channels}"
    )
#波段检索器
def apply_channel_layout(cfg: CFG, channel_indices: Tuple[int, ...]) -> None:
    cfg.channel_indices = tuple(channel_indices)
    cfg.input_channels = len(cfg.channel_indices)
    raw_to_local = {raw_i: i for i, raw_i in enumerate(cfg.channel_indices)}
    cfg.mndwi_idx = raw_to_local.get(10, None)
    cfg.vv_idx = raw_to_local.get(6, None)
    cfg.vh_idx = raw_to_local.get(7, None)
    cfg.optical_local_indices = tuple(raw_to_local[i] for i in OPTICAL_RAW_IDX if i in raw_to_local)
    cfg.sar_local_indices = tuple(raw_to_local[i] for i in SAR_RAW_IDX if i in raw_to_local)
    rgb_fallback = [i for i in [2, 1, 0] if i in raw_to_local]
    if len(rgb_fallback) < 3:
        rgb_fallback = list(range(min(3, cfg.input_channels)))
        while len(rgb_fallback) < 3:
            rgb_fallback.append(rgb_fallback[-1] if rgb_fallback else 0)
    cfg.diagnostics_rgb_idx = tuple(rgb_fallback[:3])  # type: ignore[assignment]

#实验预设
def _exp_presets() -> Dict[str, Dict[str, Any]]:
    """
    Progressive A-F ablations described in Table 1 of English V2.

    Lovasz and TTA are held constant so each adjacent preset changes only the
    data source or port-specific component named in the manuscript.
    """
    return {
        "optical": {
            "channel_indices": (0, 1, 2, 3),  # RGB + NIR
            "use_paim": False,
            "use_edge": False,
            "boundary_ignore": False,
            "use_coastal_weight": False,
            "use_aux": False,
            "lovasz_weight": 0.30,
            "tta": True,
            "run_diagnostics": False,
        },
        "optical_sar": {
            "channel_indices": (0, 1, 2, 3, 6, 7, 8),
            "use_paim": False,
            "use_edge": False,
            "boundary_ignore": False,
            "use_coastal_weight": False,
            "use_aux": False,
            "lovasz_weight": 0.30,
            "tta": True,
            "run_diagnostics": False,
        },
        "optical_sar_indices": {
            "channel_indices": DEFAULT_CHANNEL_INDICES,
            "use_paim": False,
            "use_edge": False,
            "boundary_ignore": False,
            "use_coastal_weight": False,
            "use_aux": False,
            "lovasz_weight": 0.30,
            "tta": True,
            "run_diagnostics": False,
        },
        "optical_sar_paim": {
            "channel_indices": DEFAULT_CHANNEL_INDICES,
            "use_paim": True,
            "use_edge": False,
            "boundary_ignore": False,
            "use_coastal_weight": False,
            "use_aux": False,
            "lovasz_weight": 0.30,
            "tta": True,
            "run_diagnostics": False,
        },
        "optical_sar_paim_boundary": {
            "channel_indices": DEFAULT_CHANNEL_INDICES,
            "use_paim": True,
            "use_edge": True,
            "boundary_ignore": True,
            "use_coastal_weight": False,
            "use_aux": False,
            "lovasz_weight": 0.30,
            "tta": True,
            "run_diagnostics": False,
        },
        "optical_sar_paim_boundary_coastal": {
            "channel_indices": DEFAULT_CHANNEL_INDICES,
            "use_paim": True,
            "use_edge": True,
            "boundary_ignore": True,
            "use_coastal_weight": True,
            "use_aux": False,
            "lovasz_weight": 0.30,
            "tta": True,
            "run_diagnostics": False,
        },
        "all_eta": {
            "channel_indices": DEFAULT_CHANNEL_INDICES,
            "use_paim": True,
            "use_edge": True,
            "boundary_ignore": True,
            "use_coastal_weight": True,
            "use_aux": True,
            "lovasz_weight": 0.30,
            "tta": True,
            "run_diagnostics": False,
        },
    }
#命令行的参数设置
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Top100 port dynamic segmentation trainer")
    p.add_argument(
        "--exp",
        type=str,
        default="all_eta",
        choices=tuple(_exp_presets().keys()),
        help="Experiment preset",
    )
    p.add_argument("--save-dir", type=str, default=None, help="Output directory; default runs/<exp>")
    p.add_argument("--data-root", type=str, default=None, help="Dataset root")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--channels", type=str, default=None, help="Custom raw channel indices, e.g. 0,1,2,3,6,7,8,10")
    p.add_argument("--disable-paim", action="store_true")
    p.add_argument("--disable-edge", action="store_true")
    p.add_argument("--disable-boundary-ignore", action="store_true")
    p.add_argument("--disable-coastal", action="store_true")
    p.add_argument("--disable-lovasz", action="store_true")
    p.add_argument("--disable-tta", action="store_true")
    p.add_argument("--run-diagnostics", action="store_true", help="Force diagnostics on")
    p.add_argument("--skip-diagnostics", action="store_true", help="Force diagnostics off")
    return p
#从命令行中读取cfg的各个参数
def build_cfg_from_args(args: argparse.Namespace) -> CFG:
    cfg = CFG()
    presets = _exp_presets()
    preset = presets[args.exp]
    for k, v in preset.items():
        setattr(cfg, k, v)

    if args.channels:
        cfg.channel_indices = parse_channel_indices(args.channels)
    apply_channel_layout(cfg, cfg.channel_indices)
    cfg.experiment_name = args.exp

    if args.data_root:
        cfg.data_root = args.data_root
    if args.save_dir:
        cfg.save_dir = args.save_dir
    else:
        cfg.save_dir = os.path.join("runs", f"{args.exp}_paper_aligned")
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed

    if args.disable_paim:
        cfg.use_paim = False
    if args.disable_edge:
        cfg.use_edge = False
    if args.disable_boundary_ignore:
        cfg.boundary_ignore = False
    if args.disable_coastal:
        cfg.use_coastal_weight = False
    if args.disable_lovasz:
        cfg.lovasz_weight = 0.0
    if args.disable_tta:
        cfg.tta = False
    if args.run_diagnostics:
        cfg.run_diagnostics = True
    if args.skip_diagnostics:
        cfg.run_diagnostics = False

    if cfg.use_paim:
        missing = []
        if cfg.mndwi_idx is None:
            missing.append("MNDWI(raw 10)")
        if cfg.vv_idx is None:
            missing.append("VV(raw 6)")
        if cfg.vh_idx is None:
            missing.append("VH(raw 7)")
        if missing:
            raise ValueError(
                f"PAIM requires channels {missing}. Current channel_indices={cfg.channel_indices}"
            )
    if cfg.use_coastal_weight and cfg.mndwi_idx is None:
        print("⚠️ Coastal weighting disabled because MNDWI(raw 10) is absent in selected channels.")
        cfg.use_coastal_weight = False

    return cfg
#输出实验总结
def print_experiment_summary(cfg: CFG) -> None:
    band_names = [RAW_BAND_NAMES[i] for i in cfg.channel_indices]
    print(f"🧪 Experiment: {cfg.experiment_name}")
    print(f"   Channels(raw idx): {cfg.channel_indices}")
    print(f"   Channels(names): {band_names}")
    print(
        f"   Modules: paim={cfg.use_paim}, edge={cfg.use_edge}, "
        f"boundary_ignore={cfg.boundary_ignore}, coastal_weight={cfg.use_coastal_weight}, aux={cfg.use_aux}, "
        f"lovasz_weight={cfg.lovasz_weight}, tta={cfg.tta}"
    )
#经典兜底归一化参数
def load_or_init_stats(cfg: CFG) -> Tuple[np.ndarray, np.ndarray]:
    """
    优先级：
      1) cfg.pixel_mean/std（你 audit 的真实统计）
      2) save_dir/stats.json（已有则读）
      3) fallback（不推荐）
    """
    stats_path = os.path.join(cfg.save_dir, "stats.json")

    if cfg.pixel_mean is not None and cfg.pixel_std is not None:
        mean = np.array(cfg.pixel_mean, dtype=np.float32)
        std = np.array(cfg.pixel_std, dtype=np.float32)
        return mean, std

    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        mean = np.array(d["mean"], dtype=np.float32)
        std = np.array(d["std"], dtype=np.float32)
        return mean, std

    # fallback（仅兜底）
    fallback_mean = np.array(
        [0.086464, 0.107327, 0.108375, 0.145583, 0.130819, 0.107364,
         0.606748, 0.335456, 2.902215, 0.030503, 0.088769, -0.136326],
        dtype=np.float32
    )
    fallback_std = np.array(
        [0.048247, 0.051619, 0.066393, 0.103499, 0.10548, 0.091958,
         0.207393, 0.222391, 3.911317, 0.339044, 0.491804, 0.238622],
        dtype=np.float32
    )
    return fallback_mean, fallback_std
#保存计算的归一化参数
def save_stats(cfg: CFG, mean: np.ndarray, std: np.ndarray):
    stats_path = os.path.join(cfg.save_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, ensure_ascii=False, indent=2)
#检查numpy数组在各个分位上的值
def robust_percentiles(x: np.ndarray, qs=(0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)) -> Dict[str, float]:
    qv = np.percentile(x, qs)
    return {str(q): float(v) for q, v in zip(qs, qv)}
#输出数组的统计值
def summarize_1d(x: np.ndarray) -> Dict[str, Any]:
    x = x.astype(np.float64)
    if x.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "percentiles": {}}
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "max": float(x.max()),
        "percentiles": robust_percentiles(x.astype(np.float32)),
    }
#构建边界忽略，输入二值图像
def make_boundary_ignore_mask(binary_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(binary_mask, dtype=bool)#如果忽略半径0，返回纯黑不忽略任何的掩码
    m = (binary_mask > 0).astype(np.uint8)
    if m.max() == 0 or m.min() == 1:
        return np.zeros_like(m, dtype=bool)#如果纯港口或者背景，也纯黑
    #膨胀-俯视得到一像元的梯度
    #定义操作算子
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    #根据半径把刚刚提取的边界膨胀
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    #膨胀
    ignore = cv2.dilate(edge, k2, iterations=1).astype(bool)
    return ignore#这个就是忽略的掩膜
#提取边界，传入膨胀长度，先提边界再膨胀
def make_edge_target(binary_mask: np.ndarray, dilate: int = 1) -> np.ndarray:
    m = (binary_mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    if dilate and dilate > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        edge = cv2.dilate(edge, k2, iterations=1)
    return (edge > 0).astype(np.uint8)
#去除面积太小的前景
def _remove_small_components(bin_img: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return bin_img
    #寻找联通区域
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bin_img.astype(np.uint8), connectivity=8)
    out = np.zeros_like(bin_img, dtype=np.uint8)#新建画布
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]#取出面积
        if area >= min_area:
            out[lab == i] = 1#放回画布
    return out
#算mndwi算水，costal流程
def infer_water_mask_from_mndwi(mndwi: np.ndarray, cfg: CFG) -> Tuple[np.ndarray, float]:
    """
    从 MNDWI 自动推断水体 mask（1=water）
    - 对 mndwi 做稳健裁剪 -> [0,255] -> Otsu 得阈值 -> 回到 mndwi 阈值
    - 形态学 open/close + 去小连通块
    """
    m = mndwi.astype(np.float32)

    if cfg.water_thresh_mode == "fixed":
        thr_val = float(cfg.water_thresh)
        water = (m > thr_val).astype(np.uint8)
        return water, thr_val

    if cfg.water_thresh_mode == "quantile":
        thr_val = float(np.nanquantile(m, cfg.paim_dyn_q))
        if not np.isfinite(thr_val):
            thr_val = float(cfg.water_thresh)
        water = (m > thr_val).astype(np.uint8)
        return water, thr_val

    if cfg.water_thresh_mode != "otsu":
        raise ValueError(f"Unsupported water_thresh_mode={cfg.water_thresh_mode!r}")

    p1 = np.percentile(m, cfg.water_otsu_clip_p1)
    p99 = np.percentile(m, cfg.water_otsu_clip_p99)
    if p99 <= p1 + 1e-6:
        thr_val = float(cfg.water_thresh)
        water = (m > thr_val).astype(np.uint8)
        return water, thr_val

    mc = np.clip(m, p1, p99)
    mu8 = np.clip((mc - p1) / (p99 - p1 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    # Otsu threshold in u8
    _, thr_u8 = cv2.threshold(mu8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2 returns thresholded image in thr_u8; to get threshold value, we recompute:
    # Workaround: use cv2.threshold return signature: (retval, dst)
    retval, _dst = cv2.threshold(mu8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(retval)

    thr_val = p1 + (t / 255.0) * (p99 - p1)
    water = (m > thr_val).astype(np.uint8)

    # morph open/close
    if cfg.water_morph_open and cfg.water_morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.water_morph_open * 2 + 1, cfg.water_morph_open * 2 + 1))
        water = cv2.morphologyEx(water, cv2.MORPH_OPEN, k, iterations=1)
    if cfg.water_morph_close and cfg.water_morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.water_morph_close * 2 + 1, cfg.water_morph_close * 2 + 1))
        water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, k, iterations=1)

    water = _remove_small_components(water, cfg.water_min_area)
    return water.astype(np.uint8), float(thr_val)
#计算距水体距离图
def make_coastal_weight_from_water_mask(
    water_mask: np.ndarray,
    alpha: float,
    sigma: float,
    maxdist: float,
    land_only: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    只对“陆地且靠近水体”的像素加权（匹配：港口是陆地但临近水）
    返回:
      w_map: (H,W) float32, in [1,1+alpha]
      dist_to_water: (H,W) float32 (仅陆地像素有意义)
    """
    water = (water_mask > 0).astype(np.uint8)
    land = (1 - water).astype(np.uint8)  # land=1, water=0

    # distTransform 计算到最近 0 的距离 -> 对 land 来说就是到 water 的距离
    dist_to_water = cv2.distanceTransform(land, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    #指数衰减，岸边1+a
    w = 1.0 + alpha * np.exp(-dist_to_water / max(1e-6, sigma))
    #存在影响半径，深海内陆为1
    w = np.where(dist_to_water <= maxdist, w, 1.0).astype(np.float32)

    if land_only:
        w = np.where(water == 1, 1.0, w).astype(np.float32)

    return w, dist_to_water
#diceloss 交并iou计算公式，只是对iou的拟合
def soft_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]  # fg prob
    target_fg = (target == 1).float()

    valid = (target != ignore_index).float()
    probs = probs * valid
    target_fg = target_fg * valid

    eps = 1e-6
    inter = (probs * target_fg).sum(dim=(1, 2))
    den = probs.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()

# ---- Lovasz ---- 一个面向iou的loss函数
def lovasz_grad(gt_sorted):
    # 对排序后的真实标签求累加和，得到不同阈值下的 TP 数量
    p = gt_sorted.sum()
    if p == 0:
        return gt_sorted * 0.
    gts = gt_sorted.cumsum(0)
    intersection = p - gts
    union = p + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    # 关键步骤：离散梯度。通过差分操作，将集合函数的增量作为梯度赋值给对应的像素
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_softmax_flat(probs, labels, ignore_index=255):
    valid = labels != ignore_index
    probs = probs[valid]
    labels = labels[valid]
    if probs.numel() == 0:
        return probs.sum() * 0.

    C = probs.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        pc = probs[:, c]
        errors = (fg - pc).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return probs.sum() * 0.
    return torch.stack(losses).mean()

def lovasz_softmax_loss(logits, target, ignore_index=255):
    
    # 1. 将 Logits（模型原始输出）转为概率值
    probs = torch.softmax(logits, dim=1)
    # 2. 维度变换
    # B, C, H, W -> B, H, W, C -> (B*H*W), C
    # 这样做是为了把图像中的所有像素点排成一排，统一处理
    B, C, H, W = probs.shape
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = target.view(-1)
    return lovasz_softmax_flat(probs, labels, ignore_index=ignore_index)

#数据集
class PortDataset(Dataset):
    """
    return:
      x: (C,H,W) float (normalized)
      y_bin: (H,W) long {0,1,255}
      y_aux: (H,W) long {0..8,255}
      y_edge: (H,W) float {0,1}
      w_map: (H,W) float >=1
    """
    #初始化地址统计值排序
    def __init__(self, cfg: CFG, split: str, mean: np.ndarray, std: np.ndarray, do_augment: bool):
        self.cfg = cfg
        self.split = split
        self.do_augment = do_augment

        self.img_dir = os.path.join(cfg.data_root, split, "images")
        self.mask_dir = os.path.join(cfg.data_root, split, "masks")

        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(cfg.img_ext)])
        assert len(self.files) > 0, f"No npy found in {self.img_dir}"

        self.mean = mean.reshape(1, 1, -1).astype(np.float32)
        self.std = std.reshape(1, 1, -1).astype(np.float32)
    #返回图总数
    def __len__(self):
        return len(self.files)
    #读取单图和对应的标签
    def _read_pair(self, idx: int):
        fname = self.files[idx]
        x = np.load(os.path.join(self.img_dir, fname)).astype(np.float32)
        x = adapt_loaded_image_channels(x, self.cfg)  # HWC + selected channels

        mask_path = os.path.join(self.mask_dir, fname.replace(self.cfg.img_ext, self.cfg.mask_ext))
        y = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if y is None:
            y = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
        return x, y
    #训练时抠图？似乎没必要？
    def _random_crop(self, x, y, crop_size: int):
        H, W, _ = x.shape
        if crop_size >= H or crop_size >= W:
            return x, y
        tries = self.cfg.tries_per_crop
        for _ in range(tries):
            top = random.randint(0, H - crop_size)
            left = random.randint(0, W - crop_size)
            yc = y[top:top+crop_size, left:left+crop_size]
            if random.random() < self.cfg.prefer_port_crop:
                if (yc > 0).mean() >= self.cfg.train_fg_min_frac:
                    return x[top:top+crop_size, left:left+crop_size], yc
            else:
                return x[top:top+crop_size, left:left+crop_size], yc
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        return x[top:top+crop_size, left:left+crop_size], y[top:top+crop_size, left:left+crop_size]
    #数据增强，扰动和噪声
    def _augment(self, x, y):
        if random.random() < 0.5:
            x = np.ascontiguousarray(np.fliplr(x))
            y = np.ascontiguousarray(np.fliplr(y))
        if random.random() < 0.5:
            x = np.ascontiguousarray(np.flipud(x))
            y = np.ascontiguousarray(np.flipud(y))
        if random.random() < 0.5:
            k = random.randint(0, 3)
            x = np.ascontiguousarray(np.rot90(x, k))
            y = np.ascontiguousarray(np.rot90(y, k))

        if random.random() < 0.35:
            H, W = y.shape
            angle = random.uniform(-8, 8)
            scale = random.uniform(0.95, 1.05)
            tx = random.uniform(-0.03, 0.03) * W
            ty = random.uniform(-0.03, 0.03) * H
            M = cv2.getRotationMatrix2D((W/2, H/2), angle, scale)
            M[:, 2] += (tx, ty)
            x = cv2.warpAffine(x, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            y = cv2.warpAffine(y, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

        # optical-like channels jitter
        if random.random() < 0.3 and len(self.cfg.optical_local_indices) > 0:
            gain = random.uniform(0.9, 1.1)
            bias = random.uniform(-0.02, 0.02)
            idx = list(self.cfg.optical_local_indices)
            x[..., idx] = np.clip(x[..., idx] * gain + bias, 0.0, 1.0)

        # SAR-like noise
        if random.random() < 0.25 and len(self.cfg.sar_local_indices) > 0:
            idx = list(self.cfg.sar_local_indices)
            noise = np.random.normal(0, 0.02, size=x[..., idx].shape).astype(np.float32)
            x[..., idx] = x[..., idx] + noise

        return x, y
    #dataset【i】时触发，完整的流水线加载增强辅助任务标签
    def __getitem__(self, idx):
        x, y_raw = self._read_pair(idx)

        # crop
        if self.cfg.crop_size and self.cfg.crop_size < x.shape[0]:
            if self.split == "train":
                x, y_raw = self._random_crop(x, y_raw, self.cfg.crop_size)
            else:
                # val 使用 center crop，避免随机裁剪导致指标抖动
                H, W, _ = x.shape
                cs = self.cfg.crop_size
                top = max(0, (H - cs) // 2)
                left = max(0, (W - cs) // 2)
                x = x[top:top+cs, left:left+cs]
                y_raw = y_raw[top:top+cs, left:left+cs]

        if self.do_augment:
            x, y_raw = self._augment(x, y_raw)

        # binary (raw)
        y_bin_raw = (y_raw > 0).astype(np.uint8)

        # edge target from raw binary
        y_edge = make_edge_target(y_bin_raw, dilate=self.cfg.edge_dilate).astype(np.float32)

        # coastal weight: infer water from raw MNDWI; weight land-near-water only
        if self.cfg.use_coastal_weight and self.cfg.mndwi_idx is not None:
            mndwi = x[..., self.cfg.mndwi_idx].astype(np.float32)
            water_mask, _thr = infer_water_mask_from_mndwi(mndwi, self.cfg)
            w_map, _dist = make_coastal_weight_from_water_mask(
                water_mask,
                alpha=self.cfg.coastal_alpha,
                sigma=self.cfg.coastal_sigma,
                maxdist=self.cfg.coastal_maxdist,
                land_only=self.cfg.coastal_land_only
            )
        else:
            w_map = np.ones_like(y_bin_raw, dtype=np.float32)

        # aux label
        y_max = int(y_raw.max())
        if y_max <= 1:
            y_aux = np.full_like(y_raw, 255, dtype=np.uint8)
        elif y_max == 255 and (np.unique(y_raw).size <= 2):
            y_aux = np.full_like(y_raw, 255, dtype=np.uint8)
        else:
            y_aux = np.clip(y_raw, 0, self.cfg.num_aux_classes - 1).astype(np.uint8)

        # The manuscript relaxes only the main binary segmentation loss.
        y_bin = y_bin_raw.copy().astype(np.uint8)
        if self.cfg.boundary_ignore:
            ignore = make_boundary_ignore_mask(y_bin, self.cfg.boundary_radius)
            y_bin_ign = y_bin.copy()
            y_bin_ign[ignore] = 255
            y_bin = y_bin_ign

        # normalize
        x = (x - self.mean) / (self.std + 1e-6)
        x = np.ascontiguousarray(x.transpose(2, 0, 1))  # (C,H,W)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y_bin).long(),
            torch.from_numpy(y_aux).long(),
            torch.from_numpy(y_edge).float(),
            torch.from_numpy(w_map).float(),
        )
#paim模块
#条带池化 512*512到1*512，水平加垂直，池化算的是平均
class StripPooling(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        xh = self.conv_h(self.pool_h(x))
        #插值，再特征融合为512*512
        xh = F.interpolate(xh, size=(H, W), mode="bilinear", align_corners=False)
        xw = self.conv_w(self.pool_w(x))
        xw = F.interpolate(xw, size=(H, W), mode="bilinear", align_corners=False)
        return x + xh + xw

class PAIM(nn.Module):
    
    def __init__(self, embed_dim: int, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        #参数，特征维度，对应波段位置
        self.vv_idx = cfg.vv_idx
        self.vh_idx = cfg.vh_idx
        self.mndwi_idx = cfg.mndwi_idx
        if self.vv_idx is None or self.vh_idx is None or self.mndwi_idx is None:
            raise ValueError(
                f"PAIM expects vv/vh/mndwi in current channel subset, got vv={self.vv_idx}, vh={self.vh_idx}, mndwi={self.mndwi_idx}"
            )
        #微观，3*3卷积核，分组归一，激活函数，降维，输出0-1权重
        self.micro = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        #中维，5*5
        self.meso = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        #条带池化
        if cfg.paim_use_strip_pool:
            self.macro = StripPooling(1)
        else:
            self.macro = nn.Sequential(
                nn.AvgPool2d(kernel_size=31, stride=1, padding=15),
                nn.Conv2d(1, 1, 1),
                nn.Sigmoid()
            )
        #特征融合，加回去
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim + 3, embed_dim, 1),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1)
        )
        #gamma自适应
        self.gamma = nn.Parameter(torch.tensor([cfg.paim_gamma_init], dtype=torch.float32))
    #静态方法
    #计算局部方差
    @staticmethod
    def _local_variance(img: torch.Tensor, k: int = 5):
        pad = k // 2
        mu = F.avg_pool2d(img, k, stride=1, padding=pad)
        mu2 = mu * mu
        sqmu = F.avg_pool2d(img * img, k, stride=1, padding=pad)
        var = sqmu - mu2
        return F.relu(var)

    def forward(self, visual_feat: torch.Tensor, x_phys: torch.Tensor):
        with torch.no_grad():
            mndwi = x_phys[:, self.mndwi_idx:self.mndwi_idx+1]  # (B,1,H,W)

            # 动态阈值（避免固定 0 在你数据上失效）
            if self.cfg.paim_use_dynamic_thresh:
                # per-sample quantile threshold
                B = mndwi.shape[0]
                flat = mndwi.reshape(B, -1)
                thr = torch.quantile(flat, q=self.cfg.paim_dyn_q, dim=1).view(B, 1, 1, 1)
            else:
                thr = torch.tensor(self.cfg.water_thresh, device=mndwi.device, dtype=mndwi.dtype).view(1, 1, 1, 1)
            #水体概率图
            water_prob = torch.sigmoid((mndwi - thr) * float(self.cfg.paim_water_slope))

            vv = x_phys[:, self.vv_idx:self.vv_idx+1]
            vh = x_phys[:, self.vh_idx:self.vh_idx+1]
            sar_intensity = torch.sqrt(vv * vv + vh * vh + 1e-6)
            sar_tex = self._local_variance(sar_intensity, k=5)

        ts = visual_feat.shape[-2:]
        w_small = F.interpolate(water_prob, size=ts, mode="bilinear", align_corners=False)
        t_small = F.interpolate(sar_tex, size=ts, mode="bilinear", align_corners=False)

        a_micro = self.micro(t_small)
        a_meso = self.meso(w_small)
        a_macro = self.macro(w_small)
        #拼接
        fused = torch.cat([visual_feat, a_micro, a_meso, a_macro], dim=1)
        out = visual_feat + self.gamma * self.fusion(fused)#gamma学习中优化
        return out

#resent-unet模型
class PhysicsAwareUnetMultiTask(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        #定义unet,5层
        self.unet = smp.Unet(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.input_channels,
            classes=cfg.num_binary_classes,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )
        #注册参数
        # register mean/std for PAIM inverse-normalization
        assert cfg.pixel_mean is not None and cfg.pixel_std is not None, "cfg.pixel_mean/std must be set"
        self.register_buffer(
            "x_mean", torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(1, cfg.input_channels, 1, 1),
            persistent=False
        )
        self.register_buffer(
            "x_std", torch.tensor(cfg.pixel_std, dtype=torch.float32).view(1, cfg.input_channels, 1, 1),
            persistent=False
        )
        #paim
        self.use_paim = cfg.use_paim
        if self.use_paim:
            dummy = torch.zeros(1, cfg.input_channels, 64, 64)
            with torch.no_grad():
                feats = self.unet.encoder(dummy)
            embed_dim = feats[-1].shape[1]
            self.paim = PAIM(embed_dim=embed_dim, cfg=cfg)
        #aux
        self.use_aux = cfg.use_aux
        if self.use_aux:
            self.aux_head = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, cfg.num_aux_classes, 1)
            )
        #edge
        self.use_edge = cfg.use_edge
        if self.use_edge:
            self.edge_head = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1)
            )

    def forward(self, x):
        feats = self.unet.encoder(x)

        if self.use_paim:
            # 关键：PAIM 输入必须是物理尺度，而不是归一化尺度
            x_phys = x * (self.x_std + 1e-6) + self.x_mean
            feats[-1] = self.paim(feats[-1], x_phys)

        dec = self.unet.decoder(*feats)
        bin_logits = self.unet.segmentation_head(dec)

        if self.training:
            aux_logits = self.aux_head(dec) if self.use_aux else None
            edge_logits = self.edge_head(dec) if self.use_edge else None
            return bin_logits, aux_logits, edge_logits
        else:
            return bin_logits
#ema 影子模型
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float, device: str):
        self.decay = decay
        self.device = device
        self.ema = self._clone_model(model).to(self.device)
        self.ema.eval()

    def _clone_model(self, model):
        ema = type(model)(model.cfg)
        ema.load_state_dict(model.state_dict(), strict=True)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if k not in msd:
                continue
            if torch.is_floating_point(esd[k]):
                esd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
            else:
                esd[k].copy_(msd[k])

#loss
def weighted_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    weight_map: Optional[torch.Tensor] = None,
    ignore_index: int = 255
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index, reduction="none")
    valid = (target != ignore_index).float()
    if weight_map is None:
        num = (ce * valid).sum()
        den = valid.sum().clamp_min(1.0)
        return num / den
    wm = weight_map.float()
    num = (ce * valid * wm).sum()
    den = (valid * wm).sum().clamp_min(1.0)
    return num / den


# =========================
# 7. Metrics / TTA / Thresh Sweep
# =========================

@torch.no_grad()
def compute_iou_binary_from_pred(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255):
    # pred: (B,H,W) {0,1}
    valid = (target != ignore_index)
    ious = []
    for cls in [0, 1]:
        p = (pred == cls) & valid
        t = (target == cls) & valid
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        ious.append(inter / (union + 1e-6))
    miou = (ious[0] + ious[1]) / 2.0
    return ious[0], ious[1], miou

@torch.no_grad()
def tta_logits(model: nn.Module, x: torch.Tensor):
    model.eval()
    logits0 = model(x)

    x1 = torch.flip(x, dims=[3])
    logits1 = torch.flip(model(x1), dims=[3])

    x2 = torch.flip(x, dims=[2])
    logits2 = torch.flip(model(x2), dims=[2])

    x3 = torch.flip(x, dims=[2, 3])
    logits3 = torch.flip(model(x3), dims=[2, 3])

    return (logits0 + logits1 + logits2 + logits3) / 4.0

def make_thr_list(cfg: CFG) -> List[float]:
    a, b, s = cfg.thresh_grid
    thrs = []
    t = a
    while t <= b + 1e-9:
        thrs.append(float(t))
        t += s
    return thrs

@torch.no_grad()
def validate(cfg: CFG, model: nn.Module, loader: DataLoader, device: str, epoch: int):
    model.eval()

    # argmax metrics
    sum_iou_bg, sum_iou_fg, sum_miou = 0.0, 0.0, 0.0
    n = 0
    eps = 1e-6

    # pixel confusion (global over val)
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    # boundary IoU stats
    b_inter = 0.0
    b_union = 0.0

    # coastal non-port FPR stats
    coastal_non_port_fp = 0.0
    coastal_non_port_total = 0.0
    mean_t = None
    std_t = None
    if cfg.pixel_mean is not None and cfg.pixel_std is not None:
        mean_t = torch.tensor(cfg.pixel_mean, device=device, dtype=torch.float32).view(1, cfg.input_channels, 1, 1)
        std_t = torch.tensor(cfg.pixel_std, device=device, dtype=torch.float32).view(1, cfg.input_channels, 1, 1)

    # optional threshold sweep
    do_sweep = cfg.do_thresh_sweep and (epoch % cfg.thresh_eval_interval == 0)
    thrs = make_thr_list(cfg) if do_sweep else []
    # accumulate intersections/unions for each threshold (global across images)
    if do_sweep:
        inter_fg = np.zeros(len(thrs), dtype=np.float64)
        union_fg = np.zeros(len(thrs), dtype=np.float64)
        inter_bg = np.zeros(len(thrs), dtype=np.float64)
        union_bg = np.zeros(len(thrs), dtype=np.float64)

    pbar = tqdm(loader, desc="Val", leave=False)
    for bi, (x, y_bin, _, _, _) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y_bin = y_bin.to(device, non_blocking=True)

        logits = tta_logits(model, x) if cfg.tta else model(x)

        # argmax
        pred = torch.argmax(logits, dim=1)
        iou_bg, iou_fg, miou = compute_iou_binary_from_pred(pred, y_bin, ignore_index=255)
        sum_iou_bg += iou_bg
        sum_iou_fg += iou_fg
        sum_miou += miou
        n += 1
        pbar.set_postfix(iou_fg=f"{iou_fg:.4f}")

        # confusion stats
        pred_np = pred.detach().cpu().numpy().astype(np.uint8)
        y_np = y_bin.detach().cpu().numpy()
        valid_np = (y_np != 255)
        gt_np = (y_np == 1)

        tp += float(np.logical_and(np.logical_and(pred_np == 1, gt_np), valid_np).sum())
        fp += float(np.logical_and(np.logical_and(pred_np == 1, ~gt_np), valid_np).sum())
        tn += float(np.logical_and(np.logical_and(pred_np == 0, ~gt_np), valid_np).sum())
        fn += float(np.logical_and(np.logical_and(pred_np == 0, gt_np), valid_np).sum())

        # boundary IoU
        bs = pred_np.shape[0]
        for bj in range(bs):
            p_edge = make_edge_target(pred_np[bj].astype(np.uint8), dilate=cfg.edge_dilate).astype(bool)
            g_edge = make_edge_target(gt_np[bj].astype(np.uint8), dilate=cfg.edge_dilate).astype(bool)
            v = valid_np[bj]
            b_inter += float(np.logical_and(np.logical_and(p_edge, g_edge), v).sum())
            b_union += float(np.logical_and(np.logical_or(p_edge, g_edge), v).sum())

        # coastal non-port FPR
        if cfg.mndwi_idx is not None and mean_t is not None and std_t is not None:
            x_phys = x * (std_t + 1e-6) + mean_t
            mndwi_batch = x_phys[:, cfg.mndwi_idx].detach().cpu().numpy().astype(np.float32)
            for bj in range(pred_np.shape[0]):
                water_mask, _thr = infer_water_mask_from_mndwi(mndwi_batch[bj], cfg)
                _, dist = make_coastal_weight_from_water_mask(
                    water_mask,
                    alpha=cfg.coastal_alpha,
                    sigma=cfg.coastal_sigma,
                    maxdist=cfg.coastal_maxdist,
                    land_only=cfg.coastal_land_only
                )
                coastal_non_port = (
                    (water_mask == 0) &
                    (dist <= cfg.coastal_maxdist) &
                    (y_np[bj] == 0) &
                    valid_np[bj]
                )
                coastal_non_port_total += float(coastal_non_port.sum())
                coastal_non_port_fp += float(np.logical_and(pred_np[bj] == 1, coastal_non_port).sum())

        # threshold sweep (limited batches)
        if do_sweep and bi < cfg.thresh_sweep_max_batches:
            prob_fg = torch.softmax(logits, dim=1)[:, 1]  # (B,H,W)
            valid = (y_bin != 255)
            gt_fg = (y_bin == 1)

            # vectorized over thresholds via CPU loop (T small, B small)
            prob_fg_np = prob_fg.detach().cpu().numpy()
            valid_np_thr = valid.detach().cpu().numpy().astype(np.bool_)
            gt_fg_np_thr = gt_fg.detach().cpu().numpy().astype(np.bool_)

            for ti, t in enumerate(thrs):
                pred_fg = (prob_fg_np > t)
                pred_bg = ~pred_fg

                inter_fg[ti] += np.logical_and(np.logical_and(pred_fg, gt_fg_np_thr), valid_np_thr).sum()
                union_fg[ti] += np.logical_and(np.logical_or(pred_fg, gt_fg_np_thr), valid_np_thr).sum()

                gt_bg = ~gt_fg_np_thr
                inter_bg[ti] += np.logical_and(np.logical_and(pred_bg, gt_bg), valid_np_thr).sum()
                union_bg[ti] += np.logical_and(np.logical_or(pred_bg, gt_bg), valid_np_thr).sum()

    if n == 0:
        return {
            "iou_bg": 0.0,
            "iou_fg": 0.0,
            "miou": 0.0,
            "boundary_iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "fpr": 0.0,
            "coastal_non_port_fpr": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
            "coastal_non_port_fp": 0.0,
            "coastal_non_port_total": 0.0,
        }, None

    iou_bg = sum_iou_bg / n
    iou_fg = sum_iou_fg / n
    miou = sum_miou / n
    boundary_iou = float(b_inter / (b_union + eps))

    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float(2.0 * precision * recall / (precision + recall + eps))
    specificity = float(tn / (tn + fp + eps))
    fpr = float(fp / (fp + tn + eps))
    coastal_non_port_fpr = float(coastal_non_port_fp / (coastal_non_port_total + eps))

    sweep_result = None
    if do_sweep:
        iou_fg_t = inter_fg / (union_fg + 1e-6)
        iou_bg_t = inter_bg / (union_bg + 1e-6)
        miou_t = (iou_fg_t + iou_bg_t) / 2.0
        best = int(np.argmax(iou_fg_t))
        sweep_result = {
            "best_thr": float(thrs[best]),
            "iou_fg_best_thr": float(iou_fg_t[best]),
            "iou_bg_best_thr": float(iou_bg_t[best]),
            "miou_best_thr": float(miou_t[best]),
        }

    metrics = {
        "iou_bg": float(iou_bg),
        "iou_fg": float(iou_fg),
        "miou": float(miou),
        "boundary_iou": boundary_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "coastal_non_port_fpr": coastal_non_port_fpr,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "coastal_non_port_fp": float(coastal_non_port_fp),
        "coastal_non_port_total": float(coastal_non_port_total),
    }
    return metrics, sweep_result


# =========================
# 8. Train
# =========================

def build_scheduler(cfg: CFG, optimizer, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def train_one_epoch(cfg: CFG, model: nn.Module, ema: Optional[ModelEMA],
                    loader: DataLoader, optimizer, scheduler, scaler,
                    ce_weights: torch.Tensor, ce_criterion_aux: nn.Module,
                    epoch: int, device: str):

    model.train()
    pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)

    running = {
        "loss_total": 0.0,
        "loss_ce": 0.0,
        "loss_dice": 0.0,
        "loss_lovasz": 0.0,
        "loss_aux": 0.0,
        "loss_edge": 0.0,
    }

    bce_edge = None
    if cfg.use_edge:
        pos_w = torch.tensor([cfg.edge_pos_weight], device=device, dtype=torch.float32)
        bce_edge = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="mean")

    for i, (x, y_bin, y_aux, y_edge, w_map) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y_bin = y_bin.to(device, non_blocking=True)
        y_aux = y_aux.to(device, non_blocking=True)
        y_edge = y_edge.to(device, non_blocking=True)
        w_map = w_map.to(device, non_blocking=True)

        # coastal 权重延后启用
        w_use = None
        if cfg.use_coastal_weight and epoch > cfg.coastal_start_epoch:
            w_use = w_map

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            bin_logits, aux_logits, edge_logits = model(x)

            loss_ce = weighted_ce_loss(
                bin_logits, y_bin, class_weights=ce_weights,
                weight_map=w_use,
                ignore_index=255
            )
            loss_dice = soft_dice_loss_from_logits(bin_logits, y_bin, ignore_index=255) * cfg.dice_weight
            loss_lovasz = torch.zeros((), device=device, dtype=loss_ce.dtype)
            loss_aux_term = torch.zeros((), device=device, dtype=loss_ce.dtype)
            loss_edge_term = torch.zeros((), device=device, dtype=loss_ce.dtype)

            if cfg.lovasz_weight > 0 and epoch > cfg.lovasz_start_epoch:
                loss_lovasz = cfg.lovasz_weight * lovasz_softmax_loss(bin_logits, y_bin, ignore_index=255)

            if cfg.use_aux and aux_logits is not None and (y_aux != 255).any():
                if epoch > cfg.aux_warmup_epochs:
                    ramp = min(1.0, (epoch - cfg.aux_warmup_epochs + 1) / 10.0)
                    loss_aux = ce_criterion_aux(aux_logits, y_aux)
                    loss_aux_term = (cfg.aux_weight * ramp) * loss_aux

            if cfg.use_edge and edge_logits is not None and epoch > cfg.edge_start_epoch:
                edge_log = edge_logits.squeeze(1)
                loss_edge = bce_edge(edge_log, y_edge)
                loss_edge_term = cfg.edge_weight * loss_edge

            loss = loss_ce + loss_dice + loss_lovasz + loss_aux_term + loss_edge_term

        scaler.scale(loss / cfg.accum_steps).backward()

        if cfg.grad_clip and (i % cfg.accum_steps == 0):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if i % cfg.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        running["loss_total"] += float(loss.item())
        running["loss_ce"] += float(loss_ce.item())
        running["loss_dice"] += float(loss_dice.item())
        running["loss_lovasz"] += float(loss_lovasz.item())
        running["loss_aux"] += float(loss_aux_term.item())
        running["loss_edge"] += float(loss_edge_term.item())
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

    den = max(1, len(loader))
    return {k: v / den for k, v in running.items()}


# =========================
# 9. Diagnostics (输出下一步要检查的数据)
# =========================

def _overlay_mask(rgb: np.ndarray, m: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    red = np.zeros_like(out); red[..., 2] = 255
    sel = (m > 0)
    out[sel] = (out[sel] * (1 - alpha) + red[sel] * alpha).astype(np.uint8)
    return out

def _to_rgb8(x_hwc: np.ndarray, rgb_idx: Tuple[int, int, int]) -> np.ndarray:
    rgb = x_hwc[..., list(rgb_idx)].astype(np.float32)
    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    rgb01 = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb01 * 255).astype(np.uint8)

def run_dataset_diagnostics(cfg: CFG):
    """
    直接扫文件（不依赖 DataLoader），输出：
      - pos_frac / cc_count / cc_area_median 分布
      - fg/bg 的 mndwi/vv/vh 均值
      - Otsu 阈值分布、水体比例分布、coastal band(陆地近水带)比例
      - 可视化样例：GT、water mask、weight map
    """
    print("\n🔎 Running dataset diagnostics...")
    vis_root = os.path.join(cfg.save_dir, "diagnostics_vis")
    ensure_dir(vis_root)

    for split in ["train", "val"]:
        img_dir = os.path.join(cfg.data_root, split, "images")
        msk_dir = os.path.join(cfg.data_root, split, "masks")
        files = sorted([f for f in os.listdir(img_dir) if f.endswith(cfg.img_ext)])
        if cfg.diagnostics_max_samples and cfg.diagnostics_max_samples > 0:
            random.seed(42)
            files = random.sample(files, min(cfg.diagnostics_max_samples, len(files)))

        pos_fracs = []
        cc_counts = []
        cc_area_meds = []

        thr_list = []
        water_fracs = []
        coastal_land_fracs = []

        mndwi_fg = []
        mndwi_bg = []
        vv_fg = []
        vv_bg = []
        vh_fg = []
        vh_bg = []

        save_vis_n = max(0, int(cfg.diagnostics_save_vis))
        vis_dir = os.path.join(vis_root, split)
        ensure_dir(vis_dir)

        for k, fname in enumerate(tqdm(files, desc=f"Diag {split}")):
            x = np.load(os.path.join(img_dir, fname)).astype(np.float32)
            x = adapt_loaded_image_channels(x, cfg)  # HWC + selected channels
            y = cv2.imread(os.path.join(msk_dir, fname.replace(cfg.img_ext, cfg.mask_ext)), cv2.IMREAD_GRAYSCALE)
            if y is None:
                y = np.zeros((x.shape[0], x.shape[1]), np.uint8)

            m = (y > 0).astype(np.uint8)
            pos_fracs.append(float(m.mean()))

            num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            comps = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([], np.int32)
            cc_counts.append(int(comps.size))
            cc_area_meds.append(float(np.median(comps)) if comps.size else 0.0)

            # water mask + coastal band
            if cfg.mndwi_idx is not None:
                mndwi = x[..., cfg.mndwi_idx].astype(np.float32)
                water_mask, thr = infer_water_mask_from_mndwi(mndwi, cfg)
                thr_list.append(float(thr))
                water_fracs.append(float(water_mask.mean()))

                w_map, dist = make_coastal_weight_from_water_mask(
                    water_mask,
                    alpha=cfg.coastal_alpha,
                    sigma=cfg.coastal_sigma,
                    maxdist=cfg.coastal_maxdist,
                    land_only=cfg.coastal_land_only
                )
                land = (water_mask == 0)
                coastal_band = land & (dist <= cfg.coastal_maxdist)
                coastal_land_fracs.append(float(coastal_band.mean()))
            else:
                mndwi = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
                water_mask = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
                w_map = np.ones((x.shape[0], x.shape[1]), dtype=np.float32)

            # fg/bg channel means
            if (m > 0).any():
                mndwi_fg.append(float(mndwi[m > 0].mean()))
                if cfg.vv_idx is not None:
                    vv_fg.append(float(x[..., cfg.vv_idx][m > 0].mean()))
                if cfg.vh_idx is not None:
                    vh_fg.append(float(x[..., cfg.vh_idx][m > 0].mean()))
            if (m == 0).any():
                mndwi_bg.append(float(mndwi[m == 0].mean()))
                if cfg.vv_idx is not None:
                    vv_bg.append(float(x[..., cfg.vv_idx][m == 0].mean()))
                if cfg.vh_idx is not None:
                    vh_bg.append(float(x[..., cfg.vh_idx][m == 0].mean()))

            # save visuals
            if save_vis_n > 0 and k < save_vis_n:
                rgb8 = _to_rgb8(x, cfg.diagnostics_rgb_idx)
                ov = _overlay_mask(rgb8, m)

                # water overlay (blue)
                blue = np.zeros_like(ov); blue[..., 0] = 255
                selw = (water_mask > 0)
                ov2 = ov.copy()
                ov2[selw] = (ov2[selw] * 0.6 + blue[selw] * 0.4).astype(np.uint8)

                # weight map heat
                wm = w_map.copy()
                wm = (wm - 1.0) / max(1e-6, float(cfg.coastal_alpha))
                wm = np.clip(wm, 0, 1)
                wm_u8 = (wm * 255).astype(np.uint8)
                heat = cv2.applyColorMap(wm_u8, cv2.COLORMAP_JET)
                heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

                # stack
                canvas = np.concatenate([ov2, heat], axis=1)
                out_path = os.path.join(vis_dir, fname.replace(cfg.img_ext, ".png"))
                cv2.imwrite(out_path, canvas[..., ::-1])

        out = {
            "n_files": len(files),
            "pos_frac": summarize_1d(np.array(pos_fracs, np.float32)),
            "cc_count": summarize_1d(np.array(cc_counts, np.float32)),
            "cc_area_median": summarize_1d(np.array(cc_area_meds, np.float32)),
            "mndwi_fg_mean": float(np.mean(mndwi_fg)) if len(mndwi_fg) else None,
            "mndwi_bg_mean": float(np.mean(mndwi_bg)) if len(mndwi_bg) else None,
            "vv_fg_mean": float(np.mean(vv_fg)) if len(vv_fg) else None,
            "vv_bg_mean": float(np.mean(vv_bg)) if len(vv_bg) else None,
            "vh_fg_mean": float(np.mean(vh_fg)) if len(vh_fg) else None,
            "vh_bg_mean": float(np.mean(vh_bg)) if len(vh_bg) else None,
            "water_otsu_thr": summarize_1d(np.array(thr_list, np.float32)),
            "water_frac": summarize_1d(np.array(water_fracs, np.float32)),
            "coastal_land_band_frac": summarize_1d(np.array(coastal_land_fracs, np.float32)),
        }

        out_path = os.path.join(cfg.save_dir, f"diagnostics_{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"✅ Diagnostics saved: {out_path}")


# =========================
# 10. Checkpoint / Resume
# =========================

def save_checkpoint(cfg: CFG, model: nn.Module, ema: Optional[ModelEMA],
                    optimizer, scheduler, scaler,
                    epoch: int, best_fg_iou: float, path: str):
    ckpt = {
        "cfg": asdict(cfg),
        "epoch": epoch,
        "best_fg_iou": best_fg_iou,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.ema.state_dict() if ema is not None else None,
    }
    torch.save(ckpt, path)

def try_resume(cfg: CFG, model, ema, optimizer, scheduler, scaler) -> Tuple[int, float]:
    ckpt_path = os.path.join(cfg.save_dir, "last.ckpt")
    if not os.path.exists(ckpt_path):
        return 1, 0.0

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.ema.load_state_dict(ckpt["ema"], strict=True)

    start_epoch = int(ckpt["epoch"]) + 1
    best_fg = float(ckpt.get("best_fg_iou", 0.0))
    print(f"🔁 Resumed from {ckpt_path} | start_epoch={start_epoch}, best_fg_iou={best_fg:.4f}")
    return start_epoch, best_fg


# =========================
# 11. Main
# =========================

def main():
    args = build_arg_parser().parse_args()
    cfg = build_cfg_from_args(args)
    ensure_dir(cfg.save_dir)
    print_experiment_summary(cfg)

    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")
    print(f"🔒 Reproducibility seed={cfg.seed} (deterministic on)")

    mean_raw, std_raw = load_or_init_stats(cfg)
    mean, std = slice_stats_to_channels(mean_raw, std_raw, cfg)
    save_stats(cfg, mean, std)

    # 强制把 mean/std 写回 cfg（PAIM 反归一化需要）
    cfg.pixel_mean = mean.tolist()
    cfg.pixel_std = std.tolist()

    # persist cfg AFTER runtime adaptation
    with open(os.path.join(cfg.save_dir, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    if cfg.run_diagnostics:
        run_dataset_diagnostics(cfg)

    train_ds = PortDataset(cfg, split="train", mean=mean, std=std, do_augment=True)
    val_ds = PortDataset(cfg, split="val", mean=mean, std=std, do_augment=False)

    train_gen = torch.Generator()
    train_gen.manual_seed(cfg.seed)
    val_gen = torch.Generator()
    val_gen.manual_seed(cfg.seed + 10007)
    worker_init = make_seed_worker()

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init, generator=train_gen
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg.batch_size // 2), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        worker_init_fn=worker_init, generator=val_gen
    )

    model = PhysicsAwareUnetMultiTask(cfg).to(device)

    ce_weights = torch.tensor([cfg.ce_bg_weight, cfg.ce_fg_weight], device=device, dtype=torch.float32)

    aux_weights = torch.ones(cfg.num_aux_classes, device=device, dtype=torch.float32)
    ce_criterion_aux = nn.CrossEntropyLoss(weight=aux_weights, ignore_index=255)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    ema = ModelEMA(model, decay=cfg.ema_decay, device=device) if cfg.ema_decay and cfg.ema_decay > 0 else None

    start_epoch, best_fg_iou = try_resume(cfg, model, ema, optimizer, scheduler, scaler)

    metrics_path = os.path.join(cfg.save_dir, "val_metrics.jsonl")
    print("🚀 Start training...")
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            cfg, model, ema, train_loader, optimizer, scheduler, scaler,
            ce_weights, ce_criterion_aux, epoch, device
        )

        eval_model = ema.ema if ema is not None else model
        val_metrics, sweep = validate(cfg, eval_model, val_loader, device, epoch)
        iou_bg = float(val_metrics["iou_bg"])
        iou_fg = float(val_metrics["iou_fg"])
        miou = float(val_metrics["miou"])
        boundary_iou = float(val_metrics["boundary_iou"])

        dt = time.time() - t0
        msg = (
            f"\nEpoch {epoch:03d}/{cfg.epochs} | "
            f"loss={train_stats['loss_total']:.4f} (ce={train_stats['loss_ce']:.4f}, dice={train_stats['loss_dice']:.4f}, "
            f"lovasz={train_stats['loss_lovasz']:.4f}, aux={train_stats['loss_aux']:.4f}, edge={train_stats['loss_edge']:.4f}) | "
            f"PortIoU={iou_fg:.4f} | BackgroundIoU={iou_bg:.4f} | mIoU={miou:.4f} | BoundaryIoU={boundary_iou:.4f} | "
            f"Precision={val_metrics['precision']:.4f} | Recall={val_metrics['recall']:.4f} | F1={val_metrics['f1']:.4f} | "
            f"Specificity={val_metrics['specificity']:.4f} | FPR={val_metrics['fpr']:.4f} | "
            f"CoastalNonPortFPR={val_metrics['coastal_non_port_fpr']:.4f} | {dt:.1f}s"
        )
        print(msg)

        rec = {
            "epoch": int(epoch),
            "loss": float(train_stats["loss_total"]),
            "loss_ce": float(train_stats["loss_ce"]),
            "loss_dice": float(train_stats["loss_dice"]),
            "loss_lovasz": float(train_stats["loss_lovasz"]),
            "loss_aux": float(train_stats["loss_aux"]),
            "loss_edge": float(train_stats["loss_edge"]),
            "iou_fg": float(iou_fg),
            "iou_bg": float(iou_bg),
            "miou": float(miou),
            "boundary_iou": float(boundary_iou),
            "precision": float(val_metrics["precision"]),
            "recall": float(val_metrics["recall"]),
            "f1": float(val_metrics["f1"]),
            "specificity": float(val_metrics["specificity"]),
            "fpr": float(val_metrics["fpr"]),
            "coastal_non_port_fpr": float(val_metrics["coastal_non_port_fpr"]),
            "tp": float(val_metrics["tp"]),
            "fp": float(val_metrics["fp"]),
            "tn": float(val_metrics["tn"]),
            "fn": float(val_metrics["fn"]),
            "coastal_non_port_fp": float(val_metrics["coastal_non_port_fp"]),
            "coastal_non_port_total": float(val_metrics["coastal_non_port_total"]),
        }
        if sweep is not None:
            rec.update({f"sweep_{k}": v for k, v in sweep.items()})
            print(f"   ↳ ThrSweep best_thr={sweep['best_thr']:.3f} | iou_fg={sweep['iou_fg_best_thr']:.4f} | miou={sweep['miou_best_thr']:.4f}")

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        save_checkpoint(cfg, model, ema, optimizer, scheduler, scaler, epoch, best_fg_iou,
                        os.path.join(cfg.save_dir, "last.ckpt"))

        if iou_fg > best_fg_iou:
            best_fg_iou = iou_fg
            save_checkpoint(cfg, model, ema, optimizer, scheduler, scaler, epoch, best_fg_iou,
                            os.path.join(cfg.save_dir, "best_fg.ckpt"))
            print(f"🌟 Best updated: IoU_fg={best_fg_iou:.4f}")

    print(f"\n✅ Done. Best foreground IoU: {best_fg_iou:.4f}")
    print(f"📄 Val metrics log: {metrics_path}")
    print(f"📄 Diagnostics: {os.path.join(cfg.save_dir, 'diagnostics_train.json')} / diagnostics_val.json")
    print(f"🖼️ Diagnostics vis: {os.path.join(cfg.save_dir, 'diagnostics_vis')}")


if __name__ == "__main__":
    main()

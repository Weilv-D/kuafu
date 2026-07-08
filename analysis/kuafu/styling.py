# -*- coding: utf-8 -*-
"""
KUAFU 统一视觉系统 — 调色板 / rcParams / 绘图工具
  Okabe-Ito 色盲安全色板 + 出版级科学图排版. 依赖 numpy, matplotlib.
"""
import os as _os
import numpy as np
import matplotlib
# 仅在无 GUI 环境强制 Agg；交互式会话可通过 MPLBACKEND 或 DISPLAY 覆盖
if not _os.environ.get("MPLBACKEND") and "DISPLAY" not in _os.environ:
    matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
# 路径
# ============================================================
# analysis/ 根目录 = 本文件上两级 (kuafu/styling.py -> analysis/)
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUTPUT = _os.path.join(ROOT, "output")

# ============================================================
# 调色板 — Okabe-Ito 色盲安全 + 精心调校
# ============================================================
C = {
    # 主色 (Okabe-Ito 基色, 微调饱和度和亮度)
    "blue":   "#0C6EC7",
    "orange": "#D46A00",
    "green":  "#007F5F",
    "yellow": "#DEB800",
    "sky":    "#4B9FD5",
    "verm":   "#C73E1D",
    "pink":   "#B7668C",
    "gray":   "#7B7B7B",
    # 辅助色
    "red":    "#C1272D",
    "teal":   "#008C8C",
    "navy":   "#1B3A5C",
    "cream":  "#FEFAF2",
    "slate":  "#4A5568",
    "gold":   "#B8860B",
    # 背景色
    "bg":        "#FAFBFC",
    "panel":     "#FFFFFF",
    "grid":      "#E5E7EB",
    "grid_major":"#D1D5DB",
}

# 别名 (兼容旧导入)
OI = C
PALETTE = C

# 语义色映射
SAFE   = C["green"]
CAUTION= C["orange"]
DANGER = C["verm"]
ACCENT = C["blue"]
HIGHLIGHT = C["red"]
NEUTRAL  = C["gray"]

# 数据序列色 (8色, 色盲安全)
SERIES = [C["blue"], C["verm"], C["green"], C["orange"],
          C["pink"], C["sky"], C["teal"], C["slate"]]

# D0 专用色映射
D0_COLORS = {
    58:  C["green"],     # 驻留态 - 绿色(安全)
    70:  C["teal"],      # 补偿区下界
    85:  C["sky"],
    100: C["blue"],      # 补偿区中段
    115: C["navy"],
    130: C["pink"],      # 峰值补偿
    145: C["orange"],    # 峰值扭矩
    150: C["verm"],      # 爬阶中段
    175: C["red"],
    207: C["gray"],      # 上限
}

# ============================================================
# 全局样式 — 出版级科学图
# ============================================================
rcParams.update({
    # 字体
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Microsoft YaHei", "SimHei", "Source Han Sans SC",
                          "DejaVu Sans", "Arial"],
    "font.size":        10,
    "font.weight":      "normal",
    # 标题层级
    "axes.titlesize":    "large",
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "axes.labelsize":    "medium",
    "axes.labelweight":  "normal",
    "axes.labelpad":     8,
    # 图形外观
    "axes.linewidth":    0.8,
    "axes.edgecolor":    "#555555",
    "axes.facecolor":    C["panel"],
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "axes.grid":         True,
    "axes.grid.axis":    "both",
    "axes.axisbelow":    True,
    "grid.color":        C["grid"],
    "grid.linewidth":    0.5,
    "grid.linestyle":    "-",
    "grid.alpha":        0.6,
    # 刻度
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  4.5,
    "ytick.major.size":  4.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.size":  2.5,
    "ytick.minor.size":  2.5,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.labelsize":   "small",
    "ytick.labelsize":   "small",
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    # 图例
    "legend.fontsize":   8,
    "legend.title_fontsize": 9,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#CCCCCC",
    "legend.fancybox":   True,
    "legend.borderpad":  0.6,
    "legend.labelspacing":0.35,
    "legend.handlelength":1.8,
    "legend.handletextpad":0.6,
    # 图形
    "figure.facecolor":  "white",
    "figure.edgecolor":  "white",
    "figure.dpi":        100,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "savefig.pad_inches":0.08,
    "savefig.facecolor": "white",
    # 数学排版
    "mathtext.fontset":  "dejavusans",
    "mathtext.default":  "regular",
    "axes.unicode_minus": False,
    # 线条
    "lines.linewidth":   1.8,
    "lines.markersize":  6,
    "lines.markeredgewidth":0.6,
    # 色条
    "image.cmap":        "viridis",
    "image.interpolation":"bilinear",
})

# ============================================================
# 绘图工具
# ============================================================
def savefig(fig, name):
    """保存图表到 analysis/output/ 目录."""
    _os.makedirs(OUTPUT, exist_ok=True)
    path = _os.path.join(OUTPUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"[图] output/{name}")

def ax_clean(ax):
    """双边轴线 + 浅色网格."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.8)
    ax.tick_params(which="both", colors="#444444")
    ax.set_facecolor(C["panel"])

def add_zone(ax, xlim, color, alpha=0.06, label=None):
    """在 x 轴上画半透明区域."""
    ax.axvspan(*xlim, facecolor=color, alpha=alpha, zorder=0)
    if label:
        ax.text(np.mean(xlim), ax.get_ylim()[1]*0.96, label,
                ha="center", fontsize=8, color=color, alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

def annotate_point(ax, x, y, text, offset=(8, 8), color=HIGHLIGHT,
                   arrowstyle="->", fontsize=8):
    """在数据点上标注释箭头."""
    ax.annotate(text, (x, y), xytext=offset, textcoords="offset points",
                fontsize=fontsize, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle=arrowstyle, color=color, lw=1.2,
                               connectionstyle="arc3,rad=0.2"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85, lw=0.8))

def set_title_style(ax, text):
    """统一标题样式."""
    ax.set_title(text, fontsize=11, fontweight="bold", pad=14, loc="center",
                 color="#222222")

def final_style(fig, suptitle=None):
    """收尾: 全局标题 + 紧凑布局."""
    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold",
                      color="#111111", y=0.995, x=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.985] if suptitle else [0, 0, 1, 1],
                     pad=1.5, h_pad=2.0, w_pad=2.0)

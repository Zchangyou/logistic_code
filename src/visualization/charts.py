"""
通用科研图表模块（阶段三）
General Research Charts Module (Phase 3)
"""
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.risk.factors import RiskCategory, RiskFactorRegistry, CATEGORY_NAMES
from src.visualization.style import apply_research_style, save_figure


# 风险类别颜色
CATEGORY_COLORS = {
    RiskCategory.MATERIAL_SHORTAGE:      "#E74C3C",  # 红
    RiskCategory.SUPPLIER_CONCENTRATION: "#9B59B6",  # 紫
    RiskCategory.LOGISTICS_DISRUPTION:   "#3498DB",  # 蓝
    RiskCategory.DEMAND_VOLATILITY:      "#F39C12",  # 橙
}


def plot_rpn_ranking(
    registry: RiskFactorRegistry,
    output_dir: str = "outputs/figures",
) -> plt.Figure:
    """绘制 FMEA RPN 排序水平柱状图 (F3-1)。

    Args:
        registry: RiskFactorRegistry 对象
        output_dir: 图表输出目录

    Returns:
        matplotlib Figure 对象
    """
    apply_research_style()

    factors = registry.get_rpn_ranking()  # 已按 RPN 降序
    names = [f"{f.name}\n({f.factor_id})" for f in factors]
    rpns = [f.rpn for f in factors]
    colors = [CATEGORY_COLORS[f.category] for f in factors]

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(
        range(len(factors)),
        rpns,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    # 数值标注
    for bar, rpn in zip(bars, rpns):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rpn:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()  # 最高 RPN 在顶部

    ax.set_xlabel("风险优先数 RPN (Risk Priority Number)", fontsize=11)
    ax.set_title(
        "风险优先数排序 (FMEA Risk Priority Number Ranking)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    # 图例
    legend_patches = [
        mpatches.Patch(
            color=CATEGORY_COLORS[cat],
            label=f"{CATEGORY_NAMES[cat][0]} ({CATEGORY_NAMES[cat][1]})",
        )
        for cat in RiskCategory
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )

    ax.set_xlim(0, max(rpns) * 1.15)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, "F3-1_rpn_ranking", output_dir)
    print("  [F3-1] RPN排序图已生成")
    return fig


def plot_fmea_matrix(
    registry: RiskFactorRegistry,
    output_dir: str = "outputs/figures",
) -> plt.Figure:
    """绘制 FMEA 风险矩阵散点气泡图 (F3-2)。

    Args:
        registry: RiskFactorRegistry 对象
        output_dir: 图表输出目录

    Returns:
        matplotlib Figure 对象
    """
    apply_research_style()

    factors = registry.get_all_factors()

    occurrence = [f.occurrence_prob * 10 for f in factors]
    severity = [f.severity for f in factors]
    rpn = [f.rpn for f in factors]
    colors = [CATEGORY_COLORS[f.category] for f in factors]
    fids = [f.factor_id for f in factors]

    # 气泡大小：RPN / 5 映射到面积
    sizes = [r / 5 * 40 for r in rpn]

    fig, ax = plt.subplots(figsize=(11, 8))

    sc = ax.scatter(
        occurrence, severity,
        s=sizes,
        c=colors,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

    # 四象限分割线
    ax.axvline(x=5, color="#AAAAAA", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.axhline(y=7, color="#AAAAAA", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

    # 象限标注（右上角最危险）
    quad_style = dict(fontsize=8, color="#888888", alpha=0.85, style="italic")
    ax.text(8.5, 9.6, "高风险区\nHigh Risk", ha="center", **quad_style)
    ax.text(2.0, 9.6, "低概率高危\nLow-prob High-sev", ha="center", **quad_style)
    ax.text(8.5, 3.0, "高概率低危\nHigh-prob Low-sev", ha="center", **quad_style)
    ax.text(2.0, 3.0, "低风险区\nLow Risk", ha="center", **quad_style)

    # 标注 Top-5 RPN 因素
    sorted_factors = sorted(factors, key=lambda f: f.rpn, reverse=True)
    top5_ids = {f.factor_id for f in sorted_factors[:5]}
    for i, f in enumerate(factors):
        if f.factor_id in top5_ids:
            ax.annotate(
                f.factor_id,
                xy=(occurrence[i], severity[i]),
                xytext=(occurrence[i] + 0.3, severity[i] + 0.25),
                fontsize=8,
                color="#333333",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
            )

    ax.set_xlabel("发生概率 (Occurrence) [0–10]", fontsize=11)
    ax.set_ylabel("影响严重度 (Severity) [0–10]", fontsize=11)
    ax.set_title(
        "FMEA风险矩阵 (FMEA Risk Matrix)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)

    # 图例
    legend_patches = [
        mpatches.Patch(
            color=CATEGORY_COLORS[cat],
            label=f"{CATEGORY_NAMES[cat][0]} ({CATEGORY_NAMES[cat][1]})",
        )
        for cat in RiskCategory
    ]
    # 气泡大小说明
    size_legend = [
        plt.scatter([], [], s=rpn_val / 5 * 40, c="#888888", alpha=0.6,
                    label=f"RPN≈{rpn_val}")
        for rpn_val in [100, 300, 500]
    ]
    ax.legend(
        handles=legend_patches + size_legend,
        loc="lower left",
        fontsize=8.5,
        framealpha=0.9,
        ncol=1,
    )

    plt.tight_layout()
    save_figure(fig, "F3-2_fmea_matrix", output_dir)
    print("  [F3-2] FMEA风险矩阵图已生成")
    return fig

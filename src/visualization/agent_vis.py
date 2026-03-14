"""
AI智能体模块可视化
Agent Module Visualization

产出图表：
- F5-1 多期风险趋势折线图：关键节点6期风险评分变化 + 预警阈值标注
- F5-2 策略推荐效果对比图：不同策略风险降低效果横向对比
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.visualization.style import apply_research_style, save_figure

def setup_matplotlib():
    apply_research_style()

matplotlib.use("Agg")


# ------------------------------------------------------------------ #
# F5-1：多期风险趋势折线图
# ------------------------------------------------------------------ #
def plot_risk_trend(
    node_risk_trend: Dict[str, List[float]],
    warning_events: List[Dict],
    first_warning_period: Dict[str, int],
    periods: int = 6,
    output_prefix: str = "outputs/figures/F5-1_risk_trend",
) -> None:
    """
    绘制关键节点6期风险评分变化趋势图，标注预警阈值和触发点。

    Args:
        node_risk_trend: {node_id: [score_p1, score_p2, ..., score_p6]}
        warning_events: 预警事件列表（dict格式）
        first_warning_period: {node_id: 首次预警期次}
        periods: 期数
        output_prefix: 输出路径前缀
    """
    setup_matplotlib()

    # 选取风险最高的前6个节点
    avg_scores = {nid: np.mean(scores) for nid, scores in node_risk_trend.items() if scores}
    sorted_nodes = sorted(avg_scores, key=avg_scores.get, reverse=True)[:6]

    period_labels = [f"Q{i}" for i in range(1, periods + 1)]
    x = np.arange(1, periods + 1)

    # 配色方案
    colors = ["#e74c3c", "#e67e22", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    fig, ax = plt.subplots(figsize=(12, 6))

    node_cn_names = {
        "T3-SI": "芯片晶圆\nChip Wafer",
        "T3-RE": "稀土材料\nRare Earth",
        "T2-ECU": "ECU控制单元\nECU",
        "T2-SN": "传感器模组\nSensor",
        "T1-E": "电子电气集成\nElec. Sys.",
        "T2-E2": "涡轮增压器\nTurbocharger",
        "T3-CU": "铜材\nCopper",
        "OEM": "总装厂\nOEM",
    }

    for i, node_id in enumerate(sorted_nodes):
        scores = node_risk_trend.get(node_id, [0.0] * periods)
        # 补齐到periods长度
        if len(scores) < periods:
            scores = scores + [scores[-1]] * (periods - len(scores))
        scores = scores[:periods]

        label = node_cn_names.get(node_id, node_id)
        ax.plot(
            x, scores,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2.0,
            marker="o",
            markersize=6,
            label=label,
            zorder=3,
        )

        # 标注首次预警点
        first_p = first_warning_period.get(node_id)
        if first_p and 1 <= first_p <= periods:
            score_at_first = scores[first_p - 1]
            ax.annotate(
                "!",
                xy=(first_p, score_at_first),
                xytext=(first_p + 0.15, score_at_first + 0.015),
                fontsize=11, fontweight="bold",
                color=colors[i % len(colors)],
                zorder=5,
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="white",
                          edgecolor=colors[i % len(colors)], linewidth=1.0),
            )

    # 预警阈值线
    ax.axhline(y=0.35, color="#f39c12", linewidth=1.5, linestyle="--", alpha=0.8, zorder=2)
    ax.axhline(y=0.45, color="#e67e22", linewidth=1.5, linestyle="--", alpha=0.8, zorder=2)
    ax.axhline(y=0.55, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.8, zorder=2)

    # 阈值标签
    ax.text(periods + 0.05, 0.35, "关注\nWatch", fontsize=8, color="#f39c12",
            va="center", ha="left")
    ax.text(periods + 0.05, 0.45, "警告\nWarning", fontsize=8, color="#e67e22",
            va="center", ha="left")
    ax.text(periods + 0.05, 0.55, "严重\nCritical", fontsize=8, color="#e74c3c",
            va="center", ha="left")

    # 背景分区
    ax.axhspan(0.35, 0.45, alpha=0.05, color="#f39c12", zorder=1)
    ax.axhspan(0.45, 0.55, alpha=0.07, color="#e67e22", zorder=1)
    ax.axhspan(0.55, 1.0,  alpha=0.09, color="#e74c3c", zorder=1)

    ax.set_xlim(0.5, periods + 0.8)
    ax.set_ylim(0.0, max(0.75, max(max(s) for s in node_risk_trend.values() if s) + 0.05))
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, fontsize=10)
    ax.set_xlabel("运营周期  /  Operation Period", fontsize=11)
    ax.set_ylabel("综合风险评分  /  Composite Risk Score", fontsize=11)
    ax.set_title(
        "关键节点多期风险评分趋势\nMulti-Period Risk Score Trend of Key Nodes",
        fontsize=13, fontweight="bold", pad=12,
    )

    # 图例
    legend = ax.legend(
        loc="upper left", fontsize=8.5, framealpha=0.9,
        ncol=2, bbox_to_anchor=(0.01, 0.99),
    )

    # 添加阈值图例
    watch_patch  = mpatches.Patch(color="#f39c12", alpha=0.6, label="关注阈值 Watch (0.35)")
    warn_patch   = mpatches.Patch(color="#e67e22", alpha=0.6, label="警告阈值 Warning (0.45)")
    crit_patch   = mpatches.Patch(color="#e74c3c", alpha=0.6, label="严重阈值 Critical (0.55)")
    ax2_legend = ax.legend(
        handles=[watch_patch, warn_patch, crit_patch],
        loc="lower right", fontsize=8, framealpha=0.9,
    )
    ax.add_artist(legend)

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_dir = os.path.dirname(output_prefix) or "outputs/figures"
    out_name = os.path.basename(output_prefix)
    save_figure(fig, out_name, out_dir)
    plt.close(fig)
    print(f"[F5-1] 多期风险趋势图已保存至 {output_prefix}.pdf/png")


# ------------------------------------------------------------------ #
# F5-2：策略推荐效果对比图
# ------------------------------------------------------------------ #
def plot_strategy_comparison(
    comparison_data: List[Dict],
    node_name: str,
    current_risk: float,
    output_prefix: str = "outputs/figures/F5-2_strategy_comparison",
) -> None:
    """
    绘制策略推荐效果对比图（分组柱状图）。

    Args:
        comparison_data: 来自 StrategyRecommender.compare_strategies_for_node() 的列表
        node_name: 目标节点名称
        current_risk: 当前风险评分
        output_prefix: 输出路径前缀
    """
    setup_matplotlib()

    if not comparison_data:
        print("[WARNING] 策略对比数据为空，跳过 F5-2")
        return

    names = [d["name"] for d in comparison_data]
    names_en = [d["name_en"] for d in comparison_data]
    risk_reductions = [d["risk_reduction_pct"] for d in comparison_data]
    risk_afters = [d["risk_after"] for d in comparison_data]
    lead_times = [d["lead_time_days"] for d in comparison_data]
    feasibilities = [d["feasibility"] * 100 for d in comparison_data]
    cost_levels = [d["cost_level"] for d in comparison_data]

    n = len(names)
    x = np.arange(n)
    bar_width = 0.22

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左图：风险降低效果对比 ---
    colors_main = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71", "#9b59b6"][:n]

    bars1 = ax1.bar(
        x - bar_width / 2, risk_reductions,
        width=bar_width * 1.5, color=colors_main, alpha=0.85,
        edgecolor="white", linewidth=0.8, label="风险降低率 (%)\nRisk Reduction (%)",
    )

    # 叠加：处置后风险评分（右Y轴）
    ax1b = ax1.twinx()
    ax1b.plot(
        x, risk_afters, color="#2c3e50", marker="D", markersize=8,
        linewidth=2.0, linestyle="-", label="处置后风险评分\nPost-treatment Score",
        zorder=5,
    )
    ax1b.axhline(y=current_risk, color="#e74c3c", linewidth=1.5, linestyle="--",
                 alpha=0.7, label=f"当前风险 {current_risk:.3f}\nCurrent Risk")
    ax1b.set_ylabel("风险评分  /  Risk Score", fontsize=10)
    ax1b.set_ylim(0, max(current_risk * 1.2, 0.8))

    # 数值标注
    for bar, val in zip(bars1, risk_reductions):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax1.set_xticks(x)
    tick_labels = [f"{n[:6]}\n{e[:12]}" for n, e in zip(names, names_en)]
    ax1.set_xticklabels(tick_labels, fontsize=8)
    ax1.set_ylabel("风险降低率 (%)  /  Risk Reduction (%)", fontsize=10)
    ax1.set_title(
        f"策略风险降低效果对比\nStrategy Risk Reduction Comparison\n({node_name})",
        fontsize=11, fontweight="bold",
    )
    ax1.set_ylim(0, max(risk_reductions) * 1.3 + 5)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.5)

    # --- 右图：可行性 vs 实施周期 气泡图 ---
    bubble_sizes = [200 + rr * 8 for rr in risk_reductions]
    scatter = ax2.scatter(
        lead_times, feasibilities,
        s=bubble_sizes, c=cost_levels,
        cmap="RdYlGn_r", alpha=0.80,
        edgecolors="black", linewidths=0.8,
        vmin=1, vmax=5, zorder=4,
    )

    for i, name in enumerate(names):
        ax2.annotate(
            name[:8], (lead_times[i], feasibilities[i]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7.5, ha="left",
        )

    cbar = plt.colorbar(scatter, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("成本等级  /  Cost Level", fontsize=9)
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(["很低", "低", "中", "高", "很高"])

    ax2.set_xlabel("实施周期 (天)  /  Lead Time (days)", fontsize=10)
    ax2.set_ylabel("可行性评分 (%)  /  Feasibility Score (%)", fontsize=10)
    ax2.set_title(
        "策略可行性 vs 实施周期\nFeasibility vs Lead Time\n(气泡大小 = 风险降低率)",
        fontsize=11, fontweight="bold",
    )
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 象限注释
    lt_mid = np.median(lead_times)
    fs_mid = np.median(feasibilities)
    ax2.axvline(lt_mid, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(fs_mid, color="gray", linestyle=":", alpha=0.5)
    ax2.text(lt_mid * 0.2, fs_mid * 1.02, "快速高效\nQuick & Effective",
             fontsize=7, color="green", alpha=0.7)
    ax2.text(lt_mid * 1.3, fs_mid * 0.92, "长周期低效\nSlow & Less Effective",
             fontsize=7, color="red", alpha=0.7)

    plt.tight_layout(pad=2.0)

    out_dir = os.path.dirname(output_prefix) or "outputs/figures"
    out_name = os.path.basename(output_prefix)
    save_figure(fig, out_name, out_dir)
    plt.close(fig)
    print(f"[F5-2] 策略对比图已保存至 {output_prefix}.pdf/png")

"""
风险传播与综合评估可视化模块
Risk Propagation and Assessment Visualization Module
"""
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.visualization.style import apply_research_style, save_figure, TIER_COLORS, RISK_COLORS
from src.risk.propagation import SIRResult, NodeState
from src.risk.assessment import RiskAssessmentResult


# SIR 颜色
SIR_COLORS = {
    NodeState.SUSCEPTIBLE: "#2ECC71",   # 绿
    NodeState.INFECTED:    "#E74C3C",   # 红
    NodeState.RECOVERED:   "#3498DB",   # 蓝
}

SIR_LINE_COLORS = {
    "S": "#2ECC71",
    "I": "#E74C3C",
    "R": "#3498DB",
}


def _build_tier_layout(network) -> Dict[str, tuple]:
    """构建分层布局：tier作为y轴（tier 0在顶，tier 3在底），x轴均匀分布。

    Args:
        network: SupplyChainNetwork 对象

    Returns:
        节点ID → (x, y) 坐标字典
    """
    tier_nodes: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
    for node in network.get_all_nodes():
        tier_nodes[node.tier].append(node.node_id)

    pos = {}
    y_positions = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}

    for tier, nodes in tier_nodes.items():
        n = len(nodes)
        for i, nid in enumerate(sorted(nodes)):
            x = (i - (n - 1) / 2) * 1.5
            pos[nid] = (x, y_positions[tier])

    return pos


def plot_sir_dynamics(sir_result: SIRResult, output_dir: str) -> None:
    """绘制SIR传播动态曲线 (F4-1)。

    Args:
        sir_result: S1芯片场景的 SIRResult 对象
        output_dir: 图表输出目录
    """
    apply_research_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = sir_result.time_steps
    ax.plot(steps, sir_result.s_counts, color=SIR_LINE_COLORS["S"],
            linewidth=2.5, label="易感节点 S (Susceptible)", marker="o", markersize=3)
    ax.plot(steps, sir_result.i_counts, color=SIR_LINE_COLORS["I"],
            linewidth=2.5, label="感染节点 I (Infected)", marker="s", markersize=3)
    ax.plot(steps, sir_result.r_counts, color=SIR_LINE_COLORS["R"],
            linewidth=2.5, label="恢复节点 R (Recovered)", marker="^", markersize=3)

    # 标注峰值
    peak_step = int(np.argmax(sir_result.i_counts))
    peak_val = sir_result.max_affected_count
    ax.axvline(x=peak_step, color=SIR_LINE_COLORS["I"], linestyle="--",
               alpha=0.6, linewidth=1.5)
    ax.annotate(
        f"峰值 {peak_val} 节点\n(Peak: Step {peak_step})",
        xy=(peak_step, peak_val),
        xytext=(peak_step + 2, peak_val + 0.5),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
        color="gray",
    )

    ax.set_xlabel("时间步 (Time Steps)", fontsize=11)
    ax.set_ylabel("节点数量 (Number of Nodes)", fontsize=11)
    ax.set_title(
        f"芯片停供场景风险传播动态\n(SIR Propagation - {sir_result.scenario_name_en})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # 添加统计注释
    stats_text = (
        f"最大感染数: {sir_result.max_affected_count}\n"
        f"影响持续: {sir_result.impact_duration} 步\n"
        f"恢复时间: {sir_result.recovery_time} 步"
    )
    ax.text(
        0.98, 0.97, stats_text, transform=ax.transAxes,
        va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    fig.tight_layout()
    save_figure(fig, "F4-1_sir_dynamics", output_dir)
    plt.close(fig)
    print("  F4-1 SIR传播动态曲线 已生成")


def plot_cascade_snapshots(network, sir_result: SIRResult, output_dir: str) -> None:
    """绘制级联效应传播快照 (F4-2)，展示 t=0,5,10,20 四个时刻的网络状态。

    Args:
        network: SupplyChainNetwork 对象
        sir_result: SIRResult 对象
        output_dir: 图表输出目录
    """
    import networkx as nx

    apply_research_style()

    snapshots_t = [0, 5, 10, 20]
    fig, axes = plt.subplots(1, 4, figsize=(20, 7))
    fig.suptitle(
        "级联效应传播快照 (Cascade Propagation Snapshots)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    G = network.get_graph()
    pos = _build_tier_layout(network)

    # 过滤掉不在pos中的节点（防止孤立节点）
    valid_nodes = [n for n in G.nodes() if n in pos]
    G_sub = G.subgraph(valid_nodes)

    max_t = len(sir_result.node_states_history) - 1

    for ax_idx, t in enumerate(snapshots_t):
        ax = axes[ax_idx]
        t_actual = min(t, max_t)
        snapshot = sir_result.node_states_history[t_actual]

        node_colors = []
        for node in G_sub.nodes():
            state = snapshot.get(node, NodeState.SUSCEPTIBLE)
            node_colors.append(SIR_COLORS[state])

        nx.draw_networkx_nodes(
            G_sub, pos, ax=ax,
            node_color=node_colors,
            node_size=300,
            alpha=0.9,
        )
        nx.draw_networkx_edges(
            G_sub, pos, ax=ax,
            edge_color="#AAAAAA",
            arrows=True,
            arrowsize=10,
            width=0.8,
            alpha=0.5,
            connectionstyle="arc3,rad=0.05",
        )
        nx.draw_networkx_labels(
            G_sub, pos, ax=ax,
            font_size=5,
            font_color="black",
        )

        # 统计
        n_i = sum(1 for st in snapshot.values() if st == NodeState.INFECTED)
        n_r = sum(1 for st in snapshot.values() if st == NodeState.RECOVERED)

        ax.set_title(f"t = {t}\nI={n_i}, R={n_r}", fontsize=10, fontweight="bold")
        ax.axis("off")

        # 层级标签
        y_labels = {3.0: "T3原材料", 2.0: "T2零部件", 1.0: "T1集成商", 0.0: "T0总装"}
        for y_val, label in y_labels.items():
            ax.text(-13, y_val, label, va="center", ha="left", fontsize=6.5,
                    color="gray", style="italic")

    # 图例
    legend_patches = [
        mpatches.Patch(color=SIR_COLORS[NodeState.SUSCEPTIBLE], label="易感 S (Susceptible)"),
        mpatches.Patch(color=SIR_COLORS[NodeState.INFECTED],    label="感染 I (Infected)"),
        mpatches.Patch(color=SIR_COLORS[NodeState.RECOVERED],   label="恢复 R (Recovered)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    save_figure(fig, "F4-2_cascade_snapshots", output_dir)
    plt.close(fig)
    print("  F4-2 级联效应传播快照 已生成")


def plot_scenario_comparison(comparison_df: pd.DataFrame, output_dir: str) -> None:
    """绘制三类扰动场景对比图 (F4-3)。

    Args:
        comparison_df: ScenarioRunner.compare_scenarios() 返回的 DataFrame
        output_dir: 图表输出目录
    """
    apply_research_style()

    # 只取 S1, S3, S4
    scenario_ids = ["S1", "S3", "S4"]
    labels_cn = {
        "S1": "芯片停供\n(Chip Shortage)",
        "S3": "华东中断\n(East China)",
        "S4": "需求冲击\n(Demand Shock)",
    }
    metrics = ["impact_range", "impact_depth", "impact_duration"]
    metric_labels = {
        "impact_range":    "影响范围\n(Affected Nodes)",
        "impact_depth":    "影响深度\n(Tier Depth)",
        "impact_duration": "持续步数\n(Duration Steps)",
    }
    metric_colors = ["#E74C3C", "#F39C12", "#3498DB"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_ids))
    width = 0.25
    offsets = [-width, 0, width]

    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        values = []
        for sid in scenario_ids:
            if sid in comparison_df.index:
                values.append(float(comparison_df.loc[sid, metric]))
            else:
                values.append(0.0)
        bars = ax.bar(
            x + offsets[i], values, width,
            label=metric_labels[metric],
            color=color, alpha=0.85, edgecolor="white",
        )
        # 数值标签
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([labels_cn[sid] for sid in scenario_ids], fontsize=10)
    ax.set_ylabel("指标数值 (Metric Value)", fontsize=11)
    ax.set_title(
        "三类扰动场景传播效应对比\n(Risk Propagation Comparison - Three Scenarios)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_figure(fig, "F4-3_scenario_comparison", output_dir)
    plt.close(fig)
    print("  F4-3 三类扰动场景对比图 已生成")


def plot_risk_heatmap(
    network,
    assessment_results: List[RiskAssessmentResult],
    output_dir: str,
) -> None:
    """绘制供应链风险热力图 (F4-4)，节点按风险等级着色。

    Args:
        network: SupplyChainNetwork 对象
        assessment_results: RiskAssessmentResult 列表
        output_dir: 图表输出目录
    """
    import networkx as nx

    apply_research_style()

    G = network.get_graph()
    pos = _build_tier_layout(network)
    valid_nodes = [n for n in G.nodes() if n in pos]
    G_sub = G.subgraph(valid_nodes)

    # 构建风险映射
    risk_map = {r.node_id: r for r in assessment_results}

    fig, ax = plt.subplots(figsize=(16, 10))

    # 节点颜色和大小
    node_colors = []
    node_sizes = []
    for node in G_sub.nodes():
        result = risk_map.get(node)
        if result:
            node_colors.append(result.risk_color)
            node_sizes.append(300 + result.composite_score * 800)
        else:
            node_colors.append("#2ECC71")
            node_sizes.append(300)

    nx.draw_networkx_nodes(
        G_sub, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="white",
        linewidths=1.5,
    )
    nx.draw_networkx_edges(
        G_sub, pos, ax=ax,
        edge_color="#CCCCCC",
        arrows=True,
        arrowsize=12,
        width=1.0,
        alpha=0.6,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_labels(
        G_sub, pos, ax=ax,
        font_size=7,
        font_color="black",
        font_weight="bold",
    )

    # 标注高风险节点分数
    high_risk = [r for r in assessment_results if r.risk_level in ("high", "medium")]
    for result in high_risk[:8]:  # 最多标注8个
        node = result.node_id
        if node in pos:
            x, y = pos[node]
            ax.annotate(
                f"{result.composite_score:.2f}",
                xy=(x, y),
                xytext=(x + 0.3, y + 0.2),
                fontsize=7,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    # 层级标签
    tier_label_y = {3.0: "T3\n原材料", 2.0: "T2\n零部件", 1.0: "T1\n集成商", 0.0: "T0\n总装"}
    for y_val, label in tier_label_y.items():
        ax.text(-14, y_val, label, va="center", ha="center", fontsize=9,
                color="gray", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", alpha=0.8))

    ax.set_title(
        "供应链综合风险评估热力图\n(Supply Chain Risk Heatmap)",
        fontsize=14, fontweight="bold",
    )

    # 图例
    legend_patches = [
        mpatches.Patch(color=RISK_COLORS["high"],   label="高风险 High (≥0.65)"),
        mpatches.Patch(color=RISK_COLORS["medium"], label="中风险 Medium (0.45-0.65)"),
        mpatches.Patch(color=RISK_COLORS["low"],    label="低风险 Low (0.25-0.45)"),
        mpatches.Patch(color=RISK_COLORS["safe"],   label="安全 Safe (<0.25)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=9, framealpha=0.9)
    ax.axis("off")

    fig.tight_layout()
    save_figure(fig, "F4-4_risk_heatmap", output_dir)
    plt.close(fig)
    print("  F4-4 风险热力图 已生成")


def plot_bayesian_dag(
    bn_model,
    inference_result: Dict[str, float],
    output_dir: str,
) -> None:
    """绘制贝叶斯网络因果图 (F4-5)。

    Args:
        bn_model: SupplyChainBayesianNet 对象
        inference_result: 推断结果字典 {节点名: P(node=1)}
        output_dir: 图表输出目录
    """
    import networkx as nx
    import matplotlib.colors as mcolors
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    apply_research_style()

    edges = bn_model.get_structure_for_visualization()
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # 节点布局（手动设定，DAG层级从左到右）
    node_pos = {
        "MaterialShortage":      (0, 2),
        "SupplierConcentration":  (0, 0),
        "LogisticsDisruption":    (0, -2),
        "CapacityShortage":       (3, 1),
        "DeliveryDelay":          (6, 0),
        "ProductionHalt":         (9, 0),
    }

    # 节点中文标签
    node_labels = {
        "MaterialShortage":      "材料短缺\nMaterial Shortage",
        "SupplierConcentration":  "供应商集中\nSupplier Concentration",
        "LogisticsDisruption":    "物流中断\nLogistics Disruption",
        "CapacityShortage":       "产能不足\nCapacity Shortage",
        "DeliveryDelay":          "交货延迟\nDelivery Delay",
        "ProductionHalt":         "生产停工\nProduction Halt",
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    # 节点颜色根据风险概率（蓝→红渐变）
    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=0, vmax=1)

    node_colors = []
    for node in G.nodes():
        prob = inference_result.get(node, 0.3)
        node_colors.append(cmap(norm(prob)))

    # 绘制节点
    nx.draw_networkx_nodes(
        G, node_pos, ax=ax,
        node_color=node_colors,
        node_size=2500,
        alpha=0.9,
        edgecolors="white",
        linewidths=2,
    )
    # 绘制边
    nx.draw_networkx_edges(
        G, node_pos, ax=ax,
        edge_color="#555555",
        arrows=True,
        arrowsize=20,
        width=2,
        connectionstyle="arc3,rad=0.0",
        min_source_margin=30,
        min_target_margin=30,
    )
    # 标签
    nx.draw_networkx_labels(
        G, node_pos, labels=node_labels, ax=ax,
        font_size=8,
        font_color="black",
        font_weight="bold",
    )

    # 在节点旁标注概率
    for node, (x, y) in node_pos.items():
        prob = inference_result.get(node, 0.0)
        ax.text(x, y - 0.55, f"P(risk)={prob:.2f}",
                ha="center", va="top", fontsize=8,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # 颜色条
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("P(风险=1) Risk Probability", fontsize=9)

    ax.set_title(
        "风险因素贝叶斯因果图（芯片短缺场景）\n(Bayesian Risk Causality Network - Chip Shortage)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")

    fig.tight_layout()
    save_figure(fig, "F4-5_bayesian_dag", output_dir)
    plt.close(fig)
    print("  F4-5 贝叶斯网络因果图 已生成")


def plot_fuzzy_radar(
    fuzzy_results: pd.DataFrame,
    node_ids: List[str],
    output_dir: str,
) -> None:
    """绘制关键节点模糊综合评价雷达图 (F4-6)。

    Args:
        fuzzy_results: FuzzyRiskEvaluator.evaluate_all_nodes() 返回的 DataFrame
        node_ids: 要绘制的节点ID列表（建议 T3-SI, T2-ECU, T3-RE, OEM）
        output_dir: 图表输出目录
    """
    apply_research_style()

    dimensions = [
        "capacity_utilization",
        "inventory_level",
        "on_time_delivery",
        "logistics_reliability",
        "supplier_concentration",
    ]
    dim_labels = [
        "产能利用率\nCapacity\nUtilization",
        "库存水平\nInventory\nLevel",
        "准时交货率\nOn-time\nDelivery",
        "物流可靠性\nLogistics\nReliability",
        "供应商集中度\nSupplier\nConcentration",
    ]

    n_dim = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    colors = ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    fuzzy_map = fuzzy_results.set_index("node_id") if "node_id" in fuzzy_results.columns else fuzzy_results

    plotted = []
    for i, node_id in enumerate(node_ids):
        if node_id not in fuzzy_map.index:
            continue

        row = fuzzy_map.loc[node_id]
        values = []
        for dim in dimensions:
            if dim in row:
                v = float(row[dim])
                # 对需要反向的维度（高值=高风险）进行转换用于显示
                if dim in ("inventory_level", "on_time_delivery", "logistics_reliability"):
                    # 这些维度在DataFrame中已是原始值，需要反转显示
                    v = 1.0 - float(np.clip(v, 0.0, 1.0)) if dim != "inventory_level" else 1.0 - v / 90.0
                    v = float(np.clip(v, 0.0, 1.0))
            else:
                v = 0.3
            values.append(v)

        values += values[:1]  # 闭合
        color = colors[i % len(colors)]
        ax.plot(angles, values, color=color, linewidth=2, label=node_id)
        ax.fill(angles, values, color=color, alpha=0.15)
        plotted.append(node_id)

    # 坐标轴设置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7, color="gray")
    ax.grid(True, alpha=0.3)

    ax.set_title(
        "关键节点模糊综合评价雷达图\n(Fuzzy Comprehensive Evaluation - Key Nodes)",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1), fontsize=10)

    fig.tight_layout()
    save_figure(fig, "F4-6_fuzzy_radar", output_dir)
    plt.close(fig)
    print("  F4-6 模糊评价雷达图 已生成")


def plot_accuracy_report(confusion: Dict, output_dir: str) -> None:
    """绘制准确率验证混淆矩阵图 (F4-7)。

    Args:
        confusion: get_confusion_matrix() 返回的混淆矩阵字典
        output_dir: 图表输出目录
    """
    import seaborn as sns

    apply_research_style()

    tp = confusion.get("tp", 0)
    fp = confusion.get("fp", 0)
    tn = confusion.get("tn", 0)
    fn = confusion.get("fn", 0)
    accuracy = confusion.get("accuracy", 0.0)
    fnr = confusion.get("false_negative_rate", 0.0)
    fpr = confusion.get("false_positive_rate", 0.0)

    # 混淆矩阵数组
    # 行=预测，列=实际
    cm_array = np.array([[tp, fn], [fp, tn]])

    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.4)

    # 主图：混淆矩阵热力图
    ax_cm = fig.add_subplot(gs[0:2, 0])
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm,
        xticklabels=["实际风险\nActual Risk", "实际正常\nActual Normal"],
        yticklabels=["预测风险\nPred Risk", "预测正常\nPred Normal"],
        annot_kws={"size": 14, "weight": "bold"},
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
    )
    ax_cm.set_title(
        "混淆矩阵\n(Confusion Matrix)",
        fontsize=12, fontweight="bold",
    )
    ax_cm.set_xlabel("实际标签 (Actual Label)", fontsize=10)
    ax_cm.set_ylabel("预测标签 (Predicted Label)", fontsize=10)

    # 右侧：统计指标文本
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.axis("off")

    stats_lines = [
        ("准确率 (Accuracy)",        f"{accuracy:.1%}"),
        ("假阴性率 (False Neg Rate)", f"{fnr:.1%}"),
        ("假阳性率 (False Pos Rate)", f"{fpr:.1%}"),
        ("", ""),
        ("真阳性 TP", f"{tp}"),
        ("假阳性 FP", f"{fp}"),
        ("真阴性 TN", f"{tn}"),
        ("假阴性 FN", f"{fn}"),
    ]

    y_start = 0.95
    for label, value in stats_lines:
        if not label:
            y_start -= 0.08
            continue
        color = "#E74C3C" if "False" in label or "假" in label else "#2C3E50"
        if "准确率" in label:
            color = "#27AE60" if accuracy >= 0.88 else "#E74C3C"
        ax_stats.text(0.05, y_start, label, transform=ax_stats.transAxes,
                      fontsize=10, color=color, va="top")
        ax_stats.text(0.75, y_start, value, transform=ax_stats.transAxes,
                      fontsize=10, color=color, va="top", fontweight="bold", ha="right")
        y_start -= 0.12

    # 准确率达标标注
    target_label = "目标 ≥ 88%" if accuracy >= 0.88 else "未达目标 88%"
    target_color = "#27AE60" if accuracy >= 0.88 else "#E74C3C"
    ax_stats.text(0.5, 0.05, target_label,
                  transform=ax_stats.transAxes,
                  fontsize=11, fontweight="bold", ha="center",
                  color=target_color,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0FFF0" if accuracy >= 0.88 else "#FFF0F0"))

    # 右下：精确率、召回率等
    ax_extra = fig.add_subplot(gs[1, 1])
    ax_extra.axis("off")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    extra_stats = [
        ("精确率 Precision",  f"{precision:.1%}"),
        ("召回率 Recall",     f"{recall:.1%}"),
        ("F1分数 F1-Score",   f"{f1:.1%}"),
    ]
    y_start = 0.80
    for label, value in extra_stats:
        ax_extra.text(0.05, y_start, label, transform=ax_extra.transAxes, fontsize=10, va="top")
        ax_extra.text(0.75, y_start, value, transform=ax_extra.transAxes, fontsize=10,
                      va="top", fontweight="bold", ha="right", color="#2980B9")
        y_start -= 0.25

    fig.suptitle(
        "风险识别准确率验证报告\n(Risk Detection Accuracy Validation Report)",
        fontsize=13, fontweight="bold",
    )

    save_figure(fig, "F4-7_accuracy_report", output_dir)
    plt.close(fig)
    print("  F4-7 准确率验证图 已生成")

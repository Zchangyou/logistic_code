"""
供应链网络可视化模块
Supply Chain Network Visualization Module
"""
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from .style import apply_research_style, save_figure, TIER_COLORS, TIER_NAMES


def _get_tier_color(tier: int) -> str:
    """获取层级对应的颜色。"""
    return TIER_COLORS.get(tier, '#95A5A6')


def create_f2_1_centrality_distribution(
    topology_report: Dict[str, Any],
    output_dir: str = 'outputs/figures',
) -> plt.Figure:
    """F2-1：节点中心性分布图（度中心性与介数中心性 Top-15 柱状图）。

    Args:
        topology_report: TopologyAnalyzer.get_report() 的输出
        output_dir: 图表输出目录

    Returns:
        matplotlib Figure 对象
    """
    apply_research_style()
    nodes_data = topology_report.get('nodes', {})
    if not nodes_data:
        raise ValueError("topology_report 中无节点数据")

    # 构建排序列表
    records = []
    for nid, metrics in nodes_data.items():
        records.append({
            'node_id': nid,
            'label': metrics.get('name_en', nid),
            'tier': metrics.get('tier', 3),
            'degree_centrality': metrics.get('degree_centrality', 0.0),
            'betweenness_centrality': metrics.get('betweenness_centrality', 0.0),
        })

    top_n = min(15, len(records))
    top_degree = sorted(records, key=lambda x: x['degree_centrality'], reverse=True)[:top_n]
    top_between = sorted(records, key=lambda x: x['betweenness_centrality'], reverse=True)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        '节点中心性分布 (Node Centrality Distribution)',
        fontsize=14, fontweight='bold', y=1.01
    )

    def _plot_bar(ax: plt.Axes, data: List[Dict], value_key: str, title_cn: str, title_en: str) -> None:
        labels = [d['label'] for d in data]
        values = [d[value_key] for d in data]
        colors = [_get_tier_color(d['tier']) for d in data]
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('中心性值 (Centrality Score)', fontsize=10)
        ax.set_title(f'{title_cn}\n({title_en})', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        # 数值标注
        for bar, val in zip(bars, values):
            ax.text(
                val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8
            )

    _plot_bar(axes[0], top_degree, 'degree_centrality', '度中心性 Top-15', 'Degree Centrality')
    _plot_bar(axes[1], top_between, 'betweenness_centrality', '介数中心性 Top-15', 'Betweenness Centrality')

    # 图例
    legend_patches = [
        mpatches.Patch(color=TIER_COLORS[t], label=f'T{t}: {TIER_NAMES[t][1]}')
        for t in sorted(TIER_COLORS.keys())
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    plt.tight_layout()
    save_figure(fig, 'F2-1_centrality_distribution', output_dir)
    return fig


def create_f2_2_degree_distribution(
    topology_report: Dict[str, Any],
    output_dir: str = 'outputs/figures',
) -> plt.Figure:
    """F2-2：网络度分布图（双对数坐标散点图 + 幂律参考线）。

    Args:
        topology_report: TopologyAnalyzer.get_report() 的输出
        output_dir: 图表输出目录

    Returns:
        matplotlib Figure 对象
    """
    apply_research_style()
    nodes_data = topology_report.get('nodes', {})
    if not nodes_data:
        raise ValueError("topology_report 中无节点数据")

    # 收集度值（使用 degree_centrality 还原度：dc * (N-1)）
    n_nodes = len(nodes_data)
    degrees = []
    for nid, metrics in nodes_data.items():
        dc = metrics.get('degree_centrality', 0.0)
        d = round(dc * (n_nodes - 1))
        if d > 0:
            degrees.append(d)

    if not degrees:
        degrees = [1] * n_nodes  # fallback

    # 频次统计
    from collections import Counter
    degree_counts = Counter(degrees)
    deg_vals = sorted(degree_counts.keys())
    freq_vals = [degree_counts[d] / n_nodes for d in deg_vals]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(
        '度分布图 (Degree Distribution)',
        fontsize=14, fontweight='bold'
    )

    ax.scatter(deg_vals, freq_vals, color='#3498DB', s=60, zorder=5,
               label='观测度频率 (Observed)')

    # 拟合幂律参考线（log-log 线性回归）
    if len(deg_vals) >= 3:
        log_x = np.log10([float(d) for d in deg_vals])
        log_y = np.log10([f for f in freq_vals])
        finite_mask = np.isfinite(log_x) & np.isfinite(log_y)
        if finite_mask.sum() >= 2:
            coeffs = np.polyfit(log_x[finite_mask], log_y[finite_mask], 1)
            x_fit = np.linspace(min(deg_vals), max(deg_vals), 100)
            y_fit = 10 ** (coeffs[1]) * x_fit ** coeffs[0]
            ax.plot(x_fit, y_fit, color='#E74C3C', linestyle='--', linewidth=1.5,
                    label=f'幂律拟合 (Power Law, γ={-coeffs[0]:.2f})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('度 (Degree)', fontsize=11)
    ax.set_ylabel('频率 (Frequency)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'F2-2_degree_distribution', output_dir)
    return fig


def create_f2_3_robustness_curve(
    robustness_data: Dict[str, Any],
    output_dir: str = 'outputs/figures',
) -> plt.Figure:
    """F2-3：抗毁性衰退曲线（随机故障 vs 定向攻击）。

    Args:
        robustness_data: VulnerabilityAnalyzer.robustness_analysis() 的输出
        output_dir: 图表输出目录

    Returns:
        matplotlib Figure 对象
    """
    apply_research_style()

    random_curve = robustness_data.get('random', [])
    targeted_curve = robustness_data.get('targeted', [])

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        '网络抗毁性分析 (Network Robustness Analysis)',
        fontsize=14, fontweight='bold'
    )

    if random_curve:
        rx = [p[0] for p in random_curve]
        ry = [p[1] for p in random_curve]
        ax.plot(rx, ry, color='#3498DB', linewidth=2.0, marker='o', markersize=4,
                label='随机故障 (Random Failure)')

    if targeted_curve:
        tx = [p[0] for p in targeted_curve]
        ty = [p[1] for p in targeted_curve]
        ax.plot(tx, ty, color='#E74C3C', linewidth=2.0, marker='s', markersize=4,
                linestyle='--', label='定向攻击 (Targeted Attack)')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('移除节点比例 (Fraction of Nodes Removed)', fontsize=11)
    ax.set_ylabel('最大连通分量比例 (Giant Component Ratio)', fontsize=11)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6, label='50% 阈值')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 填充两曲线之间的区域表示差距
    if random_curve and targeted_curve:
        min_len = min(len(random_curve), len(targeted_curve))
        rx_s = [random_curve[i][0] for i in range(min_len)]
        ry_s = [random_curve[i][1] for i in range(min_len)]
        ty_s = [targeted_curve[i][1] for i in range(min_len)]
        ax.fill_between(rx_s, ty_s, ry_s, alpha=0.1, color='#E74C3C',
                        label='攻击差距 (Attack Gap)')

    plt.tight_layout()
    save_figure(fig, 'F2-3_robustness_curve', output_dir)
    return fig


def create_f2_4_supply_concentration(
    network,
    hhi_scores: Dict[str, float],
    output_dir: str = 'outputs/figures',
) -> plt.Figure:
    """F2-4：供应集中度热力图（按层级×节点矩阵展示 HHI 指数）。

    Args:
        network: SupplyChainNetwork 对象
        hhi_scores: VulnerabilityAnalyzer.compute_hhi() 的输出
        output_dir: 图表输出目录

    Returns:
        matplotlib Figure 对象
    """
    apply_research_style()

    # 按层级分组节点（T1/T2/T3，T0通常无上游）
    tiers_to_show = [1, 2, 3]
    tier_node_groups: Dict[int, List] = {t: [] for t in tiers_to_show}

    for node_data in network.get_all_nodes():
        if node_data.tier in tier_node_groups:
            tier_node_groups[node_data.tier].append(node_data)

    # 构建矩阵：行=层级，列=节点
    all_nodes_ordered: List = []
    row_labels: List[str] = []
    row_indices: List[int] = []  # 记录每个节点所在层级在行中的索引

    tier_row_data: Dict[int, List[float]] = {}
    tier_col_labels: Dict[int, List[str]] = {}

    for tier in tiers_to_show:
        nodes = sorted(tier_node_groups[tier], key=lambda n: n.node_id)
        tier_col_labels[tier] = [n.name_en if len(n.name_en) <= 12 else n.node_id for n in nodes]
        tier_row_data[tier] = [hhi_scores.get(n.node_id, 0.0) for n in nodes]

    # 找最长列以统一矩阵宽度
    max_cols = max(len(v) for v in tier_col_labels.values()) if tier_col_labels else 1

    # 填充矩阵（不足部分填 NaN）
    matrix = []
    combined_col_labels: List[str] = []
    for tier in tiers_to_show:
        row = tier_row_data[tier]
        row_padded = row + [float('nan')] * (max_cols - len(row))
        matrix.append(row_padded)
        # 列标签用层级3（原材料）的标签，或综合所有层级的节点名
        if not combined_col_labels:
            combined_col_labels = tier_col_labels[tier] + [''] * (max_cols - len(tier_col_labels[tier]))

    # 实际上每行有不同的节点，重建列标签为每行最多节点数
    # 用 T3 节点数作为最多列
    combined_col_labels = [f'节点{i+1}' for i in range(max_cols)]
    row_tick_labels = [
        f'T{t}: {TIER_NAMES[t][0]}\n({TIER_NAMES[t][1]})' for t in tiers_to_show
    ]

    import pandas as pd
    # 更好的列标签：用 T3 节点的名称，其他行按顺序填入，超出部分用空
    # 重建：每行按节点顺序，列标签用数字
    col_labels_per_tier = {
        tier: tier_col_labels[tier] for tier in tiers_to_show
    }
    # 为了对齐，列标签使用最多节点所在层（通常 T3）
    max_tier = max(tiers_to_show, key=lambda t: len(tier_col_labels[t]))
    col_labels_main = tier_col_labels[max_tier]
    if len(col_labels_main) < max_cols:
        col_labels_main = col_labels_main + [f'col{i}' for i in range(max_cols - len(col_labels_main))]

    matrix_np = np.array(matrix, dtype=float)
    df = pd.DataFrame(matrix_np, index=row_tick_labels, columns=col_labels_main[:max_cols])

    # 绘图
    fig_width = max(10, max_cols * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    fig.suptitle(
        '供应集中度分布 (Supply Concentration Heatmap)',
        fontsize=14, fontweight='bold'
    )

    mask = np.isnan(matrix_np)
    sns.heatmap(
        df, ax=ax,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        annot=True, fmt='.2f', annot_kws={'size': 8},
        linewidths=0.5, linecolor='white',
        mask=mask,
        cbar_kws={'label': 'HHI 集中度指数 (HHI Score)', 'shrink': 0.8},
    )

    ax.set_xlabel('节点 (Node)', fontsize=10)
    ax.set_ylabel('供应层级 (Supply Tier)', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

    plt.tight_layout()
    save_figure(fig, 'F2-4_supply_concentration', output_dir)
    return fig


def create_phase2_figures(
    network,
    topology_report: Dict[str, Any],
    vulnerability_report: Dict[str, Any],
    output_dir: str = 'outputs/figures',
) -> Dict[str, plt.Figure]:
    """生成阶段二的全部4张科研图表。

    Args:
        network: SupplyChainNetwork 对象
        topology_report: TopologyAnalyzer.get_report() 的输出
        vulnerability_report: 包含 robustness / hhi_scores 的脆弱性报告字典
        output_dir: 图表输出目录

    Returns:
        图表名称到 Figure 对象的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    figures = {}

    print("\n[F2-1] 生成节点中心性分布图...")
    figures['F2-1'] = create_f2_1_centrality_distribution(topology_report, output_dir)
    plt.close(figures['F2-1'])

    print("\n[F2-2] 生成网络度分布图...")
    figures['F2-2'] = create_f2_2_degree_distribution(topology_report, output_dir)
    plt.close(figures['F2-2'])

    print("\n[F2-3] 生成抗毁性衰退曲线...")
    robustness_data = vulnerability_report.get('robustness', {})
    figures['F2-3'] = create_f2_3_robustness_curve(robustness_data, output_dir)
    plt.close(figures['F2-3'])

    print("\n[F2-4] 生成供应集中度热力图...")
    hhi_scores = vulnerability_report.get('hhi_scores', {})
    figures['F2-4'] = create_f2_4_supply_concentration(network, hhi_scores, output_dir)
    plt.close(figures['F2-4'])

    return figures

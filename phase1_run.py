"""
阶段一验证脚本：基础设施与网络本体模型
Phase 1 Validation: Infrastructure & Network Ontology Model
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# 加入项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.visualization.style import apply_research_style, save_figure, TIER_COLORS, TIER_NAMES
from src.network.models import SupplyChainNetwork, NodeType, EdgeType
from src.network.multilayer import MultiLayerNetwork, LAYER_CONFIGS
from src.network.builder import SupplyChainCaseLoader


def load_network() -> SupplyChainNetwork:
    """加载汽车发动机案例网络。"""
    print("正在加载汽车发动机供应链案例...")
    net = SupplyChainCaseLoader.load_auto_engine_case()
    summary = net.summary()
    print(f"  网络名称: {summary['name']}")
    print(f"  节点总数: {summary['node_count']}")
    print(f"  边总数:   {summary['edge_count']}")
    print(f"  层级分布: {summary['tier_distribution']}")
    return net


def build_hierarchical_layout(net: SupplyChainNetwork) -> dict:
    """构建分层布局（T3在底部，T0在顶部）。"""
    graph = net.get_graph()
    pos = {}
    tier_nodes = {t: net.get_nodes_by_tier(t) for t in range(4)}

    y_positions = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}  # tier -> y坐标

    for tier, nodes in tier_nodes.items():
        y = y_positions[tier]
        n = len(nodes)
        xs = np.linspace(0, 1, n) if n > 1 else [0.5]
        for i, node in enumerate(sorted(nodes, key=lambda x: x.node_id)):
            pos[node.node_id] = (xs[i], y)

    return pos


def plot_f1_1(net: SupplyChainNetwork, output_dir: str) -> None:
    """生成 F1-1 供应链网络拓扑结构图。"""
    print("\n生成 F1-1 供应链网络拓扑结构图...")

    apply_research_style()
    fig, ax = plt.subplots(figsize=(16, 11))

    graph = net.get_graph()
    # 只绘制物料流边
    material_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d.get('edge_type') == EdgeType.MATERIAL_SUPPLY.value
    ]
    subgraph = graph.edge_subgraph(material_edges).copy()
    # 确保所有节点都在
    for n_id in graph.nodes():
        if n_id not in subgraph:
            subgraph.add_node(n_id, **graph.nodes[n_id])

    pos = build_hierarchical_layout(net)

    # 高风险节点
    high_risk_nodes = {'T3-SI', 'T3-RE'}

    # 节点颜色和大小
    node_colors = []
    node_sizes = []
    for node_id in subgraph.nodes():
        node_data = net.get_node(node_id)
        if node_id in high_risk_nodes:
            node_colors.append('#E74C3C')
            node_sizes.append(800)
        else:
            node_colors.append(TIER_COLORS.get(node_data.tier, '#95A5A6'))
            # 节点大小基于出度+入度
            degree = graph.in_degree(node_id) + graph.out_degree(node_id)
            node_sizes.append(200 + degree * 80)

    # 边的粗细
    edge_widths = []
    for u, v in subgraph.edges():
        dep = graph[u][v].get('dependency_strength', 0.5)
        edge_widths.append(dep * 3)

    # 绘制边
    nx.draw_networkx_edges(
        subgraph, pos, ax=ax,
        edge_color='#7F8C8D', alpha=0.6,
        width=edge_widths,
        arrows=True,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.05',
        min_source_margin=15,
        min_target_margin=15,
    )

    # 绘制节点
    nx.draw_networkx_nodes(
        subgraph, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
    )

    # 节点标签
    nx.draw_networkx_labels(
        subgraph, pos, ax=ax,
        font_size=8,
        font_weight='bold',
    )

    # 层级横线和标注
    y_positions = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
    for tier, y in y_positions.items():
        zh, en = TIER_NAMES[tier]
        ax.axhline(y=y, color='#BDC3C7', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.text(-0.08, y, f'T{tier}\n{zh}', ha='right', va='center',
                fontsize=9, color=TIER_COLORS[tier], fontweight='bold')

    # 图例
    legend_handles = [
        mpatches.Patch(color=TIER_COLORS[t], label=f'T{t} {TIER_NAMES[t][0]} ({TIER_NAMES[t][1]})')
        for t in range(4)
    ]
    legend_handles.append(
        mpatches.Patch(color='#E74C3C', label='高风险节点 (High Risk Node)')
    )
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              framealpha=0.9, edgecolor='#BDC3C7')

    ax.set_title(
        '供应链网络拓扑结构图\n(Auto Engine Supply Chain Network Topology)',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.axis('off')
    ax.set_xlim(-0.15, 1.1)
    ax.set_ylim(-0.3, 3.4)

    plt.tight_layout()
    save_figure(fig, 'F1-1_network_topology', output_dir)
    plt.close()
    print("  F1-1 完成")


def plot_f1_2(net: SupplyChainNetwork, output_dir: str) -> None:
    """生成 F1-2 多层耦合网络结构图。"""
    print("\n生成 F1-2 多层耦合网络结构图...")

    apply_research_style()
    ml_net = MultiLayerNetwork(net)
    pos = build_hierarchical_layout(net)

    layer_keys = ['material', 'collaboration', 'logistics']
    layer_titles = {
        'material':      '物料依赖层\n(Material Dependency Layer)',
        'collaboration': '供应商协作层\n(Collaboration Layer)',
        'logistics':     '物流运输层\n(Logistics/Capital Layer)',
    }
    layer_edge_colors = {
        'material':      '#E74C3C',
        'collaboration': '#3498DB',
        'logistics':     '#2ECC71',
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    def draw_layer(ax, g, title, edge_color, show_labels=True):
        # 节点颜色
        node_colors = []
        node_sizes = []
        for node_id in g.nodes():
            node_data = net.get_node(node_id)
            if node_data:
                node_colors.append(TIER_COLORS.get(node_data.tier, '#95A5A6'))
                node_sizes.append(150 + g.degree(node_id) * 60)
            else:
                node_colors.append('#95A5A6')
                node_sizes.append(150)

        # 边粗细
        edge_widths = []
        for u, v in g.edges():
            dep = g[u][v].get('dependency_strength', 0.5)
            edge_widths.append(dep * 2.5)

        nx.draw_networkx_edges(
            g, pos, ax=ax,
            edge_color=edge_color, alpha=0.7,
            width=edge_widths if edge_widths else [1],
            arrows=True, arrowsize=12,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.05',
            min_source_margin=12, min_target_margin=12,
        )
        nx.draw_networkx_nodes(
            g, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes, alpha=0.85,
        )
        if show_labels:
            nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)

        # 层级标注
        for tier, y in {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}.items():
            ax.axhline(y=y, color='#BDC3C7', linestyle='--', alpha=0.3, linewidth=0.6)

        edge_count = g.number_of_edges()
        ax.set_title(f'{title}\n边数: {edge_count} (Edges: {edge_count})',
                     fontsize=11, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-0.12, 1.08)
        ax.set_ylim(-0.3, 3.35)

    # 三个子图
    for i, key in enumerate(layer_keys):
        row, col = divmod(i, 2)
        g = ml_net.get_layer(key)
        draw_layer(axes[row][col], g, layer_titles[key], layer_edge_colors[key])

    # 第四个子图：综合图
    ax4 = axes[1][1]
    all_nodes = list(net.get_graph().nodes(data=True))
    g_combined = nx.DiGraph()
    for node_id, attrs in all_nodes:
        g_combined.add_node(node_id, **attrs)

    for key, color in layer_edge_colors.items():
        g = ml_net.get_layer(key)
        for u, v, d in g.edges(data=True):
            if g_combined.has_edge(u, v):
                g_combined[u][v]['layers'] = g_combined[u][v].get('layers', []) + [key]
            else:
                g_combined.add_edge(u, v, **d, layers=[key], edge_color=color)

    node_colors_all = []
    for node_id in g_combined.nodes():
        nd = net.get_node(node_id)
        node_colors_all.append(TIER_COLORS.get(nd.tier if nd else 0, '#95A5A6'))

    edge_colors_all = []
    for u, v in g_combined.edges():
        layers = g_combined[u][v].get('layers', ['material'])
        if len(layers) > 1:
            edge_colors_all.append('#9B59B6')  # 紫色=多层重叠
        else:
            edge_colors_all.append(layer_edge_colors.get(layers[0], '#7F8C8D'))

    nx.draw_networkx_edges(
        g_combined, pos, ax=ax4,
        edge_color=edge_colors_all, alpha=0.6,
        width=1.5, arrows=True, arrowsize=10,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.05',
        min_source_margin=10, min_target_margin=10,
    )
    nx.draw_networkx_nodes(
        g_combined, pos, ax=ax4,
        node_color=node_colors_all,
        node_size=150, alpha=0.85,
    )
    nx.draw_networkx_labels(g_combined, pos, ax=ax4, font_size=6)

    for tier, y in {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}.items():
        ax4.axhline(y=y, color='#BDC3C7', linestyle='--', alpha=0.3, linewidth=0.6)

    legend_handles = [
        mpatches.Patch(color=layer_edge_colors['material'], label='物料流 (Material)'),
        mpatches.Patch(color=layer_edge_colors['collaboration'], label='信息流 (Information)'),
        mpatches.Patch(color=layer_edge_colors['logistics'], label='资金流 (Capital)'),
        mpatches.Patch(color='#9B59B6', label='多层重叠 (Multi-layer)'),
    ]
    ax4.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    ax4.set_title('综合耦合网络\n(Integrated Multi-layer Network)', fontsize=11, fontweight='bold')
    ax4.axis('off')
    ax4.set_xlim(-0.12, 1.08)
    ax4.set_ylim(-0.3, 3.35)

    fig.suptitle(
        '供应链多层耦合网络结构图\n(Multi-layer Coupled Supply Chain Network)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    save_figure(fig, 'F1-2_multilayer_network', output_dir)
    plt.close()
    print("  F1-2 完成")


def generate_csv_files(net: SupplyChainNetwork) -> None:
    """生成节点和边的 CSV 属性文件。"""
    print("\n生成 CSV 属性文件...")

    # node_attributes.csv
    rows = []
    for node in net.get_all_nodes():
        rows.append({
            'node_id': node.node_id,
            'name': node.name,
            'name_en': node.name_en,
            'node_type': node.node_type.value,
            'tier': node.tier,
            'location': node.location,
            'region': node.region,
            'capacity_limit': node.capacity_limit,
            'lead_time': node.lead_time,
            'substitutability': node.substitutability,
        })
    df_nodes = pd.DataFrame(rows)
    df_nodes.to_csv('data/cases/auto_engine/node_attributes.csv', index=False, encoding='utf-8-sig')
    print("  已保存: data/cases/auto_engine/node_attributes.csv")

    # edge_attributes.csv
    rows = []
    for edge in net.get_all_edges():
        rows.append({
            'source': edge.source,
            'target': edge.target,
            'edge_type': edge.edge_type.value,
            'supply_volume': edge.supply_volume,
            'dependency_strength': edge.dependency_strength,
        })
    df_edges = pd.DataFrame(rows)
    df_edges.to_csv('data/cases/auto_engine/edge_attributes.csv', index=False, encoding='utf-8-sig')
    print("  已保存: data/cases/auto_engine/edge_attributes.csv")


def generate_report(net: SupplyChainNetwork) -> None:
    """生成阶段一分析报告 JSON。"""
    print("\n生成阶段一分析报告...")

    os.makedirs('outputs/reports', exist_ok=True)

    summary = net.summary()
    graph = net.get_graph()

    # 计算基础图论指标
    material_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d.get('edge_type') == EdgeType.MATERIAL_SUPPLY.value
    ]
    mg = graph.edge_subgraph(material_edges)

    report = {
        'phase': 1,
        'phase_name': '基础设施与网络本体模型',
        'case': summary,
        'graph_metrics': {
            'is_dag': nx.is_directed_acyclic_graph(mg),
            'material_edges': len(material_edges),
            'density': round(nx.density(graph), 4),
        },
        'tier_details': {
            str(t): [n.node_id for n in net.get_nodes_by_tier(t)]
            for t in range(4)
        },
        'high_risk_nodes': ['T3-SI', 'T3-RE'],
        'outputs': {
            'figures': ['F1-1_network_topology.pdf/png', 'F1-2_multilayer_network.pdf/png'],
            'data': ['node_attributes.csv', 'edge_attributes.csv', 'network.json', 'risk_scenarios.json'],
        },
        'validation': {
            'node_count_match': summary['node_count'] == 25,
            'tier_count_match': len(summary['tier_distribution']) == 4,
        }
    }

    report_path = 'outputs/reports/phase1_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {report_path}")

    # 验收检查
    print("\n=== 验收检查 ===")
    print(f"  节点数量 = {summary['node_count']} (期望 25): {'OK' if summary['node_count'] == 25 else 'FAIL'}")
    print(f"  层级数量 = {len(summary['tier_distribution'])} (期望 4): {'OK' if len(summary['tier_distribution']) == 4 else 'FAIL'}")
    print(f"  DAG结构 = {report['graph_metrics']['is_dag']}: {'OK' if report['graph_metrics']['is_dag'] else 'FAIL'}")
    print(f"  图表输出: OK")


def main():
    """主函数：运行阶段一验证。"""
    print("=" * 60)
    print("阶段一：基础设施与网络本体模型")
    print("Phase 1: Infrastructure & Network Ontology Model")
    print("=" * 60)

    # 切换到项目目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载网络
    net = load_network()

    # 2. 生成 CSV 文件
    generate_csv_files(net)

    # 3. 生成图表
    plot_f1_1(net, output_dir)
    plot_f1_2(net, output_dir)

    # 4. 生成报告
    generate_report(net)

    print("\n" + "=" * 60)
    print("阶段一开发完成！")
    print(f"图表输出目录: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == '__main__':
    main()

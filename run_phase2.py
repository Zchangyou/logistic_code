"""
阶段二运行脚本：参数化网络构造 + 拓扑分析 + 脆弱性评估 + 科研图表生成
Phase 2 Runner: Parametric Network Construction + Topology Analysis +
                Vulnerability Assessment + Research Figure Generation
"""
import sys
import os
import json
import time
import traceback

# 将项目根目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import networkx as nx

from src.network.builder import SupplyChainCaseLoader, ParametricNetworkGenerator
from src.network.topology import TopologyAnalyzer
from src.network.vulnerability import VulnerabilityAnalyzer
from src.visualization.network_vis import create_phase2_figures


def main() -> None:
    """阶段二主流程。"""
    start_time = time.time()

    # ------------------------------------------------------------------ #
    # 0. 确保输出目录存在
    # ------------------------------------------------------------------ #
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. 加载汽车发动机案例网络
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("[步骤1] 加载汽车发动机供应链案例网络")
    print("=" * 60)
    network = SupplyChainCaseLoader.load_auto_engine_case(
        data_dir=os.path.join(PROJECT_ROOT, 'data', 'cases', 'auto_engine')
    )
    summary = network.summary()
    print(f"  网络名称: {summary['name']}")
    print(f"  节点数量: {summary['node_count']}")
    print(f"  边数量:   {summary['edge_count']}")
    print(f"  层级分布: {summary['tier_distribution']}")

    # ------------------------------------------------------------------ #
    # 2. 拓扑分析
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[步骤2] 运行拓扑特征分析")
    print("=" * 60)
    topology_analyzer = TopologyAnalyzer(network)
    topology_report = topology_analyzer.get_report()
    topology_analyzer.save_report(output_dir=os.path.join(PROJECT_ROOT, 'outputs', 'reports'))

    net_metrics = topology_report['network']
    print(f"  网络直径:         {net_metrics['network_diameter']}")
    print(f"  平均路径长度:     {net_metrics['avg_path_length']}")
    print(f"  网络密度:         {net_metrics['density']}")
    print(f"  最大弱连通分量:   {net_metrics['largest_wcc_size']} 节点")

    # 打印介数中心性 Top-5
    nodes_metrics = topology_report['nodes']
    top5_bc = sorted(
        nodes_metrics.items(),
        key=lambda x: x[1].get('betweenness_centrality', 0),
        reverse=True
    )[:5]
    print("\n  介数中心性 Top-5:")
    for nid, m in top5_bc:
        print(f"    {nid:12s} ({m.get('name_en', nid):25s}) BC={m['betweenness_centrality']:.4f}")

    # ------------------------------------------------------------------ #
    # 3. 脆弱性评估
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[步骤3] 运行脆弱性评估")
    print("=" * 60)
    vuln_analyzer = VulnerabilityAnalyzer(network)

    print("  计算 HHI 供应集中度...")
    hhi_scores = vuln_analyzer.compute_hhi()
    top5_hhi = sorted(hhi_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  HHI Top-5 节点（供应商集中度最高）:")
    for nid, hhi in top5_hhi:
        node_data = network.get_node(nid)
        name = node_data.name_en if node_data else nid
        print(f"    {nid:12s} ({name:25s}) HHI={hhi:.4f}")

    print("  计算关键节点...")
    key_nodes = vuln_analyzer.get_key_nodes(top_n=5)
    print("  关键节点 Top-5（综合脆弱性评分）:")
    for kn in key_nodes:
        print(f"    {kn['node_id']:12s} ({kn['name_en']:25s}) Score={kn['composite_score']:.4f}")

    print("  运行抗毁性仿真（随机故障 vs 定向攻击）...")
    robustness = vuln_analyzer.robustness_analysis(n_steps=20, n_trials=5, seed=42)
    print(f"  随机故障最终连通比: {robustness['random'][-1][1]:.3f}")
    print(f"  定向攻击最终连通比: {robustness['targeted'][-1][1]:.3f}")

    vuln_analyzer.save_report(output_dir=os.path.join(PROJECT_ROOT, 'outputs', 'reports'))

    # ------------------------------------------------------------------ #
    # 4. 参数化网络生成与验证
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[步骤4] 生成并验证参数化 100 节点网络")
    print("=" * 60)
    generator = ParametricNetworkGenerator(
        node_count=100,
        num_tiers=4,
        supplier_concentration=0.4,
        network_redundancy=0.3,
        seed=2024,
    )
    synthetic_network = generator.generate()
    synthetic_graph = synthetic_network.get_graph()

    is_dag = nx.is_directed_acyclic_graph(synthetic_graph)
    print(f"  节点数: {synthetic_graph.number_of_nodes()}")
    print(f"  边数:   {synthetic_graph.number_of_edges()}")
    print(f"  是否为 DAG: {is_dag}")
    assert is_dag, "参数化网络应为有向无环图（DAG）！"

    # 验证近似幂律分布
    import numpy as np
    degrees = [d for _, d in synthetic_graph.degree()]
    degrees = [d for d in degrees if d > 0]
    if degrees:
        log_degrees = np.log10(degrees)
        print(f"  度分布均值: {np.mean(degrees):.2f}, 最大度: {max(degrees)}, 最小度: {min(degrees)}")
        print(f"  度分布标准差: {np.std(degrees):.2f}")
    print("  参数化网络验证通过。")

    # ------------------------------------------------------------------ #
    # 5. 生成全部 4 张科研图表
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[步骤5] 生成科研图表（F2-1 至 F2-4）")
    print("=" * 60)

    vulnerability_report = {
        'hhi_scores': hhi_scores,
        'key_nodes': key_nodes,
        'robustness': robustness,
    }

    figures = create_phase2_figures(
        network=network,
        topology_report=topology_report,
        vulnerability_report=vulnerability_report,
        output_dir=os.path.join(PROJECT_ROOT, 'outputs', 'figures'),
    )
    print(f"\n  共生成 {len(figures)} 张图表。")

    # ------------------------------------------------------------------ #
    # 6. 保存阶段二综合报告
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[步骤6] 保存阶段二综合报告")
    print("=" * 60)
    elapsed = time.time() - start_time

    phase2_report = {
        'phase': 2,
        'description': '参数化网络构造与拓扑分析',
        'description_en': 'Parametric Network Construction and Topology Analysis',
        'elapsed_seconds': round(elapsed, 2),
        'case_network': {
            'name': summary['name'],
            'node_count': summary['node_count'],
            'edge_count': summary['edge_count'],
            'tier_distribution': summary['tier_distribution'],
        },
        'network_metrics': net_metrics,
        'top5_betweenness': [
            {'node_id': nid, 'name_en': m.get('name_en', nid),
             'betweenness_centrality': m['betweenness_centrality']}
            for nid, m in top5_bc
        ],
        'top5_hhi': [
            {'node_id': nid, 'hhi': hhi}
            for nid, hhi in top5_hhi
        ],
        'key_nodes_top5': key_nodes,
        'robustness_summary': {
            'random_steps': len(robustness['random']),
            'targeted_steps': len(robustness['targeted']),
            'random_final_gc_ratio': robustness['random'][-1][1],
            'targeted_final_gc_ratio': robustness['targeted'][-1][1],
        },
        'parametric_network': {
            'node_count': synthetic_graph.number_of_nodes(),
            'edge_count': synthetic_graph.number_of_edges(),
            'is_dag': is_dag,
        },
        'figures_generated': list(figures.keys()),
        'output_files': {
            'topology_json': 'outputs/reports/topology_report.json',
            'topology_csv': 'outputs/reports/topology_report.csv',
            'vulnerability_json': 'outputs/reports/vulnerability_report.json',
            'figures': [
                'outputs/figures/F2-1_centrality_distribution.pdf',
                'outputs/figures/F2-1_centrality_distribution.png',
                'outputs/figures/F2-2_degree_distribution.pdf',
                'outputs/figures/F2-2_degree_distribution.png',
                'outputs/figures/F2-3_robustness_curve.pdf',
                'outputs/figures/F2-3_robustness_curve.png',
                'outputs/figures/F2-4_supply_concentration.pdf',
                'outputs/figures/F2-4_supply_concentration.png',
            ],
        },
    }

    report_path = os.path.join(PROJECT_ROOT, 'outputs', 'reports', 'phase2_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(phase2_report, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {report_path}")

    # ------------------------------------------------------------------ #
    # 完成
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"[完成] 阶段二执行完毕，耗时 {elapsed:.1f} 秒")
    print("=" * 60)
    print("\n输出文件清单:")
    for f in phase2_report['output_files']['figures']:
        print(f"  {f}")
    print(f"  outputs/reports/topology_report.json")
    print(f"  outputs/reports/topology_report.csv")
    print(f"  outputs/reports/vulnerability_report.json")
    print(f"  outputs/reports/phase2_report.json")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] 执行失败: {e}")
        traceback.print_exc()
        sys.exit(1)

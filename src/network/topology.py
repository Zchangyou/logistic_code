"""
供应链网络拓扑特征分析模块
Supply Chain Network Topology Analysis Module
"""
import os
import json
import csv
from typing import Dict, Any, List

import networkx as nx
import numpy as np

from .models import SupplyChainNetwork


class TopologyAnalyzer:
    """拓扑特征分析器 (Topology Analyzer)

    计算供应链网络的各类拓扑指标：中心性、聚类系数、网络直径等。

    Attributes:
        network: SupplyChainNetwork 对象
    """

    def __init__(self, network: SupplyChainNetwork) -> None:
        """初始化拓扑分析器。

        Args:
            network: 待分析的 SupplyChainNetwork 对象
        """
        self.network = network
        self._graph: nx.DiGraph = network.get_graph()
        self._undirected: nx.Graph = self._graph.to_undirected()
        self._metrics: Dict[str, Any] = {}

    def _compute_centrality(self) -> None:
        """计算各类中心性指标。"""
        g = self._graph
        u = self._undirected

        in_deg = nx.in_degree_centrality(g)
        out_deg = nx.out_degree_centrality(g)
        deg = nx.degree_centrality(u)
        betweenness = nx.betweenness_centrality(g, normalized=True)
        clustering = nx.clustering(u)
        pagerank = nx.pagerank(g, alpha=0.85)

        for node_id in g.nodes():
            if node_id not in self._metrics:
                self._metrics[node_id] = {}
            self._metrics[node_id].update({
                'in_degree_centrality': round(in_deg.get(node_id, 0.0), 6),
                'out_degree_centrality': round(out_deg.get(node_id, 0.0), 6),
                'degree_centrality': round(deg.get(node_id, 0.0), 6),
                'betweenness_centrality': round(betweenness.get(node_id, 0.0), 6),
                'clustering_coefficient': round(clustering.get(node_id, 0.0), 6),
                'pagerank': round(pagerank.get(node_id, 0.0), 6),
            })

    def _compute_network_metrics(self) -> Dict[str, Any]:
        """计算网络级别指标（直径、平均路径长度）。

        Returns:
            包含 network_diameter 和 avg_path_length 的字典
        """
        g = self._graph
        # 使用最大弱连通分量
        wcc = max(nx.weakly_connected_components(g), key=len)
        subgraph = g.subgraph(wcc).copy()
        ug = subgraph.to_undirected()

        try:
            diameter = nx.diameter(ug)
        except Exception:
            diameter = -1

        try:
            avg_path = nx.average_shortest_path_length(ug)
        except Exception:
            avg_path = -1.0

        return {
            'network_diameter': diameter,
            'avg_path_length': round(float(avg_path), 4),
            'node_count': g.number_of_nodes(),
            'edge_count': g.number_of_edges(),
            'density': round(nx.density(g), 6),
            'largest_wcc_size': len(wcc),
        }

    def get_report(self) -> Dict[str, Any]:
        """生成完整的拓扑分析报告。

        Returns:
            包含每个节点指标和网络级别指标的字典
        """
        self._compute_centrality()
        network_level = self._compute_network_metrics()

        # 附加节点的 tier 信息
        for node_id in self._graph.nodes():
            node_data = self.network.get_node(node_id)
            if node_data:
                self._metrics[node_id]['tier'] = node_data.tier
                self._metrics[node_id]['name'] = node_data.name
                self._metrics[node_id]['name_en'] = node_data.name_en

        return {
            'nodes': self._metrics,
            'network': network_level,
        }

    def save_report(self, output_dir: str = 'outputs/reports') -> None:
        """保存拓扑报告为 JSON 和 CSV 文件。

        Args:
            output_dir: 输出目录路径
        """
        os.makedirs(output_dir, exist_ok=True)
        report = self.get_report()

        # 保存 JSON
        json_path = os.path.join(output_dir, 'topology_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'  已保存: {json_path}')

        # 保存 CSV
        csv_path = os.path.join(output_dir, 'topology_report.csv')
        nodes_data = report['nodes']
        if nodes_data:
            fieldnames = ['node_id'] + list(next(iter(nodes_data.values())).keys())
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for node_id, metrics in nodes_data.items():
                    row = {'node_id': node_id}
                    row.update(metrics)
                    writer.writerow(row)
        print(f'  已保存: {csv_path}')

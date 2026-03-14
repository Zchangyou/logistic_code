"""
多层耦合网络模型
Multi-layer Coupled Network Model
"""
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from .models import SupplyChainNetwork, EdgeType


@dataclass
class LayerConfig:
    """层配置 (Layer configuration)"""
    name: str
    name_en: str
    edge_types: List[EdgeType]
    color: str


# 三层网络配置
LAYER_CONFIGS = {
    'material': LayerConfig(
        name='物料依赖层',
        name_en='Material Dependency Layer',
        edge_types=[EdgeType.MATERIAL_SUPPLY],
        color='#E74C3C',
    ),
    'collaboration': LayerConfig(
        name='供应商协作层',
        name_en='Supplier Collaboration Layer',
        edge_types=[EdgeType.INFO_TRANSFER],
        color='#3498DB',
    ),
    'logistics': LayerConfig(
        name='物流运输层',
        name_en='Logistics Transport Layer',
        edge_types=[EdgeType.CAPITAL_FLOW],
        color='#2ECC71',
    ),
}


class MultiLayerNetwork:
    """多层耦合供应链网络 (Multi-layer Coupled Supply Chain Network)

    在基础供应链网络之上，构建物料依赖层、供应商协作层、物流运输层三层子网络，
    并计算层间耦合矩阵。

    Attributes:
        base_network: 基础供应链网络
        layers: 各层子网络字典
    """

    def __init__(self, base_network: SupplyChainNetwork) -> None:
        """初始化多层网络。

        Args:
            base_network: 基础 SupplyChainNetwork 对象
        """
        self.base_network = base_network
        self._layers: Dict[str, nx.DiGraph] = {}
        self._build_layers()

    def _build_layers(self) -> None:
        """从基础网络构建各层子网络。"""
        base_graph = self.base_network.get_graph()
        nodes = list(base_graph.nodes(data=True))

        for layer_key, config in LAYER_CONFIGS.items():
            g = nx.DiGraph(
                name=config.name,
                name_en=config.name_en,
            )
            # 所有节点在各层都存在
            for node_id, attrs in nodes:
                g.add_node(node_id, **attrs)

            # 只添加该层对应类型的边
            for edge in self.base_network.get_all_edges():
                if edge.edge_type in config.edge_types:
                    g.add_edge(
                        edge.source, edge.target,
                        **edge.to_dict()
                    )

            self._layers[layer_key] = g

    def get_material_layer(self) -> nx.DiGraph:
        """获取物料依赖层子网络。

        Returns:
            物料依赖层 DiGraph
        """
        return self._layers['material']

    def get_collaboration_layer(self) -> nx.DiGraph:
        """获取供应商协作层子网络（信息流）。

        Returns:
            供应商协作层 DiGraph
        """
        return self._layers['collaboration']

    def get_logistics_layer(self) -> nx.DiGraph:
        """获取物流运输层子网络（资金流，方向与物料流相反）。

        Returns:
            物流运输层 DiGraph
        """
        return self._layers['logistics']

    def get_layer(self, layer_key: str) -> nx.DiGraph:
        """获取指定层网络。

        Args:
            layer_key: 层标识符 ('material', 'collaboration', 'logistics')

        Returns:
            对应层的 DiGraph
        """
        return self._layers[layer_key]

    def get_layer_configs(self) -> Dict[str, LayerConfig]:
        """获取所有层配置。"""
        return LAYER_CONFIGS

    def get_layer_coupling_matrix(self) -> np.ndarray:
        """计算层间耦合矩阵。

        Returns:
            3x3 numpy 矩阵，元素为两层间共享边数量
        """
        layer_keys = ['material', 'collaboration', 'logistics']
        n = len(layer_keys)
        matrix = np.zeros((n, n))

        for i, k1 in enumerate(layer_keys):
            g1 = self._layers[k1]
            matrix[i][i] = g1.number_of_edges()
            for j, k2 in enumerate(layer_keys):
                if i != j:
                    g2 = self._layers[k2]
                    shared = len(set(g1.nodes()) & set(g2.nodes()))
                    matrix[i][j] = shared

        return matrix

    def layer_summary(self) -> Dict[str, Dict]:
        """返回各层网络摘要。"""
        summary = {}
        for key, g in self._layers.items():
            config = LAYER_CONFIGS[key]
            summary[key] = {
                'name': config.name,
                'name_en': config.name_en,
                'nodes': g.number_of_nodes(),
                'edges': g.number_of_edges(),
            }
        return summary

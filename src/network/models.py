"""
供应链网络核心数据模型
Core data models for supply chain network
"""
import json
import networkx as nx
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, ClassVar
from enum import Enum


class NodeType(Enum):
    """节点类型枚举 (Node type enumeration)"""
    RAW_MATERIAL = "raw_material"      # 原材料供应商 T3
    PARTS_SUPPLIER = "parts_supplier"  # 零部件供应商 T2
    INTEGRATOR = "integrator"          # 分系统集成商 T1
    ASSEMBLY = "assembly"              # 总装厂 T0


class EdgeType(Enum):
    """边类型枚举 (Edge type enumeration)"""
    MATERIAL_SUPPLY = "material_supply"  # 物料供应
    INFO_TRANSFER = "info_transfer"      # 信息传递
    CAPITAL_FLOW = "capital_flow"        # 资金流转


@dataclass
class NodeData:
    """节点数据类 (Node data class)

    Attributes:
        node_id: 节点唯一标识符
        name: 中文名称
        name_en: 英文名称
        node_type: 节点类型（NodeType枚举）
        tier: 供应链层级（0=总装厂, 3=原材料）
        location: 地理位置（城市）
        capacity_limit: 产能上限（相对值 0-1）
        lead_time: 供货周期（天）
        substitutability: 可替代性等级（0=不可替代, 1=完全可替代）
        region: 所在区域
        attributes: 扩展属性字典
    """
    node_id: str
    name: str
    name_en: str
    node_type: NodeType
    tier: int
    location: str
    capacity_limit: float
    lead_time: int
    substitutability: float
    region: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        d = asdict(self)
        d['node_type'] = self.node_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeData':
        """从字典反序列化。"""
        data = data.copy()
        data['node_type'] = NodeType(data['node_type'])
        return cls(**data)


@dataclass
class EdgeData:
    """边数据类 (Edge data class)

    Attributes:
        source: 源节点ID
        target: 目标节点ID
        edge_type: 边类型（EdgeType枚举）
        supply_volume: 供应量（相对值 0-1）
        dependency_strength: 依赖强度（0=无依赖, 1=完全依赖）
        attributes: 扩展属性字典
    """
    source: str
    target: str
    edge_type: EdgeType
    supply_volume: float
    dependency_strength: float
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        d = asdict(self)
        d['edge_type'] = self.edge_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeData':
        """从字典反序列化。"""
        data = data.copy()
        data['edge_type'] = EdgeType(data['edge_type'])
        return cls(**data)


class SupplyChainNetwork:
    """供应链网络核心类 (Supply Chain Network Core Class)

    封装 networkx 有向图，提供供应链网络的建模、存储与查询能力。

    Attributes:
        name: 网络中文名称
        name_en: 网络英文名称
        graph: 底层 networkx 有向图
    """

    def __init__(self, name: str, name_en: str) -> None:
        """初始化供应链网络。

        Args:
            name: 网络中文名称
            name_en: 网络英文名称
        """
        self.name = name
        self.name_en = name_en
        self._graph = nx.DiGraph(name=name, name_en=name_en)
        self._nodes: Dict[str, NodeData] = {}
        self._edges: Dict[tuple, List[EdgeData]] = {}

    def add_node(self, node: NodeData) -> None:
        """添加节点。

        Args:
            node: NodeData 节点数据对象
        """
        self._nodes[node.node_id] = node
        self._graph.add_node(
            node.node_id,
            **node.to_dict()
        )

    def add_edge(self, edge: EdgeData) -> None:
        """添加边。

        Args:
            edge: EdgeData 边数据对象
        """
        key = (edge.source, edge.target, edge.edge_type.value)
        if key not in self._edges:
            self._edges[key] = []
        self._edges[key].append(edge)

        # networkx 中同一对节点只保留一条边（按edge_type区分用属性标记）
        if not self._graph.has_edge(edge.source, edge.target):
            self._graph.add_edge(
                edge.source, edge.target,
                **edge.to_dict()
            )
        else:
            # 如果已有边，更新依赖强度取最大值
            existing = self._graph[edge.source][edge.target]
            if edge.dependency_strength > existing.get('dependency_strength', 0):
                self._graph[edge.source][edge.target].update(edge.to_dict())

    def get_node(self, node_id: str) -> Optional[NodeData]:
        """获取节点数据。

        Args:
            node_id: 节点ID

        Returns:
            NodeData 对象，不存在则返回 None
        """
        return self._nodes.get(node_id)

    def get_nodes_by_tier(self, tier: int) -> List[NodeData]:
        """获取指定层级的所有节点。

        Args:
            tier: 层级编号（0-3）

        Returns:
            该层级的 NodeData 列表
        """
        return [n for n in self._nodes.values() if n.tier == tier]

    def get_graph(self) -> nx.DiGraph:
        """获取底层 networkx 有向图。

        Returns:
            networkx DiGraph 对象
        """
        return self._graph

    def get_all_nodes(self) -> List[NodeData]:
        """获取所有节点。"""
        return list(self._nodes.values())

    def get_all_edges(self) -> List[EdgeData]:
        """获取所有边（去重）。"""
        seen = set()
        result = []
        for edges in self._edges.values():
            for e in edges:
                k = (e.source, e.target, e.edge_type.value)
                if k not in seen:
                    seen.add(k)
                    result.append(e)
        return result

    def summary(self) -> Dict[str, Any]:
        """返回网络摘要信息。"""
        tier_counts = {}
        for node in self._nodes.values():
            tier_counts[node.tier] = tier_counts.get(node.tier, 0) + 1
        return {
            'name': self.name,
            'name_en': self.name_en,
            'node_count': len(self._nodes),
            'edge_count': self._graph.number_of_edges(),
            'tier_distribution': tier_counts,
        }

    def to_dict(self) -> Dict[str, Any]:
        """序列化整个网络为字典。"""
        return {
            'name': self.name,
            'name_en': self.name_en,
            'nodes': [n.to_dict() for n in self._nodes.values()],
            'edges': [e.to_dict() for e in self.get_all_edges()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SupplyChainNetwork':
        """从字典反序列化网络。

        Args:
            data: 序列化的网络字典

        Returns:
            SupplyChainNetwork 对象
        """
        net = cls(data['name'], data['name_en'])
        for node_dict in data['nodes']:
            net.add_node(NodeData.from_dict(node_dict))
        for edge_dict in data['edges']:
            net.add_edge(EdgeData.from_dict(edge_dict))
        return net

    def save_json(self, filepath: str) -> None:
        """保存网络到 JSON 文件。

        Args:
            filepath: 输出文件路径
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> 'SupplyChainNetwork':
        """从 JSON 文件加载网络。

        Args:
            filepath: JSON 文件路径

        Returns:
            SupplyChainNetwork 对象
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class NetworkData:
    """模块间数据传递对象 (Inter-module data transfer object)

    标准化网络数据接口，供风险模块等下游使用。

    Attributes:
        network: SupplyChainNetwork 对象
        metadata: 附加元数据
    """
    network: SupplyChainNetwork
    metadata: Dict[str, Any] = field(default_factory=dict)

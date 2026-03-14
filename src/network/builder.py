"""
供应链网络构造器
Supply Chain Network Builder
"""
import os
import csv
import json
import math
import random
from typing import Optional, List, Dict, Tuple
import numpy as np
from .models import (
    SupplyChainNetwork, NodeData, EdgeData, NodeType, EdgeType
)


class SupplyChainCaseLoader:
    """供应链案例加载器 (Supply Chain Case Loader)

    从预定义的案例数据文件加载真实供应链网络。
    """

    BASE_DIR: str = 'data/cases'

    @classmethod
    def load_auto_engine_case(cls, data_dir: Optional[str] = None) -> SupplyChainNetwork:
        """加载汽车发动机供应链案例网络。

        Args:
            data_dir: 案例数据目录，默认为 data/cases/auto_engine/

        Returns:
            SupplyChainNetwork 对象
        """
        if data_dir is None:
            data_dir = os.path.join(cls.BASE_DIR, 'auto_engine')

        json_path = os.path.join(data_dir, 'network.json')
        return SupplyChainNetwork.load_json(json_path)


# 层级对应的节点类型
_TIER_NODE_TYPES = {
    0: NodeType.ASSEMBLY,
    1: NodeType.INTEGRATOR,
    2: NodeType.PARTS_SUPPLIER,
    3: NodeType.RAW_MATERIAL,
}

# 层级对应的中英文名称前缀
_TIER_PREFIXES = {
    0: ('总装厂', 'Assembly'),
    1: ('集成商', 'Integrator'),
    2: ('零部件商', 'Parts'),
    3: ('原材料商', 'RawMat'),
}


class ParametricNetworkGenerator:
    """参数化供应链网络生成器 (Parametric Supply Chain Network Generator)

    基于参数生成具有幂律度分布的有向无环供应链网络，用于仿真实验。

    Attributes:
        node_count: 网络节点总数
        num_tiers: 层级数量（默认4，T0-T3）
        supplier_concentration: 供应商集中度 (0-1)，越高则每个下游节点的供应商越少
        network_redundancy: 网络冗余度 (0-1)，越高则备用供应路径越多
        seed: 随机种子
    """

    def __init__(
        self,
        node_count: int = 50,
        num_tiers: int = 4,
        supplier_concentration: float = 0.5,
        network_redundancy: float = 0.3,
        seed: int = 42,
    ) -> None:
        """初始化参数化网络生成器。

        Args:
            node_count: 节点总数
            num_tiers: 层级数量
            supplier_concentration: 供应商集中度 (0=分散, 1=高度集中)
            network_redundancy: 网络冗余度 (0=无冗余, 1=高冗余)
            seed: 随机种子
        """
        self.node_count = node_count
        self.num_tiers = num_tiers
        self.supplier_concentration = supplier_concentration
        self.network_redundancy = network_redundancy
        self.seed = seed

    def _compute_tier_sizes(self) -> List[int]:
        """按层级分配节点数量。

        T0=1, T1~10%, T2~30%, T3~60%（四层时）
        其余层级按等比分配。

        Returns:
            各层节点数列表，索引0=T0（总装厂）
        """
        n = self.node_count
        if self.num_tiers == 4:
            t0 = 1
            t1 = max(1, round(n * 0.10))
            t2 = max(1, round(n * 0.30))
            t3 = max(1, n - t0 - t1 - t2)
            return [t0, t1, t2, t3]
        else:
            sizes = [1]
            remaining = n - 1
            ratios = [0.10, 0.30, 0.60]
            for i in range(1, self.num_tiers - 1):
                ratio = ratios[min(i - 1, len(ratios) - 1)]
                sizes.append(max(1, round(remaining * ratio)))
            last = max(1, n - sum(sizes))
            sizes.append(last)
            return sizes

    def _assign_degree_weights(self, tier_nodes: List[str], rng: random.Random) -> Dict[str, float]:
        """用优先连接原则为节点分配连接权重，产生幂律分布。

        Args:
            tier_nodes: 节点ID列表
            rng: 随机数生成器

        Returns:
            节点ID到连接权重的映射
        """
        weights: Dict[str, float] = {}
        # 初始权重随机化
        for nid in tier_nodes:
            weights[nid] = rng.uniform(0.5, 1.5)
        # 迭代放大（模拟优先连接效应）
        for _ in range(max(1, len(tier_nodes) // 3)):
            chosen = rng.choices(tier_nodes, weights=[weights[n] for n in tier_nodes], k=1)[0]
            weights[chosen] += 1.0
        return weights

    def generate(self) -> SupplyChainNetwork:
        """生成参数化供应链网络。

        Returns:
            SupplyChainNetwork 对象（有向无环图）
        """
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        tier_sizes = self._compute_tier_sizes()
        net = SupplyChainNetwork(
            name=f"参数化供应链网络({self.node_count}节点)",
            name_en=f"Parametric Supply Chain Network ({self.node_count} nodes)",
        )

        # 1. 生成节点
        tier_node_ids: List[List[str]] = []
        global_idx = 0
        for tier, count in enumerate(tier_sizes):
            prefix_cn, prefix_en = _TIER_PREFIXES.get(tier, (f'T{tier}', f'T{tier}'))
            node_type = _TIER_NODE_TYPES.get(tier, NodeType.RAW_MATERIAL)
            nodes_in_tier = []
            for i in range(count):
                nid = f"N{tier}-{i}"
                substitutability = float(np_rng.uniform(0.2, 0.9))
                # 层级越高（原材料）可替代性越低（受集中度影响）
                if tier == self.num_tiers - 1:
                    substitutability *= (1.0 - self.supplier_concentration * 0.5)
                node = NodeData(
                    node_id=nid,
                    name=f"{prefix_cn}{i+1}",
                    name_en=f"{prefix_en}-{i+1}",
                    node_type=node_type,
                    tier=tier,
                    location=f"City-{global_idx}",
                    capacity_limit=float(np_rng.uniform(0.7, 1.0)),
                    lead_time=int(np_rng.integers(5, 60)),
                    substitutability=round(substitutability, 3),
                    region=f"Region-{tier}",
                )
                net.add_node(node)
                nodes_in_tier.append(nid)
                global_idx += 1
            tier_node_ids.append(nodes_in_tier)

        # 2. 生成层间边（确保DAG：从高层级 -> 低层级，即从T(k+1)->T(k)）
        for tier in range(len(tier_sizes) - 1, 0, -1):
            upstream_nodes = tier_node_ids[tier]      # 供应商层（更高tier编号）
            downstream_nodes = tier_node_ids[tier - 1]  # 采购层（更低tier编号）

            # 用优先连接为上游节点分配吸引力权重
            up_weights = self._assign_degree_weights(upstream_nodes, rng)

            # 每个下游节点至少连接一个上游节点（保证连通）
            connected_upstream: set = set()
            for down_nid in downstream_nodes:
                # 基础连接数：受集中度影响（集中度高 -> 少供应商）
                base_links = max(1, round(len(upstream_nodes) / len(downstream_nodes)))
                max_links = max(base_links, round(base_links * (1 + self.network_redundancy * 2)))
                n_links = rng.randint(
                    max(1, round(base_links * (1 - self.supplier_concentration))),
                    min(max_links, len(upstream_nodes)),
                )
                # 优先连接权重采样
                pop = upstream_nodes
                w = [up_weights[n] for n in pop]
                chosen = rng.choices(pop, weights=w, k=min(n_links, len(pop)))
                chosen = list(dict.fromkeys(chosen))  # 去重保序

                for up_nid in chosen:
                    connected_upstream.add(up_nid)
                    dep = float(np_rng.uniform(0.3, 1.0))
                    edge = EdgeData(
                        source=up_nid,
                        target=down_nid,
                        edge_type=EdgeType.MATERIAL_SUPPLY,
                        supply_volume=round(dep * 0.9, 3),
                        dependency_strength=round(dep, 3),
                    )
                    net.add_edge(edge)

            # 确保每个上游节点至少有一条输出边
            for up_nid in upstream_nodes:
                if up_nid not in connected_upstream:
                    down_nid = rng.choice(downstream_nodes)
                    dep = float(np_rng.uniform(0.3, 0.7))
                    edge = EdgeData(
                        source=up_nid,
                        target=down_nid,
                        edge_type=EdgeType.MATERIAL_SUPPLY,
                        supply_volume=round(dep * 0.9, 3),
                        dependency_strength=round(dep, 3),
                    )
                    net.add_edge(edge)

        return net

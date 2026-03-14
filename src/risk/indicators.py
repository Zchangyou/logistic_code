"""
风险量化指标计算模块
Risk Quantitative Indicator Calculation Module
"""
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.network.models import SupplyChainNetwork
from src.risk.factors import RiskCategory, RiskFactor, RiskFactorRegistry


class RiskIndicatorCalculator:
    """风险量化指标计算器 (Risk Indicator Calculator)

    基于网络拓扑信息计算各节点的风险量化指标，并对风险因素注册表进行丰富化。

    Attributes:
        network: SupplyChainNetwork 对象
        topology_report: 拓扑分析报告字典
    """

    def __init__(self, network: SupplyChainNetwork, topology_report: Dict[str, Any]) -> None:
        """初始化风险指标计算器。

        Args:
            network: SupplyChainNetwork 对象
            topology_report: TopologyAnalyzer.get_report() 返回的字典
        """
        self.network = network
        self.topology_report = topology_report
        self._graph = network.get_graph()
        self._node_metrics = topology_report.get("nodes", {})

    # ------------------------------------------------------------------
    # 四类风险指标计算
    # ------------------------------------------------------------------

    def calculate_material_shortage_index(self, node_id: str) -> float:
        """计算材料短缺指数。

        基于：可替代性（反向）、上游入边数量（供应商数量反映多元化程度）、交货周期。

        Args:
            node_id: 节点ID

        Returns:
            材料短缺指数 0-1
        """
        node = self.network.get_node(node_id)
        if node is None:
            return 0.0

        # 可替代性越低，短缺风险越高
        sub_risk = 1.0 - node.substitutability

        # 上游供应商数量：越少风险越高
        in_edges = list(self._graph.in_edges(node_id))
        supplier_count = max(len(in_edges), 1)
        supplier_risk = 1.0 / supplier_count  # 1供应商=1.0, 2=0.5, 3=0.33...

        # 交货周期越长，短缺暴露时间越长
        lead_time_risk = min(node.lead_time / 90.0, 1.0)  # 90天为上限

        index = 0.5 * sub_risk + 0.3 * supplier_risk + 0.2 * lead_time_risk
        return round(float(np.clip(index, 0.0, 1.0)), 4)

    def calculate_concentration_index(self, node_id: str) -> float:
        """计算供应商集中度指数（HHI变体）。

        使用入边的 dependency_strength 计算 HHI 指数。

        Args:
            node_id: 节点ID

        Returns:
            集中度指数 0-1
        """
        in_edges = list(self._graph.in_edges(node_id, data=True))
        if not in_edges:
            return 0.0

        deps = [data.get("dependency_strength", 0.0) for _, _, data in in_edges]
        total = sum(deps)
        if total == 0:
            return 0.0

        shares = [d / total for d in deps]
        hhi = sum(s ** 2 for s in shares)  # 范围 [1/n, 1]

        # 归一化到 0-1：单一供应商 HHI=1, 均匀分散趋近于 0
        n = len(shares)
        min_hhi = 1.0 / n if n > 0 else 1.0
        if n == 1:
            normalized = 1.0
        else:
            normalized = (hhi - min_hhi) / (1.0 - min_hhi)

        return round(float(np.clip(normalized, 0.0, 1.0)), 4)

    def calculate_logistics_risk_index(self, node_id: str) -> float:
        """计算物流风险指数。

        基于：介数中心性（瓶颈节点）、跨区域边比例、交货周期。

        Args:
            node_id: 节点ID

        Returns:
            物流风险指数 0-1
        """
        node = self.network.get_node(node_id)
        if node is None:
            return 0.0

        # 介数中心性（归一化后已在0-1）
        betweenness = self._node_metrics.get(node_id, {}).get("betweenness_centrality", 0.0)

        # 跨区域边比例：上游供应商中跨区域比例越高，物流风险越大
        in_edges = list(self._graph.in_edges(node_id))
        cross_region = 0
        for src, _ in in_edges:
            src_node = self.network.get_node(src)
            if src_node and src_node.region != node.region:
                cross_region += 1
        cross_ratio = cross_region / max(len(in_edges), 1)

        # 交货周期越长，物流中断影响越大
        lead_time_risk = min(node.lead_time / 90.0, 1.0)

        index = 0.3 * betweenness + 0.4 * cross_ratio + 0.3 * lead_time_risk
        return round(float(np.clip(index, 0.0, 1.0)), 4)

    def calculate_demand_volatility_index(self, node_id: str) -> float:
        """计算需求波动指数。

        基于：出度（下游越多，需求放大效应越大）、层级（越靠近OEM，需求直接暴露越大）。

        Args:
            node_id: 节点ID

        Returns:
            需求波动指数 0-1
        """
        node = self.network.get_node(node_id)
        if node is None:
            return 0.0

        # 出度：下游连接数越多，需求波动传导渠道越多
        out_degree = self._graph.out_degree(node_id)
        max_out = max(self._graph.out_degree(n) for n in self._graph.nodes())
        out_ratio = out_degree / max(max_out, 1)

        # 层级：tier越小（越靠近OEM），直接面临终端需求波动
        tier = node.tier
        tier_factor = (4 - tier) / 4.0  # T0=1.0, T1=0.75, T2=0.5, T3=0.25

        index = 0.5 * out_ratio + 0.5 * tier_factor
        return round(float(np.clip(index, 0.0, 1.0)), 4)

    # ------------------------------------------------------------------
    # 注册表丰富化
    # ------------------------------------------------------------------

    def enrich_registry(self, registry: RiskFactorRegistry) -> RiskFactorRegistry:
        """基于网络指标重新计算风险因素的 occurrence_prob 并更新 RPN。

        融合策略：原始 occurrence_prob 权重 0.6 + 网络指标 0.4

        Args:
            registry: 待丰富化的 RiskFactorRegistry

        Returns:
            丰富化后的 RiskFactorRegistry（原地修改并返回）
        """
        category_index_map = {
            RiskCategory.MATERIAL_SHORTAGE:      self.calculate_material_shortage_index,
            RiskCategory.SUPPLIER_CONCENTRATION: self.calculate_concentration_index,
            RiskCategory.LOGISTICS_DISRUPTION:   self.calculate_logistics_risk_index,
            RiskCategory.DEMAND_VOLATILITY:       self.calculate_demand_volatility_index,
        }

        for factor in registry.get_all_factors():
            calc_fn = category_index_map.get(factor.category)
            if calc_fn is None:
                continue
            network_index = calc_fn(factor.node_id)
            # 融合：保留原始专家判断权重 0.6，网络拓扑指标权重 0.4
            enriched_prob = 0.6 * factor.occurrence_prob + 0.4 * network_index
            factor.occurrence_prob = round(float(np.clip(enriched_prob, 0.0, 1.0)), 4)
            factor.calculate_rpn()

        return registry

    # ------------------------------------------------------------------
    # 指标汇总表
    # ------------------------------------------------------------------

    def get_indicator_table(self) -> pd.DataFrame:
        """返回所有节点的风险指标汇总表。

        Returns:
            DataFrame，行为节点，列为四类风险指标及节点基本信息
        """
        rows = []
        for node in self.network.get_all_nodes():
            nid = node.node_id
            rows.append({
                "node_id": nid,
                "name": node.name,
                "tier": node.tier,
                "region": node.region,
                "material_shortage_index": self.calculate_material_shortage_index(nid),
                "concentration_index": self.calculate_concentration_index(nid),
                "logistics_risk_index": self.calculate_logistics_risk_index(nid),
                "demand_volatility_index": self.calculate_demand_volatility_index(nid),
            })
        df = pd.DataFrame(rows)
        df.sort_values(["tier", "node_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

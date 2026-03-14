"""
仿真数据生成模块
Simulation Data Generator Module
"""
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from src.network.models import SupplyChainNetwork


@dataclass
class NodePeriodData:
    """节点周期数据类 (Node Period Data)

    Attributes:
        node_id: 节点ID
        period: 周期编号（1-6）
        capacity_utilization: 产能利用率 0-1
        inventory_level: 库存天数 0-90
        on_time_delivery: 准时交付率 0-1
        supplier_count: 有效供应商数量
        lead_time: 实际交货周期（天）
        demand_volatility: 需求变异系数 0-1
        logistics_reliability: 物流可靠性 0-1
        is_risk_embedded: 是否预埋风险
    """
    node_id: str
    period: int
    capacity_utilization: float
    inventory_level: float
    on_time_delivery: float
    supplier_count: int
    lead_time: int
    demand_volatility: float
    logistics_reliability: float
    is_risk_embedded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "node_id": self.node_id,
            "period": self.period,
            "capacity_utilization": self.capacity_utilization,
            "inventory_level": self.inventory_level,
            "on_time_delivery": self.on_time_delivery,
            "supplier_count": self.supplier_count,
            "lead_time": self.lead_time,
            "demand_volatility": self.demand_volatility,
            "logistics_reliability": self.logistics_reliability,
            "is_risk_embedded": self.is_risk_embedded,
        }


class SimulationDataGenerator:
    """6期仿真数据生成器 (6-Period Simulation Data Generator)

    为汽车发动机供应链案例的所有节点生成6期运营数据，
    按规格预埋8个已知风险点。

    Attributes:
        seed: 随机种子（默认42）
    """

    RISK_NODES = {
        "T3-SI", "T3-RE", "T2-ECU", "T2-E2",
        "T2-SN", "T1-E", "OEM", "T3-CU",
    }

    def __init__(self, seed: int = 42) -> None:
        """初始化生成器。

        Args:
            seed: 随机数种子，保证可复现性
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 预埋风险节点生成
    # ------------------------------------------------------------------

    def _gen_T3_SI(self, period: int) -> NodePeriodData:
        """芯片晶圆：产能紧绷，交货周期持续增长。"""
        lead = min(60 + (period - 1) * 6, 90)  # 60→90 天渐增
        cap = 0.95 + self._rng.uniform(0, 0.04)
        return NodePeriodData(
            node_id="T3-SI", period=period,
            capacity_utilization=round(float(np.clip(cap, 0.95, 1.0)), 4),
            inventory_level=round(float(self._rng.uniform(5, 15)), 2),
            on_time_delivery=round(float(self._rng.uniform(0.70, 0.80)), 4),
            supplier_count=1,
            lead_time=lead,
            demand_volatility=round(float(self._rng.uniform(0.15, 0.25)), 4),
            logistics_reliability=round(float(self._rng.uniform(0.70, 0.80)), 4),
            is_risk_embedded=True,
        )

    def _gen_T3_RE(self, period: int) -> NodePeriodData:
        """稀土材料：高度集中，单一供应商。"""
        cap = 0.90 + self._rng.uniform(0, 0.07)
        return NodePeriodData(
            node_id="T3-RE", period=period,
            capacity_utilization=round(float(np.clip(cap, 0.90, 1.0)), 4),
            inventory_level=round(float(self._rng.uniform(10, 20)), 2),
            on_time_delivery=round(float(self._rng.uniform(0.75, 0.85)), 4),
            supplier_count=1,
            lead_time=21,
            demand_volatility=round(float(self._rng.uniform(0.10, 0.20)), 4),
            logistics_reliability=round(float(self._rng.uniform(0.80, 0.88)), 4),
            is_risk_embedded=True,
        )

    def _gen_T2_ECU(self, period: int) -> NodePeriodData:
        """ECU控制单元：库存持续下降，准时率下滑。"""
        inventory = max(50.0 - (period - 1) * 7.0, 15.0)  # 50→15 渐降
        otd = max(0.95 - (period - 1) * 0.05, 0.70)       # 0.95→0.70
        return NodePeriodData(
            node_id="T2-ECU", period=period,
            capacity_utilization=round(float(self._rng.uniform(0.80, 0.90)), 4),
            inventory_level=round(float(inventory + self._rng.uniform(-2, 2)), 2),
            on_time_delivery=round(float(np.clip(otd + self._rng.uniform(-0.02, 0.02), 0.68, 0.97)), 4),
            supplier_count=2,
            lead_time=40,
            demand_volatility=round(float(self._rng.uniform(0.15, 0.25)), 4),
            logistics_reliability=round(float(self._rng.uniform(0.75, 0.85)), 4),
            is_risk_embedded=True,
        )

    def _gen_T2_E2(self, period: int) -> NodePeriodData:
        """涡轮增压器：物流可靠性持续偏低。"""
        return NodePeriodData(
            node_id="T2-E2", period=period,
            capacity_utilization=round(float(self._rng.uniform(0.70, 0.82)), 4),
            inventory_level=round(float(self._rng.uniform(20, 35)), 2),
            on_time_delivery=round(float(self._rng.uniform(0.72, 0.80)), 4),
            supplier_count=2,
            lead_time=35,
            demand_volatility=round(float(self._rng.uniform(0.10, 0.20)), 4),
            logistics_reliability=0.65,   # 固定偏低
            is_risk_embedded=True,
        )

    def _gen_T2_SN(self, period: int) -> NodePeriodData:
        """传感器模组：受RE和SI影响，产能偏高。"""
        cap = 0.85 + self._rng.uniform(0, 0.10)
        return NodePeriodData(
            node_id="T2-SN", period=period,
            capacity_utilization=round(float(np.clip(cap, 0.85, 1.0)), 4),
            inventory_level=round(float(self._rng.uniform(15, 30)), 2),
            on_time_delivery=round(float(self._rng.uniform(0.75, 0.85)), 4),
            supplier_count=3,
            lead_time=30,
            demand_volatility=round(float(self._rng.uniform(0.15, 0.25)), 4),
            logistics_reliability=round(float(self._rng.uniform(0.78, 0.88)), 4),
            is_risk_embedded=True,
        )

    def _gen_T1_E(self, period: int) -> NodePeriodData:
        """电子电气集成商：准时率下降趋势。"""
        otd = max(0.90 - (period - 1) * 0.03, 0.75)  # 0.90→0.75
        return NodePeriodData(
            node_id="T1-E", period=period,
            capacity_utilization=round(float(self._rng.uniform(0.75, 0.88)), 4),
            inventory_level=round(float(self._rng.uniform(20, 35)), 2),
            on_time_delivery=round(float(np.clip(otd + self._rng.uniform(-0.02, 0.02), 0.73, 0.92)), 4),
            supplier_count=3,
            lead_time=21,
            demand_volatility=round(float(self._rng.uniform(0.12, 0.22)), 4),
            logistics_reliability=round(float(self._rng.uniform(0.80, 0.90)), 4),
            is_risk_embedded=True,
        )

    def _gen_OEM(self, period: int) -> NodePeriodData:
        """总装厂：需求波动明显（季节性）。"""
        return NodePeriodData(
            node_id="OEM", period=period,
            capacity_utilization=round(float(self._rng.uniform(0.75, 0.92)), 4),
            inventory_level=round(float(self._rng.uniform(15, 30)), 2),
            on_time_delivery=round(float(self._rng.uniform(0.85, 0.95)), 4),
            supplier_count=3,
            lead_time=30,
            demand_volatility=0.35,  # 固定高波动
            logistics_reliability=round(float(self._rng.uniform(0.85, 0.93)), 4),
            is_risk_embedded=True,
        )

    def _gen_T3_CU(self, period: int) -> NodePeriodData:
        """铜材：产能偏高，少量供应商。"""
        cap = 0.85 + self._rng.uniform(0, 0.10)
        return NodePeriodData(
            node_id="T3-CU", period=period,
            capacity_utilization=round(float(np.clip(cap, 0.85, 1.0)), 4),
            inventory_level=round(float(self._rng.uniform(20, 40)), 2),
            on_time_delivery=round(float(self._rng.uniform(0.80, 0.90)), 4),
            supplier_count=2,
            lead_time=14,
            demand_volatility=round(float(self._rng.uniform(0.10, 0.20)), 4),
            logistics_reliability=round(float(self._rng.uniform(0.82, 0.92)), 4),
            is_risk_embedded=True,
        )

    # ------------------------------------------------------------------
    # 普通节点生成
    # ------------------------------------------------------------------

    def _gen_normal(self, node_id: str, network: SupplyChainNetwork, period: int) -> NodePeriodData:
        """生成普通节点的正常范围数据。

        Args:
            node_id: 节点ID
            network: 供应链网络对象
            period: 周期编号

        Returns:
            NodePeriodData 对象
        """
        node = network.get_node(node_id)
        lead_time = node.lead_time if node else 21
        # 上游供应商数量来自网络入边
        graph = network.get_graph()
        supplier_count = max(graph.in_degree(node_id), 1)

        return NodePeriodData(
            node_id=node_id, period=period,
            capacity_utilization=round(float(self._rng.uniform(0.65, 0.88)), 4),
            inventory_level=round(float(self._rng.uniform(25, 60)), 2),
            on_time_delivery=round(float(self._rng.uniform(0.85, 0.96)), 4),
            supplier_count=supplier_count,
            lead_time=lead_time,
            demand_volatility=round(float(self._rng.uniform(0.05, 0.18)), 4),
            logistics_reliability=round(float(self._rng.uniform(0.85, 0.96)), 4),
            is_risk_embedded=False,
        )

    # ------------------------------------------------------------------
    # 主生成接口
    # ------------------------------------------------------------------

    def generate(self, network: SupplyChainNetwork) -> List[NodePeriodData]:
        """生成所有节点的6期仿真数据。

        Args:
            network: SupplyChainNetwork 对象

        Returns:
            List[NodePeriodData]，共 节点数×6 条记录
        """
        # 复位随机数生成器以保证可复现
        self._rng = np.random.default_rng(self.seed)

        risk_generators = {
            "T3-SI":  self._gen_T3_SI,
            "T3-RE":  self._gen_T3_RE,
            "T2-ECU": self._gen_T2_ECU,
            "T2-E2":  self._gen_T2_E2,
            "T2-SN":  self._gen_T2_SN,
            "T1-E":   self._gen_T1_E,
            "OEM":    self._gen_OEM,
            "T3-CU":  self._gen_T3_CU,
        }

        all_data: List[NodePeriodData] = []
        for node in network.get_all_nodes():
            nid = node.node_id
            gen_fn = risk_generators.get(nid)
            for period in range(1, 7):
                if gen_fn is not None:
                    record = gen_fn(period)
                else:
                    record = self._gen_normal(nid, network, period)
                all_data.append(record)

        return all_data

    def to_dataframe(self, data: List[NodePeriodData]) -> pd.DataFrame:
        """将仿真数据列表转换为 DataFrame。

        Args:
            data: NodePeriodData 列表

        Returns:
            pandas DataFrame
        """
        rows = [d.to_dict() for d in data]
        df = pd.DataFrame(rows)
        df.sort_values(["node_id", "period"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def save_csv(self, data: List[NodePeriodData], output_path: str) -> None:
        """保存仿真数据到 CSV 文件。

        Args:
            data: NodePeriodData 列表
            output_path: 输出文件路径（含文件名）
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = self.to_dataframe(data)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  已保存仿真数据: {output_path}  ({len(df)} 条记录)")

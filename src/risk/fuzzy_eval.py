"""
模糊综合风险评价模块
Fuzzy Comprehensive Risk Evaluation Module
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# 风险等级标签
RISK_LEVELS = ["very_low", "low", "medium", "high", "very_high"]
RISK_LEVEL_CENTERS = [0.1, 0.25, 0.5, 0.7, 0.9]  # 各等级代表值（用于重心法解模糊）


def _trimf(x: float, abc: List[float]) -> float:
    """三角隶属度函数。

    Args:
        x: 输入值
        abc: [a, b, c] 三角形顶点

    Returns:
        隶属度值 0-1
    """
    a, b, c = abc
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0


def _trapmf(x: float, abcd: List[float]) -> float:
    """梯形隶属度函数。

    Args:
        x: 输入值
        abcd: [a, b, c, d] 梯形顶点

    Returns:
        隶属度值 0-1
    """
    a, b, c, d = abcd
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (d - x) / (d - c) if d != c else 1.0


def _compute_memberships(value: float) -> List[float]:
    """计算输入值对5个风险等级的隶属度。

    风险等级划分（输入值范围 0-1）：
    - very_low:  [0.0, 0.0, 0.10, 0.20]  梯形
    - low:       [0.10, 0.25, 0.40]       三角
    - medium:    [0.30, 0.50, 0.65]       三角
    - high:      [0.55, 0.70, 0.85]       三角
    - very_high: [0.75, 0.88, 1.0, 1.0]  梯形

    Args:
        value: 输入值 0-1

    Returns:
        5个等级的隶属度列表 [very_low, low, medium, high, very_high]
    """
    v = float(np.clip(value, 0.0, 1.0))
    memberships = [
        _trapmf(v, [0.0, 0.0, 0.10, 0.20]),    # very_low
        _trimf(v, [0.10, 0.25, 0.40]),           # low
        _trimf(v, [0.30, 0.50, 0.65]),           # medium
        _trimf(v, [0.55, 0.70, 0.85]),           # high
        _trapmf(v, [0.75, 0.88, 1.0, 1.0]),     # very_high
    ]
    return memberships


class FuzzyRiskEvaluator:
    """模糊综合风险评价器 (Fuzzy Comprehensive Risk Evaluator)

    使用加权模糊综合评价法对供应链节点进行多维度风险评估。

    维度权重：
    - capacity_utilization: 0.25（产能利用率）
    - inventory_level:      0.20（库存水平，反向）
    - on_time_delivery:     0.20（准时交货率，反向）
    - logistics_reliability:0.15（物流可靠性，反向）
    - supplier_concentration:0.20（供应商集中度 HHI）
    """

    # 各维度权重
    WEIGHTS: Dict[str, float] = {
        "capacity_utilization":   0.30,
        "inventory_level":        0.25,
        "on_time_delivery":       0.25,
        "logistics_reliability":  0.15,
        "supplier_concentration": 0.05,
    }

    def __init__(self) -> None:
        """初始化模糊评价器。"""
        pass

    def _normalize_indicators(self, raw: Dict[str, float]) -> Dict[str, float]:
        """将原始指标归一化为风险分（越大风险越高）。

        Args:
            raw: 原始指标字典

        Returns:
            归一化后的风险指标字典（0-1，越大表示风险越高）
        """
        cap = float(np.clip(raw.get("capacity_utilization", 0.5), 0.0, 1.0))
        inv = float(np.clip(raw.get("inventory_level", 45.0), 0.0, 90.0)) / 90.0
        otd = float(np.clip(raw.get("on_time_delivery", 0.9), 0.0, 1.0))
        log = float(np.clip(raw.get("logistics_reliability", 0.9), 0.0, 1.0))
        hhi = float(np.clip(raw.get("supplier_concentration", 0.5), 0.0, 1.0))

        return {
            "capacity_utilization":   cap,           # 高利用率 → 高风险
            "inventory_level":        1.0 - inv,     # 低库存 → 高风险
            "on_time_delivery":       1.0 - otd,     # 低OTD → 高风险
            "logistics_reliability":  1.0 - log,     # 低可靠性 → 高风险
            "supplier_concentration": hhi,            # 高集中 → 高风险
        }

    def evaluate_node(
        self, node_id: str, indicators: Dict[str, float]
    ) -> Dict:
        """对单个节点进行模糊综合风险评价。

        Args:
            node_id: 节点ID（用于标识，不影响计算）
            indicators: 原始指标字典，包含：
                - capacity_utilization (0-1)
                - inventory_level (0-90)
                - on_time_delivery (0-1)
                - logistics_reliability (0-1)
                - supplier_concentration (0-1, HHI)

        Returns:
            包含 risk_score、risk_level、membership_vector 的字典
        """
        norm = self._normalize_indicators(indicators)

        # 加权综合隶属度向量
        combined = np.zeros(5)
        for dim, weight in self.WEIGHTS.items():
            val = norm.get(dim, 0.5)
            memberships = np.array(_compute_memberships(val))
            combined += weight * memberships

        # 归一化综合隶属度（模糊向量归一化）
        total = combined.sum()
        if total > 0:
            combined = combined / total

        # 重心法解模糊：风险分 = Σ(隶属度 × 等级代表值)
        centers = np.array(RISK_LEVEL_CENTERS)
        risk_score = float(np.dot(combined, centers))
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # 确定风险等级（最大隶属度原则）
        max_idx = int(np.argmax(combined))
        risk_level = RISK_LEVELS[max_idx]

        return {
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "membership_vector": combined.tolist(),
        }

    def evaluate_all_nodes(
        self, simulation_df: pd.DataFrame, period: int = 6
    ) -> pd.DataFrame:
        """对所有节点使用指定期数据进行模糊综合评价。

        Args:
            simulation_df: 仿真数据 DataFrame（含 node_id, period 等列）
            period: 使用的期次（默认第6期）

        Returns:
            包含 node_id, risk_score, risk_level, membership_vector 的 DataFrame
        """
        period_df = simulation_df[simulation_df["period"] == period].copy()

        rows = []
        for _, row in period_df.iterrows():
            node_id = row["node_id"]

            # 计算供应商集中度 HHI（单供应商=1，多供应商按均匀分配近似）
            supplier_count = max(int(row.get("supplier_count", 1)), 1)
            hhi = 1.0 / supplier_count  # 均匀分配近似

            indicators = {
                "capacity_utilization":   float(row["capacity_utilization"]),
                "inventory_level":        float(row["inventory_level"]),
                "on_time_delivery":       float(row["on_time_delivery"]),
                "logistics_reliability":  float(row["logistics_reliability"]),
                "supplier_concentration": hhi,
            }

            result = self.evaluate_node(node_id, indicators)
            rows.append({
                "node_id": node_id,
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "membership_vector": result["membership_vector"],
                "capacity_utilization": float(row["capacity_utilization"]),
                "inventory_level": float(row["inventory_level"]),
                "on_time_delivery": float(row["on_time_delivery"]),
                "logistics_reliability": float(row["logistics_reliability"]),
                "supplier_concentration": hhi,
            })

        return pd.DataFrame(rows)

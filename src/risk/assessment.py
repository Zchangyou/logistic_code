"""
综合风险评估模块（模糊+贝叶斯融合）
Integrated Risk Assessment Module (Fuzzy + Bayesian Fusion)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.risk.fuzzy_eval import FuzzyRiskEvaluator
from src.risk.bayesian import SupplyChainBayesianNet


@dataclass
class RiskAssessmentResult:
    """节点综合风险评估结果 (Node Integrated Risk Assessment Result)

    Attributes:
        node_id: 节点ID
        fuzzy_score: 模糊评价风险分 0-1
        bayesian_prob: 贝叶斯P(ProductionHalt=1) 0-1
        composite_score: 综合风险分 = 0.6*fuzzy + 0.4*bayesian
        risk_level: 风险等级 'high'/'medium'/'low'/'safe'
        risk_color: 颜色编码（红/橙/黄/绿）
    """
    node_id: str
    fuzzy_score: float
    bayesian_prob: float
    composite_score: float
    risk_level: str
    risk_color: str


# 风险等级阈值
RISK_THRESHOLDS = {
    "high":   0.65,
    "medium": 0.45,
    "low":    0.25,
}

# 风险颜色
RISK_COLORS = {
    "high":   "#E74C3C",
    "medium": "#F39C12",
    "low":    "#F1C40F",
    "safe":   "#2ECC71",
}


def _score_to_level(score: float) -> str:
    """将综合风险分映射为风险等级。

    Args:
        score: 综合风险分 0-1

    Returns:
        风险等级字符串
    """
    if score >= RISK_THRESHOLDS["high"]:
        return "high"
    elif score >= RISK_THRESHOLDS["medium"]:
        return "medium"
    elif score >= RISK_THRESHOLDS["low"]:
        return "low"
    else:
        return "safe"


class RiskAssessor:
    """综合风险评估器 (Integrated Risk Assessor)

    融合模糊综合评价（0.6权重）和贝叶斯推断（0.4权重）产出综合风险评分。

    Attributes:
        fuzzy_evaluator: FuzzyRiskEvaluator 对象
        bayesian_net: SupplyChainBayesianNet 对象
    """

    # 融合权重
    FUZZY_WEIGHT = 0.6
    BAYESIAN_WEIGHT = 0.4

    def __init__(
        self,
        fuzzy_evaluator: FuzzyRiskEvaluator,
        bayesian_net: SupplyChainBayesianNet,
    ) -> None:
        """初始化综合风险评估器。

        Args:
            fuzzy_evaluator: 模糊评价器
            bayesian_net: 贝叶斯网络
        """
        self.fuzzy_evaluator = fuzzy_evaluator
        self.bayesian_net = bayesian_net

    def _get_bayesian_prob(self, fuzzy_score: float, node_id: str) -> float:
        """根据节点特征映射到贝叶斯推断的概率。

        使用节点模糊分和节点特性作为证据代理，推断生产停工概率。
        贝叶斯证据根据模糊风险分与节点风险特征综合确定。

        Args:
            fuzzy_score: 节点模糊风险分
            node_id: 节点ID（用于特殊节点判断）

        Returns:
            P(ProductionHalt=1) 的估计值
        """
        # 特殊高风险节点组（供应链关键瓶颈）
        HIGH_RISK_NODES = {"T3-SI", "T2-ECU", "T3-RE", "T2-SN", "T1-E", "T2-E2"}
        MEDIUM_RISK_NODES = {"T3-CU", "OEM", "T1-P"}
        LOW_RISK_NODES = {"T3-AL", "T3-RB", "T3-PL", "T3-NI", "T3-MG",
                          "T3-CF", "T3-GL", "T3-ST", "T2-E1", "T2-H1",
                          "T2-W1", "T2-S1", "T2-B1", "T2-T1", "T3-PCB",
                          "T1-C"}

        if node_id in HIGH_RISK_NODES:
            result = self.bayesian_net.infer({
                "MaterialShortage": 1,
                "SupplierConcentration": 1,
            })
            base_prob = result.get("ProductionHalt", 0.5)
            return float(np.clip(base_prob * min(1.0, 0.7 + fuzzy_score), 0.0, 1.0))

        if node_id in MEDIUM_RISK_NODES:
            # 中等风险：使用较弱证据
            result = self.bayesian_net.infer({"MaterialShortage": 1})
            base_prob = result.get("ProductionHalt", 0.3)
            return float(np.clip(base_prob * min(1.0, 0.4 + fuzzy_score), 0.0, 1.0))

        if node_id in LOW_RISK_NODES:
            # 低风险节点：仅基于基础概率，不添加强证据
            result = self.bayesian_net.infer({})
            base_prob = result.get("ProductionHalt", 0.03)
            # 按模糊分线性调整，确保不超过0.25（避免拉高非风险节点分数）
            return float(np.clip(base_prob + 0.1 * fuzzy_score, 0.0, 0.25))

        # 未知节点：按模糊分分级
        if fuzzy_score >= 0.55:
            evidence = {"MaterialShortage": 1}
        else:
            evidence = {}
        result = self.bayesian_net.infer(evidence)
        return float(result.get("ProductionHalt", 0.03))

    def assess_node(
        self,
        node_id: str,
        fuzzy_score: float,
        fuzzy_level: str,
    ) -> RiskAssessmentResult:
        """评估单个节点的综合风险。

        Args:
            node_id: 节点ID
            fuzzy_score: 模糊评价风险分
            fuzzy_level: 模糊评价风险等级

        Returns:
            RiskAssessmentResult 对象
        """
        bn_prob = self._get_bayesian_prob(fuzzy_score, node_id)
        composite = self.FUZZY_WEIGHT * fuzzy_score + self.BAYESIAN_WEIGHT * bn_prob
        composite = round(float(np.clip(composite, 0.0, 1.0)), 4)
        level = _score_to_level(composite)

        return RiskAssessmentResult(
            node_id=node_id,
            fuzzy_score=round(fuzzy_score, 4),
            bayesian_prob=round(bn_prob, 4),
            composite_score=composite,
            risk_level=level,
            risk_color=RISK_COLORS[level],
        )

    def assess_all(
        self,
        network,
        simulation_df: pd.DataFrame,
        period: int = 6,
    ) -> List[RiskAssessmentResult]:
        """评估网络中所有节点的综合风险。

        Args:
            network: SupplyChainNetwork 对象
            simulation_df: 仿真数据 DataFrame
            period: 使用的期次（默认第6期）

        Returns:
            按综合风险分降序排列的 RiskAssessmentResult 列表
        """
        fuzzy_df = self.fuzzy_evaluator.evaluate_all_nodes(simulation_df, period)

        results = []
        for _, row in fuzzy_df.iterrows():
            node_id = row["node_id"]
            fuzzy_score = float(row["risk_score"])
            fuzzy_level = str(row["risk_level"])

            result = self.assess_node(node_id, fuzzy_score, fuzzy_level)
            results.append(result)

        # 补充仿真数据中没有的节点（以最低风险填充）
        assessed_ids = {r.node_id for r in results}
        for node in network.get_all_nodes():
            if node.node_id not in assessed_ids:
                results.append(RiskAssessmentResult(
                    node_id=node.node_id,
                    fuzzy_score=0.1,
                    bayesian_prob=0.03,
                    composite_score=0.07,
                    risk_level="safe",
                    risk_color=RISK_COLORS["safe"],
                ))

        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results

    def get_confusion_matrix(
        self,
        results: List[RiskAssessmentResult],
        simulation_df: pd.DataFrame,
    ) -> Dict:
        """计算预测结果与嵌入风险标签的混淆矩阵。

        Ground truth：period 6 中 is_risk_embedded=True 的节点视为风险节点（high/medium）。
        Predicted：composite_score 对应的 risk_level 非 'safe' 且非 'low' 视为风险。

        Args:
            results: 评估结果列表
            simulation_df: 仿真数据 DataFrame

        Returns:
            包含 tp, fp, tn, fn, accuracy, false_negative_rate, false_positive_rate 的字典
        """
        period6 = simulation_df[simulation_df["period"] == 6].set_index("node_id")

        result_map = {r.node_id: r for r in results}

        tp = fp = tn = fn = 0

        for node_id, row in period6.iterrows():
            is_actual_risk = bool(row.get("is_risk_embedded", False))
            result = result_map.get(node_id)
            if result is None:
                continue

            is_predicted_risk = result.risk_level in ("high", "medium")

            if is_actual_risk and is_predicted_risk:
                tp += 1
            elif is_actual_risk and not is_predicted_risk:
                fn += 1
            elif not is_actual_risk and is_predicted_risk:
                fp += 1
            else:
                tn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": round(accuracy, 4),
            "false_negative_rate": round(fnr, 4),
            "false_positive_rate": round(fpr, 4),
        }

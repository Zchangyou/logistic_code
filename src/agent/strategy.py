"""
防控策略推荐模块
Risk Prevention Strategy Recommendation Module

功能：
- 构建防控策略知识图谱（策略 ↔ 风险类型 ↔ 网络特征）
- 根据诊断报告和节点数据自动推荐策略组合
- 策略效果量化评估（风险降低 / 成本 / 实施周期）
- 支持多策略对比分析（用于 F5-2 图表）
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

from src.agent.knowledge_base import SupplyChainKnowledgeBase


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class StrategyOption:
    """单个防控策略选项"""
    strategy_id: str
    name: str
    name_en: str
    risk_types: List[str]
    risk_reduction: float       # 风险降低比例 [0, 1]
    cost_level: int             # 实施成本等级 1-5
    lead_time_days: int         # 实施周期（天）
    feasibility_score: float    # 可行性评分 [0, 1]
    priority_score: float       # 最终推荐优先级评分 [0, 1]
    description: str
    expected_risk_after: float  # 预期处置后风险评分


@dataclass
class StrategyPlan:
    """策略推荐方案"""
    target_node_id: str
    target_node_name: str
    current_risk_score: float
    risk_type: str
    recommended_strategies: List[StrategyOption]
    combined_risk_reduction: float     # 组合策略总风险降低
    expected_risk_after_combo: float
    implementation_order: List[str]    # 按优先级排序的策略ID
    cost_estimate: str                 # 低/中/高
    rationale: str


# ------------------------------------------------------------------ #
# 主推荐类
# ------------------------------------------------------------------ #
class StrategyRecommender:
    """防控策略推荐引擎"""

    NODE_NAMES = {
        "T3-SI": "芯片晶圆供应商", "T3-RE": "稀土材料供应商",
        "T2-ECU": "ECU控制单元", "T2-SN": "传感器模组",
        "T1-E": "电子电气系统集成商", "T2-E2": "涡轮增压器",
        "T3-CU": "铜材供应商", "OEM": "总装厂",
    }

    # 节点风险类型映射
    NODE_RISK_TYPES = {
        "T3-SI": "材料短缺",
        "T3-RE": "供应商集中",
        "T2-ECU": "材料短缺",     # 级联短缺归类材料短缺
        "T2-SN": "材料短缺",
        "T1-E": "供应商集中",
        "T2-E2": "物流中断",
        "T3-CU": "材料短缺",
        "OEM": "需求波动",
    }

    # 节点类型
    NODE_TYPES = {
        "T3-SI": "raw_material", "T3-RE": "raw_material",
        "T3-CU": "raw_material", "T3-ST": "raw_material",
        "T3-AL": "raw_material", "T3-NI": "raw_material",
        "T3-RB": "raw_material", "T3-PCB": "raw_material",
        "T3-PL": "raw_material", "T3-CF": "raw_material",
        "T3-MG": "raw_material", "T3-GL": "raw_material",
        "T2-ECU": "component", "T2-SN": "component",
        "T2-E1": "component", "T2-E2": "component",
        "T2-T1": "component", "T2-B1": "component",
        "T2-S1": "component", "T2-W1": "component", "T2-H1": "component",
        "T1-E": "integrator", "T1-P": "integrator", "T1-C": "integrator",
        "OEM": "oem",
    }

    def __init__(self, knowledge_base: Optional[SupplyChainKnowledgeBase] = None):
        self.kb = knowledge_base or SupplyChainKnowledgeBase()

    # ------------------------------------------------------------------ #
    # 主推荐接口
    # ------------------------------------------------------------------ #
    def recommend(
        self,
        node_id: str,
        current_risk_score: float,
        node_data: Optional[Dict] = None,
        top_k: int = 4,
    ) -> StrategyPlan:
        risk_type = self.NODE_RISK_TYPES.get(node_id, "综合风险")
        node_name = self.NODE_NAMES.get(node_id, node_id)

        # 获取候选策略
        raw_strategies = self.kb.get_strategies_for_risk(risk_type, node_data, top_k=top_k + 2)

        # 转换为 StrategyOption
        options = []
        for s in raw_strategies:
            feasibility = self._calc_feasibility(s, node_id, node_data)
            priority = self._calc_priority(s, feasibility, current_risk_score)
            expected_after = current_risk_score * (1.0 - s["risk_reduction"] * feasibility)
            options.append(StrategyOption(
                strategy_id=s["id"],
                name=s["name"],
                name_en=s["name_en"],
                risk_types=s["applicable_risks"],
                risk_reduction=s["risk_reduction"],
                cost_level=s["cost_level"],
                lead_time_days=s["lead_time_days"],
                feasibility_score=feasibility,
                priority_score=priority,
                description=s["description"],
                expected_risk_after=round(expected_after, 4),
            ))

        # 按优先级排序，取前 top_k
        options.sort(key=lambda x: x.priority_score, reverse=True)
        options = options[:top_k]

        # 组合效果（假设独立叠加，有衰减）
        combined_reduction = self._combined_reduction([o.risk_reduction for o in options[:2]])
        expected_after_combo = current_risk_score * (1.0 - combined_reduction)

        # 实施顺序（快见效 + 高优先级优先）
        impl_order = sorted(options, key=lambda x: x.lead_time_days)
        impl_order_ids = [o.strategy_id for o in impl_order]

        cost_levels = [o.cost_level for o in options[:2]]
        avg_cost = sum(cost_levels) / len(cost_levels) if cost_levels else 3
        cost_estimate = "低" if avg_cost <= 2 else ("高" if avg_cost >= 4 else "中")

        rationale = (
            f"针对{node_name}（风险评分 {current_risk_score:.3f}）的{risk_type}，"
            f"推荐优先实施《{options[0].name}》（可行性 {options[0].feasibility_score:.2f}），"
            f"组合前两项策略预期可将风险降至 {expected_after_combo:.3f}，"
            f"降低幅度 {combined_reduction:.1%}。"
        )

        return StrategyPlan(
            target_node_id=node_id,
            target_node_name=node_name,
            current_risk_score=current_risk_score,
            risk_type=risk_type,
            recommended_strategies=options,
            combined_risk_reduction=round(combined_reduction, 4),
            expected_risk_after_combo=round(expected_after_combo, 4),
            implementation_order=impl_order_ids,
            cost_estimate=cost_estimate,
            rationale=rationale,
        )

    def recommend_for_scenario(
        self,
        scenario_name: str,
        assessment_results: List[Dict],
        sim_data: Optional[Dict] = None,
    ) -> List[StrategyPlan]:
        """为某个风险场景生成全链路策略推荐"""
        # 取风险最高的 3 个节点
        top_nodes = sorted(assessment_results, key=lambda x: x["composite_score"], reverse=True)[:3]
        plans = []
        for node in top_nodes:
            node_data = sim_data.get(node["node_id"]) if sim_data else None
            plan = self.recommend(
                node_id=node["node_id"],
                current_risk_score=node["composite_score"],
                node_data=node_data,
                top_k=3,
            )
            plans.append(plan)
        return plans

    # ------------------------------------------------------------------ #
    # 策略对比分析（F5-2 图表数据来源）
    # ------------------------------------------------------------------ #
    def compare_strategies_for_node(
        self,
        node_id: str,
        current_risk_score: float,
        strategy_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """对指定节点的多个策略进行横向对比，返回可视化数据"""
        risk_type = self.NODE_RISK_TYPES.get(node_id, "综合风险")
        all_strategies = self.kb.strategies

        if strategy_ids:
            candidates = [s for s in all_strategies if s["id"] in strategy_ids]
        else:
            candidates = [
                s for s in all_strategies
                if risk_type in s.get("applicable_risks", [])
            ][:5]

        comparison = []
        for s in candidates:
            feasibility = self._calc_feasibility(s, node_id, None)
            risk_after = current_risk_score * (1.0 - s["risk_reduction"] * feasibility)
            comparison.append({
                "strategy_id": s["id"],
                "name": s["name"],
                "name_en": s["name_en"],
                "risk_reduction_pct": round(s["risk_reduction"] * feasibility * 100, 1),
                "risk_after": round(risk_after, 4),
                "cost_level": s["cost_level"],
                "lead_time_days": s["lead_time_days"],
                "feasibility": round(feasibility, 3),
            })

        comparison.sort(key=lambda x: x["risk_reduction_pct"], reverse=True)
        return comparison

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calc_feasibility(s: Dict, node_id: str, node_data: Optional[Dict]) -> float:
        """计算策略在当前节点的可行性分"""
        base = s["feasibility_base"]
        node_type = StrategyRecommender.NODE_TYPES.get(node_id, "component")
        applicable = s.get("applicable_node_types", [])
        if applicable and node_type not in applicable:
            base *= 0.80   # 节点类型不完全适配

        # 成本惩罚（成本越高，可行性折扣）
        cost = s.get("cost_level", 3)
        cost_penalty = 1.0 - (cost - 1) * 0.04
        return round(min(0.99, base * cost_penalty), 3)

    @staticmethod
    def _calc_priority(s: Dict, feasibility: float, current_risk: float) -> float:
        """综合优先级分"""
        risk_reduction_impact = s["risk_reduction"] * feasibility * current_risk
        speed_bonus = 1.0 / (1.0 + s["lead_time_days"] / 365.0)
        return round(risk_reduction_impact * 0.6 + feasibility * 0.2 + speed_bonus * 0.2, 4)

    @staticmethod
    def _combined_reduction(reductions: List[float]) -> float:
        """多策略组合风险降低（独立假设下，相乘叠加）"""
        if not reductions:
            return 0.0
        combined_remaining = 1.0
        for r in reductions:
            combined_remaining *= (1.0 - r * 0.8)  # 0.8 为协同效率系数
        return round(1.0 - combined_remaining, 4)

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def plan_to_dict(self, plan: StrategyPlan) -> Dict:
        return {
            "target_node_id": plan.target_node_id,
            "target_node_name": plan.target_node_name,
            "current_risk_score": plan.current_risk_score,
            "risk_type": plan.risk_type,
            "combined_risk_reduction": plan.combined_risk_reduction,
            "expected_risk_after_combo": plan.expected_risk_after_combo,
            "cost_estimate": plan.cost_estimate,
            "rationale": plan.rationale,
            "strategies": [
                {
                    "strategy_id": o.strategy_id,
                    "name": o.name,
                    "risk_reduction": o.risk_reduction,
                    "feasibility_score": o.feasibility_score,
                    "priority_score": o.priority_score,
                    "lead_time_days": o.lead_time_days,
                    "cost_level": o.cost_level,
                    "expected_risk_after": o.expected_risk_after,
                    "description": o.description,
                }
                for o in plan.recommended_strategies
            ],
        }

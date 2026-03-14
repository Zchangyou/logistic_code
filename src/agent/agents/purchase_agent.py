"""
采购智能体
Purchase Agent

功能：
- 识别单一来源与高集中采购风险
- 生成供应商替换与长协锁量建议
- 输出结构化采购处置方案 PurchaseProposal
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class PurchaseAction:
    """单节点采购处置动作 (Purchase Action for a node)"""
    node_id: str
    node_name: str
    current_supplier_count: int
    target_supplier_count: int
    concentration_signal: float
    priority: str
    action: str
    cost_index: float
    expected_risk_reduction: float


@dataclass
class PurchaseProposal:
    """采购智能体处置方案 (Purchase Agent Proposal)"""
    scenario: str
    agent_name: str = "采购策略优化智能体"
    agent_name_en: str = "Purchase Strategy Agent"
    actions: List[PurchaseAction] = field(default_factory=list)
    total_risk_reduction: float = 0.0
    total_cost_index: float = 0.0
    summary: str = ""


_NODE_NAMES = {
    "T3-SI": "芯片晶圆供应商", "T3-RE": "稀土材料供应商",
    "T2-ECU": "ECU控制单元", "T2-SN": "传感器模组",
    "T1-E": "电子电气系统集成商", "T2-E2": "涡轮增压器",
    "T3-CU": "铜材供应商", "OEM": "总装厂",
}


class PurchaseAgent:
    """采购策略优化智能体 (Purchase Strategy Optimization Agent)"""

    def analyze(
        self,
        sim_data: pd.DataFrame,
        risk_results: List[Dict],
        scenario: str = "S1_chip_shortage",
        target_nodes: Optional[List[str]] = None,
    ) -> PurchaseProposal:
        latest = (
            sim_data[sim_data["period"] == sim_data["period"].max()]
            .set_index("node_id")
        )
        risk_map = {r["node_id"]: r for r in risk_results}
        nodes = target_nodes if target_nodes else list(latest.index)

        actions: List[PurchaseAction] = []
        for node_id in nodes:
            if node_id not in latest.index:
                continue

            supplier_count = int(latest.loc[node_id, "supplier_count"])
            risk_score = float(risk_map.get(node_id, {}).get("composite_score", 0.0))
            concentration_signal = round(max(0.0, 1.0 / max(1, supplier_count)), 4)

            if supplier_count <= 1 and risk_score >= 0.45:
                priority, target, cost, red = "高", 2, 0.55, 0.18
                action = "立即启动第二供应商认证与试供，签订长协锁量以分散单一来源风险"
            elif supplier_count <= 2 and risk_score >= 0.30:
                priority, target, cost, red = "中", 3, 0.35, 0.10
                action = "开展备选供应商开发，优化采购份额分配，降低集中采购依赖"
            else:
                continue

            actions.append(PurchaseAction(
                node_id=node_id,
                node_name=_NODE_NAMES.get(node_id, node_id),
                current_supplier_count=supplier_count,
                target_supplier_count=target,
                concentration_signal=concentration_signal,
                priority=priority,
                action=action,
                cost_index=cost,
                expected_risk_reduction=red,
            ))

        prio = {"高": 0, "中": 1, "低": 2}
        actions.sort(key=lambda x: prio.get(x.priority, 3))
        top3 = actions[:3]
        total_risk_red = min(0.45, sum(a.expected_risk_reduction for a in top3))
        total_cost = sum(a.cost_index for a in top3) / max(len(top3), 1)
        high_cnt = sum(1 for a in actions if a.priority == "高")

        return PurchaseProposal(
            scenario=scenario,
            actions=actions,
            total_risk_reduction=round(total_risk_red, 4),
            total_cost_index=round(total_cost, 4),
            summary=(
                f"采购智能体：识别 {len(actions)} 个节点存在采购集中风险，"
                f"其中 {high_cnt} 个高优先级；"
                f"综合处置预期降低风险 {total_risk_red:.1%}，"
                f"综合成本指数 {total_cost:.2f}。"
            ),
        )


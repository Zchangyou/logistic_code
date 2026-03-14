"""
多智能体协同决策模块
Multi-Agent Coordination Module

功能：
- 管理库存/物流/需求/协调 四个智能体的生命周期
- 实现智能体间通信协议（消息传递接口）
- 运行全流程协同决策并生成综合处置报告
- 执行处置结果反馈闭环（更新网络节点风险分）
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
import os

import pandas as pd
import numpy as np

from src.agent.agents.inventory_agent import InventoryAgent, InventoryProposal
from src.agent.agents.logistics_agent import LogisticsAgent, LogisticsProposal
from src.agent.agents.demand_agent import DemandAgent, DemandProposal
from src.agent.agents.coordinator import CoordinatorAgent, IntegratedPlan


# --------------------------------------------------------------------------- #
# 消息协议
# --------------------------------------------------------------------------- #
@dataclass
class AgentMessage:
    """智能体间通信消息 (Inter-Agent Message)"""
    sender: str            # 发送方智能体名称
    receiver: str          # 接收方（"coordinator" / "broadcast"）
    msg_type: str          # "proposal" / "query" / "ack" / "conflict"
    payload: Any           # 消息内容
    timestamp: int = 0     # 消息序号


@dataclass
class DisposalReport:
    """全链路处置报告 (Full-chain Disposal Report)"""
    scenario: str
    inventory_proposal: InventoryProposal
    logistics_proposal: LogisticsProposal
    demand_proposal: DemandProposal
    integrated_plan: IntegratedPlan
    messages: List[AgentMessage] = field(default_factory=list)
    pre_disposal_risk_scores: Dict[str, float] = field(default_factory=dict)
    post_disposal_risk_scores: Dict[str, float] = field(default_factory=dict)
    overall_risk_reduction: float = 0.0

    def to_dict(self) -> Dict:
        def _plan_to_dict(plan: IntegratedPlan) -> Dict:
            return {
                "scenario": plan.scenario,
                "summary": plan.summary,
                "total_expected_risk_reduction": plan.total_expected_risk_reduction,
                "total_cost_index": plan.total_cost_index,
                "inventory_contribution": plan.inventory_contribution,
                "logistics_contribution": plan.logistics_contribution,
                "demand_contribution": plan.demand_contribution,
                "conflicts": [
                    {"node_id": c.node_id, "type": c.conflict_type, "resolution": c.resolution}
                    for c in plan.conflicts
                ],
                "actions": [
                    {
                        "node_id": a.node_id,
                        "node_name": a.node_name,
                        "priority": a.priority,
                        "source_agents": a.source_agents,
                        "combined_action": a.combined_action,
                        "cost_index": a.cost_index,
                        "expected_risk_reduction": a.expected_risk_reduction,
                        "scores": {
                            "risk_reduction": a.score_risk_reduction,
                            "cost": a.score_cost,
                            "efficiency": a.score_efficiency,
                            "feasibility": a.score_feasibility,
                            "urgency": a.score_urgency,
                        },
                    }
                    for a in plan.integrated_actions
                ],
            }

        return {
            "scenario": self.scenario,
            "inventory_summary": self.inventory_proposal.summary,
            "logistics_summary": self.logistics_proposal.summary,
            "demand_summary": self.demand_proposal.summary,
            "integrated_plan": _plan_to_dict(self.integrated_plan),
            "pre_disposal_risk_scores": self.pre_disposal_risk_scores,
            "post_disposal_risk_scores": self.post_disposal_risk_scores,
            "overall_risk_reduction": self.overall_risk_reduction,
            "message_count": len(self.messages),
        }


# --------------------------------------------------------------------------- #
# 多智能体协同系统
# --------------------------------------------------------------------------- #
class MultiAgentSystem:
    """多智能体协同系统 (Multi-Agent Coordination System)

    管理四个智能体的生命周期与协作流程：
    1. 并行运行三个领域智能体（库存/物流/需求）
    2. 收集提案，传递给协调智能体
    3. 协调智能体消解冲突，生成综合处置计划
    4. 执行反馈闭环，更新网络风险评分
    """

    def __init__(self) -> None:
        self.inventory_agent = InventoryAgent()
        self.logistics_agent = LogisticsAgent()
        self.demand_agent = DemandAgent()
        self.coordinator = CoordinatorAgent()
        self._message_log: List[AgentMessage] = []
        self._tick = 0

    # ------------------------------------------------------------------ #
    # 主协调流程
    # ------------------------------------------------------------------ #
    def run(
        self,
        sim_data: pd.DataFrame,
        risk_results: List[Dict],
        scenario: str = "S1_chip_shortage",
        target_nodes: Optional[List[str]] = None,
        budget_limit: float = 2.0,
    ) -> DisposalReport:
        """运行完整多智能体协同处置流程。

        Args:
            sim_data: 6期仿真数据
            risk_results: 综合风险评估结果
            scenario: 场景名称
            target_nodes: 限定分析节点列表
            budget_limit: 总成本预算上限

        Returns:
            DisposalReport: 完整处置报告（含处置前后风险对比）
        """
        self._message_log.clear()
        self._tick = 0

        # 处置前风险评分
        pre_scores = {r["node_id"]: r["composite_score"] for r in risk_results}

        print("  [多智能体] 启动协同处置流程 ...")
        print(f"  [多智能体] 场景: {scenario}，分析节点数: "
              f"{len(target_nodes) if target_nodes else '全部'}")

        # Step 1: 三领域智能体并行分析
        print("  [库存智能体] 分析中 ...")
        inv_proposal = self.inventory_agent.analyze(
            sim_data, risk_results, scenario, target_nodes
        )
        self._send(AgentMessage("inventory", "coordinator", "proposal", inv_proposal))

        print("  [物流智能体] 分析中 ...")
        log_proposal = self.logistics_agent.analyze(
            sim_data, risk_results, scenario, target_nodes
        )
        self._send(AgentMessage("logistics", "coordinator", "proposal", log_proposal))

        print("  [需求智能体] 分析中 ...")
        dem_proposal = self.demand_agent.analyze(
            sim_data, risk_results, scenario, target_nodes
        )
        self._send(AgentMessage("demand", "coordinator", "proposal", dem_proposal))

        # Step 2: 协调智能体整合
        print("  [协调智能体] 整合方案、消解冲突 ...")
        integrated_plan = self.coordinator.coordinate(
            inv_proposal, log_proposal, dem_proposal,
            scenario=scenario, budget_limit=budget_limit,
        )
        self._send(AgentMessage("coordinator", "broadcast", "ack", integrated_plan))

        # Step 3: 反馈闭环——更新处置后风险评分
        post_scores = self._apply_disposal(pre_scores, integrated_plan)
        overall_reduction = self._calc_overall_reduction(pre_scores, post_scores)

        print(f"  [多智能体] 处置完成！综合风险降低: {overall_reduction:.1%}")

        return DisposalReport(
            scenario=scenario,
            inventory_proposal=inv_proposal,
            logistics_proposal=log_proposal,
            demand_proposal=dem_proposal,
            integrated_plan=integrated_plan,
            messages=list(self._message_log),
            pre_disposal_risk_scores=pre_scores,
            post_disposal_risk_scores=post_scores,
            overall_risk_reduction=round(overall_reduction, 4),
        )

    # ------------------------------------------------------------------ #
    # 反馈闭环
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_disposal(
        pre_scores: Dict[str, float],
        plan: IntegratedPlan,
    ) -> Dict[str, float]:
        """将综合处置方案应用于网络节点，更新风险评分。

        被处置节点的风险分按 expected_risk_reduction 按比例降低；
        未被直接处置但在传播路径上的节点，施加0.05的间接降低效应。

        Args:
            pre_scores: 处置前节点风险评分字典
            plan: 综合处置计划

        Returns:
            处置后节点风险评分字典
        """
        post = dict(pre_scores)
        direct_nodes = set()

        for action in plan.integrated_actions:
            nid = action.node_id
            if nid in post:
                reduction = action.expected_risk_reduction
                post[nid] = max(0.05, post[nid] * (1.0 - reduction))
                direct_nodes.add(nid)

        # 间接效应：下游节点受益（仅一层）
        _downstream = {
            "T3-SI": ["T2-ECU"], "T3-RE": ["T2-SN", "T2-ECU"],
            "T2-ECU": ["T1-E"], "T2-SN": ["T1-E"],
            "T1-E": ["OEM"], "T1-P": ["OEM"], "T1-C": ["OEM"],
        }
        for nid in direct_nodes:
            for downstream in _downstream.get(nid, []):
                if downstream in post and downstream not in direct_nodes:
                    post[downstream] = max(0.05, post[downstream] * 0.95)

        return {k: round(v, 4) for k, v in post.items()}

    @staticmethod
    def _calc_overall_reduction(
        pre: Dict[str, float], post: Dict[str, float]
    ) -> float:
        """计算整体风险降低比例（取高风险节点的加权平均）。"""
        nodes = [n for n in pre if pre[n] >= 0.45]
        if not nodes:
            nodes = list(pre.keys())
        pre_avg = float(np.mean([pre[n] for n in nodes]))
        post_avg = float(np.mean([post.get(n, pre[n]) for n in nodes]))
        return max(0, (pre_avg - post_avg) / pre_avg) if pre_avg > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 消息通信
    # ------------------------------------------------------------------ #
    def _send(self, msg: AgentMessage) -> None:
        msg.timestamp = self._tick
        self._tick += 1
        self._message_log.append(msg)

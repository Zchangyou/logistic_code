"""
协调智能体
Coordinator Agent

功能：
- 汇聚三大领域智能体的处置方案
- 冲突检测（同一节点的重叠 / 资源竞争）
- 多目标权衡（成本 × 效率 × 风险降低 × 实施难度）
- 输出综合处置计划 IntegratedPlan
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from src.agent.agents.inventory_agent import InventoryProposal
from src.agent.agents.logistics_agent import LogisticsProposal
from src.agent.agents.demand_agent import DemandProposal


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class ConflictRecord:
    """冲突记录 (Conflict Record between proposals)"""
    node_id: str
    conflict_type: str      # 资源竞争/目标冲突/冗余
    agents_involved: List[str]
    description: str
    resolution: str


@dataclass
class IntegratedAction:
    """综合处置动作（协调整合后）"""
    node_id: str
    node_name: str
    priority: str               # 高/中/低
    source_agents: List[str]    # 来源智能体
    combined_action: str        # 综合处置描述
    cost_index: float
    expected_risk_reduction: float
    # 多目标评分（用于 F6-2 雷达图）
    score_risk_reduction: float   # 风险降低效果 0-10
    score_cost: float             # 成本合理性（越低成本→越高分）0-10
    score_efficiency: float       # 实施效率 0-10
    score_feasibility: float      # 可行性 0-10
    score_urgency: float          # 紧迫性 0-10


@dataclass
class IntegratedPlan:
    """综合处置计划 (Integrated Disposal Plan)"""
    scenario: str
    integrated_actions: List[IntegratedAction] = field(default_factory=list)
    conflicts: List[ConflictRecord] = field(default_factory=list)
    total_expected_risk_reduction: float = 0.0
    total_cost_index: float = 0.0
    implementation_sequence: List[str] = field(default_factory=list)  # 节点ID排序
    summary: str = ""
    # 三智能体各自贡献
    inventory_contribution: float = 0.0
    logistics_contribution: float = 0.0
    demand_contribution: float = 0.0


# --------------------------------------------------------------------------- #
# 主协调智能体
# --------------------------------------------------------------------------- #
class CoordinatorAgent:
    """协调智能体 (Coordinator Agent)

    整合库存/物流/需求三个领域智能体的方案，
    执行冲突检测与多目标权衡，生成统一处置计划。
    """

    # 各智能体在不同风险场景中的可信度权重
    _SCENARIO_WEIGHTS = {
        "S1_chip_shortage": {"inventory": 0.50, "logistics": 0.25, "demand": 0.25},
        "S2_rare_earth":    {"inventory": 0.45, "logistics": 0.30, "demand": 0.25},
        "S3_region":        {"inventory": 0.25, "logistics": 0.55, "demand": 0.20},
        "S4_demand_shock":  {"inventory": 0.25, "logistics": 0.25, "demand": 0.50},
    }

    def coordinate(
        self,
        inventory_proposal: InventoryProposal,
        logistics_proposal: LogisticsProposal,
        demand_proposal: DemandProposal,
        scenario: str = "S1_chip_shortage",
        budget_limit: float = 2.0,  # 总成本指数上限
    ) -> IntegratedPlan:
        """整合三方案并生成综合处置计划。

        Args:
            inventory_proposal: 库存智能体输出
            logistics_proposal: 物流智能体输出
            demand_proposal: 需求智能体输出
            scenario: 场景名称
            budget_limit: 总成本指数预算上限

        Returns:
            IntegratedPlan: 综合处置计划
        """
        weights = self._SCENARIO_WEIGHTS.get(
            scenario, {"inventory": 0.4, "logistics": 0.3, "demand": 0.3}
        )

        # 汇聚各方案的节点级动作
        all_actions: Dict[str, Dict] = {}

        for action in inventory_proposal.actions:
            _upsert(all_actions, action.node_id, {
                "node_name": action.node_name,
                "inventory": action,
                "combined_priority_score": _priority_score(action.priority) * weights["inventory"],
                "total_cost": action.cost_index * weights["inventory"],
                "total_risk_red": action.expected_risk_reduction * weights["inventory"],
            })

        for action in logistics_proposal.actions:
            _upsert(all_actions, action.node_id, {
                "node_name": action.node_name,
                "logistics": action,
                "combined_priority_score": _priority_score(action.priority) * weights["logistics"],
                "total_cost": action.cost_index * weights["logistics"],
                "total_risk_red": action.expected_risk_reduction * weights["logistics"],
            })

        for action in demand_proposal.actions:
            _upsert(all_actions, action.node_id, {
                "node_name": action.node_name,
                "demand": action,
                "combined_priority_score": _priority_score(action.warning_level) * weights["demand"],
                "total_cost": action.cost_index * weights["demand"],
                "total_risk_red": action.expected_risk_reduction * weights["demand"],
            })

        # 冲突检测
        conflicts = self._detect_conflicts(all_actions)

        # 生成整合动作
        integrated: List[IntegratedAction] = []
        cumulative_cost = 0.0

        sorted_nodes = sorted(
            all_actions.items(),
            key=lambda x: x[1].get("combined_priority_score", 0),
            reverse=True,
        )

        for node_id, info in sorted_nodes:
            node_cost = info.get("total_cost", 0.3)
            if cumulative_cost + node_cost > budget_limit:
                continue  # 超预算，跳过

            action = self._build_integrated_action(node_id, info, weights)
            integrated.append(action)
            cumulative_cost += action.cost_index

        # 按优先级排序
        integrated.sort(key=lambda x: _priority_score(x.priority), reverse=True)

        # 实施顺序（先高优先级，再低成本）
        impl_seq = [a.node_id for a in integrated]

        total_risk_red = min(
            0.60,
            sum(a.expected_risk_reduction for a in integrated[:5])
        )
        total_cost = sum(a.cost_index for a in integrated) / max(len(integrated), 1)

        return IntegratedPlan(
            scenario=scenario,
            integrated_actions=integrated,
            conflicts=conflicts,
            total_expected_risk_reduction=round(total_risk_red, 4),
            total_cost_index=round(total_cost, 4),
            implementation_sequence=impl_seq,
            inventory_contribution=round(inventory_proposal.total_risk_reduction * weights["inventory"], 4),
            logistics_contribution=round(logistics_proposal.total_risk_reduction * weights["logistics"], 4),
            demand_contribution=round(demand_proposal.total_risk_reduction * weights["demand"], 4),
            summary=(
                f"协调智能体：整合三方案后生成 {len(integrated)} 条综合处置动作，"
                f"检测到 {len(conflicts)} 处冲突并已消解；"
                f"预期综合风险降低 {total_risk_red:.1%}，"
                f"综合成本指数 {total_cost:.2f}（预算上限 {budget_limit:.1f}）。"
            ),
        )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_conflicts(all_actions: Dict[str, Dict]) -> List[ConflictRecord]:
        """检测多智能体方案间的冲突。"""
        conflicts = []
        for node_id, info in all_actions.items():
            agents = [k for k in ("inventory", "logistics", "demand") if k in info]
            if len(agents) >= 2:
                # 检查成本重叠
                costs = sum(
                    getattr(info.get(a, None), "cost_index", 0) or 0.0
                    for a in agents
                )
                if costs > 1.0:
                    conflicts.append(ConflictRecord(
                        node_id=node_id,
                        conflict_type="资源竞争",
                        agents_involved=agents,
                        description=f"节点 {node_id} 同时涉及{len(agents)}个智能体，总成本指数{costs:.2f}>1.0",
                        resolution="按场景权重分配预算，优先执行高优先级动作",
                    ))
        return conflicts

    @staticmethod
    def _build_integrated_action(
        node_id: str,
        info: Dict,
        weights: Dict[str, float],
    ) -> IntegratedAction:
        """构建单节点的综合处置动作。"""
        agents = [k for k in ("inventory", "logistics", "demand") if k in info]
        parts = []
        cost_sum, risk_red_sum = 0.0, 0.0

        for ag in agents:
            obj = info[ag]
            w = weights.get(ag, 0.33)
            cost_sum += getattr(obj, "cost_index", 0.3) * w
            risk_red_sum += getattr(obj, "expected_risk_reduction", 0.1) * w
            desc = getattr(obj, "action", "") or ""
            if desc:
                parts.append(f"[{_agent_cn(ag)}]{desc[:50]}")

        # 综合优先级
        score = info.get("combined_priority_score", 0)
        priority = "高" if score >= 0.15 else ("中" if score >= 0.08 else "低")

        # 多目标评分（0-10）
        risk_score = min(10, risk_red_sum * 40)
        cost_score = max(0, 10 - cost_sum * 10)
        eff_score = 7.0 if priority == "高" else (5.0 if priority == "中" else 3.0)
        feas_score = min(10, 8 - len(agents) * 0.5)  # 多智能体协调稍降可行性
        urgency_score = 10 if priority == "高" else (6 if priority == "中" else 3)

        return IntegratedAction(
            node_id=node_id,
            node_name=info.get("node_name", node_id),
            priority=priority,
            source_agents=[_agent_cn(a) for a in agents],
            combined_action="；".join(parts) if parts else "综合处置",
            cost_index=round(cost_sum, 4),
            expected_risk_reduction=round(risk_red_sum, 4),
            score_risk_reduction=round(risk_score, 2),
            score_cost=round(cost_score, 2),
            score_efficiency=round(eff_score, 2),
            score_feasibility=round(feas_score, 2),
            score_urgency=round(urgency_score, 2),
        )


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _priority_score(priority: str) -> float:
    return {"高": 3.0, "严重": 3.0, "警告": 2.0, "中": 2.0, "关注": 1.0, "低": 1.0}.get(priority, 1.0)


def _agent_cn(agent: str) -> str:
    return {"inventory": "库存", "logistics": "物流", "demand": "需求"}.get(agent, agent)


def _upsert(d: Dict, key: str, updates: Dict) -> None:
    """将 updates 合并到 d[key]（累加数值字段）。"""
    if key not in d:
        d[key] = updates.copy()
    else:
        for k, v in updates.items():
            if k in ("combined_priority_score", "total_cost", "total_risk_red"):
                d[key][k] = d[key].get(k, 0.0) + v
            elif k not in d[key]:
                d[key][k] = v

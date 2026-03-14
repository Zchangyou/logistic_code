"""
SIR风险传播模型
SIR-based Risk Propagation Model for Supply Chain
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np


class NodeState(Enum):
    """节点状态枚举 (Node State Enumeration)"""
    SUSCEPTIBLE = "S"   # 正常
    INFECTED = "I"      # 受影响
    RECOVERED = "R"     # 已恢复


@dataclass
class SIRResult:
    """SIR仿真结果 (SIR Simulation Result)

    Attributes:
        scenario_name: 场景中文名
        scenario_name_en: 场景英文名
        time_steps: 时间步列表
        s_counts: 每步S节点数
        i_counts: 每步I节点数
        r_counts: 每步R节点数
        node_states_history: 每步各节点状态
        final_affected_count: 最终受影响节点数
        max_affected_count: 最大同时感染节点数
        recovery_time: 感染归零步数
        impact_depth: 感染到达的最大层级深度
        impact_duration: 至少有1个感染节点的持续步数
    """
    scenario_name: str
    scenario_name_en: str
    time_steps: List[int]
    s_counts: List[int]
    i_counts: List[int]
    r_counts: List[int]
    node_states_history: List[Dict[str, NodeState]]
    final_affected_count: int
    max_affected_count: int
    recovery_time: int
    impact_depth: int
    impact_duration: int


class SIRPropagationModel:
    """改进SIR风险传播模型 (Improved SIR Risk Propagation Model)

    基于供应链有向图实现SIR传播仿真，支持正向（供应中断）和反向（需求冲击）传播。

    Attributes:
        network: SupplyChainNetwork 对象
        beta: 传播概率基础值
        gamma: 恢复概率基础值
        rng: 随机数生成器
    """

    def __init__(self, network, beta: float = 0.4, gamma: float = 0.15, seed: int = 42):
        """初始化SIR传播模型。

        Args:
            network: SupplyChainNetwork 对象
            beta: 传播概率（沿边传播基础概率）
            gamma: 恢复概率
            seed: 随机种子
        """
        self.network = network
        self.beta = beta
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)

    def _get_tier(self, node_id: str) -> int:
        """获取节点层级。"""
        node = self.network.get_node(node_id)
        return node.tier if node is not None else -1

    def _get_substitutability(self, node_id: str) -> float:
        """获取节点可替代性。"""
        node = self.network.get_node(node_id)
        return node.substitutability if node is not None else 0.5

    def run(
        self,
        initial_infected: List[str],
        n_steps: int = 30,
        scenario_name: str = "",
        scenario_name_en: str = "",
        use_reversed: bool = False,
    ) -> SIRResult:
        """运行SIR仿真。

        传播规则：每个I节点对每个易感邻居以 beta * dependency_strength 的概率传播。
        恢复规则：每个I节点以 gamma * substitutability 的概率恢复。

        Args:
            initial_infected: 初始感染节点ID列表
            n_steps: 仿真步数
            scenario_name: 场景中文名
            scenario_name_en: 场景英文名
            use_reversed: 是否使用反向图（需求冲击反向传播）

        Returns:
            SIRResult 对象
        """
        G = self.network.get_graph()
        if use_reversed:
            G = G.reverse(copy=True)

        all_nodes = list(G.nodes())

        # 初始化状态
        states: Dict[str, NodeState] = {n: NodeState.SUSCEPTIBLE for n in all_nodes}
        for node_id in initial_infected:
            if node_id in states:
                states[node_id] = NodeState.INFECTED

        time_steps = []
        s_counts = []
        i_counts = []
        r_counts = []
        history = []

        for step in range(n_steps + 1):
            s = sum(1 for s in states.values() if s == NodeState.SUSCEPTIBLE)
            i = sum(1 for s in states.values() if s == NodeState.INFECTED)
            r = sum(1 for s in states.values() if s == NodeState.RECOVERED)
            time_steps.append(step)
            s_counts.append(s)
            i_counts.append(i)
            r_counts.append(r)
            history.append(dict(states))

            if i == 0:
                # 感染已消退，填充剩余步
                for remaining in range(step + 1, n_steps + 1):
                    time_steps.append(remaining)
                    s_counts.append(s)
                    i_counts.append(0)
                    r_counts.append(r)
                    history.append(dict(states))
                break

            if step == n_steps:
                break

            # 计算新感染和新恢复
            new_infected = set()
            new_recovered = set()

            infected_nodes = [n for n, st in states.items() if st == NodeState.INFECTED]

            for node in infected_nodes:
                # 尝试感染下游邻居（在G中为successors）
                for neighbor in G.successors(node):
                    if states.get(neighbor) == NodeState.SUSCEPTIBLE:
                        edge_data = G.get_edge_data(node, neighbor) or {}
                        dep = edge_data.get("dependency_strength", 0.5)
                        prob = self.beta * dep
                        if self.rng.random() < prob:
                            new_infected.add(neighbor)

                # 尝试恢复
                sub = self._get_substitutability(node)
                recover_prob = self.gamma * max(sub, 0.1)
                if self.rng.random() < recover_prob:
                    new_recovered.add(node)

            # 先恢复，再感染（避免同步问题）
            for node in new_recovered:
                states[node] = NodeState.RECOVERED
            for node in new_infected:
                # 只感染仍处于S状态的节点
                if states[node] == NodeState.SUSCEPTIBLE:
                    states[node] = NodeState.INFECTED

        # 计算统计指标
        final_affected = sum(1 for st in states.values() if st != NodeState.SUSCEPTIBLE)
        max_infected = max(i_counts)

        # 恢复时间：I归零的步数
        recovery_time = n_steps
        for t, ic in enumerate(i_counts):
            if t > 0 and ic == 0:
                recovery_time = t
                break

        # 影响持续时间
        impact_duration = sum(1 for ic in i_counts if ic > 0)

        # 最大影响层级深度
        impact_depth = self._calc_impact_depth(history, use_reversed)

        return SIRResult(
            scenario_name=scenario_name,
            scenario_name_en=scenario_name_en,
            time_steps=time_steps,
            s_counts=s_counts,
            i_counts=i_counts,
            r_counts=r_counts,
            node_states_history=history,
            final_affected_count=final_affected,
            max_affected_count=max_infected,
            recovery_time=recovery_time,
            impact_depth=impact_depth,
            impact_duration=impact_duration,
        )

    def _calc_impact_depth(
        self, history: List[Dict[str, NodeState]], use_reversed: bool
    ) -> int:
        """计算感染到达的最大层级跨度。"""
        affected_nodes = set()
        for snapshot in history:
            for node, state in snapshot.items():
                if state in (NodeState.INFECTED, NodeState.RECOVERED):
                    affected_nodes.add(node)

        if not affected_nodes:
            return 0

        tiers = [self._get_tier(n) for n in affected_nodes if self._get_tier(n) >= 0]
        if not tiers:
            return 0
        return max(tiers) - min(tiers) + 1

    def run_scenario_s1_chip(self) -> SIRResult:
        """场景S1: T3-SI芯片晶圆停供"""
        return self.run(
            initial_infected=["T3-SI"],
            n_steps=30,
            scenario_name="芯片晶圆停供",
            scenario_name_en="Chip Wafer Shortage",
        )

    def run_scenario_s2_rare_earth(self) -> SIRResult:
        """场景S2: T3-RE稀土材料集中风险"""
        return self.run(
            initial_infected=["T3-RE"],
            n_steps=30,
            scenario_name="稀土材料集中",
            scenario_name_en="Rare Earth Concentration",
        )

    def run_scenario_s3_east_china(self) -> SIRResult:
        """场景S3: 华东区域中断 — 4节点同时受影响"""
        return self.run(
            initial_infected=["T2-E2", "T2-W1", "T3-RB", "T2-SN"],
            n_steps=30,
            scenario_name="华东区域中断",
            scenario_name_en="East China Disruption",
        )

    def run_scenario_s4_demand_shock(self) -> SIRResult:
        """场景S4: 需求冲击 — OEM订单骤增，反向传播至上游"""
        return self.run(
            initial_infected=["OEM"],
            n_steps=30,
            scenario_name="需求骤增冲击",
            scenario_name_en="Demand Shock",
            use_reversed=True,
        )

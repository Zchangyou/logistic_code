"""
库存风险感知智能体
Inventory Risk Awareness Agent

功能：
- 分析关键节点库存水位与安全库存缺口
- 计算物料缓冲储备建议量（天）
- 输出结构化库存处置建议 InventoryProposal
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class InventoryAction:
    """单节点库存处置动作 (Inventory Action for a node)"""
    node_id: str
    node_name: str
    current_inventory_days: float
    recommended_inventory_days: float
    gap_days: float               # 正数=需补库，负数=过剩
    priority: str                 # 高/中/低
    action: str                   # 处置描述
    cost_index: float             # 相对成本指数 0-1
    expected_risk_reduction: float  # 预期风险降低比例 0-1


@dataclass
class InventoryProposal:
    """库存智能体处置方案 (Inventory Agent Proposal)"""
    scenario: str
    agent_name: str = "库存风险感知智能体"
    agent_name_en: str = "Inventory Risk Agent"
    actions: List[InventoryAction] = field(default_factory=list)
    total_risk_reduction: float = 0.0
    total_cost_index: float = 0.0
    summary: str = ""


# --------------------------------------------------------------------------- #
# 静态配置
# --------------------------------------------------------------------------- #
_SAFETY_DAYS = {
    "raw_material": 45,
    "component": 30,
    "integrator": 20,
    "oem": 15,
}

_RISK_MULTIPLIER = {"high": 1.5, "medium": 1.2, "low": 1.0, "safe": 0.8}

_NODE_NAMES = {
    "T3-SI": "芯片晶圆供应商", "T3-RE": "稀土材料供应商",
    "T2-ECU": "ECU控制单元", "T2-SN": "传感器模组",
    "T1-E": "电子电气系统集成商", "T2-E2": "涡轮增压器",
    "T3-CU": "铜材供应商", "OEM": "总装厂",
    "T3-AL": "铸造铝合金", "T3-ST": "特种钢材",
    "T3-NI": "镍基合金", "T3-RB": "合成橡胶",
    "T3-PCB": "印制电路板", "T3-PL": "工程塑料",
    "T3-CF": "碳纤维材料", "T3-MG": "镁合金",
    "T3-GL": "特种玻璃", "T2-E1": "发动机缸体",
    "T2-T1": "变速箱总成", "T2-B1": "制动系统",
    "T2-S1": "悬挂系统", "T2-W1": "转向器",
    "T2-H1": "线束总成", "T1-P": "动力总成集成商",
    "T1-C": "底盘系统集成商",
}

_NODE_TYPES = {
    "T3-SI": "raw_material", "T3-RE": "raw_material",
    "T3-CU": "raw_material", "T3-ST": "raw_material",
    "T3-AL": "raw_material", "T3-NI": "raw_material",
    "T3-RB": "raw_material", "T3-PCB": "raw_material",
    "T3-PL": "raw_material", "T3-CF": "raw_material",
    "T3-MG": "raw_material", "T3-GL": "raw_material",
    "T2-ECU": "component", "T2-SN": "component",
    "T2-E1": "component", "T2-E2": "component",
    "T2-T1": "component", "T2-B1": "component",
    "T2-S1": "component", "T2-W1": "component",
    "T2-H1": "component",
    "T1-E": "integrator", "T1-P": "integrator", "T1-C": "integrator",
    "OEM": "oem",
}


# --------------------------------------------------------------------------- #
# 主智能体类
# --------------------------------------------------------------------------- #
class InventoryAgent:
    """库存风险感知智能体 (Inventory Risk Awareness Agent)

    基于仿真运营数据与综合风险评估结果，计算各节点安全库存缺口，
    并按优先级输出补库建议。
    """

    def analyze(
        self,
        sim_data: pd.DataFrame,
        risk_results: List[Dict],
        scenario: str = "S1_chip_shortage",
        target_nodes: Optional[List[str]] = None,
    ) -> InventoryProposal:
        """分析库存风险，生成处置方案。

        Args:
            sim_data: 6期仿真数据 DataFrame，含 node_id / period / inventory_level
            risk_results: 综合风险评估结果列表（来自 assessment 模块）
            scenario: 当前分析场景名称
            target_nodes: 限定分析的节点列表；None 表示分析全部节点

        Returns:
            InventoryProposal: 结构化库存处置方案
        """
        latest = (
            sim_data[sim_data["period"] == sim_data["period"].max()]
            .set_index("node_id")
        )
        risk_map = {r["node_id"]: r for r in risk_results}

        nodes = target_nodes if target_nodes else list(latest.index)
        actions: List[InventoryAction] = []

        for node_id in nodes:
            if node_id not in latest.index:
                continue

            current_inv = float(latest.loc[node_id, "inventory_level"])
            risk_info = risk_map.get(node_id, {})
            risk_level = risk_info.get("risk_level", "safe")
            node_type = _NODE_TYPES.get(node_id, "component")

            base_days = _SAFETY_DAYS.get(node_type, 30)
            recommended = base_days * _RISK_MULTIPLIER.get(risk_level, 1.0)
            gap = recommended - current_inv

            if gap <= 0:
                continue  # 库存充足，跳过

            if gap > 15:
                priority, expected_red, cost_idx = "高", 0.25, 0.70
                action = f"紧急补充安全库存至{recommended:.0f}天（当前{current_inv:.0f}天，缺口{gap:.0f}天）"
            elif gap > 5:
                priority, expected_red, cost_idx = "中", 0.15, 0.40
                action = f"计划性补库至{recommended:.0f}天（缺口{gap:.0f}天）"
            else:
                priority, expected_red, cost_idx = "低", 0.05, 0.20
                action = f"适度增加库存至{recommended:.0f}天"

            actions.append(InventoryAction(
                node_id=node_id,
                node_name=_NODE_NAMES.get(node_id, node_id),
                current_inventory_days=round(current_inv, 1),
                recommended_inventory_days=round(recommended, 1),
                gap_days=round(gap, 1),
                priority=priority,
                action=action,
                cost_index=cost_idx,
                expected_risk_reduction=expected_red,
            ))

        _prio = {"高": 0, "中": 1, "低": 2}
        actions.sort(key=lambda x: _prio.get(x.priority, 3))

        top3 = actions[:3]
        total_risk_red = min(0.50, sum(a.expected_risk_reduction for a in top3))
        total_cost = sum(a.cost_index for a in top3) / max(len(top3), 1)
        high_cnt = sum(1 for a in actions if a.priority == "高")

        return InventoryProposal(
            scenario=scenario,
            actions=actions,
            total_risk_reduction=round(total_risk_red, 4),
            total_cost_index=round(total_cost, 4),
            summary=(
                f"库存智能体：发现 {len(actions)} 个节点存在安全库存缺口，"
                f"其中 {high_cnt} 个高优先级；"
                f"综合处置预期降低风险 {total_risk_red:.1%}，"
                f"综合成本指数 {total_cost:.2f}。"
            ),
        )

"""
需求预测智能体
Demand Forecasting Agent

功能：
- 基于6期仿真数据检测需求趋势（线性回归斜率）
- 量化牛鞭效应放大系数（各层级需求变异系数之比）
- 输出需求侧风险预警与调控建议 DemandProposal
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class DemandAction:
    """单节点需求处置动作 (Demand Action for a node)"""
    node_id: str
    node_name: str
    demand_volatility_trend: float   # 需求波动系数近期趋势斜率（每期变化量）
    current_volatility: float        # 最新期波动系数
    bullwhip_ratio: float            # 牛鞭放大比（vs上游）
    warning_level: str               # 正常/关注/警告/严重
    action: str                      # 处置描述
    cost_index: float                # 相对成本指数 0-1
    expected_risk_reduction: float   # 预期风险降低比例 0-1


@dataclass
class BullwhipAnalysis:
    """牛鞭效应分析结果 (Bullwhip Effect Analysis)"""
    tier_volatilities: Dict[str, float]   # 层级 → 平均需求波动系数
    amplification_ratios: Dict[str, float]  # 层级 → 相对T0放大倍数
    max_amplification: float
    most_amplified_tier: str


@dataclass
class DemandProposal:
    """需求智能体处置方案 (Demand Agent Proposal)"""
    scenario: str
    agent_name: str = "需求预测智能体"
    agent_name_en: str = "Demand Forecasting Agent"
    actions: List[DemandAction] = field(default_factory=list)
    bullwhip: Optional[BullwhipAnalysis] = None
    total_risk_reduction: float = 0.0
    total_cost_index: float = 0.0
    summary: str = ""


# --------------------------------------------------------------------------- #
# 静态配置
# --------------------------------------------------------------------------- #
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

# 节点所属层级
_NODE_TIERS = {
    "OEM": 0,
    "T1-P": 1, "T1-C": 1, "T1-E": 1,
    "T2-E1": 2, "T2-E2": 2, "T2-T1": 2, "T2-B1": 2,
    "T2-S1": 2, "T2-W1": 2, "T2-ECU": 2, "T2-H1": 2, "T2-SN": 2,
    "T3-AL": 3, "T3-ST": 3, "T3-NI": 3, "T3-RB": 3,
    "T3-CU": 3, "T3-RE": 3, "T3-SI": 3, "T3-PCB": 3,
    "T3-PL": 3, "T3-CF": 3, "T3-MG": 3, "T3-GL": 3,
}

_VOLATILITY_THRESHOLDS = {
    "severe":  0.35,
    "warning": 0.25,
    "concern": 0.18,
}


# --------------------------------------------------------------------------- #
# 主智能体类
# --------------------------------------------------------------------------- #
class DemandAgent:
    """需求预测智能体 (Demand Forecasting Agent)

    基于多期仿真数据，识别需求趋势与牛鞭效应，
    提出需求侧风险调控建议。
    """

    def analyze(
        self,
        sim_data: pd.DataFrame,
        risk_results: List[Dict],
        scenario: str = "S1_chip_shortage",
        target_nodes: Optional[List[str]] = None,
    ) -> DemandProposal:
        """分析需求侧风险，生成处置方案。

        Args:
            sim_data: 6期仿真数据 DataFrame（含 demand_volatility 字段）
            risk_results: 综合风险评估结果列表
            scenario: 当前分析场景名称
            target_nodes: 限定分析节点列表

        Returns:
            DemandProposal: 结构化需求侧处置方案
        """
        nodes = target_nodes if target_nodes else sim_data["node_id"].unique().tolist()
        risk_map = {r["node_id"]: r for r in risk_results}
        actions: List[DemandAction] = []

        for node_id in nodes:
            node_series = sim_data[sim_data["node_id"] == node_id].sort_values("period")
            if node_series.empty:
                continue

            vols = node_series["demand_volatility"].values
            current_vol = float(vols[-1])

            # 趋势斜率（线性回归）
            periods = np.arange(len(vols), dtype=float)
            if len(vols) >= 2:
                slope = float(np.polyfit(periods, vols, 1)[0])
            else:
                slope = 0.0

            # 牛鞭放大比（与层级0的波动之比）
            tier = _NODE_TIERS.get(node_id, 2)
            oem_vol = float(
                sim_data[(sim_data["node_id"] == "OEM") &
                         (sim_data["period"] == sim_data["period"].max())]
                ["demand_volatility"].iloc[0]
                if len(sim_data[sim_data["node_id"] == "OEM"]) > 0
                else 0.10
            )
            bullwhip_ratio = current_vol / oem_vol if oem_vol > 0 else 1.0

            # 预警等级
            if current_vol >= _VOLATILITY_THRESHOLDS["severe"]:
                level = "严重"
                expected_red, cost_idx = 0.20, 0.55
                action = (
                    f"实施订单平滑策略与信息共享机制，"
                    f"将需求波动系数从{current_vol:.2f}控制至0.20以下"
                )
            elif current_vol >= _VOLATILITY_THRESHOLDS["warning"]:
                level = "警告"
                expected_red, cost_idx = 0.12, 0.35
                action = f"与下游客户签订需求稳定协议，控制波动至{_VOLATILITY_THRESHOLDS['concern']:.2f}以下"
            elif current_vol >= _VOLATILITY_THRESHOLDS["concern"]:
                level = "关注"
                expected_red, cost_idx = 0.06, 0.20
                action = "持续监测需求波动趋势，预备快速响应预案"
            else:
                continue  # 需求正常

            # 趋势恶化加权
            if slope > 0.01:
                action += "（趋势上升，建议优先处置）"

            actions.append(DemandAction(
                node_id=node_id,
                node_name=_NODE_NAMES.get(node_id, node_id),
                demand_volatility_trend=round(slope, 5),
                current_volatility=round(current_vol, 4),
                bullwhip_ratio=round(bullwhip_ratio, 3),
                warning_level=level,
                action=action,
                cost_index=cost_idx,
                expected_risk_reduction=expected_red,
            ))

        # 牛鞭效应分析
        bullwhip = self._analyze_bullwhip(sim_data, nodes)

        _wt = {"严重": 0, "警告": 1, "关注": 2}
        actions.sort(key=lambda x: _wt.get(x.warning_level, 3))

        top3 = actions[:3]
        total_risk_red = min(0.40, sum(a.expected_risk_reduction for a in top3))
        total_cost = sum(a.cost_index for a in top3) / max(len(top3), 1)
        severe_cnt = sum(1 for a in actions if a.warning_level == "严重")

        return DemandProposal(
            scenario=scenario,
            actions=actions,
            bullwhip=bullwhip,
            total_risk_reduction=round(total_risk_red, 4),
            total_cost_index=round(total_cost, 4),
            summary=(
                f"需求智能体：发现 {len(actions)} 个节点存在需求波动风险，"
                f"其中 {severe_cnt} 个严重等级；"
                f"牛鞭效应最大放大倍数 {bullwhip.max_amplification:.1f}×（出现在{bullwhip.most_amplified_tier}层）；"
                f"综合处置预期降低风险 {total_risk_red:.1%}。"
            ),
        )

    @staticmethod
    def _analyze_bullwhip(sim_data: pd.DataFrame, nodes: List[str]) -> BullwhipAnalysis:
        """计算各层级需求波动系数与牛鞭放大比。"""
        latest = sim_data[sim_data["period"] == sim_data["period"].max()]
        tier_vols: Dict[str, List[float]] = {
            "T0-总装厂": [], "T1-集成商": [], "T2-零部件": [], "T3-原材料": []
        }
        _tier_keys = {0: "T0-总装厂", 1: "T1-集成商", 2: "T2-零部件", 3: "T3-原材料"}

        for _, row in latest.iterrows():
            tier = _NODE_TIERS.get(row["node_id"], 2)
            key = _tier_keys.get(tier, "T2-零部件")
            tier_vols[key].append(float(row["demand_volatility"]))

        avg_vols = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in tier_vols.items()
        }

        base = avg_vols.get("T0-总装厂", 0.10)
        ratios = {
            k: round(v / base, 3) if base > 0 else 1.0
            for k, v in avg_vols.items()
        }

        max_amp = max(ratios.values())
        most_amp = max(ratios, key=ratios.get)

        return BullwhipAnalysis(
            tier_volatilities={k: round(v, 4) for k, v in avg_vols.items()},
            amplification_ratios=ratios,
            max_amplification=round(max_amp, 2),
            most_amplified_tier=most_amp,
        )

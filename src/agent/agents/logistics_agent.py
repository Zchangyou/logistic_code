"""
物流异常分析智能体
Logistics Anomaly Analysis Agent

功能：
- 识别物流可靠性低的供应链链路（logistics_reliability < 阈值）
- 分析区域集中风险（华东/华南/华北等区域的节点集中度）
- 推荐备用物流路线与资源调配方案
- 输出结构化 LogisticsProposal
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import pandas as pd


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class LogisticsAction:
    """单条物流处置动作 (Logistics Action for a node/link)"""
    node_id: str
    node_name: str
    current_reliability: float
    target_reliability: float
    region: str                     # 所属地理区域
    risk_type: str                  # 物流脆弱/区域集中/跨境风险
    priority: str                   # 高/中/低
    action: str                     # 处置描述
    backup_routes: List[str]        # 推荐备用路线
    cost_index: float               # 相对成本指数 0-1
    expected_risk_reduction: float  # 预期风险降低比例 0-1


@dataclass
class LogisticsProposal:
    """物流智能体处置方案 (Logistics Agent Proposal)"""
    scenario: str
    agent_name: str = "物流异常分析智能体"
    agent_name_en: str = "Logistics Anomaly Agent"
    actions: List[LogisticsAction] = field(default_factory=list)
    regional_risks: Dict[str, float] = field(default_factory=dict)  # 区域→集中度
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

# 节点所在区域（来自附录A的地理位置）
_NODE_REGIONS = {
    "T3-SI": "台湾（跨境）", "T3-RE": "华北（包头）",
    "T3-AL": "西南（昆明）", "T3-ST": "东北（鞍山）",
    "T3-NI": "西北（金昌）", "T3-RB": "华东（南京）",
    "T3-CU": "华东（铜陵）", "T3-PCB": "华南（珠海）",
    "T3-PL": "华东（宁波）", "T3-CF": "华东（威海）",
    "T3-MG": "华北（府谷）", "T3-GL": "华东（蚌埠）",
    "T2-E1": "华中（长沙）", "T2-E2": "华东（无锡）",
    "T2-T1": "西南（重庆）", "T2-B1": "华东（烟台）",
    "T2-S1": "华南（柳州）", "T2-W1": "华东（苏州）",
    "T2-ECU": "华南（深圳）", "T2-H1": "华北（天津）",
    "T2-SN": "华东（合肥）", "T1-P": "华东（上海）",
    "T1-C": "华中（武汉）", "T1-E": "华南（深圳）",
    "OEM": "东北（长春）",
}

# 区域→可用备用运输方式
_BACKUP_ROUTES = {
    "台湾（跨境）": ["空运直达海关保税区", "转道日本韩国中转采购"],
    "华东（无锡）": ["高铁货运（无锡→长春）", "联合物流商调度区域仓"],
    "华东（合肥）": ["省内公路转国铁联运", "华东区域集散中心备货"],
    "华东（苏州）": ["苏沪公路快线", "上海港海运转内河"],
    "华东（南京）": ["宁沪高速公路运输", "长三角区域备用仓"],
}

_RELIABILITY_THRESHOLDS = {
    "high_risk": 0.70,   # 低于此值为高风险
    "medium_risk": 0.82, # 低于此值为中风险
    "target": 0.90,      # 目标物流可靠性
}


# --------------------------------------------------------------------------- #
# 主智能体类
# --------------------------------------------------------------------------- #
class LogisticsAgent:
    """物流异常分析智能体 (Logistics Anomaly Analysis Agent)

    分析供应链各节点物流可靠性，识别异常链路，
    提出备用路线与资源调配建议。
    """

    def analyze(
        self,
        sim_data: pd.DataFrame,
        risk_results: List[Dict],
        scenario: str = "S1_chip_shortage",
        target_nodes: Optional[List[str]] = None,
    ) -> LogisticsProposal:
        """分析物流风险，生成处置方案。

        Args:
            sim_data: 6期仿真数据 DataFrame
            risk_results: 综合风险评估结果列表
            scenario: 当前分析场景名称
            target_nodes: 限定分析节点列表

        Returns:
            LogisticsProposal: 结构化物流处置方案
        """
        latest = (
            sim_data[sim_data["period"] == sim_data["period"].max()]
            .set_index("node_id")
        )
        risk_map = {r["node_id"]: r for r in risk_results}

        nodes = target_nodes if target_nodes else list(latest.index)
        actions: List[LogisticsAction] = []

        for node_id in nodes:
            if node_id not in latest.index:
                continue

            reliability = float(latest.loc[node_id, "logistics_reliability"])
            risk_info = risk_map.get(node_id, {})
            region = _NODE_REGIONS.get(node_id, "未知区域")

            if reliability < _RELIABILITY_THRESHOLDS["high_risk"]:
                priority = "高"
                risk_type = "物流严重脆弱" if "跨境" not in region else "跨境物流高风险"
                expected_red, cost_idx = 0.20, 0.65
                action = (
                    f"立即启用备用物流路线，目标可靠性 {_RELIABILITY_THRESHOLDS['target']:.0%}，"
                    f"当前{reliability:.0%}"
                )
            elif reliability < _RELIABILITY_THRESHOLDS["medium_risk"]:
                priority = "中"
                risk_type = "区域物流脆弱"
                expected_red, cost_idx = 0.12, 0.40
                action = (
                    f"与物流商签订应急保障协议，分散运输渠道，"
                    f"可靠性提升至 {_RELIABILITY_THRESHOLDS['target']:.0%}"
                )
            else:
                continue  # 物流正常，跳过

            backup = _BACKUP_ROUTES.get(region, [f"增加{region}区域备用承运商", "建立区域缓冲仓库"])

            actions.append(LogisticsAction(
                node_id=node_id,
                node_name=_NODE_NAMES.get(node_id, node_id),
                current_reliability=round(reliability, 3),
                target_reliability=_RELIABILITY_THRESHOLDS["target"],
                region=region,
                risk_type=risk_type,
                priority=priority,
                action=action,
                backup_routes=backup,
                cost_index=cost_idx,
                expected_risk_reduction=expected_red,
            ))

        # 计算区域集中度风险
        regional_risks = self._calc_regional_risk(latest, nodes)

        _prio = {"高": 0, "中": 1, "低": 2}
        actions.sort(key=lambda x: _prio.get(x.priority, 3))

        top3 = actions[:3]
        total_risk_red = min(0.45, sum(a.expected_risk_reduction for a in top3))
        total_cost = sum(a.cost_index for a in top3) / max(len(top3), 1)
        high_cnt = sum(1 for a in actions if a.priority == "高")

        return LogisticsProposal(
            scenario=scenario,
            actions=actions,
            regional_risks=regional_risks,
            total_risk_reduction=round(total_risk_red, 4),
            total_cost_index=round(total_cost, 4),
            summary=(
                f"物流智能体：发现 {len(actions)} 条物流脆弱链路，"
                f"其中 {high_cnt} 条高优先级；"
                f"华东区域集中度最高（{regional_risks.get('华东', 0):.1%}）；"
                f"综合处置预期降低风险 {total_risk_red:.1%}。"
            ),
        )

    @staticmethod
    def _calc_regional_risk(latest: pd.DataFrame, nodes: List[str]) -> Dict[str, float]:
        """计算各区域节点集中度风险指数。"""
        region_counts: Dict[str, int] = {}
        total = 0
        for node_id in nodes:
            region = _NODE_REGIONS.get(node_id, "其他")
            # 归一到一级区域（华东/华南/华北/华中/东北/西南/西北/跨境）
            key = region.split("（")[0]
            region_counts[key] = region_counts.get(key, 0) + 1
            total += 1

        return {k: round(v / total, 3) for k, v in region_counts.items()} if total > 0 else {}

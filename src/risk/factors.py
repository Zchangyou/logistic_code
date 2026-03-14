"""
风险因素分类与FMEA量化模块
Risk Factor Classification and FMEA Quantification Module
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict


class RiskCategory(Enum):
    """风险类别枚举 (Risk Category Enumeration)"""
    MATERIAL_SHORTAGE = "material_shortage"            # 材料短缺
    SUPPLIER_CONCENTRATION = "supplier_concentration"  # 供应商集中
    LOGISTICS_DISRUPTION = "logistics_disruption"      # 物流中断
    DEMAND_VOLATILITY = "demand_volatility"            # 需求波动


# 风险类别中英文名称映射
CATEGORY_NAMES: Dict[RiskCategory, tuple] = {
    RiskCategory.MATERIAL_SHORTAGE:       ("材料短缺", "Material Shortage"),
    RiskCategory.SUPPLIER_CONCENTRATION:  ("供应商集中", "Supplier Concentration"),
    RiskCategory.LOGISTICS_DISRUPTION:    ("物流中断", "Logistics Disruption"),
    RiskCategory.DEMAND_VOLATILITY:       ("需求波动", "Demand Volatility"),
}


@dataclass
class RiskFactor:
    """风险因素数据类 (Risk Factor Data Class)

    Attributes:
        factor_id: 风险因素唯一标识符
        name: 中文名称
        name_en: 英文名称
        category: 风险类别（RiskCategory枚举）
        node_id: 所属供应链节点ID
        description: 风险描述
        occurrence_prob: 发生概率 0-1
        severity: 影响严重度 0-10
        detectability: 可探测性 0-10（10=极难探测）
        rpn: 风险优先数 = occurrence_prob*10 * severity * detectability
    """
    factor_id: str
    name: str
    name_en: str
    category: RiskCategory
    node_id: str
    description: str
    occurrence_prob: float = 0.0
    severity: float = 0.0
    detectability: float = 0.0
    rpn: float = 0.0

    def calculate_rpn(self) -> float:
        """计算并更新风险优先数 RPN。

        RPN = occurrence_prob * 10 * severity * detectability

        Returns:
            计算得到的 RPN 值
        """
        self.rpn = self.occurrence_prob * 10 * self.severity * self.detectability
        return self.rpn

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "name_en": self.name_en,
            "category": self.category.value,
            "node_id": self.node_id,
            "description": self.description,
            "occurrence_prob": self.occurrence_prob,
            "severity": self.severity,
            "detectability": self.detectability,
            "rpn": self.rpn,
        }


class RiskFactorRegistry:
    """风险因素注册表 (Risk Factor Registry)

    管理所有风险因素，提供多维度查询能力。
    """

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._factors: Dict[str, RiskFactor] = {}

    def register(self, factor: RiskFactor) -> None:
        """注册一个风险因素。

        Args:
            factor: RiskFactor 对象
        """
        factor.calculate_rpn()
        self._factors[factor.factor_id] = factor

    def get_factor(self, factor_id: str) -> Optional[RiskFactor]:
        """根据ID获取风险因素。

        Args:
            factor_id: 风险因素ID

        Returns:
            RiskFactor 对象，不存在则返回 None
        """
        return self._factors.get(factor_id)

    def get_factors_by_category(self, category: RiskCategory) -> List[RiskFactor]:
        """按类别获取风险因素列表。

        Args:
            category: 风险类别

        Returns:
            该类别的 RiskFactor 列表
        """
        return [f for f in self._factors.values() if f.category == category]

    def get_factors_by_node(self, node_id: str) -> List[RiskFactor]:
        """按节点ID获取风险因素列表。

        Args:
            node_id: 供应链节点ID

        Returns:
            该节点的 RiskFactor 列表
        """
        return [f for f in self._factors.values() if f.node_id == node_id]

    def get_all_factors(self) -> List[RiskFactor]:
        """获取所有风险因素。

        Returns:
            所有 RiskFactor 的列表，按 RPN 降序排列
        """
        return sorted(self._factors.values(), key=lambda f: f.rpn, reverse=True)

    def get_rpn_ranking(self) -> List[RiskFactor]:
        """按 RPN 降序返回风险因素排名。

        Returns:
            RPN 降序排列的 RiskFactor 列表
        """
        return self.get_all_factors()

    @classmethod
    def build_auto_engine_registry(cls) -> 'RiskFactorRegistry':
        """构建汽车发动机供应链案例的风险因素注册表。

        RPN设计原则：芯片短缺 > 稀土集中 > 物流脆弱 > 需求波动

        Returns:
            预填充的 RiskFactorRegistry 对象
        """
        registry = cls()

        factors = [
            # --- 材料短缺类 ---
            RiskFactor(
                factor_id="RF-SI-01",
                name="芯片晶圆短缺",
                name_en="Chip Wafer Shortage",
                category=RiskCategory.MATERIAL_SHORTAGE,
                node_id="T3-SI",
                description="半导体晶圆产能不足，全球供应链依赖少数代工厂，导致ECU及传感器模组断供风险极高",
                occurrence_prob=0.8,
                severity=9.5,
                detectability=8.0,
            ),
            RiskFactor(
                factor_id="RF-ECU-01",
                name="ECU级联短缺",
                name_en="ECU Cascading Shortage",
                category=RiskCategory.MATERIAL_SHORTAGE,
                node_id="T2-ECU",
                description="ECU控制单元依赖芯片晶圆与稀土材料双重供应，上游短缺触发级联断供风险",
                occurrence_prob=0.7,
                severity=8.5,
                detectability=7.0,
            ),
            RiskFactor(
                factor_id="RF-SN-01",
                name="传感器多重风险",
                name_en="Sensor Module Multi-source Risk",
                category=RiskCategory.MATERIAL_SHORTAGE,
                node_id="T2-SN",
                description="传感器模组同时依赖铜材、稀土、PCB三类原材料，多重上游风险叠加",
                occurrence_prob=0.6,
                severity=7.5,
                detectability=7.0,
            ),
            RiskFactor(
                factor_id="RF-CU-01",
                name="铜材潜在短缺",
                name_en="Copper Material Potential Shortage",
                category=RiskCategory.MATERIAL_SHORTAGE,
                node_id="T3-CU",
                description="铜价波动与矿产供应不稳定，影响线束总成与传感器模组的生产成本和交付",
                occurrence_prob=0.5,
                severity=6.5,
                detectability=6.0,
            ),
            RiskFactor(
                factor_id="RF-ST-01",
                name="特种钢材短缺",
                name_en="Special Steel Shortage",
                category=RiskCategory.MATERIAL_SHORTAGE,
                node_id="T3-ST",
                description="特种钢材受出口政策与能源价格影响，供应稳定性下降",
                occurrence_prob=0.4,
                severity=6.0,
                detectability=5.0,
            ),
            # --- 供应商集中类 ---
            RiskFactor(
                factor_id="RF-RE-01",
                name="稀土集中供应风险",
                name_en="Rare Earth Supply Concentration",
                category=RiskCategory.SUPPLIER_CONCENTRATION,
                node_id="T3-RE",
                description="稀土材料高度集中于少数地区供应商，地缘政治与出口管制风险显著",
                occurrence_prob=0.7,
                severity=9.0,
                detectability=7.5,
            ),
            RiskFactor(
                factor_id="RF-T1E-01",
                name="电子电气上游汇聚风险",
                name_en="E/E Upstream Concentration Risk",
                category=RiskCategory.SUPPLIER_CONCENTRATION,
                node_id="T1-E",
                description="电子电气系统集成商的上游供应商（ECU、传感器）高度集中于特定区域",
                occurrence_prob=0.65,
                severity=8.0,
                detectability=6.5,
            ),
            # --- 物流中断类 ---
            RiskFactor(
                factor_id="RF-E2-01",
                name="涡轮增压器物流脆弱",
                name_en="Turbocharger Logistics Vulnerability",
                category=RiskCategory.LOGISTICS_DISRUPTION,
                node_id="T2-E2",
                description="涡轮增压器供应商地处华东，区域性物流中断（如台风、港口拥堵）风险突出",
                occurrence_prob=0.6,
                severity=7.5,
                detectability=6.5,
            ),
            RiskFactor(
                factor_id="RF-NI-01",
                name="镍基合金物流中断",
                name_en="Nickel Alloy Logistics Disruption",
                category=RiskCategory.LOGISTICS_DISRUPTION,
                node_id="T3-NI",
                description="镍基合金原料产地偏远（西北内陆），长途运输周期长，物流中断影响涡轮增压器生产",
                occurrence_prob=0.45,
                severity=6.0,
                detectability=5.5,
            ),
            RiskFactor(
                factor_id="RF-RB-01",
                name="合成橡胶物流风险",
                name_en="Synthetic Rubber Logistics Risk",
                category=RiskCategory.LOGISTICS_DISRUPTION,
                node_id="T3-RB",
                description="合成橡胶运输受制于危化品监管与港口限制，物流可靠性偏低",
                occurrence_prob=0.5,
                severity=5.5,
                detectability=5.5,
            ),
            # --- 需求波动类 ---
            RiskFactor(
                factor_id="RF-OEM-01",
                name="总装厂需求波动",
                name_en="OEM Demand Volatility",
                category=RiskCategory.DEMAND_VOLATILITY,
                node_id="OEM",
                description="整车市场受季节性与宏观经济影响，总装厂需求波动向上游传导，形成牛鞭效应",
                occurrence_prob=0.5,
                severity=5.5,
                detectability=4.5,
            ),
            RiskFactor(
                factor_id="RF-S1-01",
                name="悬挂系统需求不确定",
                name_en="Suspension System Demand Uncertainty",
                category=RiskCategory.DEMAND_VOLATILITY,
                node_id="T2-S1",
                description="悬挂系统受整车平台切换与消费者偏好变化影响，需求预测难度较高",
                occurrence_prob=0.35,
                severity=4.5,
                detectability=4.0,
            ),
        ]

        for factor in factors:
            registry.register(factor)

        return registry

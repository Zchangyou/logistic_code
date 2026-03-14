"""
供应链风险贝叶斯网络
Supply Chain Risk Bayesian Network
"""
from typing import Dict, List, Tuple

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class SupplyChainBayesianNet:
    """供应链风险贝叶斯网络 (Supply Chain Risk Bayesian Network)

    DAG结构：
        MaterialShortage   → CapacityShortage
        SupplierConcentration → CapacityShortage
        CapacityShortage   → DeliveryDelay
        LogisticsDisruption → DeliveryDelay
        DeliveryDelay      → ProductionHalt

    所有变量为二值：0=正常，1=风险。

    Attributes:
        model: pgmpy BayesianNetwork 对象
        inference: VariableElimination 推断引擎
    """

    NODES: List[str] = [
        "MaterialShortage",
        "SupplierConcentration",
        "LogisticsDisruption",
        "CapacityShortage",
        "DeliveryDelay",
        "ProductionHalt",
    ]

    def __init__(self) -> None:
        """初始化并构建贝叶斯网络。"""
        self.model: BayesianNetwork = None
        self.inference: VariableElimination = None
        self._build_model()

    def _build_model(self) -> None:
        """构建贝叶斯网络结构与CPD。

        变量均为二值（0=正常，1=风险）。
        CPD使用专家知识手动设定。
        pgmpy TabularCPD中，values矩阵：
        - 行对应目标变量的取值（0行→P(var=0), 1行→P(var=1)）
        - 列对应父节点状态的笛卡尔积，顺序按父节点列表反向排列（最后父节点变化最快）
        """
        # 网络结构
        edges = [
            ("MaterialShortage", "CapacityShortage"),
            ("SupplierConcentration", "CapacityShortage"),
            ("CapacityShortage", "DeliveryDelay"),
            ("LogisticsDisruption", "DeliveryDelay"),
            ("DeliveryDelay", "ProductionHalt"),
        ]
        self.model = BayesianNetwork(edges)

        # ---- 先验 CPD ----
        # P(MaterialShortage=1) = 0.35
        cpd_ms = TabularCPD(
            variable="MaterialShortage",
            variable_card=2,
            values=[[0.65], [0.35]],
        )

        # P(SupplierConcentration=1) = 0.30
        cpd_sc = TabularCPD(
            variable="SupplierConcentration",
            variable_card=2,
            values=[[0.70], [0.30]],
        )

        # P(LogisticsDisruption=1) = 0.25
        cpd_ld = TabularCPD(
            variable="LogisticsDisruption",
            variable_card=2,
            values=[[0.75], [0.25]],
        )

        # ---- CapacityShortage | MaterialShortage, SupplierConcentration ----
        # 列顺序（pgmpy约定，父变量状态：最后列父变量变化最快）：
        # 父变量顺序: [MaterialShortage, SupplierConcentration]
        # 列 0: MS=0, SC=0 → P(CS=1) = 0.05
        # 列 1: MS=0, SC=1 → P(CS=1) = 0.55
        # 列 2: MS=1, SC=0 → P(CS=1) = 0.60
        # 列 3: MS=1, SC=1 → P(CS=1) = 0.90
        cpd_cap = TabularCPD(
            variable="CapacityShortage",
            variable_card=2,
            values=[
                [0.95, 0.45, 0.40, 0.10],  # P(CS=0)
                [0.05, 0.55, 0.60, 0.90],  # P(CS=1)
            ],
            evidence=["MaterialShortage", "SupplierConcentration"],
            evidence_card=[2, 2],
        )

        # ---- DeliveryDelay | CapacityShortage, LogisticsDisruption ----
        # 列 0: CS=0, LD=0 → P(DD=1) = 0.05
        # 列 1: CS=0, LD=1 → P(DD=1) = 0.60
        # 列 2: CS=1, LD=0 → P(DD=1) = 0.65
        # 列 3: CS=1, LD=1 → P(DD=1) = 0.92
        cpd_dd = TabularCPD(
            variable="DeliveryDelay",
            variable_card=2,
            values=[
                [0.95, 0.40, 0.35, 0.08],  # P(DD=0)
                [0.05, 0.60, 0.65, 0.92],  # P(DD=1)
            ],
            evidence=["CapacityShortage", "LogisticsDisruption"],
            evidence_card=[2, 2],
        )

        # ---- ProductionHalt | DeliveryDelay ----
        # 列 0: DD=0 → P(PH=1) = 0.03
        # 列 1: DD=1 → P(PH=1) = 0.78
        cpd_ph = TabularCPD(
            variable="ProductionHalt",
            variable_card=2,
            values=[
                [0.97, 0.22],  # P(PH=0)
                [0.03, 0.78],  # P(PH=1)
            ],
            evidence=["DeliveryDelay"],
            evidence_card=[2],
        )

        # 添加CPD到模型
        self.model.add_cpds(cpd_ms, cpd_sc, cpd_ld, cpd_cap, cpd_dd, cpd_ph)

        # 验证模型
        assert self.model.check_model(), "贝叶斯网络模型验证失败"

        # 初始化推断引擎
        self.inference = VariableElimination(self.model)

    def infer(self, evidence: Dict[str, int]) -> Dict[str, float]:
        """给定证据进行贝叶斯推断。

        Args:
            evidence: 观测证据字典，如 {'MaterialShortage': 1}

        Returns:
            各节点风险概率字典 {node_name: P(node=1)}
        """
        results = {}
        query_nodes = [n for n in self.NODES if n not in evidence]

        for node in query_nodes:
            try:
                q = self.inference.query(variables=[node], evidence=evidence, show_progress=False)
                results[node] = float(q.values[1])  # P(node=1)
            except Exception:
                results[node] = 0.0

        # 已知证据节点直接赋值
        for node, val in evidence.items():
            results[node] = float(val)

        return results

    def get_chip_shortage_scenario(self) -> Dict[str, float]:
        """芯片短缺场景：MaterialShortage=1, SupplierConcentration=1。

        Returns:
            各节点风险概率字典
        """
        return self.infer({
            "MaterialShortage": 1,
            "SupplierConcentration": 1,
        })

    def get_logistics_disruption_scenario(self) -> Dict[str, float]:
        """物流中断场景：LogisticsDisruption=1。

        Returns:
            各节点风险概率字典
        """
        return self.infer({"LogisticsDisruption": 1})

    def get_structure_for_visualization(self) -> List[Tuple[str, str]]:
        """返回DAG边列表用于绘图。

        Returns:
            边列表 [(source, target), ...]
        """
        return list(self.model.edges())

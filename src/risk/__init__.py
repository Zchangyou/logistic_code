"""
风险识别与评估模块
Risk Identification and Assessment Module

公开接口：
- RiskCategory: 风险类别枚举
- RiskFactor: 风险因素数据类
- RiskFactorRegistry: 风险因素注册表
- RiskIndicatorCalculator: 风险量化指标计算器
- NodeState: SIR节点状态枚举
- SIRResult: SIR仿真结果数据类
- SIRPropagationModel: SIR风险传播模型
- FuzzyRiskEvaluator: 模糊综合风险评价器
- SupplyChainBayesianNet: 供应链风险贝叶斯网络
- RiskAssessmentResult: 综合风险评估结果数据类
- RiskAssessor: 综合风险评估器
"""
from src.risk.factors import RiskCategory, RiskFactor, RiskFactorRegistry, CATEGORY_NAMES
from src.risk.indicators import RiskIndicatorCalculator
from src.risk.propagation import NodeState, SIRResult, SIRPropagationModel
from src.risk.fuzzy_eval import FuzzyRiskEvaluator
from src.risk.bayesian import SupplyChainBayesianNet
from src.risk.assessment import RiskAssessmentResult, RiskAssessor

__all__ = [
    # 原有接口
    "RiskCategory",
    "RiskFactor",
    "RiskFactorRegistry",
    "CATEGORY_NAMES",
    "RiskIndicatorCalculator",
    # 阶段4新增
    "NodeState",
    "SIRResult",
    "SIRPropagationModel",
    "FuzzyRiskEvaluator",
    "SupplyChainBayesianNet",
    "RiskAssessmentResult",
    "RiskAssessor",
]

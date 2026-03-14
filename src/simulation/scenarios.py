"""
预设风险场景运行器
Preset Risk Scenario Runner
"""
from typing import Dict

import pandas as pd

from src.risk.propagation import SIRPropagationModel, SIRResult


class ScenarioRunner:
    """风险场景运行器 (Risk Scenario Runner)

    封装4个预设场景的SIR仿真，提供统一的运行与对比接口。

    Attributes:
        network: SupplyChainNetwork 对象
        propagation_model: SIRPropagationModel 对象
    """

    SCENARIO_IDS = ["S1", "S2", "S3", "S4"]
    METRIC_LABELS = {
        "impact_range":    ("影响范围", "Impact Range (nodes)"),
        "impact_depth":    ("影响深度", "Impact Depth (tiers)"),
        "impact_duration": ("持续时间", "Duration (steps)"),
    }

    def __init__(self, network, propagation_model: SIRPropagationModel) -> None:
        """初始化场景运行器。

        Args:
            network: SupplyChainNetwork 对象
            propagation_model: SIRPropagationModel 对象
        """
        self.network = network
        self.propagation_model = propagation_model

    def run_all_scenarios(self) -> Dict[str, SIRResult]:
        """运行全部4个预设场景。

        Returns:
            场景ID → SIRResult 的字典
        """
        results = {}
        print("  运行场景 S1: 芯片晶圆停供...")
        results["S1"] = self.propagation_model.run_scenario_s1_chip()

        print("  运行场景 S2: 稀土材料集中风险...")
        results["S2"] = self.propagation_model.run_scenario_s2_rare_earth()

        print("  运行场景 S3: 华东区域中断...")
        results["S3"] = self.propagation_model.run_scenario_s3_east_china()

        print("  运行场景 S4: 需求骤增冲击...")
        results["S4"] = self.propagation_model.run_scenario_s4_demand_shock()

        return results

    def compare_scenarios(self, results: Dict[str, SIRResult]) -> pd.DataFrame:
        """生成场景对比 DataFrame。

        Args:
            results: run_all_scenarios() 返回的结果字典

        Returns:
            DataFrame，列为场景ID，行为影响指标
        """
        rows = []
        for sid, result in results.items():
            rows.append({
                "scenario_id":     sid,
                "scenario_name":   result.scenario_name,
                "scenario_name_en": result.scenario_name_en,
                "impact_range":    result.max_affected_count,
                "impact_depth":    result.impact_depth,
                "impact_duration": result.impact_duration,
                "final_affected":  result.final_affected_count,
                "recovery_time":   result.recovery_time,
            })

        df = pd.DataFrame(rows)
        df.set_index("scenario_id", inplace=True)
        return df

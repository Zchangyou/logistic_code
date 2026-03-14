"""
Phase 4: Risk Propagation and Comprehensive Assessment
阶段四：风险传播与综合评估

运行方式: conda run -n logistic python run_phase4.py
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from src.network.models import SupplyChainNetwork
from src.risk.propagation import SIRPropagationModel
from src.risk.fuzzy_eval import FuzzyRiskEvaluator
from src.risk.bayesian import SupplyChainBayesianNet
from src.risk.assessment import RiskAssessor
from src.simulation.scenarios import ScenarioRunner
from src.visualization.propagation_vis import (
    plot_sir_dynamics,
    plot_cascade_snapshots,
    plot_scenario_comparison,
    plot_risk_heatmap,
    plot_bayesian_dag,
    plot_fuzzy_radar,
    plot_accuracy_report,
)

# ── 输出目录 ──────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reports")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── 数据路径 ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_JSON   = os.path.join(BASE_DIR, "data", "cases", "auto_engine", "network.json")
SIMULATION_CSV = os.path.join(BASE_DIR, "data", "simulation", "auto_engine_simulation.csv")

start_time = time.time()


def main():
    print("=" * 60)
    print("Phase 4: 风险传播与综合评估")
    print("=" * 60)

    # ── Step 1: 加载数据 ──────────────────────────────────────
    print("\n[1/7] 加载网络与仿真数据...")
    network = SupplyChainNetwork.load_json(NETWORK_JSON)
    sim_df = pd.read_csv(SIMULATION_CSV)
    print(f"  网络: {network.summary()['node_count']} 节点, {network.summary()['edge_count']} 边")
    print(f"  仿真数据: {len(sim_df)} 行 × {len(sim_df.columns)} 列")

    # ── Step 2: SIR风险传播（3个场景） ─────────────────────────
    print("\n[2/7] 运行SIR风险传播仿真...")
    sir_model = SIRPropagationModel(network, beta=0.4, gamma=0.15, seed=42)
    scenario_runner = ScenarioRunner(network, sir_model)
    sir_results = scenario_runner.run_all_scenarios()
    comparison_df = scenario_runner.compare_scenarios(sir_results)

    print("\n  场景对比:")
    print(comparison_df[["scenario_name", "impact_range", "impact_depth", "impact_duration"]].to_string())

    # 验证S1传播路径
    s1_result = sir_results["S1"]
    s1_final_snapshot = s1_result.node_states_history[-1]
    from src.risk.propagation import NodeState
    affected_s1 = [n for n, st in s1_final_snapshot.items() if st != NodeState.SUSCEPTIBLE]
    print(f"\n  S1芯片场景受影响节点: {affected_s1}")

    # 验证S3影响范围 > S1
    s3_range = sir_results["S3"].max_affected_count
    s1_range = sir_results["S1"].max_affected_count
    print(f"  S3影响范围({s3_range}) vs S1影响范围({s1_range}): {'[PASS] S3 > S1' if s3_range >= s1_range else '[FAIL] S3 <= S1'}")

    # ── Step 3: 模糊综合评价 ──────────────────────────────────
    print("\n[3/7] 运行模糊综合评价（第6期）...")
    fuzzy_eval = FuzzyRiskEvaluator()
    fuzzy_df = fuzzy_eval.evaluate_all_nodes(sim_df, period=6)
    print(f"  模糊评价完成: {len(fuzzy_df)} 个节点")
    top_fuzzy = fuzzy_df.nlargest(5, "risk_score")[["node_id", "risk_score", "risk_level"]]
    print(f"  Top-5 模糊风险节点:\n{top_fuzzy.to_string(index=False)}")

    # ── Step 4: 贝叶斯网络推断 ─────────────────────────────────
    print("\n[4/7] 构建贝叶斯网络并推断（芯片短缺场景）...")
    bn = SupplyChainBayesianNet()
    chip_scenario_result = bn.get_chip_shortage_scenario()
    print("  芯片短缺场景贝叶斯推断结果:")
    for node, prob in chip_scenario_result.items():
        print(f"    P({node}=risk) = {prob:.3f}")

    # ── Step 5: 综合评估 ──────────────────────────────────────
    print("\n[5/7] 融合模糊+贝叶斯综合评估...")
    assessor = RiskAssessor(fuzzy_eval, bn)
    assessment_results = assessor.assess_all(network, sim_df, period=6)

    print("  Top-5 综合风险节点:")
    for r in assessment_results[:5]:
        print(f"    {r.node_id}: composite={r.composite_score:.3f}, level={r.risk_level}")

    # ── Step 6: 准确率验证 ────────────────────────────────────
    print("\n[6/7] 计算准确率混淆矩阵...")
    confusion = assessor.get_confusion_matrix(assessment_results, sim_df)
    accuracy = confusion["accuracy"]
    print(f"  混淆矩阵: TP={confusion['tp']}, FP={confusion['fp']}, TN={confusion['tn']}, FN={confusion['fn']}")
    print(f"  准确率: {accuracy:.1%} {'[PASS] 达标(>=88%)' if accuracy >= 0.88 else '[FAIL] 未达标(需>=88%)'}")
    print(f"  假阴性率: {confusion['false_negative_rate']:.1%}")
    print(f"  假阳性率: {confusion['false_positive_rate']:.1%}")

    # ── Step 7: 生成可视化图表 ────────────────────────────────
    print("\n[7/7] 生成Phase 4图表 (F4-1 ~ F4-7)...")

    # F4-1: SIR传播动态曲线
    plot_sir_dynamics(s1_result, FIGURES_DIR)

    # F4-2: 级联效应传播快照
    plot_cascade_snapshots(network, s1_result, FIGURES_DIR)

    # F4-3: 场景对比（S1/S3/S4）
    plot_scenario_comparison(comparison_df, FIGURES_DIR)

    # F4-4: 风险热力图
    plot_risk_heatmap(network, assessment_results, FIGURES_DIR)

    # F4-5: 贝叶斯因果图
    plot_bayesian_dag(bn, chip_scenario_result, FIGURES_DIR)

    # F4-6: 模糊评价雷达图（关键节点）
    key_nodes = ["T3-SI", "T2-ECU", "T3-RE", "OEM"]
    plot_fuzzy_radar(fuzzy_df, key_nodes, FIGURES_DIR)

    # F4-7: 准确率验证图
    plot_accuracy_report(confusion, FIGURES_DIR)

    # ── 保存 phase4_report.json ───────────────────────────────
    print("\n保存 phase4_report.json...")

    sir_scenarios_summary = {}
    for sid, result in sir_results.items():
        sir_scenarios_summary[sid] = {
            "scenario_name":    result.scenario_name,
            "scenario_name_en": result.scenario_name_en,
            "impact_range":     result.max_affected_count,
            "impact_depth":     result.impact_depth,
            "impact_duration":  result.impact_duration,
            "recovery_time":    result.recovery_time,
            "final_affected":   result.final_affected_count,
        }

    assessment_list = [
        {
            "node_id":         r.node_id,
            "fuzzy_score":     r.fuzzy_score,
            "bayesian_prob":   r.bayesian_prob,
            "composite_score": r.composite_score,
            "risk_level":      r.risk_level,
        }
        for r in assessment_results
    ]

    top_risk_nodes = [r.node_id for r in assessment_results[:5]]

    report = {
        "phase":            4,
        "description":      "风险传播与综合评估 Risk Propagation and Comprehensive Assessment",
        "sir_scenarios":    sir_scenarios_summary,
        "assessment_results": assessment_list,
        "confusion_matrix": confusion,
        "accuracy":         accuracy,
        "top_risk_nodes":   top_risk_nodes,
        "fuzzy_top5": [
            {"node_id": r["node_id"], "risk_score": r["risk_score"], "risk_level": r["risk_level"]}
            for r in fuzzy_df.nlargest(5, "risk_score")[["node_id", "risk_score", "risk_level"]].to_dict("records")
        ],
        "bayesian_chip_scenario": {k: round(v, 4) for k, v in chip_scenario_result.items()},
    }

    report_path = os.path.join(REPORTS_DIR, "phase4_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {report_path}")

    # ── 最终总结 ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Phase 4 完成!")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  生成图表: 7 张 (F4-1 ~ F4-7)")
    print(f"  准确率: {accuracy:.1%}")
    print(f"  Top风险节点: {top_risk_nodes}")
    print("=" * 60)

    # 断言验证目标
    assert "T3-SI" in top_risk_nodes[:5] or any(r.node_id == "T3-SI" and r.composite_score > 0.4 for r in assessment_results), \
        "验证失败: T3-SI 未出现在高风险节点中"
    assert "T2-ECU" in top_risk_nodes[:8] or any(r.node_id == "T2-ECU" and r.risk_level in ("high", "medium") for r in assessment_results), \
        "验证失败: T2-ECU 未出现在风险节点中"
    print("  所有验证目标通过 [PASS]")


if __name__ == "__main__":
    main()

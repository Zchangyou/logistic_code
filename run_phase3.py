"""
阶段三执行脚本：风险因素分类、FMEA量化与仿真数据生成
Phase 3 Runner: Risk Factor Classification, FMEA Quantification, Simulation Data Generation
"""
import json
import os
import sys

# 将项目根目录加入 Python 路径
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.network.models import SupplyChainNetwork
from src.risk.factors import RiskFactorRegistry
from src.risk.indicators import RiskIndicatorCalculator
from src.simulation.data_generator import SimulationDataGenerator
from src.visualization.charts import plot_rpn_ranking, plot_fmea_matrix


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
NETWORK_JSON    = os.path.join(_PROJECT_ROOT, "data", "cases", "auto_engine", "network.json")
TOPOLOGY_JSON   = os.path.join(_PROJECT_ROOT, "outputs", "reports", "topology_report.json")
SIM_CSV         = os.path.join(_PROJECT_ROOT, "data", "simulation", "auto_engine_simulation.csv")
FIGURES_DIR     = os.path.join(_PROJECT_ROOT, "outputs", "figures")
REPORTS_DIR     = os.path.join(_PROJECT_ROOT, "outputs", "reports")
PHASE3_REPORT   = os.path.join(REPORTS_DIR, "phase3_report.json")


def load_network() -> SupplyChainNetwork:
    """加载汽车发动机供应链网络。"""
    print(f"\n[1] 加载网络: {NETWORK_JSON}")
    network = SupplyChainNetwork.load_json(NETWORK_JSON)
    summary = network.summary()
    print(f"    节点数: {summary['node_count']}, 边数: {summary['edge_count']}")
    return network


def load_topology_report() -> dict:
    """加载拓扑分析报告。"""
    print(f"\n[2] 加载拓扑报告: {TOPOLOGY_JSON}")
    with open(TOPOLOGY_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)
    print(f"    拓扑报告节点数: {len(report.get('nodes', {}))}")
    return report


def build_risk_registry(network: SupplyChainNetwork, topology_report: dict) -> RiskFactorRegistry:
    """构建并丰富化风险因素注册表。"""
    print("\n[3] 构建风险因素注册表 ...")
    registry = RiskFactorRegistry.build_auto_engine_registry()
    print(f"    已注册风险因素: {len(registry.get_all_factors())} 个")

    print("    使用网络指标丰富化 occurrence_prob ...")
    calculator = RiskIndicatorCalculator(network, topology_report)
    registry = calculator.enrich_registry(registry)
    print("    丰富化完成")

    return registry, calculator


def print_rpn_ranking(registry: RiskFactorRegistry) -> None:
    """打印 RPN 排名，并验证四类风险的大小关系。"""
    print("\n[4] RPN 排名（降序）:")
    factors = registry.get_rpn_ranking()
    for rank, f in enumerate(factors, 1):
        print(f"    #{rank:2d}  {f.factor_id:<12s}  RPN={f.rpn:7.2f}  "
              f"[{f.category.value}]  {f.name}")

    # 验证四类风险 RPN 均值关系
    from src.risk.factors import RiskCategory
    avg = {}
    for cat in RiskCategory:
        cat_factors = registry.get_factors_by_category(cat)
        max_rpn = max((f.rpn for f in cat_factors), default=0)
        avg[cat] = max_rpn

    print("\n    各类别最高 RPN:")
    for cat, val in sorted(avg.items(), key=lambda x: -x[1]):
        print(f"      {cat.value:<28s}: {val:.2f}")

    # 验证顺序
    material  = avg[RiskCategory.MATERIAL_SHORTAGE]
    supplier  = avg[RiskCategory.SUPPLIER_CONCENTRATION]
    logistics = avg[RiskCategory.LOGISTICS_DISRUPTION]
    demand    = avg[RiskCategory.DEMAND_VOLATILITY]

    ok = (material > supplier) and (supplier > logistics) and (logistics > demand)
    status = "PASS" if ok else "WARN"
    print(f"\n    [验证] 材料>{supplier:.1f} > 物流>{logistics:.1f} > 需求>{demand:.1f}: [{status}]")


def generate_simulation(network: SupplyChainNetwork) -> None:
    """生成并保存6期仿真数据。"""
    print("\n[5] 生成仿真数据 ...")
    gen = SimulationDataGenerator(seed=42)
    data = gen.generate(network)
    gen.save_csv(data, SIM_CSV)

    df = gen.to_dataframe(data)
    risk_count = df[df["is_risk_embedded"]]["node_id"].nunique()
    print(f"    仿真数据: {len(df)} 条（{df['node_id'].nunique()} 节点 × {df['period'].nunique()} 期）")
    print(f"    预埋风险节点数: {risk_count}")


def generate_figures(registry: RiskFactorRegistry) -> None:
    """生成两张阶段三图表。"""
    print("\n[6] 生成图表 ...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_rpn_ranking(registry, FIGURES_DIR)
    plot_fmea_matrix(registry, FIGURES_DIR)


def save_phase3_report(
    registry: RiskFactorRegistry,
    calculator: RiskIndicatorCalculator,
    network: SupplyChainNetwork,
) -> None:
    """保存阶段三结构化报告。"""
    print(f"\n[7] 保存报告: {PHASE3_REPORT}")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # RPN 排名列表
    rpn_ranking = [
        {
            "factor_id": f.factor_id,
            "name": f.name,
            "name_en": f.name_en,
            "rpn": round(f.rpn, 4),
            "occurrence_prob": f.occurrence_prob,
            "severity": f.severity,
            "detectability": f.detectability,
            "category": f.category.value,
            "node_id": f.node_id,
        }
        for f in registry.get_rpn_ranking()
    ]

    # 仿真摘要
    sim_df = pd.read_csv(SIM_CSV)
    simulation_summary = {
        "total_records": int(len(sim_df)),
        "nodes": int(sim_df["node_id"].nunique()),
        "periods": int(sim_df["period"].nunique()),
        "embedded_risks_nodes": int(sim_df[sim_df["is_risk_embedded"]]["node_id"].nunique()),
    }

    # 指标汇总表
    indicator_df = calculator.get_indicator_table()
    indicator_table = indicator_df.to_dict(orient="records")

    report = {
        "phase": 3,
        "description": "风险因素分类、FMEA量化与仿真数据生成",
        "rpn_ranking": rpn_ranking,
        "simulation_summary": simulation_summary,
        "indicator_table": indicator_table,
    }

    with open(PHASE3_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"    已保存: {PHASE3_REPORT}")


def main() -> None:
    """阶段三主流程。"""
    print("=" * 60)
    print("  阶段三：风险因素分类、FMEA量化与仿真数据生成")
    print("  Phase 3: Risk Factors, FMEA & Simulation Data")
    print("=" * 60)

    network = load_network()
    topology_report = load_topology_report()
    registry, calculator = build_risk_registry(network, topology_report)
    print_rpn_ranking(registry)
    generate_simulation(network)
    generate_figures(registry)
    save_phase3_report(registry, calculator, network)

    print("\n" + "=" * 60)
    print("  阶段三完成！")
    print(f"  图表目录: {FIGURES_DIR}")
    print(f"  报告路径: {PHASE3_REPORT}")
    print(f"  仿真数据: {SIM_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd

from src.network.models import SupplyChainNetwork
from src.network.topology import TopologyAnalyzer
from src.risk.assessment import RiskAssessor
from src.risk.bayesian import SupplyChainBayesianNet
from src.risk.factors import RiskFactorRegistry
from src.risk.fuzzy_eval import FuzzyRiskEvaluator
from src.risk.indicators import RiskIndicatorCalculator
from src.risk.propagation import SIRPropagationModel


ROOT = Path(__file__).resolve().parents[2]
NETWORK_JSON = ROOT / "data" / "cases" / "auto_engine" / "network.json"
SIM_CSV = ROOT / "data" / "simulation" / "auto_engine_simulation.csv"


def _load_base_data():
    network = SupplyChainNetwork.load_json(str(NETWORK_JSON))
    sim_df = pd.read_csv(SIM_CSV)
    topo_report = TopologyAnalyzer(network).get_report()
    return network, sim_df, topo_report


def test_risk_factor_registry_rpn_order():
    registry = RiskFactorRegistry.build_auto_engine_registry()
    ranked = registry.get_rpn_ranking()
    assert len(ranked) >= 12
    assert ranked[0].factor_id == "RF-SI-01"
    assert ranked[0].rpn > ranked[-1].rpn


def test_indicator_table_and_enrichment():
    network, _, topo_report = _load_base_data()
    calc = RiskIndicatorCalculator(network, topo_report)
    table = calc.get_indicator_table()

    assert len(table) == 25
    for col in [
        "material_shortage_index",
        "concentration_index",
        "logistics_risk_index",
        "demand_volatility_index",
    ]:
        assert table[col].between(0.0, 1.0).all()

    registry = RiskFactorRegistry.build_auto_engine_registry()
    calc.enrich_registry(registry)
    assert all(0.0 <= f.occurrence_prob <= 1.0 for f in registry.get_all_factors())


def test_sir_scenarios_basic_behavior():
    network, _, _ = _load_base_data()
    sir = SIRPropagationModel(network, beta=0.4, gamma=0.15, seed=42)

    s1 = sir.run_scenario_s1_chip()
    s3 = sir.run_scenario_s3_east_china()

    assert len(s1.time_steps) == 31
    assert len(s1.s_counts) == 31
    assert s1.max_affected_count >= 1
    assert s3.max_affected_count >= s1.max_affected_count
    assert s1.impact_depth >= 1


def test_bayesian_and_assessment_pipeline():
    network, sim_df, _ = _load_base_data()
    fuzzy = FuzzyRiskEvaluator()
    bn = SupplyChainBayesianNet()
    assessor = RiskAssessor(fuzzy, bn)

    chip = bn.get_chip_shortage_scenario()
    logistics = bn.get_logistics_disruption_scenario()
    assert chip["MaterialShortage"] == 1.0
    assert chip["SupplierConcentration"] == 1.0
    assert logistics["LogisticsDisruption"] == 1.0
    assert 0.0 <= chip["ProductionHalt"] <= 1.0
    assert 0.0 <= logistics["ProductionHalt"] <= 1.0

    results = assessor.assess_all(network, sim_df, period=6)
    assert len(results) == 25
    assert results[0].composite_score >= results[-1].composite_score
    assert all(0.0 <= r.composite_score <= 1.0 for r in results)

    confusion = assessor.get_confusion_matrix(results, sim_df)
    assert confusion["accuracy"] >= 0.88

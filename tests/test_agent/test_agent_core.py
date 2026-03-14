import json
from pathlib import Path

import pandas as pd

from src.agent.early_warning import EarlyWarningSystem
from src.agent.knowledge_base import SupplyChainKnowledgeBase
from src.agent.multi_agent import MultiAgentSystem
from src.agent.strategy import StrategyRecommender


ROOT = Path(__file__).resolve().parents[2]
SIM_CSV = ROOT / "data" / "simulation" / "auto_engine_simulation.csv"
PHASE4_REPORT = ROOT / "outputs" / "reports" / "phase4_report.json"


def _load_inputs():
    sim_df = pd.read_csv(SIM_CSV)
    with open(PHASE4_REPORT, "r", encoding="utf-8") as f:
        phase4 = json.load(f)
    return sim_df, phase4


def test_knowledge_base_search_and_summary():
    kb = SupplyChainKnowledgeBase()
    summary = kb.summary()

    assert summary["total_events"] >= 1
    assert summary["total_strategies"] >= 1

    results = kb.search("芯片短缺 ECU", top_k=3)
    assert len(results) >= 1
    assert any("芯片" in doc.title for doc in results)


def test_early_warning_generates_events_and_key_trends():
    sim_df, _ = _load_inputs()
    ew = EarlyWarningSystem(df=sim_df)
    report = ew.analyze()

    assert report.total_warnings > 0
    assert "T3-SI" in report.first_warning_period
    trends = ew.get_key_node_trends(report)
    assert "T3-SI" in trends
    assert len(trends["T3-SI"]) == 6


def test_strategy_recommender_outputs_valid_plan():
    recommender = StrategyRecommender()
    plan = recommender.recommend(node_id="T3-SI", current_risk_score=0.58, top_k=3)

    assert plan.target_node_id == "T3-SI"
    assert len(plan.recommended_strategies) == 3
    assert plan.combined_risk_reduction > 0
    assert plan.expected_risk_after_combo < plan.current_risk_score

    comparison = recommender.compare_strategies_for_node("T3-SI", current_risk_score=0.58)
    assert len(comparison) >= 1
    assert "risk_reduction_pct" in comparison[0]


def test_multi_agent_system_reduces_risk():
    sim_df, phase4 = _load_inputs()
    risk_results = phase4["assessment_results"]
    target_nodes = [r["node_id"] for r in risk_results if r["composite_score"] >= 0.3]

    mas = MultiAgentSystem()
    report = mas.run(
        sim_data=sim_df,
        risk_results=risk_results,
        scenario="S1_chip_shortage",
        target_nodes=target_nodes,
        budget_limit=2.5,
    )

    assert len(report.messages) == 5
    assert len(report.purchase_proposal.actions) >= 1
    assert len(report.integrated_plan.integrated_actions) >= 1
    assert report.overall_risk_reduction > 0
    assert report.post_disposal_risk_scores["T3-SI"] <= report.pre_disposal_risk_scores["T3-SI"]

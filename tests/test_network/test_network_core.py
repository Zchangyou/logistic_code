from pathlib import Path

from src.network.models import SupplyChainNetwork
from src.network.topology import TopologyAnalyzer
from src.network.vulnerability import VulnerabilityAnalyzer


ROOT = Path(__file__).resolve().parents[2]
NETWORK_JSON = ROOT / "data" / "cases" / "auto_engine" / "network.json"


def test_load_auto_engine_network_summary():
    network = SupplyChainNetwork.load_json(str(NETWORK_JSON))
    summary = network.summary()

    assert summary["node_count"] == 25
    assert summary["edge_count"] == 38
    assert summary["tier_distribution"] == {0: 1, 1: 3, 2: 9, 3: 12}


def test_network_json_roundtrip(tmp_path):
    network = SupplyChainNetwork.load_json(str(NETWORK_JSON))
    out = tmp_path / "roundtrip.json"
    network.save_json(str(out))

    loaded = SupplyChainNetwork.load_json(str(out))
    assert loaded.summary() == network.summary()


def test_topology_and_vulnerability_outputs_are_valid():
    network = SupplyChainNetwork.load_json(str(NETWORK_JSON))

    topo = TopologyAnalyzer(network).get_report()
    assert "nodes" in topo and "network" in topo
    assert "OEM" in topo["nodes"]
    assert topo["network"]["node_count"] == 25
    assert 0.0 <= topo["nodes"]["OEM"]["betweenness_centrality"] <= 1.0

    vuln = VulnerabilityAnalyzer(network)
    robust = vuln.robustness_analysis(n_steps=10, n_trials=2, seed=42)
    assert len(robust["random"]) >= 2
    assert len(robust["targeted"]) >= 2
    assert robust["random"][0][1] == 1.0
    assert robust["targeted"][0][1] == 1.0

    hhi = vuln.compute_hhi()
    assert "T2-T1" in hhi
    assert 0.0 <= hhi["T2-T1"] <= 1.0

    key_nodes = vuln.get_key_nodes(top_n=5)
    assert len(key_nodes) == 5
    assert all("node_id" in item and "composite_score" in item for item in key_nodes)

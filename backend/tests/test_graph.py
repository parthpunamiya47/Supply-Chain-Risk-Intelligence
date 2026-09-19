"""SupplyChain Sentinel - Stage 3: graph & propagation tests (pytest).

The three propagation scenarios double as living documentation of the formula:

    propagated(n) = R_source * prod over path edges (weight_e * DECAY)
    DECAY = 0.7, MAX_HOPS = 4, contributions < 0.01 pruned

Run from the project root:  .venv/Scripts/python -m pytest backend/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.graph import EDGES, NODES, build_graph, to_react_flow  # noqa: E402
from backend.propagation import DECAY, MAX_HOPS, propagate  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------ structure ----
def test_graph_structure():
    g = build_graph()
    assert g.number_of_nodes() == len(NODES)
    assert g.number_of_edges() == len(EDGES)
    types = {d["type"] for _, d in g.nodes(data=True)}
    assert {"supplier", "factory", "warehouse", "customer", "port", "route"} <= types
    # capacity present where applicable, absent for customers
    assert all("capacity" in d for _, d in g.nodes(data=True) if d["type"] != "customer")
    # every edge has a relationship type and a 0-1 dependency weight
    for _, _, d in g.edges(data=True):
        assert d["relationship"] in {"supplies", "produces_for", "stores_to", "ships_via", "serves"}
        assert 0 < d["weight"] <= 1
    # DAG layering: suppliers have no incoming edges, customers no outgoing
    assert all(g.in_degree(n) == 0 for n, d in g.nodes(data=True) if d["type"] == "supplier")
    assert all(g.out_degree(n) == 0 for n, d in g.nodes(data=True) if d["type"] == "customer")


def test_react_flow_serialization():
    flow = to_react_flow()
    assert {n["id"] for n in flow["nodes"]} == {n["id"] for n in NODES}
    assert len(flow["edges"]) == len(EDGES)
    assert all("position" in n and "x" in n["position"] and "y" in n["position"] for n in flow["nodes"])
    # every React Flow edge references existing nodes
    ids = {n["id"] for n in flow["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in flow["edges"])


# ------------------------------------------------------------------ formula ----
def test_formula_direct_successor():
    # sup_c -0.85-> fac_2: propagated = R * 0.85 * 0.7
    for r in (0.85, 0.5):
        res = propagate("sup_c", r)
        fac2 = next(a for a in res["affected_nodes"] if a["node_id"] == "fac_2")
        assert fac2["hops"] == 1
        assert fac2["propagated_risk"] == pytest.approx(round(r * 0.85 * DECAY, 4), abs=1e-9)
        assert fac2["propagation_path"] == ["sup_c", "fac_2"]


def test_formula_multi_hop():
    # sup_c -0.85-> fac_2 -0.60-> route_rail: R * 0.85*0.7 * 0.60*0.7
    r = 0.85
    res = propagate("sup_c", r)
    rail = next(a for a in res["affected_nodes"] if a["node_id"] == "route_rail")
    expected = round(r * (0.85 * DECAY) * (0.60 * DECAY), 4)
    assert rail["hops"] == 2
    assert rail["propagated_risk"] == pytest.approx(expected, abs=1e-9)
    assert rail["propagation_path"] == ["sup_c", "fac_2", "route_rail"]


def test_max_hops_enforced():
    # any affected node sits at most MAX_HOPS edges from the source
    res = propagate("sup_a", 0.95)
    assert res["affected_nodes"], "sup_a should reach downstream nodes"
    assert all(a["hops"] <= MAX_HOPS for a in res["affected_nodes"])
    assert res["summary"]["max_hops_to_any_node"] <= MAX_HOPS


# ------------------------------------------------------- three scenarios ----
def test_scenario_1_critical_supplier_downstream_cascade():
    """Supplier C at 0.85 -> factory, rail, warehouses, customers."""
    res = propagate("sup_c", 0.85)
    assert res["original_risk"] == 0.85
    ids = {a["node_id"] for a in res["affected_nodes"]}
    assert {"fac_2", "route_rail", "wh_2", "wh_3", "cus_2", "cus_3"} <= ids

    # strictly decreasing along the dominant path
    by_id = {a["node_id"]: a["propagated_risk"] for a in res["affected_nodes"]}
    assert by_id["fac_2"] > by_id["route_rail"] > by_id["wh_2"]
    # all propagated risks are bounded by the source risk
    assert all(0 < v <= 0.85 for v in by_id.values())
    # customers are reachable within the hop budget
    assert res["summary"]["customer_nodes_affected"] >= 1
    # every affected node carries a readable dominant path
    assert all(len(a["propagation_path"]) >= 2 for a in res["affected_nodes"])


def test_scenario_2_low_risk_event_stays_local():
    """Supplier D at 0.15 (weight 0.40 to fac_2): only strong enough for the
    first hop -> 0.15*0.40*0.7 = 0.042; deeper hops fall under pruning."""
    res = propagate("sup_d", 0.15)
    ids = {a["node_id"] for a in res["affected_nodes"]}
    assert "fac_2" in ids
    fac2 = next(a for a in res["affected_nodes"] if a["node_id"] == "fac_2")
    assert fac2["propagated_risk"] == pytest.approx(round(0.15 * 0.40 * DECAY, 4), abs=1e-9)
    # 2nd hop would be 0.042 * 0.6*0.7 = 0.01764 -> 3rd hop 0.0052 < 0.01 prune
    deeper = [a for a in res["affected_nodes"] if a["hops"] >= 3]
    assert deeper == []
    # a weak event must affect fewer nodes than the strong event in scenario 1
    strong = propagate("sup_c", 0.85)
    assert res["summary"]["nodes_affected"] < strong["summary"]["nodes_affected"]


def test_scenario_3_middle_mile_port_disruption():
    """Port of Long Beach at 0.90 hits both warehouses it feeds and their
    customers; source node itself is never in the affected list."""
    res = propagate("port_lb", 0.90)
    ids = {a["node_id"] for a in res["affected_nodes"]}
    assert "port_lb" not in ids
    assert {"wh_1", "wh_2", "cus_1", "cus_2"} <= ids
    wh1 = next(a for a in res["affected_nodes"] if a["node_id"] == "wh_1")
    assert wh1["propagated_risk"] == pytest.approx(round(0.90 * 0.75 * DECAY, 4), abs=1e-9)
    # cus_1 is reached only via wh_1 (weight 0.90 -> strongest customer link)
    cus1 = next(a for a in res["affected_nodes"] if a["node_id"] == "cus_1")
    assert cus1["propagation_path"] == ["port_lb", "wh_1", "cus_1"]
    assert cus1["propagated_risk"] == pytest.approx(
        round(0.90 * (0.75 * DECAY) * (0.90 * DECAY), 4), abs=1e-9
    )


def test_scenario_event_at_factory_propagates_both_branches():
    """Factory 1 at 0.60 reaches port + warehouses + customers (branching)."""
    res = propagate("fac_1", 0.60)
    ids = {a["node_id"] for a in res["affected_nodes"]}
    assert {"port_lb", "wh_1", "wh_2", "wh_3", "cus_1", "cus_2", "cus_3"} <= ids
    # no upstream contamination: suppliers must not be affected
    assert not ids & {"sup_a", "sup_b", "sup_c", "sup_d"}


# ------------------------------------------------------------------ errors ----
def test_unknown_node_returns_404(client):
    res = client.post("/api/graph/propagate-risk", json={"node_id": "sup_z", "risk_probability": 0.5})
    assert res.status_code == 404


def test_risk_probability_out_of_range_returns_422(client):
    for bad in (1.5, -0.1):
        res = client.post(
            "/api/graph/propagate-risk", json={"node_id": "sup_a", "risk_probability": bad}
        )
        assert res.status_code == 422


def test_missing_fields_return_422(client):
    res = client.post("/api/graph/propagate-risk", json={"node_id": "sup_a"})
    assert res.status_code == 422
    assert any(e["loc"][-1] == "risk_probability" for e in res.json()["detail"])


# ------------------------------------------------------------ API layer ----
def test_api_get_graph(client):
    res = client.get("/api/graph")
    assert res.status_code == 200
    data = res.json()
    assert data["node_count"] == len(NODES)
    assert data["edge_count"] == len(EDGES)
    assert len(data["tiers"]) >= 4  # supplier -> factory(+mid) -> warehouse -> customer
    assert all("position" in n for n in data["nodes"])


def test_api_propagate_roundtrip(client):
    res = client.post(
        "/api/graph/propagate-risk", json={"node_id": "sup_a", "risk_probability": 0.85}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["source"] == "sup_a"
    assert data["original_risk"] == 0.85
    assert data["parameters"]["decay"] == DECAY
    assert len(data["affected_nodes"]) > 0
    top = data["affected_nodes"][0]
    # highest propagated risk must be the immediate high-weight successor
    assert top["node_id"] == "fac_1"
    assert top["propagated_risk"] == pytest.approx(round(0.85 * 0.80 * DECAY, 4), abs=1e-9)
    assert set(top.keys()) >= {
        "node_id", "name", "type", "original_risk", "propagated_risk",
        "contribution", "hops", "propagation_path", "path_edges",
    }

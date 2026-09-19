"""SupplyChain Sentinel - Stage 3: demo supply-chain network.

A small but realistic multi-tier network (~14 nodes, 16 edges):

  Tier 0 (suppliers)   -> Tier 1 (factories) -> Tier 2 (warehouses) -> Tier 3 (customers)
  plus two optional middle-mile nodes: a sea port and a rail route.

Node dict:
  id, name, type, risk (0-1 baseline), capacity (where applicable),
  location (informational)

Edge dict:
  source, target, relationship ("supplies" | "produces_for" | "stores_to"
  | "ships_via" | "serves"), weight (0-1 dependency weight)

`weight` means: what fraction of the TARGET node's input/throughput depends on
the SOURCE node. 0.9 = near-critical dependency, 0.3 = minor. The docstring of
`backend/propagation.py` explains exactly how it is used in propagation.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

NODES: list[dict[str, Any]] = [
    # ---------------------------------------------------------- tier 0: suppliers
    {"id": "sup_a", "name": "Supplier A (Shenzhen)", "type": "supplier", "risk": 0.20,
     "capacity": 40000, "location": "Shenzhen, CN"},
    {"id": "sup_b", "name": "Supplier B (Chennai)", "type": "supplier", "risk": 0.45,
     "capacity": 25000, "location": "Chennai, IN"},
    {"id": "sup_c", "name": "Supplier C (Monterrey)", "type": "supplier", "risk": 0.30,
     "capacity": 30000, "location": "Monterrey, MX"},
    {"id": "sup_d", "name": "Supplier D (Hamburg)", "type": "supplier", "risk": 0.15,
     "capacity": 20000, "location": "Hamburg, DE"},
    # ---------------------------------------------------------- tier 1: factories
    {"id": "fac_1", "name": "Factory 1 (Guadalajara)", "type": "factory", "risk": 0.10,
     "capacity": 60000, "location": "Guadalajara, MX"},
    {"id": "fac_2", "name": "Factory 2 (Monterrey)", "type": "factory", "risk": 0.12,
     "capacity": 45000, "location": "Monterrey, MX"},
    # --------------------------------------------------- middle mile (optional) --
    {"id": "port_lb", "name": "Port of Long Beach", "type": "port", "risk": 0.25,
     "capacity": 100000, "location": "Long Beach, US"},
    {"id": "route_rail", "name": "Transcontinental Rail", "type": "route", "risk": 0.18,
     "capacity": 80000, "location": "US interior"},
    # ---------------------------------------------------------- tier 2: warehouses
    {"id": "wh_1", "name": "Warehouse 1 (Dallas)", "type": "warehouse", "risk": 0.08,
     "capacity": 35000, "location": "Dallas, US"},
    {"id": "wh_2", "name": "Warehouse 2 (Chicago)", "type": "warehouse", "risk": 0.10,
     "capacity": 30000, "location": "Chicago, US"},
    {"id": "wh_3", "name": "Warehouse 3 (Atlanta)", "type": "warehouse", "risk": 0.07,
     "capacity": 25000, "location": "Atlanta, US"},
    # ---------------------------------------------------------- tier 3: customers
    {"id": "cus_1", "name": "Customer 1 (Retail East)", "type": "customer", "risk": 0.05,
     "location": "Newark, US"},
    {"id": "cus_2", "name": "Customer 2 (Retail Midwest)", "type": "customer", "risk": 0.05,
     "location": "Indianapolis, US"},
    {"id": "cus_3", "name": "Customer 3 (Retail South)", "type": "customer", "risk": 0.05,
     "location": "Orlando, US"},
]

EDGES: list[dict[str, Any]] = [
    # suppliers -> factories (relationship "supplies")
    {"source": "sup_a", "target": "fac_1", "relationship": "supplies", "weight": 0.80},
    {"source": "sup_b", "target": "fac_1", "relationship": "supplies", "weight": 0.55},
    {"source": "sup_c", "target": "fac_2", "relationship": "supplies", "weight": 0.85},
    {"source": "sup_d", "target": "fac_2", "relationship": "supplies", "weight": 0.40},
    # factories via port / rail (middle mile)
    {"source": "fac_1", "target": "port_lb", "relationship": "ships_via", "weight": 0.70},
    {"source": "fac_2", "target": "route_rail", "relationship": "ships_via", "weight": 0.60},
    # middle mile -> warehouses
    {"source": "port_lb", "target": "wh_1", "relationship": "stores_to", "weight": 0.75},
    {"source": "port_lb", "target": "wh_2", "relationship": "stores_to", "weight": 0.50},
    {"source": "route_rail", "target": "wh_2", "relationship": "stores_to", "weight": 0.65},
    {"source": "route_rail", "target": "wh_3", "relationship": "stores_to", "weight": 0.55},
    # factories -> warehouses (direct trucks)
    {"source": "fac_1", "target": "wh_3", "relationship": "stores_to", "weight": 0.45},
    # warehouses -> customers
    {"source": "wh_1", "target": "cus_1", "relationship": "serves", "weight": 0.90},
    {"source": "wh_2", "target": "cus_2", "relationship": "serves", "weight": 0.75},
    {"source": "wh_3", "target": "cus_3", "relationship": "serves", "weight": 0.80},
    {"source": "wh_3", "target": "cus_2", "relationship": "serves", "weight": 0.35},
    {"source": "wh_1", "target": "cus_2", "relationship": "serves", "weight": 0.30},
]

NODE_TYPES = ["supplier", "factory", "port", "route", "warehouse", "customer"]


def build_graph() -> nx.DiGraph:
    """Build the demo network as a NetworkX DiGraph with typed, weighted edges."""
    g = nx.DiGraph()
    for node in NODES:
        g.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in EDGES:
        g.add_edge(edge["source"], edge["target"], **{k: v for k, v in edge.items() if k not in ("source", "target")})
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Demo network must be a DAG for layered propagation")
    return g


def to_react_flow(g: nx.DiGraph | None = None) -> dict[str, Any]:
    """Serialize the graph to a React Flow friendly JSON structure.

    Layout: vertical layers (y = tier), horizontal spacing within a tier.
    Tier is derived from longest-path depth so ports/routes slot between
    factories and warehouses naturally.
    """
    g = g or build_graph()

    # longest-path layering from sources
    tier: dict[str, int] = {}
    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        tier[node] = 0 if not preds else max(tier[p] for p in preds) + 1

    # group nodes per tier for horizontal placement
    by_tier: dict[int, list[str]] = {}
    for node, t in tier.items():
        by_tier.setdefault(t, []).append(node)

    X_SPACING, Y_SPACING, WIDTH = 260, 180, 210
    nodes_out: list[dict[str, Any]] = []
    for t, members in sorted(by_tier.items()):
        for i, node_id in enumerate(sorted(members)):
            data = dict(g.nodes[node_id])
            nodes_out.append(
                {
                    "id": node_id,
                    "position": {"x": i * X_SPACING, "y": t * Y_SPACING},
                    "data": {
                        "label": data.get("name", node_id),
                        "type": data.get("type", "unknown"),
                        "risk": data.get("risk", 0.0),
                        "capacity": data.get("capacity"),
                        "location": data.get("location"),
                    },
                    "type": "custom",  # frontend maps type -> node styling
                }
            )

    edges_out = [
        {
            "id": f"{u}->{v}",
            "source": u,
            "target": v,
            "label": g.edges[u, v].get("relationship", ""),
            "data": {"weight": g.edges[u, v].get("weight", 1.0),
                     "relationship": g.edges[u, v].get("relationship", "")},
            "animated": False,
        }
        for u, v in g.edges
    ]

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "tiers": {str(t): sorted(m) for t, m in sorted(by_tier.items())},
        "width": max(len(m) for m in by_tier.values()) * X_SPACING,
        "height": len(by_tier) * Y_SPACING,
        "node_count": g.number_of_nodes(),
        "edge_count": g.number_of_edges(),
    }

"""SupplyChain Sentinel - Stage 3: explainable risk propagation.

Formula (kept deliberately simple - no GNN, one readable equation)
================================================================

For an event at node `s` with original risk `R_s` (0-1), every node `n`
reachable downstream of `s` gets a propagated risk:

    propagated(n) = R_s * product over path edges (weight_e * DECAY)

    DECAY = 0.7          # fixed per-hop decay
    MAX_HOPS = 4         # propagation stops after 4 edges
    epsilon-min: contributions below MIN_CONTRIBUTION (0.01) are pruned

Properties (why this is explainable):
  - Monotone: riskier source or stronger dependency -> more downstream risk.
  - Distance: each extra hop multiplies by DECAY, so risk fades geometrically.
    A node 2 hops away receives DECAY^2 = 0.49x of what a direct successor gets
    (times relative edge weights).
  - Dependency: a weak dependency (weight 0.3) passes only 30% of what a
    critical dependency (weight 1.0) would at the same distance.
  - Multi-path aggregation: if several paths lead to the same node, the node's
    propagated risk is the MAXIMUM single-path contribution. Max (rather than
    sum) keeps values bounded in [0, R_s] and makes the dominant path obvious;
    the `paths` output lists all contributing paths anyway.
  - Own baseline risk is reported separately (node.risk) and is NOT mixed into
    propagated_risk, so the API answers "how much of THIS event reaches you".

Outputs per affected node:
  propagated_risk: 0-1 portion of the source risk reaching that node
  propagation_path: the argmax (dominant) path as node ids
  hops: path length in edges
  contribution: propagated_risk * 100 (convenience, same number in %)
"""

from __future__ import annotations

from typing import Any

import networkx as nx

DECAY = 0.7          # per-hop decay factor
MAX_HOPS = 4         # do not propagate further than 4 edges
MIN_CONTRIBUTION = 0.01  # prune negligible paths


def propagate(
    source: str,
    risk_probability: float,
    g: nx.DiGraph | None = None,
) -> dict[str, Any]:
    """Propagate `risk_probability` from `source` through the DAG.

    Returns {"source", "original_risk", "affected_nodes": [...], "summary"}.
    Raises KeyError for unknown source nodes.
    """
    g = g or _default_graph()
    if source not in g:
        raise KeyError(f"Unknown node_id: {source!r}")
    if not 0.0 <= risk_probability <= 1.0:
        raise ValueError("risk_probability must be between 0 and 1")

    original = float(risk_probability)

    # best (max) contribution per node + the path that produced it
    best: dict[str, tuple[float, list[str], list[tuple[str, str]]]] = {}
    all_paths: dict[str, list[dict[str, Any]]] = {}

    # DFS over all simple downstream paths, capped at MAX_HOPS edges
    stack: list[tuple[str, list[str], list[tuple[str, str]], float]] = [
        (source, [source], [], 1.0)
    ]
    while stack:
        node, path, edges, factor = stack.pop()
        if node != source:
            contribution = original * factor
            prev_best = best.get(node, (0.0, [], []))
            if contribution > prev_best[0]:
                best[node] = (contribution, path, edges)
            all_paths.setdefault(node, []).append(
                {
                    "path": path,
                    "hops": len(edges),
                    "factor": round(factor, 6),
                    "propagated_risk": round(contribution, 6),
                }
            )
        if len(path) - 1 >= MAX_HOPS:
            continue
        for succ in g.successors(node):
            w = float(g.edges[node, succ].get("weight", 1.0))
            new_factor = factor * w * DECAY
            if original * new_factor < MIN_CONTRIBUTION:
                continue
            stack.append((succ, path + [succ], edges + [(node, succ)], new_factor))

    affected = []
    for node, (contribution, path, edges) in best.items():
        affected.append(
            {
                "node_id": node,
                "name": g.nodes[node].get("name", node),
                "type": g.nodes[node].get("type", "unknown"),
                "original_risk": round(float(g.nodes[node].get("risk", 0.0)), 4),
                "propagated_risk": round(contribution, 4),
                "contribution": round(contribution * 100, 2),
                "hops": len(edges),
                "propagation_path": path,
                "path_edges": [
                    {
                        "from": u,
                        "to": v,
                        "relationship": g.edges[u, v].get("relationship", ""),
                        "weight": float(g.edges[u, v].get("weight", 1.0)),
                    }
                    for u, v in edges
                ],
                "alternative_paths": sorted(
                    all_paths.get(node, []),
                    key=lambda p: p["propagated_risk"],
                    reverse=True,
                ),
            }
        )

    affected.sort(key=lambda a: a["propagated_risk"], reverse=True)

    return {
        "source": source,
        "original_risk": round(original, 4),
        "parameters": {"decay": DECAY, "max_hops": MAX_HOPS, "min_contribution": MIN_CONTRIBUTION},
        "affected_nodes": affected,
        "summary": {
            "nodes_affected": len(affected),
            "max_propagated_risk": affected[0]["propagated_risk"] if affected else 0.0,
            "max_hops_to_any_node": max((a["hops"] for a in affected), default=0),
            "customer_nodes_affected": sum(1 for a in affected if a["type"] == "customer"),
        },
    }


_DEFAULT_GRAPH = None


def _default_graph() -> nx.DiGraph:
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        from backend.graph import build_graph

        _DEFAULT_GRAPH = build_graph()
    return _DEFAULT_GRAPH

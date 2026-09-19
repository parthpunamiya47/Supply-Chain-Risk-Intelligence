"""SupplyChain Sentinel - Stage 4: rule-based mitigation recommendation engine.

Pure heuristics - no LLM, no learned component. Every recommendation is the
output of one of six explicit rules so the reason strings can always be traced
back to a threshold the user can read in this file.

Inputs (context)
----------------
- shipment features (optional): the 11 model features -> run ml.predict.predict_risk
- graph_event (optional): {"node_id", "risk_probability"} -> graph propagation
- network facts: which suppliers share factories (alternate-supplier signal),
  and each node's graph risk (0-1) when a graph event is active.

Rule set (thresholds on the 0-100 risk scale)
---------------------------------------------
R1  SWITCH_SUPPLIER      supplier risk > 70 AND an alternate supplier exists
R2  INCREASE_SAFETY_STOCK supplier risk > 70 AND inventory low (inventory < 0.5 x demand)
R3  ALTERNATE_ROUTE      transportation risk > 70 (0-100 scale)
R4  EXPEDITE_REPLENISHMENT inventory low AND demand high (demand > 5000 units)
R5  BUFFER_LEAD_TIME     average lead time > 21 days (long exposure window)
R6  MONITOR              40 <= overall risk < 70 (medium band) and no HIGH rule fired
R0  NO_ACTION            overall risk < 40 (LOW): explicitly do nothing

Risk-reduction estimates are HONEST HEURISTICS tied to the generator's
coefficients (ml/generate_dataset.py), not measured causal effects:
  - a matched feature-rule gets a share of that feature's global permutation
    importance (models/feature_importances.json), scaled by how extreme the
    value is;
  - mitigations that cannot act on the current drivers (e.g. rerouting when
    transportation risk is already low) honestly report ~0.

Cost impacts are coarse qualitative buckets (NONE/LOW/MEDIUM/HIGH) with an
explanatory note - we do not pretend to know real dollar figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Tunables (documented so stakeholders can argue with them)
# --------------------------------------------------------------------------- #

HIGH_RISK = 70.0          # supplier/transport risk above this = act now
MEDIUM_RISK = 40.0        # monitoring band starts here
LOW_INVENTORY_RATIO = 0.5  # inventory < 0.5 x demand = low
HIGH_DEMAND = 5000         # units
LONG_LEAD_TIME = 21.0      # days
MONITOR_REDUCTION = 5.0    # a monitoring program moves the needle a little

# Estimated cost buckets with short explanations (qualitative on purpose).
COST_NOTES = {
    "NONE": "No material cost - process change only.",
    "LOW": "Minor cost - scheduling/monitoring effort, no capital outlay.",
    "MEDIUM": "Moderate cost - e.g. faster freight, extra handling, buffer stock carrying cost.",
    "HIGH": "Significant cost - re-sourcing, contractual changes, new logistics contracts.",
}


@dataclass
class Recommendation:
    rule_id: str
    action: str
    reason: str
    risk_reduction: float      # percentage points, 0-100 scale
    priority: str              # HIGH / MEDIUM / LOW / NONE
    cost_estimate: str         # NONE / LOW / MEDIUM / HIGH
    cost_note: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "action": self.action,
            "reason": self.reason,
            "risk_reduction": round(self.risk_reduction, 1),
            "priority": self.priority,
            "cost_estimate": self.cost_estimate,
            "cost_note": self.cost_note,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# Network facts used by R1 (alternate suppliers) and graph-driven escalation
# --------------------------------------------------------------------------- #


def _alternate_suppliers(node_id: str | None) -> list[dict[str, Any]]:
    """Suppliers that share a factory with `node_id` (i.e. proven alternates),
    or all other suppliers when `node_id` is None (shipment-context case)."""
    from backend.graph import EDGES, NODES

    if node_id is None:
        # shipment context: every supplier in the network is a candidate
        return [
            {"supplier_id": n["id"], "name": n["name"],
             "baseline_risk": n["risk"], "shared_factory": None}
            for n in NODES if n["type"] == "supplier"
        ]

    shared_factories = {
        e["target"] for e in EDGES if e["source"] == node_id and e["relationship"] == "supplies"
    }
    alternates: list[dict[str, Any]] = []
    for e in EDGES:
        if e["relationship"] != "supplies" or e["target"] not in shared_factories:
            continue
        if e["source"] == node_id:
            continue
        alt = next((n for n in NODES if n["id"] == e["source"]), None)
        if alt:
            alternates.append(
                {"supplier_id": alt["id"], "name": alt["name"], "baseline_risk": alt["risk"],
                 "shared_factory": e["target"]}
            )
    return alternates


def _factor_contribution(factors: list[dict[str, Any]], feature: str) -> float:
    """Signed contribution of one feature from the explainability output."""
    return next((f["contribution"] for f in factors if f["feature"] == feature), 0.0)


def _share(importance_share: float, deviation_iqr: float) -> float:
    """Heuristic risk-reduction share: importance x capped normalised deviation."""
    capped = min(max(deviation_iqr, 0.0), 3.0)
    return min(100.0, importance_share * 100.0 * (0.35 + 0.65 * capped / 3.0))


# --------------------------------------------------------------------------- #
# Rule implementations
# --------------------------------------------------------------------------- #


def _rule_switch_supplier(ctx: dict[str, Any]) -> Recommendation | None:
    supplier_risk = ctx["supplier_risk"]
    if supplier_risk <= HIGH_RISK or not ctx["has_alternates"]:
        return None
    alts = ctx["alternates"]
    best = min(alts, key=lambda a: a["baseline_risk"])
    gain = _share(0.28, 2.5)  # supplier reliability/delays carry ~28% of importance
    names = ", ".join(a["name"] for a in alts[:2])
    return Recommendation(
        rule_id="R1",
        action=f"Switch to alternate supplier ({names})",
        reason=(f"Current supplier risk {supplier_risk:.0f}/100 is above the {HIGH_RISK:.0f} "
                f"action threshold; {len(alts)} qualified alternate supplier(s) already feed "
                f"the same factory (best: {best['name']}, baseline risk {best['baseline_risk']:.2f})."),
        risk_reduction=gain,
        priority="HIGH" if supplier_risk >= 85 else "MEDIUM",
        cost_estimate="HIGH",
        cost_note=COST_NOTES["HIGH"],
        details={"alternate_suppliers": alts, "trigger_value": round(supplier_risk, 1),
                 "threshold": HIGH_RISK},
    )


def _rule_safety_stock(ctx: dict[str, Any]) -> Recommendation | None:
    supplier_risk = ctx["supplier_risk"]
    inv_ratio = ctx["inventory_ratio"]
    if supplier_risk <= HIGH_RISK or inv_ratio >= LOW_INVENTORY_RATIO:
        return None
    # more urgent when stock is thinner
    gain = _share(0.14, 2.0 if inv_ratio < 0.25 else 1.2)
    return Recommendation(
        rule_id="R2",
        action="Increase safety stock for the affected SKU",
        reason=(f"Supplier risk {supplier_risk:.0f}/100 with inventory at only "
                f"{inv_ratio:.2f}x demand (below the {LOW_INVENTORY_RATIO}x floor) - "
                "buffer stock would cover replenishment gaps during a disruption."),
        risk_reduction=gain,
        priority="HIGH",
        cost_estimate="MEDIUM",
        cost_note=COST_NOTES["MEDIUM"],
        details={"inventory_ratio": round(inv_ratio, 2), "trigger_value": round(supplier_risk, 1),
                 "threshold": HIGH_RISK},
    )


def _rule_alternate_route(ctx: dict[str, Any]) -> Recommendation | None:
    tr = ctx["transport_risk"]
    if tr <= HIGH_RISK:
        return None
    gain = _share(0.19, 2.5)  # transportation_risk ~19% of model importance
    return Recommendation(
        rule_id="R3",
        action="Reroute via alternate lane/port",
        reason=(f"Transportation risk {tr:.0f}/100 exceeds the {HIGH_RISK:.0f} threshold; "
                "the current lane is the single largest model driver for this shipment."),
        risk_reduction=gain,
        priority="HIGH" if tr >= 85 else "MEDIUM",
        cost_estimate="MEDIUM",
        cost_note=COST_NOTES["MEDIUM"],
        details={"trigger_value": round(tr, 1), "threshold": HIGH_RISK},
    )


def _rule_expedite(ctx: dict[str, Any]) -> Recommendation | None:
    inv_ratio = ctx["inventory_ratio"]
    demand = ctx["demand"]
    if inv_ratio >= LOW_INVENTORY_RATIO or demand <= HIGH_DEMAND:
        return None
    gain = _share(0.10, 1.5)
    return Recommendation(
        rule_id="R4",
        action="Expedite replenishment order",
        reason=(f"Inventory covers {inv_ratio:.2f}x demand while demand is {demand:,.0f} units "
                f"(above the {HIGH_DEMAND:,} line) - a stockout would hurt even without a "
                "disruption."),
        risk_reduction=gain,
        priority="MEDIUM",
        cost_estimate="MEDIUM",
        cost_note=COST_NOTES["MEDIUM"],
        details={"inventory_ratio": round(inv_ratio, 2), "demand": demand},
    )


def _rule_buffer_lead_time(ctx: dict[str, Any]) -> Recommendation | None:
    lead = ctx["lead_time"]
    if lead <= LONG_LEAD_TIME:
        return None
    gain = _share(0.08, 1.5)
    return Recommendation(
        rule_id="R5",
        action="Buffer the committed delivery date",
        reason=(f"Average lead time is {lead:.0f} days (above the {LONG_LEAD_TIME:.0f}-day "
                "threshold), extending the exposure window; quote a padded date to customers."),
        risk_reduction=gain,
        priority="LOW",
        cost_estimate="LOW",
        cost_note=COST_NOTES["LOW"],
        details={"trigger_value": round(lead, 1), "threshold": LONG_LEAD_TIME},
    )


def _rule_monitor(ctx: dict[str, Any]) -> Recommendation | None:
    overall = ctx["overall_risk"]
    if ctx["high_fired"] or overall < MEDIUM_RISK:
        return None
    return Recommendation(
        rule_id="R6",
        action="Enroll shipment in enhanced monitoring",
        reason=(f"Overall risk {overall:.1f}/100 sits in the MEDIUM band ({MEDIUM_RISK:.0f}-"
                f"{HIGH_RISK:.0f}) with no single HIGH-severity driver - monitoring is the "
                "cost-efficient response."),
        risk_reduction=MONITOR_REDUCTION,
        priority="MEDIUM",
        cost_estimate="LOW",
        cost_note=COST_NOTES["LOW"],
        details={"trigger_value": round(overall, 1), "band": [MEDIUM_RISK, HIGH_RISK]},
    )


def _rule_no_action(ctx: dict[str, Any]) -> Recommendation | None:
    if ctx["overall_risk"] >= MEDIUM_RISK or ctx["high_fired"]:
        return None
    return Recommendation(
        rule_id="R0",
        action="No action required",
        reason=(f"Overall risk {ctx['overall_risk']:.1f}/100 is in the LOW band (< "
                f"{MEDIUM_RISK:.0f}); re-evaluate if supplier or route conditions change."),
        risk_reduction=0.0,
        priority="NONE",
        cost_estimate="NONE",
        cost_note=COST_NOTES["NONE"],
        details={"trigger_value": round(ctx["overall_risk"], 1)},
    )


HIGH_RULES = ("R1", "R2", "R3")  # rules that indicate a HIGH-severity driver
CORE_RULES = [_rule_switch_supplier, _rule_safety_stock, _rule_alternate_route,
              _rule_expedite, _rule_buffer_lead_time]
MONITOR_RULE = _rule_monitor
NO_ACTION_RULE = _rule_no_action  # only fires when NOTHING else did

PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}


def generate_recommendations(ctx: dict[str, Any]) -> dict[str, Any]:
    """Run every rule over the prepared context and assemble the response.

    ctx keys: overall_risk, risk_level, supplier_risk, transport_risk,
    inventory_ratio, demand, lead_time, factors, has_alternates, alternates,
    high_fired (managed internally), source, node_id
    """
    recs: list[Recommendation] = []
    fired: list[str] = []

    # pass 1: driver-specific rules
    for rule in CORE_RULES:
        rec = rule(ctx)
        if rec is not None:
            recs.append(rec)
            fired.append(rec.rule_id)

    # pass 2: monitoring fallback, aware of pass-1 outcomes
    ctx["high_fired"] = any(r in fired for r in HIGH_RULES)
    rec = MONITOR_RULE(ctx)
    if rec is not None:
        recs.append(rec)
        fired.append(rec.rule_id)

    # pass 3: explicit no-action ONLY when no other rule produced advice
    if not recs:
        rec = NO_ACTION_RULE(ctx)
        if rec is not None:
            recs.append(rec)
            fired.append(rec.rule_id)

    recs.sort(key=lambda r: (PRIORITY_ORDER[r.priority], -r.risk_reduction))

    total_potential = min(100.0, sum(r.risk_reduction for r in recs if r.rule_id != "R0"))
    return {
        "context": {
            "source": ctx["source"],
            "node_id": ctx.get("node_id"),
            "risk_level": ctx["risk_level"],
            "overall_risk": round(ctx["overall_risk"], 1),
            "supplier_risk": round(ctx["supplier_risk"], 1),
            "transport_risk": round(ctx["transport_risk"], 1),
            "inventory_ratio": round(ctx["inventory_ratio"], 2),
            "demand": ctx["demand"],
            "lead_time_days": round(ctx["lead_time"], 1),
            "top_risk_factors": [f["phrase"] for f in ctx["factors"][:3]
                                 if f["contribution"] > 0],
        },
        "recommendations": [r.to_dict() for r in recs],
        "summary": {
            "rules_fired": fired,
            "count": len(recs),
            "highest_priority": recs[0].priority if recs else "NONE",
            "total_potential_risk_reduction": round(total_potential, 1),
        },
    }

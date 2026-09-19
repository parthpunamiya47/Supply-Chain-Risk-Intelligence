"""SupplyChain Sentinel - Stage 5: live verification of /api/simulation.

Starts uvicorn on 127.0.0.1:8012, runs the 4 demo scenarios (+ one custom
variant and error cases), verifies the real graph stays unchanged via a
before/after /api/graph fingerprint, then shuts the server down.

    .venv/Scripts/python scripts/verify_stage5.py
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8012
BASE = f"http://127.0.0.1:{PORT}"


def call(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def graph_fingerprint() -> dict[str, float]:
    _, g = call("GET", "/api/graph")
    return {n["id"]: n["data"]["risk"] for n in g["nodes"]}


def show(name: str, body: dict) -> None:
    status, d = call("POST", "/api/simulation", body)
    print(f"\n=== {name} (HTTP {status}) ===")
    if status != 200:
        print("  ", json.dumps(d)[:300])
        return
    print(f"  target={d['target_node']} event_risk={d['event_risk']}")
    print(f"  {d['target_node']}: {d['before_risk'][d['target_node']]} -> {d['after_risk'][d['target_node']]}")
    for a in d["affected_nodes"][:6]:
        print(f"    {a['node_id']:<10} {a['before_risk']:.2f} -> {a['after_risk']:.4f} "
              f"(delta {a['delta']:+.4f}, {a['hops']} hops)")
    if len(d["affected_nodes"]) > 6:
        print(f"    ... and {len(d['affected_nodes']) - 6} more")
    print("  sample path:", " -> ".join(d["propagation_paths"][0]["dominant_path"]))
    for r in d["recommendations"]:
        print(f"    [{r['rule_id']}] {r['priority']:<6} {r['action']}")
    s = d["summary"]
    print(f"  summary: affected={s['nodes_affected']} customers={s['customers_affected']} "
          f"max_delta={s['max_delta']} rules={s['rules_fired']} graph_modified={s['graph_modified']}")


def main() -> None:
    server = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "uvicorn",
         "backend.main:app", "--port", str(PORT), "--host", "127.0.0.1"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                if call("GET", "/api/health")[0] == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server not healthy")

        before_fp = graph_fingerprint()

        _, catalog = call("GET", "/api/simulation/scenarios")
        print("=== scenario catalog ===")
        for s in catalog["scenarios"]:
            print(f"  {s['scenario']:<28} target={s['default_target']:<8} "
                  f"event_risk={s['event_risk']}  {s['description']}")

        show("Scenario 1: supplier_failure (default sup_c @ 0.95)", {"scenario": "supplier_failure"})
        show("Scenario 2: port_closure (port_lb @ 0.95)", {"scenario": "port_closure"})
        show("Scenario 3: transportation_disruption (route_rail @ 0.90)",
             {"scenario": "transportation_disruption"})
        show("Scenario 4: demand_spike (wh_2 @ 0.55, demand x2)",
             {"scenario": "demand_spike", "demand_multiplier": 2.0})
        show("Custom: supplier_failure on sup_a @ 0.80",
             {"scenario": "supplier_failure", "target_node": "sup_a", "event_risk": 0.8})

        print("\n=== error handling ===")
        s, _ = call("POST", "/api/simulation", {"scenario": "zombie_apocalypse"})
        print("unknown scenario        ->", s)
        s, _ = call("POST", "/api/simulation", {"scenario": "supplier_failure",
                                                "target_node": "supplier_3"})
        print("unknown target node     ->", s)
        s, _ = call("POST", "/api/simulation", {"scenario": "port_closure", "target_node": "sup_a"})
        print("wrong node type         ->", s)
        s, _ = call("POST", "/api/simulation", {"scenario": "supplier_failure", "event_risk": 1.5})
        print("invalid event_risk      ->", s)

        after_fp = graph_fingerprint()
        print("\n=== graph integrity ===")
        print("real graph unchanged:", before_fp == after_fp)
        print("\nAll Stage 5 live checks completed.")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()

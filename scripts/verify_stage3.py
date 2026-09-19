"""SupplyChain Sentinel - Stage 3: live verification of graph endpoints.

Starts uvicorn on 127.0.0.1:8010, hits GET /api/graph and
POST /api/graph/propagate-risk for 4 scenarios + 2 error cases, prints a
compact report, then shuts the server down. Run from the project root:

    .venv/Scripts/python scripts/verify_stage3.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PORT = 8010
BASE = f"http://127.0.0.1:{PORT}"


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def show_scenario(name: str, node: str, risk: float) -> None:
    status, d = call("POST", "/api/graph/propagate-risk", {"node_id": node, "risk_probability": risk})
    print(f"\n=== {name}: {node} @ {risk} (HTTP {status}) ===")
    for a in d["affected_nodes"]:
        path_str = " -> ".join(a["propagation_path"])
        print(f"  {a['node_id']:<10} {a['type']:<10} hops={a['hops']} "
              f"propagated={a['propagated_risk']:<7} via {path_str}")
    print("  summary:", d["summary"])


def main() -> None:
    server = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "uvicorn",
         "backend.main:app", "--port", str(PORT), "--host", "127.0.0.1"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # wait for readiness
        for _ in range(60):
            try:
                status, _ = call("GET", "/api/health")
                if status == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not become healthy in time")

        status, g = call("GET", "/api/graph")
        print(f"=== GET /api/graph (HTTP {status}) ===")
        print("node_count:", g["node_count"], "| edge_count:", g["edge_count"])
        print("tiers:", json.dumps(g["tiers"]))
        print("sample node:", json.dumps(g["nodes"][0]))
        print("sample edge:", json.dumps(g["edges"][0]))

        show_scenario("Scenario 1 - critical supplier", "sup_c", 0.85)
        show_scenario("Scenario 2 - weak supplier event", "sup_d", 0.15)
        show_scenario("Scenario 3 - middle-mile port", "port_lb", 0.90)
        show_scenario("Bonus - branching factory", "fac_1", 0.60)

        print("\n=== error handling ===")
        s, _ = call("POST", "/api/graph/propagate-risk", {"node_id": "sup_z", "risk_probability": 0.5})
        print("unknown node ->", s)
        s, _ = call("POST", "/api/graph/propagate-risk", {"node_id": "sup_a", "risk_probability": 1.5})
        print("risk 1.5     ->", s)
        s, d = call("GET", "/api/health")
        print("health       ->", s, d["status"])
        print("\nAll Stage 3 live checks completed.")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()

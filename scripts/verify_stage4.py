"""SupplyChain Sentinel - Stage 4: live verification of /api/recommendations.

Starts uvicorn on 127.0.0.1:8011, runs six scenarios across both modes
(shipment features via the ML model, graph events via propagation), prints a
compact report and shuts the server down. Run from the project root:

    .venv/Scripts/python scripts/verify_stage4.py
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


BASE_FEATURES = {
    "supplier_id": "SUP-001",
    "supplier_reliability": 90.0,
    "previous_delays": 1,
    "average_lead_time": 10.0,
    "demand": 3000,
    "inventory_level": 5000,
    "weather_risk": 2.0,
    "transportation_risk": 2.0,
    "distance": 500,
    "supplier_capacity": 20000,
    "shipment_size": 2500,
    "historical_disruptions": 0,
}


def show(name: str, body: dict) -> None:
    status, d = call("POST", "/api/recommendations", body)
    print(f"\n=== {name} (HTTP {status}) ===")
    if status != 200:
        print("  ", json.dumps(d)[:300])
        return
    c = d["context"]
    print(f"  ctx: source={c['source']} node={c.get('node_id')} level={c['risk_level']} "
          f"overall={c['overall_risk']} supplier={c['supplier_risk']} "
          f"transport={c['transport_risk']} inv_ratio={c['inventory_ratio']}")
    for r in d["recommendations"]:
        print(f"  [{r['rule_id']}] {r['priority']:<6} -{r['risk_reduction']:>5.1f}%  "
              f"cost={r['cost_estimate']:<6} {r['action']}")
    s = d["summary"]
    print(f"  summary: rules={s['rules_fired']} total_reduction={s['total_potential_risk_reduction']}%")


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

        show("1. LOW-risk shipment (healthy supplier)", {"features": BASE_FEATURES})

        worst = dict(BASE_FEATURES,
                     supplier_reliability=50.0, previous_delays=13, average_lead_time=30.0,
                     demand=12000, inventory_level=1500, weather_risk=9.0,
                     transportation_risk=9.0, distance=8000, supplier_capacity=9000,
                     shipment_size=10000, historical_disruptions=6)
        show("2. HIGH-risk shipment (bad supplier, thin stock, risky lane)",
             {"features": worst})

        mid = dict(BASE_FEATURES, supplier_reliability=74.0, previous_delays=6,
                   average_lead_time=14.0, weather_risk=5.5, transportation_risk=5.0,
                   distance=2000, historical_disruptions=2)
        show("3. MEDIUM-risk shipment (monitoring expected)", {"features": mid})

        show("4. Graph event: sup_c @ 0.85 (supplier outage)",
             {"node_id": "sup_c", "risk_probability": 0.85})
        show("5. Graph event: port_lb @ 0.95 (port congestion)",
             {"node_id": "port_lb", "risk_probability": 0.95})
        show("6. Graph event: sup_d @ 0.15 (weak supplier signal)",
             {"node_id": "sup_d", "risk_probability": 0.15})

        print("\n=== error handling ===")
        s, _ = call("POST", "/api/recommendations", {"node_id": "sup_z", "risk_probability": 0.5})
        print("unknown node          ->", s)
        s, _ = call("POST", "/api/recommendations",
                    {"features": BASE_FEATURES, "node_id": "sup_a", "risk_probability": 0.5})
        print("mixed-mode payload    ->", s)
        s, _ = call("POST", "/api/recommendations",
                    {"features": dict(BASE_FEATURES, weather_risk=99)})
        print("invalid feature range ->", s)
        print("\nAll Stage 4 live checks completed.")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()

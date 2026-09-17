"""
capture_demo.py
Freezes real output from your running backend into a single JSON file,
so the demo page can be hosted as static files with no server.

Everything it saves is genuine output from your own system - the same
routes, scores and hotspots the live API returns. The only thing lost
is the ability to route between arbitrary new points.

Run with the backend up:
    python capture_demo.py

Produces demo/demo-data.json
"""

import json
import os

import requests

API = "http://127.0.0.1:8000"
OUT_DIR = "demo"

# Pairs worth showing. Each should demonstrate something different.
SCENARIOS = [
    {
        "id": "powai-dadar",
        "label": "Powai \u2192 Dadar",
        "note": "Crosses the Mithi corridor. The safest route avoids a hotspot "
                "for under two minutes of extra travel.",
        "origin": {"lat": 19.1200, "lon": 72.9050, "name": "Powai"},
        "dest": {"lat": 19.0176, "lon": 72.8434, "name": "Dadar"},
    },
    {
        "id": "hindmata-milan",
        "label": "Hindmata \u2192 Milan Subway",
        "note": "Both endpoints are chronic flooding locations and the "
                "corridor between them is uniformly flat, so the routes "
                "score alike \u2014 the system says so rather than inventing "
                "a benefit.",
        "origin": {"lat": 19.0060, "lon": 72.8360, "name": "Hindmata"},
        "dest": {"lat": 19.0776, "lon": 72.8420, "name": "Milan Subway"},
    },
    {
        "id": "andheri-bkc",
        "label": "Andheri \u2192 BKC",
        "note": "A common commute crossing the Mithi.",
        "origin": {"lat": 19.1190, "lon": 72.8470, "name": "Andheri Subway"},
        "dest": {"lat": 19.0640, "lon": 72.8620, "name": "Bandra-Kurla Complex"},
    },
]

RAIN_LEVELS = ["light", "moderate", "heavy", "extreme"]

# Viewport for the risk overlay - central Mumbai, tight enough to keep
# the payload reasonable.
RISK_BBOX = {
    "min_lat": 19.00, "min_lon": 72.82,
    "max_lat": 19.14, "max_lon": 72.92,
}


def get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def post(path, body):
    r = requests.post(f"{API}{path}", json=body, timeout=180)
    r.raise_for_status()
    return r.json()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[capture] Checking backend...")
    health = get("/health")
    print(f"[capture]   {health['nodes']} nodes, {health['edges']} edges")

    conditions = get("/current-conditions")

    data = {
        "generated_note": "Captured from the live FloodReroute API. All "
                          "routes, scores and hotspots are real system "
                          "output.",
        "graph": health,
        "conditions": conditions,
        "risk_bbox": RISK_BBOX,
        "scenarios": [],
        "risk_layers": {},
    }

    # Routes: every scenario at every rainfall level, so the demo's
    # rainfall control does something real rather than being decorative.
    for sc in SCENARIOS:
        entry = {k: sc[k] for k in ("id", "label", "note", "origin", "dest")}
        entry["results"] = {}
        for level in RAIN_LEVELS:
            print(f"[capture] {sc['id']} @ {level} ...")
            entry["results"][level] = post("/recommend", {
                "origin_lat": sc["origin"]["lat"],
                "origin_lon": sc["origin"]["lon"],
                "dest_lat": sc["dest"]["lat"],
                "dest_lon": sc["dest"]["lon"],
                "rainfall_level": level,
            })
        data["scenarios"].append(entry)

    # Risk overlay, one per rainfall level.
    for level in RAIN_LEVELS:
        print(f"[capture] risk overlay @ {level} ...")
        fc = get("/risk", rainfall_level=level, **RISK_BBOX)
        # Trim to keep the file servable - keep every 3rd segment. The
        # overlay is context, not the subject, and the full set is ~30k
        # features per level.
        fc["features"] = fc["features"][::3]
        data["risk_layers"][level] = fc
        print(f"[capture]   {len(fc['features'])} segments kept")

    path = os.path.join(OUT_DIR, "demo-data.json")
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    mb = os.path.getsize(path) / 1e6
    print(f"\n[capture] Wrote {path}  ({mb:.1f} MB)")
    if mb > 25:
        print("[capture] That is large for a static host. Increase the "
              "[::3] stride in the risk trim, or narrow RISK_BBOX.")
    print("[capture] Now open demo/index.html, or deploy the demo/ folder.")


if __name__ == "__main__":
    main()

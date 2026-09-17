"""
main.py
FastAPI backend for FloodReroute.

Endpoints
---------
  GET  /health                 liveness probe
  GET  /current-conditions     live rainfall + which factors are active
  GET  /risk                   risk-coloured road segments for a viewport
  GET  /geocode/suggest        type-ahead place search
  POST /geocode                address -> lat/lon
  POST /routes                 danger, fastest and safest in one call
  POST /recommend              routes + the recommendation string

Rainfall is a REQUEST parameter, not server state, so the frontend's
Light/Moderate/Heavy/Extreme selector recalculates instantly without
rescoring the graph.
"""

import json
import os
from typing import Optional, List

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph_utils import get_graph
from geocoding import search as geo_search, geocode_one
from risk import (
    compute_static_risk, edge_risk,
    risk_score_100, risk_band, RAINFALL_PRESETS,
)
from router import find_route, nearest_node, ROUTE_PROFILES, _road_name

app = FastAPI(title="FloodReroute API", version="0.3.0")

# Lock this down before shipping. "*" would let any site drive your
# server's CPU.
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

CITY_CENTER = (19.076, 72.877)
FALLBACK_RAINFALL_MM_HR = 10.0


# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------

def get_live_rainfall(lat: float, lon: float) -> float:
    """
    Current precipitation (mm/hr). Never raises - a weather API hiccup
    should not take down routing, which was the behaviour before.
    """
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "current": "precipitation", "timezone": "auto"},
            timeout=8,
        )
        resp.raise_for_status()
        return float(resp.json()["current"]["precipitation"])
    except Exception as e:
        print(f"[main] Rainfall fetch failed ({e}); "
              f"using {FALLBACK_RAINFALL_MM_HR} mm/hr.")
        return FALLBACK_RAINFALL_MM_HR


def _load_node_table(path: str, label: str) -> Optional[dict]:
    """Loads a {node_id: 0-1} JSON table, coercing keys to int."""
    if not os.path.exists(path):
        print(f"[main] No {label} data at {path} - factor disabled.")
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        table = {}
        for k, v in raw.items():
            try:
                table[int(k)] = float(v)
            except (ValueError, TypeError):
                table[k] = float(v)
        print(f"[main] Loaded {label}: {len(table)} nodes.")
        return table
    except Exception as e:
        print(f"[main] Failed to load {label}: {e}")
        return None


print("[main] Loading road graph...")
G = get_graph(mode="city")

FLOOD_HISTORY = _load_node_table("mumbai_sar_flood_history.json",
                                 "SAR flood history")
DRAINAGE = _load_node_table("mumbai_drainage.json", "drainage")
WATERLOGGING = _load_node_table("mumbai_waterlogging.json", "waterlogging")

print("[main] Computing static risk (once)...")
G = compute_static_risk(
    G,
    flood_history=FLOOD_HISTORY,
    drainage=DRAINAGE,
    waterlogging=WATERLOGGING,
)

LIVE_RAINFALL = get_live_rainfall(*CITY_CENTER)
print(f"[main] Live rainfall: {LIVE_RAINFALL} mm/hr")

_lats = [d["y"] for _, d in G.nodes(data=True)]
_lons = [d["x"] for _, d in G.nodes(data=True)]
CITY_VIEWBOX = (min(_lons), min(_lats), max(_lons), max(_lats))

ACTIVE_FACTORS = {
    "elevation": True,
    "historical": FLOOD_HISTORY is not None,
    "drainage": DRAINAGE is not None,
    "waterlogging": WATERLOGGING is not None,
    "rainfall": True,
}


def resolve_rainfall(level: Optional[str], mm_hr: Optional[float]) -> float:
    """Explicit mm/hr wins, then a named preset, then live weather."""
    if mm_hr is not None:
        return max(0.0, float(mm_hr))
    if level:
        key = level.strip().lower()
        if key not in RAINFALL_PRESETS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown rainfall level {level!r}. "
                       f"Use one of {sorted(RAINFALL_PRESETS)}.",
            )
        return RAINFALL_PRESETS[key]
    return LIVE_RAINFALL


# --------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------

class Stop(BaseModel):
    lat: float
    lon: float


class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    rainfall_level: Optional[str] = Field(
        None, description="light | moderate | heavy | extreme")
    rainfall_mm_hr: Optional[float] = None
    stops: List[Stop] = Field(default_factory=list)


class GeocodeRequest(BaseModel):
    address: str


# --------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges()}


@app.get("/current-conditions")
def current_conditions():
    """Drives the weather panel and tells the UI which factors are live."""
    return {
        "rainfall_mm_hr": LIVE_RAINFALL,
        "rainfall_presets": RAINFALL_PRESETS,
        "active_factors": ACTIVE_FACTORS,
    }


@app.post("/geocode")
def geocode(req: GeocodeRequest):
    """Best single match for a typed address or place name."""
    place = geocode_one(req.address)
    if place is None:
        raise HTTPException(
            404,
            detail=f'No routable place found for "{req.address}". '
                   f'Try adding the area, e.g. "{req.address}, Andheri".',
        )
    return place.as_dict()


@app.get("/geocode/suggest")
def geocode_suggest(q: str = Query(..., min_length=2), limit: int = 6):
    """
    Type-ahead candidates for the address box.

    Returns a list rather than one answer on purpose: a society name
    like "Sai Krupa CHS" matches many buildings across the metro, and
    silently picking one is how people get routed to the wrong address.
    """
    return {"results": [p.as_dict() for p in
                        geo_search(q, limit=limit, fast=True)]}


@app.get("/risk")
def risk_geojson(
    min_lat: float = Query(...), min_lon: float = Query(...),
    max_lat: float = Query(...), max_lon: float = Query(...),
    rainfall_level: Optional[str] = None,
    rainfall_mm_hr: Optional[float] = None,
):
    """
    Risk-coloured segments for the current viewport.

    The bounding box is REQUIRED - omitting it would walk all ~250k
    edges and serialise them, enough to hang both ends.
    """
    rainfall = resolve_rainfall(rainfall_level, rainfall_mm_hr)

    features = []
    seen = set()
    for u, v, data in G.edges(data=True):
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue

        y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
        y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
        if not ((min_lat <= y1 <= max_lat and min_lon <= x1 <= max_lon) or
                (min_lat <= y2 <= max_lat and min_lon <= x2 <= max_lon)):
            continue
        seen.add(key)

        r = edge_risk(data, rainfall)
        score = risk_score_100(r)
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[x1, y1], [x2, y2]]},
            "properties": {
                "risk_score": score,
                "risk_band": risk_band(score),
                "road_name": _road_name(data),
            },
        })

    return {"type": "FeatureCollection", "features": features,
            "rainfall_mm_hr": rainfall}


def _route_feature(result, profile):
    return {
        "profile": profile,
        # The danger route exists to be shown, never followed.
        "avoid": profile == "danger",
        "distance_km": result["distance_km"],
        "eta_min": result["travel_time_min"],
        "risk_score": result["risk_score"],
        "risk_band": result["risk_band"],
        "hotspot_count": result["hotspot_count"],
        "hotspots": result["hotspots"][:10],
        "geojson": {
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": result["coordinates"]},
            "properties": {"profile": profile,
                           "risk_score": result["risk_score"]},
        },
    }


def _compute_all_routes(req: RouteRequest):
    rainfall = resolve_rainfall(req.rainfall_level, req.rainfall_mm_hr)
    origin = nearest_node(G, req.origin_lat, req.origin_lon)
    dest = nearest_node(G, req.dest_lat, req.dest_lon)

    routes = {}
    for profile, alpha in ROUTE_PROFILES.items():
        try:
            res = find_route(G, origin, dest, alpha=alpha,
                             rainfall_mm_hr=rainfall)
        except ValueError as e:
            raise HTTPException(404, detail=str(e))
        routes[profile] = _route_feature(res, profile)

    # "Estimated Delay" in the UI is measured against the fastest route.
    baseline = routes["fastest"]["eta_min"]
    for r in routes.values():
        r["delay_min"] = round(max(0.0, r["eta_min"] - baseline), 1)
        # Safety score is the plain inverse of risk - easier to read as
        # "92% safe" than "8/100 risk" on a route card.
        r["safety_score"] = 100 - r["risk_score"]

    # How many hotspots the safest route dodges that the danger route
    # runs straight through. This is the number that makes the
    # comparison concrete rather than abstract.
    danger_spots = {h["road_name"] for h in routes["danger"]["hotspots"]}
    safest_spots = {h["road_name"] for h in routes["safest"]["hotspots"]}
    routes["safest"]["hotspots_avoided"] = len(danger_spots - safest_spots)
    routes["fastest"]["hotspots_avoided"] = len(
        danger_spots - {h["road_name"] for h in routes["fastest"]["hotspots"]}
    )
    routes["danger"]["hotspots_avoided"] = 0

    return routes, rainfall


@app.post("/routes")
def routes(req: RouteRequest):
    """
    All three route cards in one call: danger, fastest, safest.

    Multi-stop is accepted in the schema but not implemented - left
    explicit rather than silently ignored.
    """
    if req.stops:
        raise HTTPException(501, detail="Multi-stop routing not implemented.")
    routes_out, rainfall = _compute_all_routes(req)
    return {"routes": routes_out, "rainfall_mm_hr": rainfall}


@app.post("/recommend")
def recommend(req: RouteRequest):
    """
    Routes plus the recommendation logic:
      risk < 30         -> fastest
      risk >= 30        -> safest
      extreme rainfall  -> always safest, with the warning message

    The danger route is never recommended - it exists so the user can
    see what the safest route is avoiding.
    """
    if req.stops:
        raise HTTPException(501, detail="Multi-stop routing not implemented.")
    routes_out, rainfall = _compute_all_routes(req)

    fastest = routes_out["fastest"]
    is_extreme = rainfall >= RAINFALL_PRESETS["extreme"]

    if is_extreme:
        choice = "safest"
        message = (
            "Heavy rainfall detected. The fastest route passes through "
            "multiple flood-prone road segments. The safer route is "
            "recommended to reduce delay and safety risks."
        )
    elif fastest["risk_score"] < 30:
        choice = "fastest"
        message = (
            f"Flood risk on the fastest route is low "
            f"({fastest['risk_score']}/100). No detour needed."
        )
    else:
        safest = routes_out["safest"]
        gain = fastest["risk_score"] - safest["risk_score"]
        if gain <= 0:
            # No safer alternative exists on this corridor. Say so
            # rather than recommending a detour that buys nothing -
            # claiming an improvement that isn't there is worse than
            # admitting the network offers no better option.
            choice = "fastest"
            message = (
                f"Flood risk is {fastest['risk_score']}/100 on every "
                f"available route between these points — the alternatives "
                f"run through the same low-lying corridor. Taking the "
                f"shortest one."
            )
        else:
            choice = "safest"
            message = (
                f"The fastest route scores {fastest['risk_score']}/100 with "
                f"{fastest['hotspot_count']} flood hotspot(s). The safer "
                f"route drops that to {safest['risk_score']}/100 for "
                f"{safest['delay_min']} min extra, avoiding "
                f"{safest['hotspots_avoided']} flood hotspot(s)."
            )

    return {
        "routes": routes_out,
        "recommended": choice,
        "message": message,
        "rainfall_mm_hr": rainfall,
        "extreme_conditions": is_extreme,
    }
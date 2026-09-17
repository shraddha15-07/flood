"""
router.py
Risk-weighted routing over the road graph.

  - nearest_node() uses a KDTree rather than a linear scan of 101,907
    nodes per call.
  - Routes return true OSM geometry, not straight lines between
    intersections.
  - Hotspots come back with real road names.
  - Risk is read at query time so a rainfall change does not require
    rescoring the graph.
"""

import math

import networkx as nx

from risk import edge_risk, risk_score_100, risk_band, is_hotspot

# alpha is a MULTIPLIER on distance, not a flat metre penalty:
#
#     cost = length * (1 + alpha * risk)
#
# The earlier form, cost = length + alpha * risk, charged the same
# penalty for a 10 m alley as for a 500 m flooded highway, so A* was
# minimising the *count* of risky segments rather than exposure to
# them. That produced a "safest" route scoring worse than the fastest.
#
# A negative alpha inverts the cost: risky roads become CHEAPER than
# safe ones, so A* seeks them out. That gives an explicit "danger"
# route - the flood-prone path through the same corridor - which is
# what makes the safest route's advantage legible. Without it the user
# sees two similar-looking lines and no sense of what was avoided.
#
# -0.7 rather than a larger magnitude: at -1.0 or beyond, cost can go
# to zero or negative and A* stops terminating sensibly.
ROUTE_PROFILES = {
    "danger": -0.7,   # deliberately seeks flood-prone roads
    "fastest": 0.0,   # pure shortest path
    "safest": 6.0,    # max-risk road costs 7x its length
}

_KDTREE_CACHE = {}


def _build_kdtree(G):
    """
    Builds a KDTree over node positions. Longitude is scaled by
    cos(latitude) so that degrees are locally equal-distance - without
    this the tree is skewed and picks wrong nodes at Mumbai's latitude.
    """
    from scipy.spatial import cKDTree

    node_ids = list(G.nodes())
    lats = [G.nodes[n]["y"] for n in node_ids]
    lons = [G.nodes[n]["x"] for n in node_ids]
    mean_lat = sum(lats) / len(lats)
    lon_scale = math.cos(math.radians(mean_lat))

    coords = [(lon * lon_scale, lat) for lon, lat in zip(lons, lats)]
    tree = cKDTree(coords)
    return {"tree": tree, "ids": node_ids, "lon_scale": lon_scale}


def nearest_node(G, lat: float, lon: float):
    """Graph node closest to a lat/lon. O(log n) after first call."""
    key = id(G)
    if key not in _KDTREE_CACHE:
        print("[router] Building spatial index for nearest-node lookup...")
        _KDTREE_CACHE[key] = _build_kdtree(G)
    idx = _KDTREE_CACHE[key]
    _, i = idx["tree"].query((lon * idx["lon_scale"], lat))
    return idx["ids"][i]


def invalidate_index(G=None):
    """Call if the graph is rebuilt, so the stale KDTree is dropped."""
    if G is None:
        _KDTREE_CACHE.clear()
    else:
        _KDTREE_CACHE.pop(id(G), None)


def _edge_coords(G, u, v, edge):
    """
    Returns [[lon, lat], ...] for a segment, using true OSM geometry
    when present and falling back to a straight line when it isn't.
    """
    geom = edge.get("geometry")
    if geom is not None:
        try:
            if isinstance(geom, str):
                from shapely import wkt
                geom = wkt.loads(geom)
            return [[float(x), float(y)] for x, y in geom.coords]
        except Exception:
            pass
    return [
        [G.nodes[u]["x"], G.nodes[u]["y"]],
        [G.nodes[v]["x"], G.nodes[v]["y"]],
    ]


def _road_name(edge) -> str:
    """OSM 'name' can be a string, a list, or missing."""
    name = edge.get("name")
    if isinstance(name, list):
        return name[0] if name else "Unnamed road"
    if isinstance(name, str) and name.strip():
        return name
    ref = edge.get("ref")
    if isinstance(ref, str) and ref.strip():
        return ref
    return "Unnamed road"


def _pick_edge(G, u, v, rainfall_mm_hr, alpha):
    """Lowest-cost parallel edge between u and v under the given alpha."""
    candidates = G.get_edge_data(u, v)
    best, best_cost = None, float("inf")
    for edge in candidates.values():
        length = float(edge.get("length", 1.0))
        cost = max(
            length * (1.0 + alpha * edge_risk(edge, rainfall_mm_hr)),
            length * 0.05,
        )
        if cost < best_cost:
            best_cost, best = cost, edge
    return best


def find_route(G, origin_node, dest_node, alpha: float = 0.0,
               rainfall_mm_hr: float = 0.0):
    """
    A* with edge cost = length * (1 + alpha * risk).

    Returns a dict shaped for the UI's route cards, including the
    hotspot list that feeds the Flood Intelligence panel.
    """

    def cost_fn(u, v, data):
        edge = data if "length" in data else list(data.values())[0]
        length = float(edge.get("length", 1.0))
        # Clamped at 0.05 so an inverted alpha can never produce a
        # zero or negative edge weight, which would break A*.
        return max(
            length * (1.0 + alpha * edge_risk(edge, rainfall_mm_hr)),
            length * 0.05,
        )

    def heuristic(a, b):
        # Straight-line metres. Scaled by the clamp factor so it stays
        # admissible when alpha is negative and edges get cheaper.
        lat1, lon1 = G.nodes[a]["y"], G.nodes[a]["x"]
        lat2, lon2 = G.nodes[b]["y"], G.nodes[b]["x"]
        dlat = (lat2 - lat1) * 111_000
        dlon = (lon2 - lon1) * 111_000 * math.cos(math.radians(lat1))
        return math.hypot(dlat, dlon) * (0.05 if alpha < 0 else 1.0)

    try:
        path = nx.astar_path(G, origin_node, dest_node,
                             heuristic=heuristic, weight=cost_fn)
    except nx.NetworkXNoPath:
        raise ValueError("No drivable route exists between these points.")

    coords = []
    total_length = 0.0
    total_time_s = 0.0
    risk_sum = 0.0
    max_risk = 0.0
    hotspots = {}
    segments = []

    for u, v in zip(path[:-1], path[1:]):
        edge = _pick_edge(G, u, v, rainfall_mm_hr, alpha)
        if edge is None:
            continue

        seg_length = float(edge.get("length", 0.0))
        seg_risk = edge_risk(edge, rainfall_mm_hr)
        name = _road_name(edge)

        total_length += seg_length
        # Weight risk by how far you actually travel on it. An
        # unweighted mean lets many short safe links outvote one long
        # submerged one.
        risk_sum += seg_risk * seg_length
        max_risk = max(max_risk, seg_risk)

        travel_s = edge.get("travel_time")
        if travel_s is not None:
            total_time_s += float(travel_s)
        else:
            speed = float(edge.get("speed_kph", 25.0)) or 25.0
            total_time_s += (seg_length / 1000.0) / speed * 3600.0

        pts = _edge_coords(G, u, v, edge)
        if coords and pts and coords[-1] == pts[0]:
            coords.extend(pts[1:])
        else:
            coords.extend(pts)

        if is_hotspot(seg_risk):
            # Merge repeated segments of the same street into one
            # hotspot so the panel shows a road once, not fourteen times.
            entry = hotspots.get(name)
            if entry is None or seg_risk > entry["risk"]:
                hotspots[name] = {
                    "road_name": name,
                    "risk": seg_risk,
                    "lat": G.nodes[u]["y"],
                    "lon": G.nodes[u]["x"],
                    "length_m": seg_length,
                }
            else:
                entry["length_m"] += seg_length

        segments.append({
            "from": u, "to": v,
            "road_name": name,
            "length_m": round(seg_length, 1),
            "risk": round(seg_risk, 3),
        })

    n = len(segments) or 1
    avg_risk = risk_sum / total_length if total_length > 0 else 0.0

    hotspot_list = sorted(hotspots.values(), key=lambda h: -h["risk"])
    for h in hotspot_list:
        h["flood_probability"] = risk_score_100(h["risk"])
        h["severity"] = risk_band(h["flood_probability"])
        h["length_m"] = round(h["length_m"], 1)
        h["risk"] = round(h["risk"], 3)

    return {
        "path_nodes": path,
        "coordinates": coords,
        "distance_m": round(total_length, 1),
        "distance_km": round(total_length / 1000.0, 2),
        "travel_time_min": round(total_time_s / 60.0, 1),
        "avg_risk": round(avg_risk, 3),
        "risk_score": risk_score_100(avg_risk),
        "risk_band": risk_band(risk_score_100(avg_risk)),
        "max_segment_risk": round(max_risk, 3),
        "hotspot_count": len(hotspot_list),
        "hotspots": hotspot_list,
        "total_segments": n,
        "segments": segments,
    }
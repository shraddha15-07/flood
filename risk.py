"""
risk.py
Flood-risk scoring for road segments.

Two-phase design (this is the important change):

  Phase 1 - compute_static_risk(G, factors)   [slow, run ONCE at startup]
      Computes the spatial component of risk from data that does not
      change minute to minute: elevation, drainage, historical flooding,
      waterlogging reports. Stores it as edge attribute "base_risk".

  Phase 2 - apply_rainfall(base_risk, mm_hr)  [instant, run PER REQUEST]
      Blends in the rainfall term. Because this is a pure scalar
      function of a precomputed number, the frontend's rainfall
      selector (Light/Moderate/Heavy/Extreme) is free - no graph walk.

The old code recomputed all ~250k edges on every rainfall change, which
would have made the rainfall control unusable.

FACTOR REGISTRY
---------------
Weights follow the UI spec. Factors with no data loaded are dropped and
their weight is redistributed proportionally across the remaining ones,
so the model stays normalised as you add data sources.

    elevation      30%   <- SRTM, already working
    drainage       25%   <- awaiting data
    historical     20%   <- SAR cache, working (sparse)
    rainfall       15%   <- live, applied in phase 2
    waterlogging   10%   <- awaiting data
"""

from collections import defaultdict

# Spec weights. Rainfall is handled separately in phase 2.
SPATIAL_WEIGHTS = {
    "elevation": 0.30,
    "drainage": 0.25,
    "historical": 0.20,
    "waterlogging": 0.10,
}
RAINFALL_WEIGHT = 0.15

# Rainfall preset levels (mm/hr) matching the UI selector.
# Calibrated to IMD's own nowcast intensity bands (light <5, moderate
# 5-15, heavy >15 mm/hr) rather than invented thresholds, so the UI
# selector means the same thing IMD means. "Extreme" has no IMD hourly
# band; 60 reflects Mumbai deluge rates (26 Jul 2005 peaked far above).
RAINFALL_PRESETS = {
    "light": 2.5,
    "moderate": 10.0,
    "heavy": 25.0,
    "extreme": 60.0,
}
RAINFALL_SCALE_MM_HR = 40.0  # mm/hr at which the rain term saturates


def _normalise_weights(available: set) -> dict:
    """Keep only factors we have data for, rescaled to sum to 1.0."""
    active = {k: w for k, w in SPATIAL_WEIGHTS.items() if k in available}
    total = sum(active.values())
    if total <= 0:
        raise ValueError("No risk factors available - cannot score graph.")
    return {k: w / total for k, w in active.items()}


def smooth_node_values(G, node_values: dict, hops: int = 2,
                       decay: float = 0.5) -> dict:
    """
    Spreads a sparse per-node signal to nearby nodes along the road graph.

    Your SAR cache marks only 3.8% of nodes, so used raw it contributes
    almost nothing outside those exact intersections. Flooding is not
    that localised - if an intersection floods, the approaches to it are
    usually affected too. This propagates each node's value outward for
    `hops` steps, attenuated by `decay` per hop, taking the max at each
    node so strong signals dominate weak ones.

    hops=2, decay=0.5 means a node 2 intersections from a 0.6 flood node
    inherits 0.15. Tune both against ground truth once you have it.
    """
    smoothed = dict(node_values)
    frontier = dict(node_values)

    for _ in range(hops):
        next_frontier = defaultdict(float)
        for node, val in frontier.items():
            if node not in G:
                continue
            attenuated = val * decay
            if attenuated < 0.01:
                continue
            neighbours = set(G.successors(node)) | set(G.predecessors(node))
            for nbr in neighbours:
                if attenuated > next_frontier[nbr]:
                    next_frontier[nbr] = attenuated
        for node, val in next_frontier.items():
            if val > smoothed.get(node, 0.0):
                smoothed[node] = val
        frontier = dict(next_frontier)

    return smoothed


# Above this height, urban pluvial flooding is not an elevation problem -
# a node at 60 m and one at 600 m are equally safe, so both score 0.
# Normalising across the graph's true 0-705 m range instead flattens
# every street in Mumbai into the top 3% of the scale, which makes the
# elevation factor a constant and destroys the signal entirely.
ELEVATION_CEILING_M = 30.0


def _elevation_factor(G) -> dict:
    """
    Per-node elevation risk, 0-1, inverted (low ground = high risk).

    Scaled against ELEVATION_CEILING_M rather than the graph maximum.
    With a 0-705 m linear scale a 3 m node scored 0.996 and a 17 m node
    0.976 - indistinguishable. Against a 30 m ceiling they score 0.90
    and 0.43, which is the spread the router actually needs.

    Nodes with no elevation (elev_missing) are omitted from the table
    so they contribute nothing rather than scoring as sea level.
    """
    out = {}
    for n, d in G.nodes(data=True):
        elev = d.get("elevation")
        if elev in (None, "") or d.get("elev_missing"):
            continue
        try:
            elev = float(elev)
        except (TypeError, ValueError):
            continue
        clipped = min(max(elev, 0.0), ELEVATION_CEILING_M)
        out[n] = 1.0 - clipped / ELEVATION_CEILING_M
    return out


def compute_static_risk(G, flood_history: dict = None,
                        drainage: dict = None,
                        waterlogging: dict = None,
                        smooth_historical: bool = True):
    """
    Computes the rainfall-independent part of risk and stores it on each
    edge as "base_risk" (0-1). Call once at startup.

    Parameters
    ----------
    G : networkx MultiDiGraph, nodes carry 'elevation'
    flood_history : {node_id: 0-1} historical flood frequency (SAR)
    drainage : {node_id: 0-1} drainage inadequacy, 1 = worst drainage
    waterlogging : {node_id: 0-1} normalised complaint/report density

    The last two are the datasets you're supplying. Both are optional -
    pass them in as node-keyed dicts and they activate automatically.
    See the note in main.py for how to map your source data onto node IDs.
    """
    node_factors = {"elevation": _elevation_factor(G)}

    if flood_history:
        hist = flood_history
        if smooth_historical:
            before = len(hist)
            hist = smooth_node_values(G, hist, hops=2, decay=0.5)
            print(f"[risk] Historical signal smoothed: "
                  f"{before} -> {len(hist)} nodes with nonzero value.")
        node_factors["historical"] = hist

    if drainage:
        node_factors["drainage"] = drainage
    if waterlogging:
        node_factors["waterlogging"] = waterlogging

    weights = _normalise_weights(set(node_factors))
    missing = set(SPATIAL_WEIGHTS) - set(node_factors)
    if missing:
        print(f"[risk] No data for {sorted(missing)} - weight redistributed.")
    print(f"[risk] Active weights: "
          f"{ {k: round(v, 3) for k, v in weights.items()} }")

    for u, v, k, data in G.edges(keys=True, data=True):
        score = 0.0
        for name, weight in weights.items():
            table = node_factors[name]
            val_u = table.get(u, 0.0) or 0.0
            val_v = table.get(v, 0.0) or 0.0
            score += weight * ((val_u + val_v) / 2.0)
        data["base_risk"] = round(max(0.0, min(1.0, score)), 4)

    return G


def apply_rainfall(base_risk: float, rainfall_mm_hr: float) -> float:
    """
    Phase 2. Combines a precomputed base_risk with current rainfall.

    Spec-faithful additive form: rainfall contributes its own 15%.
    Note that because rainfall is uniform citywide it shifts every
    segment equally and so does not change route *ranking* - it only
    moves the absolute numbers. If you want rain to actually change
    which route wins, switch to the multiplicative form below, where
    heavy rain amplifies already-risky segments more than safe ones:

        return base_risk * (0.4 + 0.6 * rain_norm)

    Keep whichever you pick consistent between /risk and /route or the
    map colours won't match the route scores.
    """
    rain_norm = min(max(rainfall_mm_hr, 0.0) / RAINFALL_SCALE_MM_HR, 1.0)
    score = (1.0 - RAINFALL_WEIGHT) * base_risk + RAINFALL_WEIGHT * rain_norm
    return max(0.0, min(1.0, score))


def edge_risk(edge_data: dict, rainfall_mm_hr: float) -> float:
    """Convenience: risk for one edge dict at a given rainfall."""
    return apply_rainfall(edge_data.get("base_risk", 0.0), rainfall_mm_hr)


def risk_score_100(risk: float) -> int:
    """0-1 float -> 0-100 integer, which is what the UI displays."""
    return int(round(risk * 100))


def risk_band(score_100: int) -> str:
    """
    Spec classification: Low 0-30, Moderate 31-60, High 61-100.
    Returns the band key; the frontend maps these to colours.
    """
    if score_100 <= 30:
        return "low"
    if score_100 <= 60:
        return "moderate"
    return "high"


def is_hotspot(risk: float) -> bool:
    """A segment counts toward 'Number of Flood Hotspots' in the UI."""
    return risk_score_100(risk) >= 61
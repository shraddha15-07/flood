"""
add_river_overflow.py
Adds overflow exposure to an existing mumbai_drainage.json.

Overpass has been unreliable, and the overflow component matters too
much to drop: Sion, King's Circle and Kurla flood because the Mithi
overtops, not because they lack drainage. A drainage layer without it
scores those areas on slope alone and misses the actual mechanism.

The four courses below are Mumbai's main rivers, traced as coarse
polylines from their known routes. They are approximate - vertex
spacing is roughly a kilometre and the true channel meanders between
points - which is why the influence radius is generous rather than
tight. Replace them with OSM geometry when Overpass cooperates:

    ox.features_from_bbox(bbox=..., tags={"waterway": ["river", "canal"]})

This does NOT recompute slope. It reads the existing file, adds the
overflow component, and writes it back - so it runs in seconds.

Run after extract_drainage.py has produced a file:
    python add_river_overflow.py
"""

import json
import math

import numpy as np

TARGET = "mumbai_drainage.json"

# Metres from the channel centreline at which exposure reaches zero.
# Wide because the traced courses are approximate.
OVERFLOW_RADIUS = 250.0

# (lat, lon) waypoints, upstream to outfall.
RIVERS = {
    "Mithi": [
        (19.1450, 72.9080),  # Vihar Lake outfall
        (19.1290, 72.9040),  # Powai Lake overflow joins
        (19.1130, 72.8950),  # Saki Naka
        (19.0980, 72.8860),  # Marol / Andheri-Kurla Rd
        (19.0850, 72.8810),  # CST Road
        (19.0720, 72.8760),  # Kurla
        (19.0660, 72.8660),  # Bandra-Kurla Complex
        (19.0570, 72.8560),  # BKC west
        (19.0450, 72.8460),  # Dharavi
        (19.0370, 72.8380),  # Mahim
        (19.0330, 72.8320),  # Mahim Creek mouth
    ],
    "Oshiwara": [
        (19.1780, 72.8560),  # Goregaon / Aarey side
        (19.1660, 72.8460),
        (19.1550, 72.8380),
        (19.1450, 72.8290),  # Oshiwara
        (19.1380, 72.8230),  # Lokhandwala
        (19.1330, 72.8180),  # Malad Creek
    ],
    "Poisar": [
        (19.2140, 72.8820),  # Sanjay Gandhi NP
        (19.2080, 72.8700),
        (19.2020, 72.8570),
        (19.1980, 72.8460),  # Kandivali
        (19.1940, 72.8380),  # Marve Creek side
    ],
    "Dahisar": [
        (19.2320, 72.8760),  # Tulsi Lake catchment
        (19.2400, 72.8640),
        (19.2480, 72.8530),
        (19.2550, 72.8450),  # Dahisar
        (19.2620, 72.8390),  # Manori Creek
    ],
}


def densify(points, step_m=50.0):
    """Interpolate along each polyline so distance is to the line."""
    out = []
    for (lat1, lon1), (lat2, lon2) in zip(points[:-1], points[1:]):
        mid = (lat1 + lat2) / 2
        dx = (lon2 - lon1) * 111_320 * math.cos(math.radians(mid))
        dy = (lat2 - lat1) * 110_540
        n = max(int(math.hypot(dx, dy) // step_m), 1)
        for i in range(n):
            f = i / n
            out.append((lat1 + (lat2 - lat1) * f, lon1 + (lon2 - lon1) * f))
    out.append(points[-1])
    return out


def main():
    from graph_utils import get_graph
    from scipy.spatial import cKDTree

    print("[rivers] Loading road graph...")
    G = get_graph(mode="city")
    node_ids = list(G.nodes())
    lats = np.array([G.nodes[n]["y"] for n in node_ids])
    lons = np.array([G.nodes[n]["x"] for n in node_ids])
    scale = math.cos(math.radians(float(lats.mean())))

    pts = []
    for name, way in RIVERS.items():
        d = densify(way)
        pts.extend(d)
        print(f"[rivers]   {name:10s} {len(way)} waypoints -> {len(d)} samples")

    tree = cKDTree([(lon * scale, lat) for lat, lon in pts])
    dist_deg, _ = tree.query(np.column_stack([lons * scale, lats]))
    dist_m = dist_deg * 110_540
    overflow = np.clip(1.0 - dist_m / OVERFLOW_RADIUS, 0.0, 1.0)

    exposed = int((overflow > 0.01).sum())
    print(f"[rivers] Overflow-exposed nodes: {exposed} "
          f"({exposed * 100.0 / len(node_ids):.1f}%)")

    # Merge into the existing file.
    try:
        with open(TARGET) as f:
            existing = json.load(f)
        print(f"[rivers] Existing drainage entries: {len(existing)}")
    except FileNotFoundError:
        existing = {}
        print("[rivers] No existing file; writing overflow only.")

    merged = dict(existing)
    raised = 0
    for nid, ov in zip(node_ids, overflow):
        if ov <= 0.01:
            continue
        key = str(nid)
        prev = float(merged.get(key, 0.0))
        # max(), not sum: good slope must not offset riverbank exposure,
        # and two signals agreeing should not exceed either.
        if ov > prev:
            merged[key] = round(float(ov), 3)
            raised += 1

    with open(TARGET, "w") as f:
        json.dump(merged, f)

    v = np.array(list(merged.values()))
    print(f"\n[rivers] Wrote {TARGET}")
    print(f"  entries      : {len(merged)} (was {len(existing)})")
    print(f"  raised by river exposure : {raised}")
    print(f"  saturated at 1.0 : {int((v >= 0.999).sum())} "
          f"({(v >= 0.999).sum() * 100.0 / len(v):.1f}%)")
    print("  percentiles  : " + ", ".join(
        f"p{p}={np.percentile(v, p):.2f}" for p in (10, 25, 50, 75, 90)))

    # Sanity check against places known to flood from Mithi overflow.
    print("\n  Check - these should be nonzero:")
    for name, (la, lo) in {"Sion Circle": (19.0390, 72.8619),
                           "Kurla": (19.0654, 72.8794),
                           "BKC": (19.0640, 72.8620),
                           "Dharavi": (19.0430, 72.8480)}.items():
        i = int(np.argmin((lats - la) ** 2 + (lons - lo) ** 2))
        print(f"    {name:14s} {merged.get(str(node_ids[i]), 0.0)}")


if __name__ == "__main__":
    main()

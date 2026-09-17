"""
extract_drainage.py
Builds mumbai_drainage.json from terrain slope and major watercourses.

WHY NOT DISTANCE-TO-DRAIN
-------------------------
The obvious implementation - distance from each node to the nearest OSM
`waterway=drain` - does not work here, and the failure is instructive.

OSM has roughly 1,060 drain/ditch features across the graph's extent,
which spans Panvel to Kalyan. Mean distance to the nearest one came out
at 2.7 km, and 65% of nodes saturated at maximum risk. Mumbai's nullah
network is heavily under-mapped in OSM, and unevenly so: dense central
areas have mappers, outer areas do not. So "far from a mapped drain"
measures mapper coverage, not drainage capacity. No choice of threshold
rescues a signal that is not in the data.

WHAT THIS USES INSTEAD
----------------------
Two components, both from sources that are actually reliable, and both
named in section 7.3 of the proposal:

  1. SLOPE (from the DEM you already have). Water leaves a gradient and
     sits on a flat. Slope is derived from terrain, so it has no
     mapping-coverage bias at all - every node gets a real number.

  2. OVERFLOW EXPOSURE (major watercourses only). Rivers and canals are
     mapped far more completely than ditches. Being on the bank of the
     Mithi is a genuine flood mechanism: Sion and King's Circle flood
     because the river overflows, not because they lack drains.

Combined with max(), not mean - good slope should not cancel out
sitting beside a river.

WHAT THIS IS NOT
----------------
It is not BMC's stormwater drain network, which is unpublished. It is a
terrain-and-hydrology proxy for it. When you obtain the SWD GIS layer,
replace component 2 and re-run.

Run:
    python extract_drainage.py
"""

import json
import math

import numpy as np

OUTPUT = "mumbai_drainage.json"

# Slope is sampled by comparing elevation at each node against points
# this far away in four directions.
#
# 400 m is not arbitrary - it was measured. At 150 m the probe sits
# inside the building noise of a 30 m surface model and the result is
# meaningless: Hindmata, which floods to chest height every monsoon,
# read 2.67 m/100m, steeper than it read at Sion. At 400 m the
# buildings average out and real terrain appears:
#
#     Hindmata 0.38   Sion 0.38   Kurla 0.38   Milan Subway 0.88
#     Powai (dry, hilly) 2.38
#
# Every known flood spot under 0.9, dry high ground above 2.3. Beyond
# ~800 m the probes start falling outside DEM coverage.
SLOPE_PROBE_M = 400.0

# Slope in metres per 100 m at which drainage is considered adequate.
# Set from the measurements above: flood spots cluster below 0.9,
# dry ground above 2.3, so 2.0 puts the boundary in the gap.
SLOPE_ADEQUATE = 2.0



def compute_slope(lats, lons, dem_paths):
    """
    Per-node slope in metres per 100 m.

    Samples elevation at the node and at four points SLOPE_PROBE_M away
    (N/S/E/W), then takes the steepest of the two axis gradients. Nodes
    where the DEM has no coverage return NaN and are excluded rather
    than assumed flat.
    """
    from elevation import sample_elevations

    mean_lat = float(lats.mean())
    dlat = SLOPE_PROBE_M / 110_540
    dlon = SLOPE_PROBE_M / (111_320 * math.cos(math.radians(mean_lat)))

    offsets = {
        "n": (lats + dlat, lons),
        "s": (lats - dlat, lons),
        "e": (lats, lons + dlon),
        "w": (lats, lons - dlon),
    }

    # srtm.py fills probe points that fall outside the local rasters -
    # without it, nodes near a raster edge lose slope entirely.
    try:
        import srtm
        srtm_data = srtm.get_data()
    except ImportError:
        srtm_data = None
        print("[drainage]   srtm.py unavailable; edge nodes may lack slope")

    samples = {}
    for name, (la, lo) in offsets.items():
        print(f"[drainage]   sampling DEM: {name}")
        vals = sample_elevations(list(zip(la, lo)), dem_paths)
        if srtm_data is not None:
            for i, v in enumerate(vals):
                if v is None:
                    try:
                        e = srtm_data.get_elevation(float(la[i]), float(lo[i]))
                        if e is not None:
                            vals[i] = float(e)
                    except Exception:
                        pass
        samples[name] = np.array(
            [np.nan if v is None else v for v in vals], dtype=float)

    # Central difference over 2 * probe distance.
    ns = np.abs(samples["n"] - samples["s"]) / (2 * SLOPE_PROBE_M) * 100.0
    ew = np.abs(samples["e"] - samples["w"]) / (2 * SLOPE_PROBE_M) * 100.0
    return np.fmax(ns, ew)


def main():
    from graph_utils import get_graph
    from elevation import find_dem_files

    print("[drainage] Loading road graph...")
    G = get_graph(mode="city")
    node_ids = list(G.nodes())
    lats = np.array([G.nodes[n]["y"] for n in node_ids])
    lons = np.array([G.nodes[n]["x"] for n in node_ids])
    mean_lat = float(lats.mean())
    scale = math.cos(math.radians(mean_lat))
    bounds = (float(lons.min()), float(lats.min()),
              float(lons.max()), float(lats.max()))

    # ---- Component 1: slope ----
    dem_paths = find_dem_files(("dem", "."))
    if not dem_paths:
        print("[drainage] No DEM found in ./dem/ - cannot compute slope.")
        return
    print(f"[drainage] Computing slope from {len(dem_paths)} raster(s)...")
    slope = compute_slope(lats, lons, dem_paths)

    valid = ~np.isnan(slope)
    print(f"[drainage] Slope computed for {valid.sum()} nodes "
          f"({valid.sum() * 100.0 / len(node_ids):.1f}%)")
    if valid.any():
        print(f"[drainage] Slope m/100m: median {np.nanmedian(slope):.2f}, "
              f"p90 {np.nanpercentile(slope, 90):.2f}, "
              f"max {np.nanmax(slope):.2f}")

    # Flat -> 1.0, at or above SLOPE_ADEQUATE -> 0.0
    slope_risk = np.where(
        valid, np.clip(1.0 - slope / SLOPE_ADEQUATE, 0.0, 1.0), 0.0)

    # ---- Component 2: overflow exposure ----
    # Removed. This used to fetch rivers from Overpass, which failed on
    # every mirror (502s and connect timeouts) and blocked the whole
    # script for minutes each run. River exposure is now applied
    # separately by add_river_overflow.py, from traced polylines, in
    # seconds and with no network dependency.
    overflow = np.zeros(len(node_ids))

    # max(), not mean: good slope must not cancel out riverbank exposure.
    score = np.maximum(slope_risk, overflow)

    out = {str(nid): round(float(s), 3)
           for nid, s in zip(node_ids, score) if s > 0.01}

    with open(OUTPUT, "w") as f:
        json.dump(out, f)

    total = len(node_ids)
    v = np.array(list(out.values())) if out else np.array([0.0])
    print(f"\n[drainage] Wrote {OUTPUT}")
    print(f"  nonzero nodes : {len(out)} ({len(out) * 100.0 / total:.1f}%)")
    print(f"  saturated at 1.0 : {int((v >= 0.999).sum())} "
          f"({(v >= 0.999).sum() * 100.0 / max(len(v), 1):.1f}%)")
    print("  percentiles : " + ", ".join(
        f"p{p}={np.percentile(v, p):.2f}" for p in (10, 25, 50, 75, 90)))
    print("\nIf more than about a quarter sits at 1.0, raise "
          "SLOPE_ADEQUATE and re-run - a saturated factor is a constant.")
    print("\nNow run:  python add_river_overflow.py")


if __name__ == "__main__":
    main()
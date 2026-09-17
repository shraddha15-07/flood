"""
rebuild_elevation.py
Replaces every node's elevation in the cached graph with a real reading.

Two sources, in priority order:

  1. Local .tif rasters in ./dem/  - your CartoDEM and SRTM files.
     CartoDEM (P5_PAN_CD_*) is ISRO's Cartosat-1 product and is
     generally better conditioned over urban India than SRTM, so
     local rasters win wherever they have coverage.

  2. srtm.py  - auto-downloads and caches whatever tiles the local
     rasters do not reach. Confirmed working for N18E072, N19E072,
     N18E073 and N19E073, which together cover the whole road graph.

Nodes that BOTH sources fail on are marked, never guessed. The original
graph_utils.attach_elevation() filled failures with

    elevation = 2.0 + dist * 50_000

which is a distance-from-centre number with no relationship to terrain.
That is the synthetic data being removed here.

Run:
    python rebuild_elevation.py
"""

import os

import osmnx as ox

from elevation import find_dem_files, sample_elevations

SOURCE = "mumbai_graph_cache.graphml"
OUTPUT = "mumbai_graph_cache_dem.graphml"


def fill_with_srtm(points, values):
    """Fills None entries using srtm.py. Returns (values, filled_count)."""
    todo = [i for i, v in enumerate(values) if v is None]
    if not todo:
        return values, 0

    try:
        import srtm
    except ImportError:
        print("[rebuild] srtm.py not installed - run 'pip install srtm.py'")
        return values, 0

    print(f"[rebuild] Filling {len(todo)} uncovered nodes via srtm.py "
          f"(downloads tiles on first use)...")
    data = srtm.get_data()

    filled = 0
    for count, i in enumerate(todo, 1):
        lat, lon = points[i]
        try:
            v = data.get_elevation(lat, lon)
        except Exception:
            v = None
        if v is not None:
            values[i] = max(float(v), 0.0)
            filled += 1
        if count % 10000 == 0:
            print(f"           {count}/{len(todo)}...")

    return values, filled


def main():
    dem_paths = find_dem_files(("dem", "."))
    print(f"[rebuild] Local rasters: "
          f"{[os.path.basename(p) for p in dem_paths] or 'none'}")

    print(f"[rebuild] Loading {SOURCE} (several minutes)...")
    G = ox.load_graphml(SOURCE)
    for _, d in G.nodes(data=True):
        d["y"] = float(d["y"])
        d["x"] = float(d["x"])

    node_ids = list(G.nodes())
    points = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_ids]
    total = len(node_ids)

    # Report what is being replaced.
    old = []
    for _, d in G.nodes(data=True):
        try:
            old.append(float(d["elevation"]))
        except (KeyError, TypeError, ValueError):
            pass
    if old:
        whole = sum(1 for v in old if v == int(v))
        pct = whole * 100.0 / len(old)
        print(f"[rebuild] Existing values: {len(old)} nodes, "
              f"{pct:.1f}% whole-numbered, "
              f"range {min(old):.1f}-{max(old):.1f} m")
        if pct < 90:
            print("          -> mostly fractional, i.e. synthetic fallback, "
                  "not real DEM readings.")

    # Stage 1: local rasters.
    if dem_paths:
        print(f"[rebuild] Sampling {total} nodes from local rasters...")
        values = sample_elevations(points, dem_paths)
        from_raster = sum(1 for v in values if v is not None)
        print(f"[rebuild] Local rasters covered {from_raster} "
              f"({from_raster * 100.0 / total:.1f}%)")
    else:
        values = [None] * total
        from_raster = 0

    # Stage 2: srtm.py gap fill.
    values, from_srtm = fill_with_srtm(points, values)

    missing = 0
    for n, v in zip(node_ids, values):
        if v is None:
            G.nodes[n]["elevation"] = ""
            G.nodes[n]["elev_missing"] = 1
            missing += 1
        else:
            G.nodes[n]["elevation"] = float(v)
            G.nodes[n]["elev_missing"] = 0

    good = [v for v in values if v is not None]
    print("\n--- Result ---")
    print(f"  local rasters : {from_raster:6d} ({from_raster * 100.0 / total:5.1f}%)")
    print(f"  srtm.py fill  : {from_srtm:6d} ({from_srtm * 100.0 / total:5.1f}%)")
    print(f"  no data       : {missing:6d} ({missing * 100.0 / total:5.1f}%)")
    if good:
        print(f"  range         : {min(good):.0f}-{max(good):.0f} m, "
              f"mean {sum(good) / len(good):.1f} m")

    if missing:
        print("\n  Those nodes have no elevation signal and are excluded "
              "from elevation-based scoring.")

    print(f"\n[rebuild] Saving {OUTPUT}...")
    ox.save_graphml(G, OUTPUT)
    print("[rebuild] Done.\n")
    print("Swap it in once the numbers above look right:")
    print(f"    copy {SOURCE} {SOURCE}.synthetic.bak")
    print(f"    copy {OUTPUT} {SOURCE}")


if __name__ == "__main__":
    main()
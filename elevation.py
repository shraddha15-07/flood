"""
elevation.py
Attaches elevation to graph nodes by sampling local SRTM GeoTIFFs.

Replaces the srtm.py lookup in graph_utils.attach_elevation(). That
version downloaded tiles at runtime and fell back to a distance-from-
centre heuristic on any failure - which silently produced fake
elevation, and fake elevation means fake flood risk.

This version reads the rasters you already have, and instead of
inventing values for uncovered nodes it marks them and reports the
gap loudly, so a coverage hole shows up as a warning rather than as
plausible-looking wrong numbers.

Coverage of your current rasters against the cached road graph:

    mumbai_dem.tif      72.0-73.0 E, 19.0-20.0 N   ->  44.7% of nodes
    mumbai_clipped.tif  72.7-73.0 E, 19.0-19.3 N   ->  35.0% of nodes
    graph extent        72.74-73.45 E, 18.70-19.52 N

55.3% of nodes fall outside both. Missing SRTM 1-arcsec tiles:
    N18E072  (covers Colaba, Fort, Churchgate - all of south Mumbai)
    N18E073
    N19E073  (covers Thane, Navi Mumbai, Panvel)

Download from https://earthexplorer.usgs.gov (SRTM 1 Arc-Second Global)
or https://portal.opentopography.org, drop them in dem/, and this
module will mosaic whatever it finds.
"""

import glob
import os

import numpy as np

# SRTM voids are -32768. Coastal returns can also be slightly negative
# from radar error; below this we treat the pixel as unusable.
VOID_THRESHOLD = -15.0


def _open_sources(dem_paths):
    import rasterio
    srcs = []
    for p in dem_paths:
        try:
            srcs.append(rasterio.open(p))
        except Exception as e:
            print(f"[elevation] Could not open {p}: {e}")
    return srcs


def find_dem_files(search_dirs=("dem", ".")):
    """Collects every .tif in the given directories, ordered finest first."""
    paths = []
    for d in search_dirs:
        if os.path.isdir(d):
            paths.extend(sorted(glob.glob(os.path.join(d, "*.tif"))))
    # De-duplicate while preserving order.
    seen, out = set(), []
    for p in paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def sample_elevations(points, dem_paths):
    """
    Samples elevation for [(lat, lon), ...] across one or more rasters.

    Rasters are tried in order; the first one that yields a valid,
    non-void reading wins. Returns a list of float-or-None, None meaning
    no raster covered that point.
    """
    srcs = _open_sources(dem_paths)
    if not srcs:
        raise RuntimeError(f"No readable DEM found in {dem_paths}")

    for s in srcs:
        print(f"[elevation] Using {os.path.basename(s.name)} "
              f"({s.bounds.left:.3f},{s.bounds.bottom:.3f} -> "
              f"{s.bounds.right:.3f},{s.bounds.top:.3f})")

    n = len(points)
    out = [None] * n
    pending = list(range(n))

    for src in srcs:
        if not pending:
            break
        b = src.bounds
        # Only hand this raster the points that fall inside its bounds -
        # sampling out-of-bounds is slow and returns nodata anyway.
        idxs, coords = [], []
        for i in pending:
            lat, lon = points[i]
            if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
                idxs.append(i)
                coords.append((lon, lat))
        if not idxs:
            continue

        nodata = src.nodata
        for i, val in zip(idxs, src.sample(coords, 1)):
            v = float(val[0])
            if nodata is not None and v == nodata:
                continue
            if v <= VOID_THRESHOLD or not np.isfinite(v):
                continue
            out[i] = max(v, 0.0)   # clamp sub-sea-level radar noise to 0

        pending = [i for i in pending if out[i] is None]

    for s in srcs:
        s.close()
    return out


def attach_elevation_from_dem(G, dem_paths=None, strict=False):
    """
    Writes 'elevation' onto every node from local DEM rasters.

    Nodes with no coverage get elevation=None and 'elev_missing'=True.
    risk.py should skip these rather than score them - a node with
    unknown ground height must not be presented as low risk.

    strict=True raises if any node is uncovered, which is what you want
    in the cache-generation script so a bad build fails fast instead of
    shipping a half-scored graph.
    """
    if dem_paths is None:
        dem_paths = find_dem_files()
    if not dem_paths:
        raise RuntimeError(
            "No DEM .tif files found. Place mumbai_dem.tif (and the "
            "missing N18E072 / N18E073 / N19E073 tiles) in ./dem/")

    node_ids = list(G.nodes())
    points = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_ids]

    print(f"[elevation] Sampling {len(node_ids)} nodes...")
    values = sample_elevations(points, dem_paths)

    missing = 0
    for n, v in zip(node_ids, values):
        if v is None:
            G.nodes[n]["elevation"] = None
            G.nodes[n]["elev_missing"] = True
            missing += 1
        else:
            G.nodes[n]["elevation"] = v
            G.nodes[n]["elev_missing"] = False

    pct = missing * 100.0 / max(len(node_ids), 1)
    covered = [v for v in values if v is not None]
    if covered:
        print(f"[elevation] Covered {len(covered)} nodes: "
              f"min={min(covered):.1f}m max={max(covered):.1f}m "
              f"mean={sum(covered) / len(covered):.1f}m")

    if missing:
        msg = (f"[elevation] WARNING: {missing} nodes ({pct:.1f}%) have no "
               f"DEM coverage and cannot be risk-scored.")
        if strict:
            raise RuntimeError(msg)
        print(msg)

    return G


def coverage_report(G, dem_paths=None):
    """Prints which parts of the graph a DEM set does and doesn't reach."""
    import rasterio
    if dem_paths is None:
        dem_paths = find_dem_files()

    lats = [d["y"] for _, d in G.nodes(data=True)]
    lons = [d["x"] for _, d in G.nodes(data=True)]
    print(f"Graph extent: {min(lons):.4f}-{max(lons):.4f} E, "
          f"{min(lats):.4f}-{max(lats):.4f} N  ({len(lats)} nodes)")

    total = len(lats)
    covered = set()
    for p in dem_paths:
        with rasterio.open(p) as s:
            b = s.bounds
            hits = {i for i, (la, lo) in enumerate(zip(lats, lons))
                    if b.left <= lo <= b.right and b.bottom <= la <= b.top}
            covered |= hits
            print(f"  {os.path.basename(p):28s} {len(hits):6d} nodes "
                  f"({len(hits) * 100.0 / total:5.1f}%)")
    gap = total - len(covered)
    print(f"  {'UNION':28s} {len(covered):6d} nodes "
          f"({len(covered) * 100.0 / total:5.1f}%)")
    print(f"  {'UNCOVERED':28s} {gap:6d} nodes "
          f"({gap * 100.0 / total:5.1f}%)")
    return gap
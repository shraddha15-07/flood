"""
graph_utils.py
Loads the road network graph for the demo area.

Two modes:
1. load_real_graph()  -> uses OSMnx to pull actual Mumbai roads.
   Requires internet access to the OpenStreetMap/Overpass API.
   Run this on your own machine (not inside a sandboxed environment).
2. load_demo_graph()  -> builds a small synthetic grid graph that
   mimics a road network, so you can run and test the whole
   pipeline (routing, API, frontend) with zero external dependencies.

Both return a networkx.MultiDiGraph with the same edge/node
attribute shape, so nothing downstream (risk scoring, routing,
API) needs to know which one is in use.
"""

import random
import requests
import networkx as nx


def geocode_address(address: str, viewbox_bias: tuple = None):
    """
    Converts a plain-text address into (lat, lon) using OpenStreetMap's
    free Nominatim geocoding API — no API key needed.

    viewbox_bias: optional (min_lon, min_lat, max_lon, max_lat) box to
    softly bias results toward your demo area (e.g. "Dadar" ranks
    Mumbai's Dadar higher). This does NOT restrict results to that box —
    a real address outside it can still be found; whether it's actually
    usable for routing is checked separately in main.py, since that
    depends on how large an area's road network you've loaded.

    Nominatim's usage policy requires a real User-Agent identifying
    your app, and max ~1 request/second for free use — fine for this
    prototype's interactive use.
    """
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
    }
    if viewbox_bias:
        params["viewbox"] = ",".join(str(v) for v in viewbox_bias)
        # note: no "bounded" param — this only nudges ranking, doesn't
        # exclude results outside the box

    headers = {"User-Agent": "aquaroute-prototype/0.1 (student project)"}
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params=params, headers=headers, timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Could not geocode address: {address!r}")

    return float(results[0]["lat"]), float(results[0]["lon"])


def load_real_graph(center_lat: float = 19.017, center_lon: float = 72.844,
                     radius_m: int = 1500):
    """
    Pulls a real drivable road network for a SMALL area using OSMnx,
    centered on a lat/lon point with a radius in meters. Useful for
    fast local iteration/testing — for full-city coverage (so any
    Mumbai address can be routed), use load_city_graph() instead.

    We use graph_from_point (not graph_from_place) because Nominatim's
    free geocoder often can't resolve small neighborhoods (e.g.
    "Hindmata, Mumbai") to an official polygon boundary — a center
    point + radius sidesteps that entirely and is more reliable for
    an arbitrary small area.

    Free, no API key needed — but needs internet access to OSM servers.
    Install first:  pip install osmnx
    """
    import osmnx as ox

    G = ox.graph_from_point((center_lat, center_lon), dist=radius_m,
                             network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    attach_elevation(G)
    return G


def load_city_graph(place_name: str = "Greater Mumbai, Maharashtra, India",
                     cache_path: str = "mumbai_graph_cache.graphml",
                     force_refresh: bool = False):
    """
    Pulls the FULL drivable road network for Mumbai using OSMnx, so any
    address in the city can be geocoded and routed — like Uber/Google
    Maps coverage, not just one neighborhood.

    Free, no API key needed — but the FIRST run needs internet access
    to OSM/Overpass servers and will take several minutes (Mumbai's
    full network has tens of thousands of road segments). After that,
    the graph is cached to disk (cache_path) and loads in seconds on
    every restart — important since --reload restarts the process.

    Install first:  pip install osmnx

    Set force_refresh=True to re-download even if a cache file exists
    (e.g. if OSM data has since been updated).
    """
    import os
    import osmnx as ox

    if not force_refresh and os.path.exists(cache_path):
        print(f"[graph_utils] Loading cached Mumbai graph from {cache_path} ...")
        G = ox.load_graphml(cache_path)
        # graphml round-trips numeric attrs as strings sometimes; make sure
        # the ones we rely on downstream are floats.
        for _, data in G.nodes(data=True):
            data["y"] = float(data["y"])
            data["x"] = float(data["x"])
            if "elevation" in data:
                data["elevation"] = float(data["elevation"])
        for _, _, data in G.edges(data=True):
            if "length" in data:
                data["length"] = float(data["length"])
        return G

    print(f"[graph_utils] Downloading full Mumbai road network for '{place_name}' — "
          "this can take several minutes on first run...")
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    attach_elevation(G)

    print(f"[graph_utils] Caching graph to {cache_path} for fast reloads...")
    ox.save_graphml(G, cache_path)
    return G


def attach_elevation(G):
    """
    Attaches a real 'elevation' attribute to every node, using SRTM data
    (free, NASA open data, no API key) via the srtm.py package. It
    auto-downloads the relevant elevation tile(s) on first use and
    caches them locally, so it needs internet access the first time
    only.

    Install first:  pip install srtm.py

    If srtm.py isn't installed or a lookup fails (e.g. no internet),
    falls back to a distance-from-center heuristic so the pipeline
    still runs — but real routing quality depends on real elevation,
    so install srtm.py before using this for anything beyond a demo.
    """
    try:
        import srtm
        elevation_data = srtm.get_data()
        use_fallback = False
    except ImportError:
        print("[graph_utils] srtm.py not installed — run 'pip install srtm.py' "
              "for real elevation. Falling back to an approximate heuristic.")
        use_fallback = True

    lats = [d["y"] for _, d in G.nodes(data=True)]
    lons = [d["x"] for _, d in G.nodes(data=True)]
    center_lat, center_lon = sum(lats) / len(lats), sum(lons) / len(lons)

    for n, data in G.nodes(data=True):
        elevation = None
        if not use_fallback:
            try:
                elevation = elevation_data.get_elevation(data["y"], data["x"])
            except Exception:
                elevation = None
        if elevation is None:
            # Fallback: distance from area center as a rough, non-real proxy
            dist = ((data["y"] - center_lat) ** 2 + (data["x"] - center_lon) ** 2) ** 0.5
            elevation = 2.0 + dist * 50_000
        data["elevation"] = elevation


def load_demo_graph(rows: int = 8, cols: int = 8, seed: int = 42):
    """
    Builds a synthetic grid road network (rows x cols intersections)
    with randomized elevation per node, standing in for real DEM +
    OSM data. Good enough to exercise routing, risk-weighting, the
    API, and the map frontend end-to-end.
    """
    random.seed(seed)
    G = nx.MultiDiGraph()

    # Mumbai-ish bounding box (roughly central Mumbai) so the demo
    # graph plots sensibly on a real map.
    lat0, lat1 = 19.00, 19.03
    lon0, lon1 = 72.83, 72.87

    node_id = 0
    grid = {}
    for r in range(rows):
        for c in range(cols):
            lat = lat0 + (lat1 - lat0) * r / (rows - 1)
            lon = lon0 + (lon1 - lon0) * c / (cols - 1)
            # Synthetic elevation: lower toward the "center" to mimic
            # a low-lying flood-prone pocket, like Hindmata in real life.
            center_r, center_c = rows / 2, cols / 2
            dist_from_center = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
            elevation = 2.0 + dist_from_center * 1.5 + random.uniform(-0.3, 0.3)

            G.add_node(node_id, y=lat, x=lon, elevation=elevation)
            grid[(r, c)] = node_id
            node_id += 1

    def add_edge(u, v):
        y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
        y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
        length = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5 * 111_000  # rough meters
        G.add_edge(u, v, length=length)
        G.add_edge(v, u, length=length)

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                add_edge(grid[(r, c)], grid[(r, c + 1)])
            if r + 1 < rows:
                add_edge(grid[(r, c)], grid[(r + 1, c)])

    return G


def get_graph(mode: str = "demo", center_lat: float = 19.017,
              center_lon: float = 72.844, radius_m: int = 1500):
    """
    Single entry point for loading the graph.

    mode:
      "demo"  -> synthetic grid, zero dependencies, fastest to run
      "small" -> real OSM roads for a small area (fast-ish, needs internet)
      "city"  -> real OSM roads for all of Mumbai (needed for full-city
                 address search/routing; slow first run, cached after)
    """
    if mode == "city":
        return load_city_graph()
    if mode == "small":
        return load_real_graph(center_lat, center_lon, radius_m)
    return load_demo_graph()
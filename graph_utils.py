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
                     cache_path: str = None,
                     force_refresh: bool = False):
    """
    Pulls the FULL drivable road network for Mumbai using OSMnx.
    Loads from compressed .gz cache if present.
    """
    import os
    import gzip
    import pickle
    import tempfile
    import osmnx as ox

    # Fast low-memory binary pickle load (130 MB RAM vs >600 MB for GraphML)
    pickle_files = ["mumbai_graph_cache.pkl.gz", "mumbai_graph_full.pkl", "mumbai_graph.pkl"]
    for pkl in pickle_files:
        if not force_refresh and os.path.exists(pkl):
            try:
                print(f"[graph_utils] Loading binary pickle graph from {pkl}...")
                if pkl.endswith(".gz"):
                    with gzip.open(pkl, "rb") as f:
                        return pickle.load(f)
                else:
                    with open(pkl, "rb") as f:
                        return pickle.load(f)
            except Exception as e:
                print(f"[graph_utils] Failed loading pickle {pkl}: {e}")

    cache_path = cache_path or os.environ.get(
        "FLOOD_GRAPH_CACHE", "mumbai_graph_cache.graphml.gz")
    target_path = cache_path
    if (not os.path.exists(target_path)
            and cache_path == "mumbai_graph_cache.graphml.gz"
            and os.path.exists("mumbai_graph_cache.graphml")):
        target_path = "mumbai_graph_cache.graphml"

    if os.path.exists(target_path):
        with open(target_path, "rb") as cache_file:
            is_lfs_pointer = cache_file.readline().strip() == (
                b"version https://git-lfs.github.com/spec/v1")
        if is_lfs_pointer:
            cache_dir = os.path.join(
                os.environ.get("LOCALAPPDATA", tempfile.gettempdir()),
                "FloodReroute")
            os.makedirs(cache_dir, exist_ok=True)
            target_path = os.path.join(cache_dir, "mumbai_graph_cache.graphml")

    if not force_refresh and os.path.exists(target_path):
        try:
            print(f"[graph_utils] Loading cached Mumbai graph from {target_path} ...")
            G = ox.load_graphml(target_path)
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
        except Exception as e:
            print(f"[graph_utils] Cached graph could not be loaded: {e}")

    print(f"[graph_utils] Downloading full Mumbai road network for '{place_name}' — "
          "this can take several minutes on first run...")
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    attach_elevation(G)

    print(f"[graph_utils] Caching graph to {target_path} for fast reloads...")
    ox.save_graphml(G, target_path)
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
    Single entry point for loading the graph. Always prefers loading
    the cached Mumbai graph file if present.
    """
    import os
    cache_files = ["mumbai_graph_cache.pkl.gz", "mumbai_graph_full.pkl", "mumbai_graph.pkl", "mumbai_graph_cache.graphml.gz", "mumbai_graph_cache.graphml"]
    if any(os.path.exists(f) for f in cache_files):
        return load_city_graph()
    if mode == "city":
        return load_city_graph()
    if mode == "small":
        return load_real_graph(center_lat, center_lon, radius_m)
    return load_demo_graph()
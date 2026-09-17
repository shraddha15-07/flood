"""
generate_sar_cache.py
Pre-computes SAR flood frequency history for all nodes in the road network
and caches it to a JSON file. This avoids querying GEE at every startup.
"""
import os
import json
from graph_utils import get_graph
from gee_sar import init_gee, build_flood_history, get_flood_at_points_batched, MUMBAI_FLOOD_EVENTS

def main():
    print("[cache] Loading city graph...")
    G = get_graph(mode="city")
    nodes = list(G.nodes(data=True))
    node_ids = [n for n, _ in nodes]
    points = [(data['y'], data['x']) for _, data in nodes]
    print(f"[cache] Loaded {len(node_ids)} nodes to sample.")

    print("[cache] Initializing GEE...")
    try:
        init_gee()
    except Exception as e:
        print(f"[cache] GEE Initialization failed. Make sure you have authorized access.")
        print(f"Error: {e}")
        return

    print(f"[cache] Building cumulative flood frequency from {len(MUMBAI_FLOOD_EVENTS)} events...")
    freq_map = build_flood_history(MUMBAI_FLOOD_EVENTS)

    print("[cache] Sampling points in batches (this may take a few minutes)...")
    # Sample in batches of 5000 to avoid GEE payload size limits
    freq_vals = get_flood_at_points_batched(freq_map, points, batch_size=5000)

    print("[cache] Compiling results...")
    cache_data = {}
    for node_id, val in zip(node_ids, freq_vals):
        if val is not None and val > 0:
            # Store in dict; JSON keys must be strings but we'll convert to int when loading
            cache_data[str(node_id)] = round(float(val), 3)

    output_path = "mumbai_sar_flood_history.json"
    with open(output_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    print(f"[cache] Successfully saved {len(cache_data)} flood-prone nodes to {output_path}!")

if __name__ == "__main__":
    main()

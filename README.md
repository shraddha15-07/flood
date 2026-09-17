FloodReroute	—	Flood-Aware	Routing	System	for	Mumbai

A working, end-to-end flood-risk-aware routing system for Mumbai:
real road graph → SAR-enhanced risk scoring → risk-weighted A* routing → FastAPI backend → Leaflet map.

Everything used here is free — no API keys, no paid tiles, no paid hosting required.

## Features

- **Full Mumbai road network** (100k+ nodes) from OpenStreetMap via OSMnx
- **SAR-based flood detection** using Sentinel-1 change-detection via Google Earth Engine
- **Real elevation data** from SRTM (NASA, free)
- **Live rainfall** from Open-Meteo (free, no API key)
- **Risk scoring** blending SAR flood history (60%) + elevation (40%) × rainfall
- **Risk-weighted A* routing** that avoids flood-prone roads
- **Interactive Leaflet map** with color-coded risk overlay and address geocoding

## Run it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open: http://127.0.0.1:8000/static/index.html

- Type or click on the map to set origin/destination
- Adjust "alpha" to control how strongly the router avoids risky roads
  (0 = pure shortest path, higher = detours more around red segments)
- Click "Find Safe Route"

## SAR Flood History (Google Earth Engine)

The risk model uses Sentinel-1 SAR change-detection to identify historically
flood-prone locations. This data is pre-computed and cached:

1. **One-time setup** (already done):
   ```bash
   pip install earthengine-api geemap
   earthengine authenticate
   ```

2. **Generate/refresh the SAR cache** (queries GEE for 5 flood events):
   ```bash
   python generate_sar_cache.py
   ```
   This produces `mumbai_sar_flood_history.json` (~3,880 flood-prone nodes).

3. **Automatic loading**: `main.py` loads the cache at startup. If the file
   doesn't exist, risk scoring falls back to elevation-only (still works).

## Files

- `main.py` — FastAPI backend (`/risk`, `/route`, `/geocode`, `/refresh-risk` endpoints)
- `graph_utils.py` — loads the road graph (real OSMnx or synthetic demo)
- `risk.py` — risk scoring: SAR flood history + elevation + rainfall
- `router.py` — risk-weighted A* routing
- `gee_sar.py` — Google Earth Engine Sentinel-1 SAR flood detection module
- `generate_sar_cache.py` — batch pre-computes SAR flood history for all road nodes
- `mumbai_sar_flood_history.json` — cached SAR flood frequency data (3,880 nodes)
- `mumbai_graph_cache.graphml` — cached Mumbai road network (101,907 nodes)
- `static/index.html` — Leaflet frontend, free OpenStreetMap tiles


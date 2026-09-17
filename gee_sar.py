"""
gee_sar.py
Google Earth Engine — Sentinel-1 SAR flood detection for Mumbai.

Uses change-detection on VV-polarization backscatter to identify
flooded areas: pre-flood composite vs. post-flood image, threshold
on the dB difference.

How it works:
  1. Pull Sentinel-1 SAR images (C-band, VV polarization) from GEE
  2. Build median composites for a "dry reference" period and a "flood event" period
  3. Compute the backscatter difference (post − pre, in dB)
  4. Where the drop exceeds a threshold (e.g. −3 dB) → classify as flooded
  5. Sample the flood mask at road-network node locations

Water produces specular reflection (radar bounces away), so flooded
areas appear DARK (low backscatter) compared to dry land. SAR works
through clouds, unlike optical satellites — critical during monsoon.

Requires:
  pip install earthengine-api geemap numpy
  earthengine authenticate  (one-time browser login)
"""

import ee
import numpy as np


# ── GEE Initialization ────────────────────────────────────────────

_initialized = False


def init_gee(project: str = None):
    """
    Initialize the Earth Engine API. Call once at import or startup.

    project: your Google Cloud project ID (required by GEE since 2024).
             If None, tries to use the default from prior `ee.Authenticate()`.
    """
    global _initialized
    if _initialized:
        return
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        _initialized = True
        print("[gee_sar] Earth Engine initialized successfully.")
    except ee.EEException as e:
        print(f"[gee_sar] EE init failed: {e}")
        print("  Run 'earthengine authenticate' in your terminal first.")
        raise


# ── Mumbai Region of Interest ─────────────────────────────────────

def get_mumbai_bbox():
    """
    Returns the bounding box covering Greater Mumbai.
    Constructed dynamically since ee must be initialized first.
    """
    return ee.Geometry.Rectangle([72.75, 18.88, 72.99, 19.27])


# ── Core SAR Functions ────────────────────────────────────────────

def get_s1_collection(start_date: str, end_date: str,
                      geometry: ee.Geometry = None,
                      polarization: str = 'VV',
                      pass_direction: str = 'DESCENDING'):
    """
    Query Sentinel-1 GRD collection filtered by date, region, and orbit.

    Parameters:
        start_date:     'YYYY-MM-DD' start of date range
        end_date:       'YYYY-MM-DD' end of date range
        geometry:       ee.Geometry region (defaults to get_mumbai_bbox())
        polarization:   'VV' or 'VH' (VV is better for flood detection)
        pass_direction: 'ASCENDING' or 'DESCENDING'

    Returns:
        ee.ImageCollection of Sentinel-1 GRD images (single band)
    """
    if geometry is None:
        geometry = get_mumbai_bbox()

    return (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains(
                'transmitterReceiverPolarisation', polarization))
            .filter(ee.Filter.eq('orbitProperties_pass', pass_direction))
            .select(polarization))


def build_flood_mask(pre_start: str, pre_end: str,
                     post_start: str, post_end: str,
                     threshold_db: float = -3.0,
                     geometry: ee.Geometry = None):
    """
    Change-detection flood mapping.

    Steps:
      1. Median composite of pre-flood SAR images  (dry reference)
      2. Median composite of post-flood SAR images  (during/after event)
      3. Difference = post − pre  (in dB)
      4. Where difference < threshold_db → classified as flooded

    Parameters:
        pre_start / pre_end:   date range for dry-period reference
        post_start / post_end: date range for the flood event
        threshold_db:          dB drop to classify as flood (default −3.0;
                               use −5.0 for more conservative detection)
        geometry:              region of interest (defaults to Mumbai)

    Returns:
        dict with ee.Image objects:
          'pre_composite':  median VV before flood
          'post_composite': median VV during flood
          'difference':     post − pre (negative = darker = possible flood)
          'flood_mask':     binary 1 = flooded, 0 = not flooded
    """
    if geometry is None:
        geometry = get_mumbai_bbox()

    pre_collection = get_s1_collection(pre_start, pre_end, geometry)
    post_collection = get_s1_collection(post_start, post_end, geometry)

    # Median composites reduce speckle noise from single acquisitions
    pre_composite = pre_collection.median().clip(geometry)
    post_composite = post_collection.median().clip(geometry)

    # Difference in dB (negative = backscatter dropped = possible flood)
    difference = post_composite.subtract(pre_composite)

    # Threshold: where the dB drop exceeds our threshold → flood
    flood_mask = difference.lt(threshold_db).rename('flood')

    # Morphological cleanup: remove isolated flood pixels (noise)
    # and fill small holes in the flood extent
    flood_mask = flood_mask.focal_median(radius=30, units='meters')
    flood_mask = flood_mask.gt(0.5)  # re-binarize after smoothing

    return {
        'pre_composite': pre_composite,
        'post_composite': post_composite,
        'difference': difference,
        'flood_mask': flood_mask,
    }


def get_flood_at_points(flood_mask: ee.Image, points: list,
                        scale: int = 10) -> list:
    """
    Sample the flood mask at specific lat/lon points (e.g. road nodes).

    Parameters:
        flood_mask: ee.Image with binary flood values (from build_flood_mask)
        points:     list of (lat, lon) tuples
        scale:      pixel resolution in meters (Sentinel-1 = 10 m)

    Returns:
        list of float values (1.0 = flooded, 0.0 = not flooded) in the
        same order as input points. Returns None for any point that
        couldn't be sampled.
    """
    # Build a FeatureCollection of points
    features = []
    for i, (lat, lon) in enumerate(points):
        pt = ee.Geometry.Point([lon, lat])  # GEE uses [lon, lat] order!
        features.append(ee.Feature(pt, {'index': i}))

    fc = ee.FeatureCollection(features)

    # Sample the flood mask raster at each point
    sampled = flood_mask.sampleRegions(
        collection=fc,
        scale=scale,
        geometries=False,
    )

    # Pull results to Python
    results = sampled.getInfo()

    # Map back to original order
    flood_values = [None] * len(points)
    for feat in results['features']:
        idx = feat['properties']['index']
        # Find the first property key that isn't 'index' (corresponds to the image's band)
        band_keys = [k for k in feat['properties'].keys() if k != 'index']
        val = feat['properties'].get(band_keys[0], None) if band_keys else None
        flood_values[idx] = val

    return flood_values


def get_flood_at_points_batched(flood_mask: ee.Image, points: list,
                                batch_size: int = 5000,
                                scale: int = 10) -> list:
    """
    Same as get_flood_at_points, but processes in batches to avoid
    GEE's per-request size limits. Use this for large road networks
    (Mumbai's full graph has tens of thousands of nodes).

    Parameters:
        flood_mask: ee.Image (binary flood mask)
        points:     list of (lat, lon) tuples
        batch_size: max points per GEE request (default 5000)
        scale:      pixel resolution in meters

    Returns:
        list of float values, same length and order as `points`
    """
    all_results = [None] * len(points)

    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        batch_points = points[start:end]

        print(f"[gee_sar] Sampling batch {start}-{end} of {len(points)} points...")

        # Build features for this batch
        features = []
        for i, (lat, lon) in enumerate(batch_points):
            pt = ee.Geometry.Point([lon, lat])
            features.append(ee.Feature(pt, {'index': i}))

        fc = ee.FeatureCollection(features)
        sampled = flood_mask.sampleRegions(
            collection=fc, scale=scale, geometries=False
        )
        results = sampled.getInfo()

        for feat in results['features']:
            local_idx = feat['properties']['index']
            band_keys = [k for k in feat['properties'].keys() if k != 'index']
            val = feat['properties'].get(band_keys[0], None) if band_keys else None
            all_results[start + local_idx] = val

    return all_results


def build_flood_history(events: list, geometry: ee.Geometry = None,
                        threshold_db: float = -3.0):
    """
    Build a cumulative flood-frequency map from multiple historical
    flood events: for each pixel, count how many events out of N
    resulted in flooding → value between 0.0 and 1.0.

    This is the key input for training an ML risk model — locations
    that flood repeatedly have high historical flood frequency.

    Parameters:
        events: list of dicts, each with keys:
                'pre_start', 'pre_end', 'post_start', 'post_end'
                (see MUMBAI_FLOOD_EVENTS for format)
        geometry:     region (defaults to Mumbai)
        threshold_db: dB threshold for each event's flood mask

    Returns:
        ee.Image with a single band 'flood_frequency' in [0, 1]
    """
    if geometry is None:
        geometry = get_mumbai_bbox()

    masks = []
    for event in events:
        result = build_flood_mask(
            pre_start=event['pre_start'], pre_end=event['pre_end'],
            post_start=event['post_start'], post_end=event['post_end'],
            threshold_db=threshold_db, geometry=geometry,
        )
        masks.append(result['flood_mask'].toFloat())

    # Stack all masks and compute mean (= fraction of events flooded)
    stacked = ee.ImageCollection(masks).mean()
    return stacked.rename('flood_frequency').clip(geometry)


def export_flood_geotiff(flood_mask: ee.Image, filename: str,
                         geometry: ee.Geometry = None, scale: int = 10):
    """
    Export the flood mask as a GeoTIFF to Google Drive (async task).

    You'll need to download it from Drive after the export completes
    (typically takes a few minutes for Mumbai-sized areas at 10 m).

    Parameters:
        flood_mask: ee.Image to export
        filename:   output file name (no extension)
        geometry:   export region (defaults to Mumbai)
        scale:      output resolution in meters

    Returns:
        ee.batch.Task — monitor at https://code.earthengine.google.com/tasks
    """
    if geometry is None:
        geometry = get_mumbai_bbox()

    task = ee.batch.Export.image.toDrive(
        image=flood_mask.toFloat(),
        description=filename,
        folder='AquaRoute_SAR',
        fileNamePrefix=filename,
        region=geometry,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e9,
    )
    task.start()
    print(f"[gee_sar] Export started: {filename}")
    print(f"  Check progress at https://code.earthengine.google.com/tasks")
    return task


# ── Known Mumbai Flood Events ─────────────────────────────────────

# Real heavy-flood dates for Mumbai — use these as labeled events
# for SAR change-detection and ML training data.
MUMBAI_FLOOD_EVENTS = [
    {
        'name': '2024 Mumbai Monsoon Flooding (Sep)',
        'pre_start': '2024-07-01', 'pre_end': '2024-07-31',
        'post_start': '2024-09-20', 'post_end': '2024-09-30',
    },
    {
        'name': '2023 Mumbai Monsoon Flooding (Jul)',
        'pre_start': '2023-05-01', 'pre_end': '2023-06-15',
        'post_start': '2023-07-10', 'post_end': '2023-07-25',
    },
    {
        'name': '2020 Mumbai Floods (Aug)',
        'pre_start': '2020-06-01', 'pre_end': '2020-06-30',
        'post_start': '2020-08-01', 'post_end': '2020-08-10',
    },
    {
        'name': '2019 Mumbai Floods (Aug–Sep)',
        'pre_start': '2019-06-01', 'pre_end': '2019-07-15',
        'post_start': '2019-08-01', 'post_end': '2019-09-05',
    },
    {
        'name': '2017 Mumbai Floods (Aug 29)',
        'pre_start': '2017-07-01', 'pre_end': '2017-08-15',
        'post_start': '2017-08-27', 'post_end': '2017-09-05',
    },
]


# ── Quick-test / demo usage ───────────────────────────────────────

if __name__ == '__main__':
    """
    Run this file standalone to test your GEE setup and see flood
    detection results for a known Mumbai flood event.

    Usage:
        python gee_sar.py

    If you need to pass your GCP project ID, edit the init_gee() call
    below or set it via: earthengine set_project YOUR_PROJECT_ID
    """
    init_gee()  # pass project='your-gcp-project-id' if needed

    # ── Test with the 2020 flood event ──
    event = MUMBAI_FLOOD_EVENTS[2]  # 2020
    print(f"\n[demo] Processing: {event['name']}")

    result = build_flood_mask(
        pre_start=event['pre_start'], pre_end=event['pre_end'],
        post_start=event['post_start'], post_end=event['post_end'],
        threshold_db=-3.0,
    )

    # Check how many SAR images went into each composite
    pre_count = get_s1_collection(
        event['pre_start'], event['pre_end']).size().getInfo()
    post_count = get_s1_collection(
        event['post_start'], event['post_end']).size().getInfo()
    print(f"  Pre-flood images:  {pre_count}")
    print(f"  Post-flood images: {post_count}")

    # ── Sample known flood-prone locations ──
    test_points = [
        (19.0176, 72.8435),   # Hindmata Junction (notoriously floods)
        (19.0330, 72.8490),   # Sion
        (19.0760, 72.8777),   # CST area
        (19.1197, 72.9052),   # Kurla
        (19.1860, 72.8346),   # Andheri subway
    ]

    flood_vals = get_flood_at_points(result['flood_mask'], test_points)

    print(f"\n  Flood detection at known flood-prone locations:")
    labels = ['Hindmata', 'Sion', 'CST', 'Kurla', 'Andheri']
    for label, (lat, lon), val in zip(labels, test_points, flood_vals):
        status = 'FLOODED' if val and val > 0.5 else 'dry'
        print(f"    {label:12s} ({lat:.4f}, {lon:.4f}): {status}")

    # ── Build flood-frequency map from all events ──
    print(f"\n[demo] Building flood-frequency map from {len(MUMBAI_FLOOD_EVENTS)} events...")
    freq_map = build_flood_history(MUMBAI_FLOOD_EVENTS)
    freq_vals = get_flood_at_points(freq_map, test_points)

    print(f"  Historical flood frequency (0=never, 1=every event):")
    for label, val in zip(labels, freq_vals):
        pct = f"{val*100:.0f}%" if val is not None else "N/A"
        print(f"    {label:12s}: {pct}")

    print("\n[demo] Done. If the above looks reasonable, your GEE setup is working!")
    print("  Next step: wire this into risk.py to enhance flood risk scoring.")

"""
extract_waterlogging.py
Builds mumbai_waterlogging.json from documented chronic flooding spots.

WHAT THIS IS, PRECISELY
-----------------------
BMC has identified 386 chronic flooding spots (up from 225 in 2017 and
66 in 2016), but that list is not published in any machine-readable
form - it exists in pre-monsoon SWD reports, RTI responses and press
coverage. The seed list below is roughly 25 spots compiled from
published reporting, not BMC's full register.

So: this is real, attributable data, but it is a SAMPLE, not the
dataset. Treat it as a bootstrap that makes the waterlogging factor
functional, and replace it with the full list when you obtain it via:

  - RTI to the BMC Storm Water Drains department
  - the annual pre-monsoon preparedness report
  - cityresource.in/MumbaiFloods (chronic flooding spots as open GIS)

Each entry carries its severity and a source note so the provenance
stays visible. Severity reflects how consistently the location appears
in flood reporting, not a measured depth - nobody publishes depths.

Coordinates are for well-known junctions and should be spot-checked
against the map before you rely on them.

Run:
    python extract_waterlogging.py
"""

import json
import math

import numpy as np

OUTPUT = "mumbai_waterlogging.json"

# Influence radius. Waterlogging at a junction affects its approaches,
# so the signal decays over this distance rather than marking a point.
INFLUENCE_M = 400.0

# (name, lat, lon, severity 0-1, note)
CHRONIC_SPOTS = [
    # --- Island city ---
    ("Hindmata Junction, Dadar", 19.0060, 72.8360, 1.00,
     "Saucer-shaped depression; described as the single worst flooding "
     "point, floods every year, water has reached waist to chest height"),
    ("Gandhi Market, Sion", 19.0333, 72.8590, 0.95,
     "Chronic spot; overhead pipeline plus pumps failed to reduce flooding"),
    ("Sion Circle", 19.0390, 72.8619, 0.90,
     "Floods from Mithi River overflow"),
    ("King's Circle, Matunga", 19.0270, 72.8570, 0.90,
     "Mithi overflow; reliably underwater in major events"),
    ("Dadar TT", 19.0180, 72.8430, 0.80, "Reported in most heavy spells"),
    ("Parel", 19.0000, 72.8380, 0.80,
     "Holding pond built beneath St Xavier's school grounds"),
    ("Lalbaug", 18.9960, 72.8340, 0.75, "Drains toward Hindmata"),
    ("Elphinstone Road / Prabhadevi", 19.0100, 72.8300, 0.70,
     "Culvert works; open-manhole fatality here in 2017"),
    ("Matunga", 19.0270, 72.8560, 0.75, "Named among chronic zones"),
    ("Mahalaxmi", 18.9820, 72.8230, 0.60, "Major culvert cleared at station"),
    ("Byculla", 18.9760, 72.8330, 0.55, "Low-lying island-city stretch"),
    ("Wadala", 19.0170, 72.8590, 0.55, "Low-lying, near Mithi outfall"),

    # --- Western suburbs ---
    ("Milan Subway, Santacruz", 19.0776, 72.8420, 0.85,
     "Chronic; 2-crore-litre holding pond installed, impassable within "
     "hours of heavy rain before that"),
    ("Andheri Subway", 19.1190, 72.8470, 0.90,
     "Closed to traffic overnight in first 2022 heavy spell"),
    ("Khar Subway", 19.0700, 72.8390, 0.85,
     "Submerged to waist height despite BMC mitigation claims"),
    ("Lokhandwala, Andheri West", 19.1380, 72.8250, 0.75,
     "Almost all chronic spots flooded in first 2022 spell"),
    ("Oshiwara", 19.1470, 72.8320, 0.75,
     "Reported alongside Lokhandwala; near Oshiwara river"),
    ("SV Road, Bandra", 19.0550, 72.8400, 0.65,
     "Water stagnation reported along Bandra-Dahisar corridor"),
    ("Juhu", 19.1080, 72.8260, 0.60,
     "Worsened by Metro roadworks causing stagnation"),
    ("Malad Subway", 19.1860, 72.8480, 0.70, "Recurrent western-suburb spot"),
    ("Dahisar", 19.2500, 72.8590, 0.55, "Northern end of SV Road corridor"),

    # --- Eastern suburbs ---
    ("Kurla Junction", 19.0654, 72.8794, 0.85,
     "Mithi corridor; track flooding halts Central and Harbour lines"),
    ("Chembur", 19.0620, 72.8990, 0.70, "Floods regularly"),
    ("Ghatkopar", 19.0860, 72.9080, 0.65,
     "Named in Kurla-Vidyavihar-Ghatkopar-Chembur flooding belt"),
    ("Vidyavihar", 19.0790, 72.8970, 0.65, "Same belt as above"),
    ("Vikhroli", 19.1100, 72.9250, 0.50, "Eastern low-lying stretch"),
    ("Mankhurd", 19.0480, 72.9300, 0.55, "Low-lying, tidal influence"),
]


def main():
    from graph_utils import get_graph

    print(f"[waterlog] Seed list: {len(CHRONIC_SPOTS)} documented spots.")
    print("[waterlog] Loading road graph...")
    G = get_graph(mode="city")
    node_ids = list(G.nodes())
    lats = np.array([G.nodes[n]["y"] for n in node_ids])
    lons = np.array([G.nodes[n]["x"] for n in node_ids])
    mean_lat = float(lats.mean())
    scale = math.cos(math.radians(mean_lat))

    score = np.zeros(len(node_ids))

    print("[waterlog] Applying influence fields...")
    for name, lat, lon, severity, _note in CHRONIC_SPOTS:
        dx = (lons - lon) * 111_320 * scale
        dy = (lats - lat) * 110_540
        dist = np.hypot(dx, dy)

        # Gaussian falloff reaching ~0 at INFLUENCE_M. A hard cutoff
        # would create a visible ring of risk around each junction.
        influence = severity * np.exp(-((dist / (INFLUENCE_M / 2)) ** 2))
        influence[dist > INFLUENCE_M * 1.5] = 0.0

        # Max, not sum: two nearby spots should not push a node above
        # the severity of the worse one.
        score = np.maximum(score, influence)

        hit = int((influence > 0.05).sum())
        print(f"  {name:34s} sev {severity:.2f}  {hit:5d} nodes")

    out = {}
    for nid, s in zip(node_ids, score):
        if s > 0.02:
            out[str(nid)] = round(float(s), 3)

    with open(OUTPUT, "w") as f:
        json.dump(out, f)

    total = len(node_ids)
    print(f"\n[waterlog] Wrote {OUTPUT}")
    print(f"  nodes with nonzero waterlogging risk: {len(out)} "
          f"({len(out) * 100.0 / total:.1f}%)")
    print(f"  peak score: {score.max():.3f}")
    print("\nThis is a ~25-spot sample of BMC's 386. Replace it with the "
          "full register when you get it.")


if __name__ == "__main__":
    main()

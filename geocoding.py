"""
geocoding.py
Address and place lookup tuned for Mumbai.

The previous implementation called Nominatim with limit=1 and returned
the single top hit. That fails badly on the thing people actually type:
building and society names. "Sai Krupa CHS", "Lodha Amara", "Hiranandani
Gardens" either miss entirely or resolve to a same-named place in
another city, and with limit=1 the user never sees that it went wrong.

Three changes:

  1. Two providers. Photon (Komoot) is built for type-ahead and is
     far more forgiving of partial and misspelt input, which is what
     autocomplete needs. Nominatim is stricter but has better coverage
     of formal addresses. Query both, merge, dedupe.

  2. Candidates, not an answer. Return a ranked list so the UI can
     show a picker. Guessing silently is worse than asking.

  3. Mumbai bias with a hard distance gate. Results outside the road
     graph's extent are dropped, because a coordinate we cannot route
     from is not a useful answer.

Both providers are free and need no API key. Nominatim's policy asks
for a real User-Agent and roughly one request a second, so the
autocomplete path prefers Photon and only falls back to Nominatim.
"""

import math
import time
from typing import List, Optional

import requests

TIMEOUT = 6
USER_AGENT = "floodreroute/0.2 (student project; Mumbai flood routing)"

# Road graph extent, from mumbai_graph_cache.graphml. A result outside
# this box cannot be routed, so it is not offered.
GRAPH_BOUNDS = {"min_lat": 18.70, "max_lat": 19.53,
                "min_lon": 72.74, "max_lon": 73.45}

MUMBAI_CENTER = (19.076, 72.877)

_last_nominatim_call = 0.0


class Place:
    """One geocoding candidate."""

    __slots__ = ("name", "context", "lat", "lon", "kind", "source", "score")

    def __init__(self, name, context, lat, lon, kind, source, score=0.0):
        self.name = name
        self.context = context
        self.lat = lat
        self.lon = lon
        self.kind = kind
        self.source = source
        self.score = score

    def as_dict(self):
        return {
            "name": self.name,
            "context": self.context,
            "lat": self.lat,
            "lon": self.lon,
            "kind": self.kind,
            "source": self.source,
        }

    @property
    def key(self):
        """Dedupe key: same name within ~100 m is the same place."""
        return (self.name.lower().strip(), round(self.lat, 3), round(self.lon, 3))


def _in_graph(lat: float, lon: float) -> bool:
    b = GRAPH_BOUNDS
    return (b["min_lat"] <= lat <= b["max_lat"]
            and b["min_lon"] <= lon <= b["max_lon"])


def _km_from_center(lat: float, lon: float) -> float:
    clat, clon = MUMBAI_CENTER
    dlat = (lat - clat) * 111.0
    dlon = (lon - clon) * 111.0 * math.cos(math.radians(clat))
    return math.hypot(dlat, dlon)


def _photon(query: str, limit: int = 8) -> List[Place]:
    """
    Komoot's Photon. Built on OSM data but indexed for type-ahead, so
    it tolerates partial words - "hiranan" finds Hiranandani.
    """
    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={
                "q": query,
                "limit": limit,
                "lat": MUMBAI_CENTER[0],
                "lon": MUMBAI_CENTER[1],
                "lang": "en",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as e:
        print(f"[geocode] Photon failed: {e}")
        return []

    out = []
    for f in features:
        try:
            lon, lat = f["geometry"]["coordinates"]
        except (KeyError, ValueError):
            continue
        p = f.get("properties", {})
        name = p.get("name") or p.get("street") or query
        context = ", ".join(
            x for x in (p.get("district"), p.get("city"), p.get("state")) if x
        )
        out.append(Place(name, context, float(lat), float(lon),
                         p.get("osm_value") or p.get("type") or "place",
                         "photon"))
    return out


def _nominatim(query: str, limit: int = 5) -> List[Place]:
    """
    OSM Nominatim. Stricter matching, better on full postal addresses.
    Rate-limited to one call per second per their usage policy.
    """
    global _last_nominatim_call
    wait = 1.05 - (time.time() - _last_nominatim_call)
    if wait > 0:
        time.sleep(wait)

    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "jsonv2",
                "limit": limit,
                "addressdetails": 1,
                "countrycodes": "in",
                "viewbox": f"{GRAPH_BOUNDS['min_lon']},{GRAPH_BOUNDS['max_lat']},"
                           f"{GRAPH_BOUNDS['max_lon']},{GRAPH_BOUNDS['min_lat']}",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
        )
        _last_nominatim_call = time.time()
        r.raise_for_status()
        results = r.json()
    except Exception as e:
        print(f"[geocode] Nominatim failed: {e}")
        return []

    out = []
    for res in results:
        try:
            lat, lon = float(res["lat"]), float(res["lon"])
        except (KeyError, ValueError):
            continue
        addr = res.get("address", {})
        name = res.get("name") or res.get("display_name", "").split(",")[0]
        context = ", ".join(
            x for x in (addr.get("suburb"), addr.get("city_district"),
                        addr.get("city"), addr.get("state")) if x
        )
        out.append(Place(name, context, lat, lon,
                         res.get("type", "place"), "nominatim"))
    return out


def _rank(places: List[Place], query: str) -> List[Place]:
    """
    Score candidates. Closeness to the city centre matters because
    people typing a bare society name almost always mean the local one;
    an exact name match matters more.
    """
    q = query.lower().strip()
    for p in places:
        score = 0.0
        name = p.name.lower()
        if name == q:
            score += 100
        elif name.startswith(q):
            score += 60
        elif q in name:
            score += 35

        # Nearer the centre ranks higher, tapering off past ~35 km.
        score += max(0.0, 35.0 - _km_from_center(p.lat, p.lon))

        # Buildings and residential places are usually what is meant
        # when someone types a name rather than a road.
        if p.kind in ("building", "residential", "house", "apartments",
                      "neighbourhood", "suburb"):
            score += 12

        p.score = score

    return sorted(places, key=lambda x: -x.score)


def search(query: str, limit: int = 6, fast: bool = False) -> List[Place]:
    """
    Ranked candidates for a typed query.

    fast=True queries only Photon - use it for keystroke-by-keystroke
    autocomplete, where Nominatim's one-per-second limit would stall
    the UI. fast=False queries both, for the final resolve.
    """
    query = (query or "").strip()
    if len(query) < 2:
        return []

    # Bare names get a Mumbai qualifier so "Sai Krupa" does not match a
    # society in Pune. Skip if the user already named a locality.
    hinted = query
    if not any(w in query.lower() for w in ("mumbai", "bombay", "thane",
                                            "navi", "maharashtra")):
        hinted = f"{query}, Mumbai"

    places = _photon(hinted)
    if not fast:
        places += _nominatim(hinted)
        if not places:
            places = _nominatim(query)  # retry unqualified

    # Drop anything we cannot route from, then dedupe.
    seen, kept = set(), []
    for p in places:
        if not _in_graph(p.lat, p.lon):
            continue
        if p.key in seen:
            continue
        seen.add(p.key)
        kept.append(p)

    return _rank(kept, query)[:limit]


def geocode_one(query: str) -> Optional[Place]:
    """Best single match, or None. Used when the UI needs no picker."""
    results = search(query, limit=1)
    return results[0] if results else None

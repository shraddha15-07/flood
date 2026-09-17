"""
rainfall_imd.py
Live rainfall from the India Meteorological Department.

IMD does publish an official API gateway (api.imd.gov.in, reference at
/public/api_reference.html). No key is documented, but access has been
reported to be IP-whitelisted at times, so every call here degrades
gracefully and the caller always gets a usable number.

Three sources, tried in order:

  1. Station nowcast  - IMD's own intensity bands, which map exactly
                        onto the UI's Light/Moderate/Heavy selector:
                            Cat2  light     < 5 mm/hr
                            Cat7  moderate  5-15 mm/hr
                            Cat12 heavy     > 15 mm/hr
                        This is the best fit for the risk model, since
                        it is an intensity, which is what drives
                        waterlogging.
  2. AWS/ARG stations - Automatic Weather Station / Automatic Rain
                        Gauge readings with lat/lon, so you can pick
                        the nearest gauge to the route. Note the
                        documented sample payload does NOT list a
                        rainfall field, so verify what Maharashtra
                        (sid=21) actually returns before relying on it.
  3. Open-Meteo       - already working, no whitelisting, used as the
                        fallback so routing never goes down.

Calibration note: IMD calls anything over 15 mm/hr "heavy". Mumbai
deluges routinely exceed 60 mm/hr, so the saturation point in risk.py
is set above IMD's heavy threshold rather than at it.
"""

import math
import time
from typing import Optional

import requests

IMD_BASE = "https://api.imd.gov.in/api/v1"
MAHARASHTRA_SID = 21
TIMEOUT = 8

# IMD nowcast category -> representative mm/hr (band midpoints; the
# open-ended heavy band is assigned a Mumbai-realistic value).
NOWCAST_MM_HR = {
    1: 0.0,    # no weather
    2: 2.5,    # light rain < 5
    7: 10.0,   # moderate rain 5-15
    12: 35.0,  # heavy rain > 15
}

_cache = {"value": None, "source": None, "at": 0.0}
CACHE_TTL_S = 600  # IMD nowcasts update every few hours; do not hammer it


def _get(path: str, **params):
    r = requests.get(f"{IMD_BASE}/{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def from_station_nowcast(station: str = "Mumbai") -> Optional[float]:
    """
    Nowcast intensity band for a station, converted to mm/hr.

    The payload carries Cat1..Cat19 flags plus a 'color' severity code.
    We take the highest active rain category, since a station flagged
    both moderate and heavy should be treated as heavy.
    """
    try:
        data = _get("stationnowcast", id=station)
    except Exception as e:
        print(f"[imd] station nowcast failed: {e}")
        return None

    rows = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        return None

    best = None
    for row in rows:
        for cat, mm in sorted(NOWCAST_MM_HR.items(), reverse=True):
            val = row.get(f"Cat{cat}")
            if val in (None, "", "0", 0):
                continue
            if best is None or mm > best:
                best = mm
            break
    return best


def from_aws_nearest(lat: float, lon: float,
                     sid: int = MAHARASHTRA_SID) -> Optional[float]:
    """
    Nearest AWS/ARG gauge reading for a point.

    Field naming is not fully documented, so we probe a few plausible
    rainfall keys rather than assuming one. If none are present the
    function returns None and the caller falls through - which is the
    likely outcome until you confirm the Maharashtra payload shape.
    """
    try:
        data = _get("aws_data", sid=sid)
    except Exception as e:
        print(f"[imd] AWS fetch failed: {e}")
        return None

    rows = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        return None

    rain_keys = ("RAINFALL", "RF", "Rainfall", "RAIN", "PRECIPITATION",
                 "Last 24 hrs Rainfall", "RAINFALL_24HR")

    best, best_d = None, float("inf")
    for row in rows:
        try:
            slat = float(row.get("Latitude"))
            slon = float(row.get("Longitude"))
        except (TypeError, ValueError):
            continue
        d = _haversine_km(lat, lon, slat, slon)
        if d >= best_d:
            continue
        for k in rain_keys:
            if k in row and row[k] not in (None, "", "NIL"):
                try:
                    best, best_d = float(row[k]), d
                except ValueError:
                    pass
                break

    if best is None:
        print("[imd] AWS rows carried no recognised rainfall field.")
        return None
    print(f"[imd] Nearest AWS gauge {best_d:.1f} km away: {best} mm")
    return best


def from_open_meteo(lat: float, lon: float) -> Optional[float]:
    """Fallback. Already proven to work from your environment."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "current": "precipitation", "timezone": "auto"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return float(r.json()["current"]["precipitation"])
    except Exception as e:
        print(f"[imd] Open-Meteo fallback failed: {e}")
        return None


def get_rainfall_mm_hr(lat: float = 19.076, lon: float = 72.877,
                       station: str = "Mumbai", use_cache: bool = True):
    """
    Returns (mm_hr, source). Never raises.

    Cached for 10 minutes so a burst of route requests does not turn
    into a burst of IMD requests.
    """
    now = time.time()
    if use_cache and _cache["value"] is not None \
            and now - _cache["at"] < CACHE_TTL_S:
        return _cache["value"], _cache["source"]

    for fn, name in (
        (lambda: from_station_nowcast(station), "imd_nowcast"),
        (lambda: from_aws_nearest(lat, lon), "imd_aws"),
        (lambda: from_open_meteo(lat, lon), "open_meteo"),
    ):
        val = fn()
        if val is not None:
            _cache.update({"value": val, "source": name, "at": now})
            print(f"[imd] Rainfall {val} mm/hr from {name}")
            return val, name

    _cache.update({"value": 0.0, "source": "unavailable", "at": now})
    return 0.0, "unavailable"


if __name__ == "__main__":
    print(get_rainfall_mm_hr())
# Backend patch — geocoding endpoints

Add `geocoding.py` to the backend folder, then make these three edits
to `main.py`.

## 1. Replace the geocoding import

Find:

```python
from graph_utils import get_graph, geocode_address
```

Replace with:

```python
from graph_utils import get_graph
from geocoding import search as geo_search, geocode_one
```

## 2. Replace the /geocode endpoint

Find the whole existing `geocode` function and its `GeocodeRequest`
class, and replace both with:

```python
class GeocodeRequest(BaseModel):
    address: str


@app.post("/geocode")
def geocode(req: GeocodeRequest):
    """Best single match for a typed address or place name."""
    place = geocode_one(req.address)
    if place is None:
        raise HTTPException(
            404,
            detail=f'No routable place found for "{req.address}". '
                   f"Try adding the area, e.g. \"{req.address}, Andheri\".",
        )
    return place.as_dict()


@app.get("/geocode/suggest")
def geocode_suggest(q: str = Query(..., min_length=2), limit: int = 6):
    """
    Type-ahead candidates. Photon only, so it stays fast enough to call
    on each keystroke — Nominatim's one-request-per-second policy would
    stall the input box.

    Returns a list rather than one answer on purpose: "Sai Krupa CHS"
    matches a dozen societies across the metro, and picking one for the
    user silently is how people get routed to the wrong building.
    """
    return {"results": [p.as_dict() for p in
                        geo_search(q, limit=limit, fast=True)]}
```

## 3. Delete the old helper

`geocode_address` in `graph_utils.py` is now unused. Leave the file
alone or remove that one function — nothing else imports it.

## Test it

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/geocode/suggest?q=hiranandani" |
  Select-Object -ExpandProperty results |
  Format-Table name, context, lat, lon
```

You should get several Hiranandani entries with different contexts
(Powai, Thane, and so on) — which is the point. Previously this
returned one, with no indication there were others.

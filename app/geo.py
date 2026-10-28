"""UK postcode / place name -> coordinates, via the free postcodes.io API (no key needed)."""
from typing import Optional, Tuple
from urllib.parse import quote

import httpx

API = "https://api.postcodes.io"


def geocode(text: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for a UK postcode or a place name like "King's Cross"."""
    text = text.strip()
    if not text:
        return None
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{API}/postcodes/{quote(text)}")
        if r.status_code == 200:
            res = r.json()["result"]
            return res["latitude"], res["longitude"]
        r = c.get(f"{API}/places", params={"q": text, "limit": 10})
        results = r.json().get("result") if r.status_code == 200 else None
        if results:
            # Place names repeat across the UK (there's a Shoreditch in Somerset); prefer London
            res = next((p for p in results if p.get("region") == "London"), results[0])
            return res["latitude"], res["longitude"]
    return None


def midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    # Fine at city scale; no need for great-circle maths
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

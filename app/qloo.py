"""Thin client for the Qloo Taste AI hackathon API."""
import json
import os
import time
from typing import Dict, List, Optional

import httpx

# Short names the LLM uses -> Qloo entity type URNs
ENTITY_TYPES = {
    "movie": "urn:entity:movie",
    "tv_show": "urn:entity:tv_show",
    "artist": "urn:entity:artist",
    "album": "urn:entity:album",
    "book": "urn:entity:book",
    "podcast": "urn:entity:podcast",
    "videogame": "urn:entity:videogame",
    "brand": "urn:entity:brand",
    "place": "urn:entity:place",
    "destination": "urn:entity:destination",
    "person": "urn:entity:person",
}

# Places a newcomer would not "go to" for a night out
EXCLUDED_PLACE_TAGS = "urn:tag:category:place:hotel"
# Qloo's place graph also covers offices, schools, flats...; drop them by category/genre name
NON_LEISURE = (
    "lodging", "apartment", "hotel", "hostel", "residential", "school", "education center",
    "college", "university", "agency", "office", "consultan", "clinic", "hospital", "dentist",
    "bank", "pharmacy", "estate agent", "coworking", "publisher", "charity", "banquet",
    "wedding venue", "conference", "corporate", "non-profit", "bus company", "coach", "transport",
    "travel agency", "car rental", "taxi",
)


def is_leisure(entity: dict) -> bool:
    kinds = " ".join(
        t.get("name", "").lower()
        for t in entity.get("tags") or []
        if ":category:" in t.get("type", "") or ":genre:" in t.get("type", "")
    )
    return not any(word in kinds for word in NON_LEISURE)


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=os.getenv("QLOO_BASE_URL", "https://hackathon.api.qloo.com"),
        headers={"X-Api-Key": os.environ["QLOO_API_KEY"]},
        timeout=20,
    )


_cache: Dict[str, dict] = {}
_CACHE_MAX = 2000


def _get(path: str, params: dict) -> dict:
    """GET with a small in-memory cache (demo inputs repeat a lot) and backoff on 429."""
    key = path + "?" + json.dumps(params, sort_keys=True)
    if key in _cache:
        return _cache[key]
    with _client() as c:
        for attempt in range(4):
            r = c.get(path, params=params)
            if r.status_code != 429:
                break
            time.sleep(float(r.headers.get("Retry-After") or 0.5 * 2 ** attempt))
        r.raise_for_status()
    data = r.json()
    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[key] = data
    return data


def slim(entity: dict) -> dict:
    """Keep only the fields the LLM and UI need, so prompts stay small."""
    props = entity.get("properties") or {}
    query = entity.get("query") or {}
    out = {
        "name": entity.get("name"),
        "id": entity.get("entity_id"),
        "type": (entity.get("types") or [entity.get("subtype") or entity.get("type")])[0],
        "popularity": round(entity.get("popularity") or 0, 3),
    }
    if "affinity" in query:
        out["affinity"] = round(query["affinity"], 3)
    if "distance" in query:
        out["distance_m"] = int(query["distance"])
    for key in ("address", "release_year", "neighborhood", "price_level"):
        if props.get(key):
            out[key] = str(props[key])
    description = props.get("short_description") or props.get("description")
    if description:
        out["description"] = str(description)[:200]
    tags = entity.get("tags") or []
    category = next((t["name"] for t in tags if ":category:" in t.get("type", "")), None)
    if category:
        out["category"] = category
    vibe = [t["name"] for t in tags if ":ambience:" in t.get("type", "")]
    if vibe:
        out["vibe"] = vibe[:4]
    image = (props.get("image") or {}).get("url")
    if image:
        out["image"] = image
    return out


def search_entity(query: str, type: Optional[str] = None, take: int = 3) -> List[dict]:
    """Look up an entity by name to get its Qloo ID."""
    params = {"query": query, "take": take}
    if type in ENTITY_TYPES:
        params["types"] = ENTITY_TYPES[type]
    return [slim(e) for e in _get("/search", params).get("results", [])]


def search_tags(query: str, take: int = 5) -> List[dict]:
    """Find Qloo tag IDs (e.g. 'chinese restaurant' -> urn:tag:category:place:chinese_restaurant)."""
    data = _get("/v2/tags", {"filter.query": query, "take": take})
    return [
        {"name": t.get("name"), "id": t.get("id")}
        for t in (data.get("results") or {}).get("tags", [])
    ]


def get_recommendations(
    entity_ids: List[str],
    target_type: str,
    location: Optional[str] = None,
    take: int = 5,
) -> List[dict]:
    """Recommend entities of target_type for someone who likes entity_ids."""
    params = {
        "filter.type": ENTITY_TYPES.get(target_type, target_type),
        "signal.interests.entities": ",".join(entity_ids),
        "take": take,
    }
    if location:
        params["filter.location.query"] = location
    return [slim(e) for e in _get("/v2/insights", params).get("results", {}).get("entities", [])]


def nearby_places(
    entity_ids: List[str],
    lat: float,
    lon: float,
    radius_m: int = 2000,
    popularity_max: Optional[float] = None,
    tag: Optional[str] = None,
    take: int = 20,
) -> List[dict]:
    """Places within radius_m of (lat, lon), ranked by taste affinity to entity_ids."""
    params = {
        "filter.type": "urn:entity:place",
        "signal.interests.entities": ",".join(entity_ids),
        "filter.location": f"POINT({lon} {lat})",
        "filter.location.radius": radius_m,
        "filter.exclude.tags": EXCLUDED_PLACE_TAGS,
        "take": take,
    }
    if popularity_max is not None:
        params["filter.popularity.max"] = popularity_max
    if tag:
        params["filter.tags"] = tag
    entities = _get("/v2/insights", params).get("results", {}).get("entities", [])
    return [slim(e) for e in entities if is_leisure(e)]

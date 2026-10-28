"""TasteBridge ranking engine.

Turns a newcomer's home favourites into a Comfort + Bridge plan of nearby places.
Each step follows an idea from our sequential-recommendation work:

1. Two branches, loosely coupled (LOOM): "into right now" favourites form an ordered branch
   with recency decay; "always loved" favourites form an orderless branch. Each branch is
   ranked on its own and only fused at the end, so neither one dominates the other.
2. Denoising (LLM4DSR): a favourite whose recommendations share nothing with the others
   (a gift, a one-off, a wrong entity match) is down-weighted and flagged, not trusted blindly.
3. Popularity debiasing: tourist magnets are capped and penalised, more so the more the user
   wants to explore.
4. Attribution: every recommendation records which favourite actually produced it, so the
   explanation cites a real cause instead of a generic "you might like this".
5. Group fusion for Taste Twin: least-misery aggregation, so a meetup spot has to suit both.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from . import geo, qloo

RRF_K = 10  # reciprocal-rank-fusion constant; small because lists are short (20)
RECENT_HALF_LIFE = 2.0  # in list positions: the 3rd "now" favourite counts half as much as the 1st
RECENT_BRANCH_WEIGHT = 0.6  # ordered branch vs orderless branch, when both exist
NOISE_WEIGHT = 0.3
NOISE_OVERLAP = 0.05
NOISE_RELATIVE = 0.6  # flag a seed whose overlap is under 60% of its peers' median
NOISE_MIN_PEER = 0.15  # ...but only if the peers themselves agree this much
CAUSE_REPEAT_PENALTY = 0.35  # per earlier pick explained by the same favourite
QLOO_PARALLEL = 4  # per request; the hackathon key rate-limits bursts


def resolve(items: List[dict]) -> Tuple[List[dict], List[str]]:
    """Map {"name", "type"?, "when"} favourites to Qloo entities. Returns (seeds, unresolved names)."""

    def one(item):
        hits = qloo.search_entity(item["name"], item.get("type"), take=1)
        return item, hits[0] if hits else None

    seeds, missing = [], []
    with ThreadPoolExecutor(QLOO_PARALLEL) as pool:
        for item, hit in pool.map(one, items):
            if hit:
                seeds.append({
                    "input": item["name"],
                    "name": hit["name"],
                    "id": hit["id"],
                    "type": hit["type"],
                    "when": item.get("when", "always"),
                })
            else:
                missing.append(item["name"])
    return seeds, missing


def _per_seed_lists(seeds, lat, lon, radius_m, popularity_max, tag=None) -> Dict[str, List[dict]]:
    def one(seed):
        return seed["id"], qloo.nearby_places([seed["id"]], lat, lon, radius_m, popularity_max, tag)

    with ThreadPoolExecutor(QLOO_PARALLEL) as pool:
        return dict(pool.map(one, seeds))


def _denoise(seeds: List[dict], lists: Dict[str, List[dict]]) -> None:
    """Leave-one-out consistency: flag a seed whose list barely overlaps every other seed's."""
    if len(seeds) < 3:
        for s in seeds:
            s["weight"], s["noisy"] = 1.0, False
        return
    ids = {s["id"]: {p["id"] for p in lists[s["id"]]} for s in seeds}
    for s in seeds:
        own = ids[s["id"]]
        others = set().union(*(v for k, v in ids.items() if k != s["id"]))
        s["consistency"] = round(len(own & others) / max(len(own), 1), 2)
    # Noise is the odd one out, never the majority: flag at most the single least consistent
    # seed, and only when the others agree with each other clearly more than it does
    for s in seeds:
        s["noisy"], s["weight"] = False, 1.0
    worst = min(seeds, key=lambda s: s["consistency"])
    peers = sorted(t["consistency"] for t in seeds if t is not worst)
    median = peers[len(peers) // 2]
    if median >= NOISE_MIN_PEER and worst["consistency"] < max(NOISE_OVERLAP, NOISE_RELATIVE * median):
        worst["noisy"], worst["weight"] = True, NOISE_WEIGHT


def _branch_scores(seeds, lists) -> Dict[str, Dict[str, float]]:
    """RRF within one branch. Returns place_id -> {seed_id: contribution}."""
    contrib: Dict[str, Dict[str, float]] = {}
    for i, s in enumerate(seeds):
        decay = 0.5 ** (i / RECENT_HALF_LIFE) if s["when"] == "now" else 1.0
        for rank, place in enumerate(lists[s["id"]]):
            c = s["weight"] * decay / (RRF_K + rank + 1)
            contrib.setdefault(place["id"], {})[s["id"]] = c
    return contrib


def taste_scores(seeds, lists, explore: float) -> Dict[str, dict]:
    """Fuse the two branches, debias by popularity, and keep per-seed attribution."""
    places = {p["id"]: p for lst in lists.values() for p in lst}
    branches = {
        "now": [s for s in seeds if s["when"] == "now"],
        "always": [s for s in seeds if s["when"] != "now"],
    }
    weights = {"now": RECENT_BRANCH_WEIGHT, "always": 1 - RECENT_BRANCH_WEIGHT}
    if not branches["now"] or not branches["always"]:
        weights = {"now": 1.0, "always": 1.0}

    fused: Dict[str, Dict[str, float]] = {}
    for name, branch_seeds in branches.items():
        if not branch_seeds:
            continue
        contrib = _branch_scores(branch_seeds, lists)
        top = max((sum(c.values()) for c in contrib.values()), default=1.0)
        for pid, c in contrib.items():  # normalise per branch so branch sizes don't matter
            for sid, v in c.items():
                fused.setdefault(pid, {})[sid] = fused.get(pid, {}).get(sid, 0) + weights[name] * v / top

    penalty = 0.3 + 0.5 * explore
    by_name = {s["id"]: s["name"] for s in seeds}
    out = {}
    for pid, c in fused.items():
        place = places[pid]
        raw = sum(c.values())
        cause = max(c, key=c.get)
        out[pid] = {
            **place,
            "score": round(raw * (1 - penalty * place.get("popularity", 0)), 4),
            "because": by_name[cause],
            "because_share": round(c[cause] / raw, 2),
            "supported_by": [by_name[s] for s in sorted(c, key=c.get, reverse=True)],
        }
    return out


def diversify(candidates: List[dict], k: int) -> List[dict]:
    """Greedy re-rank so the plan isn't all explained by one favourite (MMR over causes)."""
    pool, picked, used = list(candidates), [], {}
    while pool and len(picked) < k:
        best = max(pool, key=lambda p: p["score"] * (1 - CAUSE_REPEAT_PENALTY * used.get(p["because"], 0)))
        pool.remove(best)
        picked.append(best)
        used[best["because"]] = used.get(best["because"], 0) + 1
    return picked


def _comfort(seeds, lat, lon, home_food: Optional[str]) -> Tuple[Optional[dict], List[dict]]:
    """Home-culture spots: same taste signal, restricted to a home-cuisine tag."""
    if not home_food:
        return None, []
    tags = [t for t in qloo.search_tags(home_food) if t["id"] and ":place:" in t["id"]]
    if not tags:
        return None, []
    tag = tags[0]
    good = [s["id"] for s in seeds if not s.get("noisy")] or [s["id"] for s in seeds]
    places = qloo.nearby_places(good, lat, lon, radius_m=3000, tag=tag["id"], take=10)
    return tag, places


PROFILE_DOMAINS = (("artist", "Artists"), ("movie", "Films"), ("tv_show", "TV"))
# Qloo tag types that read well as "your taste is..."; genre:place etc. gives "Hotel", "Cafe"
DESCRIPTOR_TYPES = (":style:", ":genre:media", ":genre:music")


def _norm(name: str) -> str:
    # Fold case, punctuation and traditional/simplified variants we've seen in Qloo names
    return "".join(ch for ch in name.lower().replace("倫", "伦") if ch.isalnum())


def taste_profile(seeds: List[dict]) -> dict:
    """Cross-domain view of the user's taste: what else people with these favourites love."""
    ids = [s["id"] for s in seeds if not s.get("noisy")] or [s["id"] for s in seeds]
    own_ids = {s["id"] for s in seeds}
    own_names = {_norm(n) for s in seeds for n in (s["name"], s["input"])}

    def is_own(e):
        # Qloo has e.g. both "Jay Chou 周杰伦" and "周杰倫"; don't recommend the user their own favourite
        n = _norm(e["name"])
        return e["id"] in own_ids or any(n and (n in o or o in n) for o in own_names)

    def domain(item):
        key, _ = item
        return key, [e for e in qloo.get_recommendations(ids, key, take=10) if not is_own(e)][:5]

    with ThreadPoolExecutor(QLOO_PARALLEL) as pool:
        tags_future = pool.submit(qloo.taste_tags, ids)
        domains = dict(pool.map(domain, PROFILE_DOMAINS))
        tags = tags_future.result()

    seen, descriptors = set(), []
    for t in tags:
        name = t["name"].strip()
        if any(kind in t["type"] for kind in DESCRIPTOR_TYPES) and _norm(name) not in seen:
            seen.add(_norm(name))  # "k pop" and "Kpop" are the same descriptor
            descriptors.append(name[:1].upper() + name[1:])
    return {
        "descriptors": descriptors[:6],
        "domains": [{"label": label, "items": [e["name"] for e in domains[key]]} for key, label in PROFILE_DOMAINS],
    }


def _noop(_event):
    pass


def _plan_for(seeds, lat, lon, explore, radius_m, tag=None):
    popularity_max = round(0.95 - 0.35 * explore, 2)
    lists = _per_seed_lists(seeds, lat, lon, radius_m, popularity_max, tag)
    _denoise(seeds, lists)
    return taste_scores(seeds, lists, explore)


def build_plan(
    favourites: List[dict],
    location: str,
    explore: float = 0.5,
    home_food: Optional[str] = None,
    radius_m: int = 2000,
    bridge_tag: Optional[str] = None,
    with_profile: bool = False,
    emit: Callable[[dict], None] = _noop,
) -> dict:
    """Comfort + Bridge plan for one person. bridge_tag (a Qloo tag id) narrows the bridge places."""
    point = geo.geocode(location)
    if not point:
        return {"error": f"Could not find location '{location}'. Try a UK postcode."}
    emit({"type": "step", "text": f"Found {location} on the map"})
    seeds, missing = resolve(favourites)
    if not seeds:
        return {"error": "None of the favourites were found in Qloo.", "unresolved": missing}
    emit({"type": "step", "text": "Matched your favourites in Qloo: " + ", ".join(s["name"] for s in seeds)})

    scored = _plan_for(seeds, *point, explore, radius_m, bridge_tag)
    noisy = [s["name"] for s in seeds if s.get("noisy")]
    emit({"type": "step", "text": (
        f"{', '.join(noisy)} doesn't fit your other favourites, so it counts less" if noisy
        else "Your favourites agree with each other, so they all count fully"
    )})
    # The profile uses the denoised seeds, so it runs after _plan_for, alongside the comfort lookup
    with ThreadPoolExecutor(2) as pool:
        profile_future = pool.submit(taste_profile, seeds) if with_profile else None
        tag, comfort = _comfort(seeds, *point, home_food)
        profile = profile_future.result() if profile_future else None

    comfort_ids = {p["id"] for p in comfort}
    bridge = diversify([p for p in scored.values() if p["id"] not in comfort_ids], k=8)
    emit({"type": "step", "text": f"Ranked {len(scored)} nearby places by your taste, tourist hotspots down-weighted"})
    n_comfort = 2 if explore < 0.34 else 1
    return {
        "location": {"query": location, "lat": point[0], "lon": point[1]},
        "seeds": seeds,
        "unresolved": missing,
        "comfort_tag": tag,
        "comfort": comfort[:3],
        "bridge": bridge,
        "profile": profile,
        "suggested_mix": {"comfort": n_comfort if comfort else 0, "bridge": 4 - n_comfort},
    }


def _common_descriptors(a_seeds: List[dict], b_seeds: List[dict]) -> List[str]:
    """Taste descriptors two people share: an honest basis for conversation starters."""
    def names(seeds):
        return [t["name"] for t in qloo.taste_tags([s["id"] for s in seeds]) if any(k in t["type"] for k in DESCRIPTOR_TYPES)]

    with ThreadPoolExecutor(2) as pool:
        a, b = pool.map(names, (a_seeds, b_seeds))
    b_lower = {n.lower() for n in b}
    return [n for n in dict.fromkeys(a) if n.lower() in b_lower][:5]


def build_twin_plan(
    me: List[dict],
    friend: List[dict],
    my_location: str,
    friend_location: str,
    explore: float = 0.5,
    radius_m: int = 1500,
    emit: Callable[[dict], None] = _noop,
) -> dict:
    """A meetup spot both people would like, near the midpoint between them."""
    a_pt, b_pt = geo.geocode(my_location), geo.geocode(friend_location)
    if not a_pt or not b_pt:
        return {"error": "Could not find one of the locations. Try UK postcodes."}
    lat, lon = geo.midpoint(a_pt, b_pt)
    emit({"type": "step", "text": "Found the midpoint between you two"})
    (a_seeds, a_missing), (b_seeds, b_missing) = resolve(me), resolve(friend)
    if not a_seeds or not b_seeds:
        return {"error": "Need at least one recognised favourite from each person."}
    emit({"type": "step", "text": "Matched both of your favourites in Qloo"})

    with ThreadPoolExecutor(2) as pool:
        fa = pool.submit(_plan_for, a_seeds, lat, lon, explore, radius_m)
        fb = pool.submit(_plan_for, b_seeds, lat, lon, explore, radius_m)
        a, b = fa.result(), fb.result()
    emit({"type": "step", "text": "Scored each place for both of you; a spot only wins if you'd both like it"})
    common = _common_descriptors(a_seeds, b_seeds)
    a_top = max((p["score"] for p in a.values()), default=1) or 1
    b_top = max((p["score"] for p in b.values()), default=1) or 1

    shared = []
    for pid in set(a) & set(b):
        na, nb = a[pid]["score"] / a_top, b[pid]["score"] / b_top
        shared.append({
            **{k: v for k, v in a[pid].items() if k not in ("score", "because", "because_share", "supported_by")},
            "score": round(min(na, nb) + 0.1 * (na + nb) / 2, 4),  # least misery, tie-broken by mean
            "because_me": a[pid]["because"],
            "because_friend": b[pid]["because"],
        })
    shared.sort(key=lambda p: p["score"], reverse=True)
    return {
        "midpoint": {"lat": lat, "lon": lon},
        "my_point": {"lat": a_pt[0], "lon": a_pt[1]},
        "friend_point": {"lat": b_pt[0], "lon": b_pt[1]},
        "common": common,
        "my_seeds": a_seeds,
        "friend_seeds": b_seeds,
        "unresolved": a_missing + b_missing,
        "shared": shared[:6],
    }

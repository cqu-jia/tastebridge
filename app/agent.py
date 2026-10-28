"""Tool-calling agent: Nemotron on Nebius Token Factory, with the TasteBridge engine and Qloo as tools.

The engine always runs first, so there is a grounded plan even if the LLM is down. The LLM then
curates it: it may re-run the engine with different settings, look up tags or cross-domain taste,
and writes the explanations. Without NEBIUS_API_KEY the app falls back to template text.
"""
import json
import os
import re
from typing import List, Optional

from openai import OpenAI

from . import engine, qloo

MAX_STEPS = 6

# Words that make an explanation read like generic AI copy rather than about this person
AI_CLICHES = [
    "vibrant", "hidden gem", "nestled", "bustling", "must-visit", "delve", "tapestry",
    "embark", "elevate", "curated", "unforgettable", "perfect blend", "something for everyone",
]

SYSTEM_PROMPT = f"""You are TasteBridge, an agent that helps international students who have just moved
to London find places that feel like theirs.

Psychology you design around:
- Acculturation research (Berry): people adapt best through *integration*, keeping home culture
  while engaging the new one. So every plan mixes COMFORT spots (a taste of home) and BRIDGE spots
  (local places that people with the same taste love).
- Optimal novelty (Berlyne): too familiar is dull, too foreign is stressful. The user's explore
  value (0 = homesick, 1 = adventurous) sets the mix.

You receive a plan computed by the TasteBridge engine from Qloo's taste graph. Each bridge place has
`because` (the favourite that produced it) and `supported_by`. Seeds flagged `noisy` looked
inconsistent with the user's other favourites.

What to do:
1. If a seed is noisy, mention it briefly and say it was given less weight.
2. If the user's note asks for something the plan doesn't cover (e.g. "quiet", "late night",
   "cheap", a different area): call search_tags to find a matching tag id, then call build_plan with
   bridge_tag set to it (and/or another location or explore value). If that returns too few places,
   keep the original plan.
3. Pick the final places: follow suggested_mix, prefer independent places over chains, and prefer
   picks explained by different favourites.
4. For each BRIDGE pick, write one sentence that cites the favourite in its `because` field and one
   concrete detail from the place data (category, vibe, description). COMFORT picks are not caused by
   a favourite: describe them as a taste of home and give one concrete detail. Never invent facts.
   The favourites are things the user likes, so phrase the link as taste ("Popular with Faye Wong
   fans: ...") and never claim anything about the artist or film themselves.

Do not use these words: {", ".join(AI_CLICHES)}.

Reply with JSON only:
{{"intro": "1-2 sentences in the user's language",
  "picks": [{{"id": "<place id>", "kind": "comfort|bridge", "why": "<one sentence>"}}],
  "note": "optional: about noisy seeds or adjustments"}}"""

TWIN_PROMPT = f"""You are TasteBridge. Two people want to meet up somewhere both would enjoy. You receive
places near their midpoint ranked with least-misery fusion (both must like it). Each place has
`because_me` and `because_friend`: the favourite of each person that led to it.

Pick the best 3 places, prefer independent places over chains. For each, write one short sentence on
why it works for both, e.g. "Popular with Jay Chou fans and Paddington fans alike; a cosy bakery 1 km
from each of you." The favourites are things the two people like (artists, films...), not people:
address the users as "you" and "your friend", never speak as if a film or character will visit.
Use only facts from the place data.

Then write 3 short, natural conversation starters a shy student could actually say, grounded in the two
people's real favourites (an overlap or an interesting contrast). One sentence each. Ask about the
other person's experience and taste ("What got you into ...?", "Which ... would you show a first-timer?").
Never mention news, releases or facts about the artists or films, and never imagine characters doing things.

Do not use these words: {", ".join(AI_CLICHES)}.

Reply with JSON only:
{{"intro": "...", "picks": [{{"id": "...", "why": "..."}}], "starters": ["...", "...", "..."]}}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "build_plan",
            "description": "Re-run the TasteBridge engine with different settings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "UK postcode or London area"},
                    "explore": {"type": "number", "description": "0 homesick .. 1 adventurous"},
                    "home_food": {"type": "string", "description": "e.g. 'chinese restaurant', 'korean'"},
                    "radius_m": {"type": "integer"},
                    "bridge_tag": {
                        "type": "string",
                        "description": "A Qloo tag id from search_tags (e.g. for 'quiet', 'live music') to narrow bridge places",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tags",
            "description": "Find Qloo tag IDs for a concept, e.g. 'quiet cafe', 'live music', 'hot pot'.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_taste",
            "description": "Cross-domain taste: what artists/movies/books people with these favourites like.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_ids": {"type": "array", "items": {"type": "string"}},
                    "target_type": {"type": "string", "enum": list(qloo.ENTITY_TYPES)},
                },
                "required": ["entity_ids", "target_type"],
            },
        },
    },
]


def llm_enabled() -> bool:
    return bool(os.getenv("NEBIUS_API_KEY") and os.getenv("NEBIUS_MODEL"))


def _llm() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        api_key=os.environ["NEBIUS_API_KEY"],
        timeout=45,
        max_retries=1,
    )


def _parse_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _compact(plan: dict) -> dict:
    """What the LLM needs to see: drop images and coordinates to keep the prompt small."""
    def strip(p):
        return {k: v for k, v in p.items() if k not in ("image", "popularity", "affinity")}

    out = {k: v for k, v in plan.items() if k not in ("comfort", "bridge", "shared", "location", "midpoint")}
    for key in ("comfort", "bridge", "shared"):
        if key in plan:
            out[key] = [strip(p) for p in plan[key]]
    return out


def _run_tools_loop(system: str, user: str, favourites: List[dict], defaults: dict, trace: list):
    """Let the LLM call tools until it answers. Returns (parsed JSON or None, latest plan or None)."""
    client, model = _llm(), os.environ["NEBIUS_MODEL"]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    latest_plan = None

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0.4
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return _parse_json(msg.content), latest_plan

        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            name, args = call.function.name, {}
            try:
                args = json.loads(call.function.arguments or "{}")
                if name == "build_plan":
                    settings = {**defaults, **{k: v for k, v in args.items() if v not in (None, "")}}
                    latest_plan = engine.build_plan(favourites, **settings)
                    result = _compact(latest_plan)
                elif name == "search_tags":
                    result = qloo.search_tags(args["query"])
                else:
                    result = qloo.get_recommendations(args["entity_ids"], args["target_type"], take=6)
            except Exception as e:  # report tool errors back so the model can recover
                result = {"error": str(e)}
            trace.append({"tool": name, "args": args})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return None, latest_plan


# ---------- template fallback (no LLM) ----------

def _why(place: dict, kind: str) -> str:
    detail = place.get("category") or ", ".join(place.get("vibe", [])[:2])
    if kind == "comfort":
        return f"A taste of home nearby: {detail.lower() or 'home-style food'}."
    tail = f" ({detail.lower()})" if detail else ""
    return f"People who love {place['because']} rate this place highly{tail}."


def _fallback_plan(plan: dict) -> dict:
    mix = plan["suggested_mix"]
    picks = [{"id": p["id"], "kind": "comfort", "why": _why(p, "comfort")} for p in plan["comfort"][: mix["comfort"]]]
    picks += [{"id": p["id"], "kind": "bridge", "why": _why(p, "bridge")} for p in plan["bridge"][: mix["bridge"]]]
    noisy = [s["input"] for s in plan["seeds"] if s.get("noisy")]
    note = f"{', '.join(noisy)} didn't fit your other favourites, so it counted less." if noisy else ""
    return {"intro": "Here's a week that mixes a taste of home with new places people like you love.", "picks": picks, "note": note}


def _fallback_twin(plan: dict) -> dict:
    picks = [
        {"id": p["id"], "why": f"Fits {p['because_me']} fans and {p['because_friend']} fans alike."}
        for p in plan["shared"][:3]
    ]
    a = [s["name"] for s in plan["my_seeds"]]
    b = [s["name"] for s in plan["friend_seeds"]]
    starters = [f"What got you into {b[0]}?", f"Have you come across {a[0]}?", "What do you miss most from home?"]
    return {"intro": "Somewhere in the middle that suits you both.", "picks": picks, "starters": starters}


# ---------- entry points ----------

def _attach(answer: dict, places: List[dict], kind_default: str) -> List[dict]:
    """Join the LLM's picks back to full place records, dropping any id it made up."""
    by_id = {p["id"]: p for p in places}
    cards = []
    for pick in answer.get("picks", []):
        place = by_id.get(pick.get("id"))
        if place:
            cards.append({**place, "kind": pick.get("kind", kind_default), "why": pick.get("why", "")})
    return cards


def plan(favourites: List[dict], location: str, explore: float, home_food: str, note: str = "") -> dict:
    defaults = {"location": location, "explore": explore, "home_food": home_food or None}
    trace = [{"tool": "build_plan", "args": defaults}]
    base = engine.build_plan(favourites, **defaults)
    if "error" in base:
        return {"error": base["error"]}

    answer, final_plan = None, base
    if llm_enabled():
        user = json.dumps({"note": note, "explore": explore, "plan": _compact(base)}, ensure_ascii=False)
        try:
            answer, newer = _run_tools_loop(SYSTEM_PROMPT, user, favourites, defaults, trace)
            final_plan = newer if newer and "error" not in newer else base
        except Exception as e:  # LLM down or misconfigured: still serve the engine's plan
            trace.append({"tool": "llm_error", "args": {"error": str(e)[:120]}})
    if not answer:
        answer = _fallback_plan(final_plan)

    places = final_plan["comfort"] + final_plan["bridge"]
    cards = _attach(answer, places, "bridge") or _attach(_fallback_plan(final_plan), places, "bridge")
    comfort_ids = {p["id"] for p in final_plan["comfort"]}
    for c in cards:  # the engine knows which list a place came from; don't trust the LLM's label
        c["kind"] = "comfort" if c["id"] in comfort_ids else "bridge"
    return {
        "intro": answer.get("intro", ""),
        "note": answer.get("note", ""),
        "cards": cards,
        "seeds": final_plan["seeds"],
        "unresolved": final_plan["unresolved"],
        "location": final_plan["location"],
        "trace": trace,
        "llm": llm_enabled(),
    }


def twin(me: List[dict], friend: List[dict], my_location: str, friend_location: str) -> dict:
    trace = [{"tool": "build_twin_plan", "args": {"my_location": my_location, "friend_location": friend_location}}]
    base = engine.build_twin_plan(me, friend, my_location, friend_location)
    if "error" in base:
        return {"error": base["error"]}

    answer = None
    if llm_enabled():
        try:
            resp = _llm().chat.completions.create(
                model=os.environ["NEBIUS_MODEL"],
                messages=[
                    {"role": "system", "content": TWIN_PROMPT},
                    {"role": "user", "content": json.dumps(_compact(base), ensure_ascii=False)},
                ],
                temperature=0.5,
            )
            answer = _parse_json(resp.choices[0].message.content)
            trace.append({"tool": "write_starters", "args": {}})
        except Exception as e:
            trace.append({"tool": "llm_error", "args": {"error": str(e)[:120]}})
    if not answer:
        answer = _fallback_twin(base)

    cards = _attach(answer, base["shared"], "shared") or _attach(_fallback_twin(base), base["shared"], "shared")
    return {
        "intro": answer.get("intro", ""),
        "cards": cards,
        "starters": answer.get("starters", []),
        "my_seeds": base["my_seeds"],
        "friend_seeds": base["friend_seeds"],
        "midpoint": base["midpoint"],
        "trace": trace,
        "llm": llm_enabled(),
    }

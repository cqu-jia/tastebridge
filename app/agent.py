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
1. If a seed is noisy, say in plain words that it didn't seem to match the user's other favourites,
   so it counted less. If none is noisy, leave note empty. Users never see the words "seed",
   "noisy", "tag", "plan" or "weight": call them favourites and talk like a friend.
2. If the user's note asks for something the plan doesn't cover (e.g. "quiet", "late night",
   "cheap", a different area): call search_tags to find a matching tag id, then call build_plan with
   bridge_tag set to it. Keep the user's location unless their note names another area. If
   build_plan answers `too_few`, the original plan stands and its places were NOT filtered for that
   request: never describe a pick as quiet, cheap etc. because of it, and say in note that you
   couldn't find enough such places nearby.
3. Pick the final places: follow suggested_mix, prefer independent places over chains, and prefer
   picks explained by different favourites.
4. For each BRIDGE pick, write one sentence that cites the favourite in its `because` field and one
   concrete detail from the place data (category, vibe, description). COMFORT picks are not caused by
   a favourite: describe them as a taste of home and give one concrete detail. Never invent facts:
   a place only fits the note's wish (quiet, good for study...) if its own data says so.
   The favourites are things the user likes, so phrase the link as taste ("Popular with Faye Wong
   fans: ...") and never claim anything about the artist or film themselves.

Do not use these words: {", ".join(AI_CLICHES)}.
Write intro, why and note in the language of the user's note (and the same script: Simplified vs
Traditional Chinese); if the note is empty, use English.

Reply with JSON only:
{{"intro": "1-2 sentences in the user's language",
  "picks": [{{"id": "<place id>", "kind": "comfort|bridge", "why": "<one sentence>"}}],
  "note": "optional: about noisy seeds or adjustments"}}"""

TWIN_PROMPT = f"""You are TasteBridge. Two people want to meet up somewhere both would enjoy. You receive
places near their midpoint ranked with least-misery fusion (both must like it). Each place has
`because_me` and `because_friend`: the favourite of each person that led to it.

Pick the best 3 places, prefer independent places over chains. For each, write one short sentence on
why it works for both, e.g. "Popular with Jay Chou fans and Paddington fans alike; a cosy bakery near
your halfway point." `distance_m` is from the halfway point, not from either person. The favourites are
things the two people like (artists, films...), not people: address the users as "you" and "your
friend", never speak as if a film or character will visit. Use only facts from the place data.

Then write 3 short, natural conversation starters a shy student could actually say, grounded in the two
people's real favourites (an overlap or an interesting contrast). One sentence each. `common` lists
taste descriptors Qloo says they share (e.g. "Artistically Rich"); use one if it helps.
The user says these lines TO the friend, so keep whose favourite is whose straight:
- `friend_favourites` are the friend's. Ask about those ("What got you into Paddington?",
  "Which Arctic Monkeys album would you show a first-timer?").
- `your_favourites` are the user's own. Only mention them as the user's ("I've had Jay Chou on repeat
  lately, have you heard him?"), never as something the friend likes.
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
        return {k: v for k, v in p.items() if k not in ("image", "popularity", "affinity", "lat", "lon")}

    skip = ("comfort", "bridge", "shared", "location", "midpoint", "my_point", "friend_point", "profile")
    out = {k: v for k, v in plan.items() if k not in skip}
    for key in ("comfort", "bridge", "shared"):
        if key in plan:
            out[key] = [strip(p) for p in plan[key]]
    return out


def _tag_label(args: dict) -> str:
    """'urn:tag:genre:place:quiet_study' -> 'quiet study'."""
    return (args.get("bridge_tag") or "").split(":")[-1].replace("_", " ")


def _narrate(name: str, args: dict, result) -> str:
    """One plain-English line per tool call, shown live in the UI."""
    if isinstance(result, dict) and "error" in result:
        return f"Nemotron tried {name} but it failed; carrying on"
    if name == "search_tags":
        found = ", ".join(list(dict.fromkeys(t["name"] for t in result))[:3]) or "nothing"
        return f"Nemotron looked up \"{args.get('query')}\" in Qloo's tags: {found}"
    if name == "build_plan":
        what = _tag_label(args) or "new settings"
        if isinstance(result, dict) and result.get("too_few"):
            return f"Nemotron tried narrowing to {what}, but too few places nearby match, so it kept the original picks"
        return f"Nemotron re-ran the plan, narrowing to {what}"
    return f"Nemotron explored {args.get('target_type', 'more')}s people with your taste love"


def _run_tools_loop(system: str, user: str, favourites: List[dict], defaults: dict, note: str, trace: list, emit,
                    min_bridge: int = 0):
    """Let the LLM call tools until it answers. Returns (parsed JSON or None, latest usable plan or None,
    names of filters that left too few places)."""
    client, model = _llm(), os.environ["NEBIUS_MODEL"]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    latest_plan, too_few = None, []

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0.4
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return _parse_json(msg.content), latest_plan, too_few

        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            name, args = call.function.name, {}
            try:
                args = json.loads(call.function.arguments or "{}")
                if name == "build_plan":
                    settings = {**defaults, **{k: v for k, v in args.items() if v not in (None, "")}}
                    # Models like to "widen" or "fix" the area; only move if the user named a new one
                    if settings["location"].strip().lower() not in note.lower():
                        settings["location"] = defaults["location"]
                    newer = engine.build_plan(favourites, **settings)
                    if "error" not in newer and len(newer["bridge"]) < min_bridge:
                        # Tell the model now, before it writes: otherwise it explains the old picks
                        # as if they matched the filter ("quiet, good for study")
                        what = _tag_label(args) or "those settings"
                        too_few.append(what)
                        result = {"too_few": True, "found": len(newer["bridge"]), "message":
                                  f"Only {len(newer['bridge'])} places match {what}; the original plan stands, unfiltered."}
                    else:
                        latest_plan = newer
                        result = _compact(newer)
                elif name == "search_tags":
                    result = qloo.search_tags(args["query"])
                else:
                    result = qloo.get_recommendations(args["entity_ids"], args["target_type"], take=6)
            except Exception as e:  # report tool errors back so the model can recover
                result = {"error": str(e)}
            trace.append({"tool": name, "args": args})
            emit({"type": "step", "text": _narrate(name, args, result)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return None, latest_plan, too_few


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


def _aliases(seed: dict) -> List[str]:
    """Names a starter might use for a favourite: what the user typed, Qloo's name, and its Latin/CJK halves."""
    names = {seed.get("input") or "", seed.get("name") or ""}
    for n in list(names):
        names.update(part.strip() for part in re.split(r"(?<=[A-Za-z])\s+(?=[^\x00-\x7f])", n))
    return [n for n in names if len(n) >= 2]


# "you" + liking verb shortly before a favourite = the starter says the friend likes it
_YOU_LIKE = r"(?:got you into|you(?:'re| are)? (?:really |also )?(?:into|enjoy|love|like)|your (?:love|taste) (?:of|for))"


def _fix_starters(starters: List[str], plan: dict) -> List[str]:
    """Replace starters that tell the friend they like one of the user's own favourites."""
    friend = {a.lower() for s in plan["friend_seeds"] for a in _aliases(s)}
    mine = [a for s in plan["my_seeds"] for a in _aliases(s) if a.lower() not in friend]
    wrong = [re.compile(_YOU_LIKE + r"[^.?!,;]{0,40}?" + re.escape(a), re.I) for a in mine]
    spare = [s for s in _fallback_twin(plan)["starters"] if s not in starters]
    fixed = []
    for line in starters:
        if any(p.search(line) for p in wrong):
            if spare:
                fixed.append(spare.pop(0))
        else:
            fixed.append(line)
    return fixed


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


def _plan_result(answer: dict, plan: dict, trace: list, final: bool) -> dict:
    places = plan["comfort"] + plan["bridge"]
    cards = _attach(answer, places, "bridge") or _attach(_fallback_plan(plan), places, "bridge")
    comfort_ids = {p["id"] for p in plan["comfort"]}
    for c in cards:  # the engine knows which list a place came from; don't trust the LLM's label
        c["kind"] = "comfort" if c["id"] in comfort_ids else "bridge"
    return {
        "intro": answer.get("intro", ""),
        "note": answer.get("note", ""),
        "cards": cards,
        "seeds": plan["seeds"],
        "unresolved": plan["unresolved"],
        "location": plan["location"],
        "profile": plan.get("profile"),
        "trace": list(trace),
        "llm": llm_enabled(),
        "final": final,
    }


def plan(favourites: List[dict], location: str, explore: float, home_food: str, note: str = "", emit=None) -> dict:
    """Engine plan first (sent as a draft), then the LLM curates it. emit() gets progress events."""
    emit = emit or (lambda _e: None)
    defaults = {"location": location, "explore": explore, "home_food": home_food or None}
    trace = [{"tool": "build_plan", "args": defaults}]
    base = engine.build_plan(favourites, **defaults, with_profile=True, emit=emit)
    if "error" in base:
        return {"error": base["error"]}
    emit({"type": "draft", "data": _plan_result(_fallback_plan(base), base, trace, final=False)})

    answer, final_plan, too_few = None, base, []
    if llm_enabled():
        emit({"type": "step", "text": "Asking Nemotron to choose and explain your picks"})
        user = json.dumps({"note": note, "explore": explore, "plan": _compact(base)}, ensure_ascii=False)
        try:
            # A narrowing tag can leave almost nothing; then the original plan is the better answer
            answer, newer, too_few = _run_tools_loop(SYSTEM_PROMPT, user, favourites, defaults, note, trace, emit,
                                                     min_bridge=base["suggested_mix"]["bridge"])
            usable = newer and "error" not in newer
            final_plan = {**newer, "profile": base["profile"]} if usable else base
        except Exception as e:  # LLM down or misconfigured: still serve the engine's plan
            trace.append({"tool": "llm_error", "args": {"error": str(e)[:120]}})
    answer = answer or _fallback_plan(final_plan)
    if too_few and final_plan is base and not answer.get("note"):
        # The page must not imply the picks match a wish we couldn't filter for
        answer = {**answer, "note": f"Not enough places nearby match \"{too_few[-1]}\", so these picks aren't filtered for it."}
    return _plan_result(answer, final_plan, trace, final=True)


def _twin_result(answer: dict, plan: dict, trace: list, final: bool) -> dict:
    cards = _attach(answer, plan["shared"], "shared") or _attach(_fallback_twin(plan), plan["shared"], "shared")
    return {
        "intro": answer.get("intro", ""),
        "cards": cards,
        "starters": _fix_starters(answer.get("starters", []), plan),
        "common": plan["common"],
        "my_seeds": plan["my_seeds"],
        "friend_seeds": plan["friend_seeds"],
        "midpoint": plan["midpoint"],
        "my_point": plan["my_point"],
        "friend_point": plan["friend_point"],
        "trace": list(trace),
        "llm": llm_enabled(),
        "final": final,
    }


def twin(me: List[dict], friend: List[dict], my_location: str, friend_location: str, emit=None) -> dict:
    emit = emit or (lambda _e: None)
    trace = [{"tool": "build_twin_plan", "args": {"my_location": my_location, "friend_location": friend_location}}]
    base = engine.build_twin_plan(me, friend, my_location, friend_location, emit=emit)
    if "error" in base:
        return {"error": base["error"]}
    emit({"type": "draft", "data": _twin_result(_fallback_twin(base), base, trace, final=False)})

    answer = None
    if llm_enabled():
        emit({"type": "step", "text": "Asking Nemotron to explain the picks and write conversation starters"})
        try:
            resp = _llm().chat.completions.create(
                model=os.environ["NEBIUS_MODEL"],
                messages=[
                    {"role": "system", "content": TWIN_PROMPT},
                    {"role": "user", "content": json.dumps({
                        "your_favourites": [s["name"] for s in base["my_seeds"]],
                        "friend_favourites": [s["name"] for s in base["friend_seeds"]],
                        **_compact(base),
                    }, ensure_ascii=False)},
                ],
                temperature=0.5,
            )
            answer = _parse_json(resp.choices[0].message.content)
            trace.append({"tool": "write_starters", "args": {}})
        except Exception as e:
            trace.append({"tool": "llm_error", "args": {"error": str(e)[:120]}})
    return _twin_result(answer or _fallback_twin(base), base, trace, final=True)

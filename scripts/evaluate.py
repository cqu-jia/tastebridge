"""Offline comparison: TasteBridge vs simpler ways to answer "where should I go?".

For 20 synthetic newcomers (different home cultures, different London areas) we take 4 picks from:
  - LLM only:          Nemotron asked directly, no tools (names checked against Qloo afterwards)
  - Qloo, city-wide:   one Qloo insights call for all favourites, filtered to "London"
  - Qloo, nearby:      the same call within 2 km of the user (no other filtering)
  - TasteBridge:       our engine (two-branch fusion, denoising, debiasing, diversity)

Metrics:
  hotspot   % of picks that are tourist magnets (Qloo popularity >= 0.9)      lower is better
  distance  mean km from the user                                              lower is better
  chains    % of picks that are chain outlets                                  lower is better
  valid     % of picks that are real leisure places we could verify in Qloo    higher is better

    .venv/bin/python scripts/evaluate.py      # writes eval/results.json, eval/summary.md, eval/chart.svg
"""
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
from app import agent, engine, geo, qloo  # noqa: E402

K = 4
OUT = Path(__file__).resolve().parent.parent / "eval"
HOTSPOT = 0.9
LONDON = (51.5074, -0.1278)
N_ROWS = 0

PERSONAS = [
    ("China", [("Jay Chou", "artist"), ("Spirited Away", "movie"), ("Faye Wong", "artist")], "WC1E 6BT", "chinese restaurant"),
    ("China", [("Mayday", "artist"), ("The Three-Body Problem", "book"), ("Nezha", "movie")], "E14 5AB", "hot pot"),
    ("Korea", [("BTS", "artist"), ("Parasite", "movie"), ("Squid Game", "tv_show")], "E1 6AN", "korean"),
    ("Korea", [("IU", "artist"), ("Crash Landing on You", "tv_show"), ("Oldboy", "movie")], "SE1 7PB", "korean"),
    ("Japan", [("Hikaru Utada", "artist"), ("Your Name", "movie"), ("Haruki Murakami", "book")], "NW1 2DB", "ramen"),
    ("India", [("A. R. Rahman", "artist"), ("3 Idiots", "movie"), ("Lagaan", "movie")], "SW9 8HE", "indian"),
    ("India", [("Arijit Singh", "artist"), ("Dangal", "movie"), ("Sacred Games", "tv_show")], "UB1 1EX", "indian"),
    ("Nigeria", [("Burna Boy", "artist"), ("Wizkid", "artist"), ("Chimamanda Ngozi Adichie", "book")], "SE15 4QL", "nigerian"),
    ("Ghana", [("Sarkodie", "artist"), ("Black Panther", "movie"), ("Stonebwoy", "artist")], "N15 4QL", "african"),
    ("Brazil", [("Anitta", "artist"), ("City of God", "movie"), ("Paulo Coelho", "book")], "SW8 1QF", "brazilian"),
    ("Mexico", [("Natalia Lafourcade", "artist"), ("Coco", "movie"), ("Roma", "movie")], "N1 9GU", "mexican"),
    ("Spain", [("Rosalia", "artist"), ("Money Heist", "tv_show"), ("Pedro Almodovar", "person")], "NW1 8NH", "spanish"),
    ("Italy", [("Maneskin", "artist"), ("Cinema Paradiso", "movie"), ("Elena Ferrante", "book")], "W8 4PT", "italian"),
    ("France", [("Daft Punk", "artist"), ("Amelie", "movie"), ("Asterix", "book")], "SW7 2AZ", "french"),
    ("Turkey", [("Tarkan", "artist"), ("Magnificent Century", "tv_show"), ("Orhan Pamuk", "book")], "N16 8JH", "turkish"),
    ("Iran", [("Googoosh", "artist"), ("A Separation", "movie"), ("Persepolis", "book")], "W2 1HB", "persian"),
    ("Vietnam", [("Son Tung M-TP", "artist"), ("The Scent of Green Papaya", "movie"), ("Pho", "brand")], "E2 8DY", "vietnamese"),
    ("Thailand", [("Lisa", "artist"), ("Bad Genius", "movie"), ("Girl From Nowhere", "tv_show")], "W1D 3QF", "thai"),
    ("USA", [("Taylor Swift", "artist"), ("Friends", "tv_show"), ("Harry Potter", "book")], "WC2R 2LS", "american"),
    ("Poland", [("Dawid Podsiadlo", "artist"), ("The Witcher", "videogame"), ("Olga Tokarczuk", "book")], "W6 0QT", "polish"),
]


def km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def raw_places(params):
    """Unfiltered Qloo insights call, as a naive integration would make it."""
    entities = qloo._get("/v2/insights", {"filter.type": "urn:entity:place", "take": K, **params})
    return [{**qloo.slim(e), "leisure": qloo.is_leisure(e)} for e in entities.get("results", {}).get("entities", [])][:K]


def llm_only(favs, location):
    """Ask the model directly, then try to find each named place in Qloo."""
    client = agent._llm()
    prompt = (f"I just moved to London and live near {location}. I love " + ", ".join(n for n, _ in favs)
              + f". Recommend exactly {K} specific places (restaurants, cafes, bars, venues...) near me that I'd like. "
              'Reply with JSON only: {"places": ["name", ...]}')
    resp = client.chat.completions.create(model=os.environ["NEBIUS_MODEL"],
                                          messages=[{"role": "user", "content": prompt}], temperature=0.3)
    names = (agent._parse_json(resp.choices[0].message.content) or {}).get("places", [])[:K]
    picks = []
    for name in names:
        hits = qloo._get("/search", {"query": f"{name} London", "types": "urn:entity:place", "take": 1}).get("results", [])
        hit = hits[0] if hits else None
        found = hit and norm(name)[:6] and norm(name)[:6] in norm(hit.get("name"))
        place = {**qloo.slim(hit), "leisure": qloo.is_leisure(hit), "llm_name": name} if found else None
        # A same-name shop in another city isn't the place the model meant
        if place and place.get("lat") is not None and km(LONDON, (place["lat"], place["lon"])) > 30:
            place = None
        picks.append(place or {"name": name, "id": None, "unverified": True})
    return picks


def tastebridge(favs, location, food):
    plan = engine.build_plan([{"name": n, "type": t} for n, t in favs], location, 0.5, food)
    mix = plan["suggested_mix"]
    picks = plan["comfort"][: mix["comfort"]] + plan["bridge"][: mix["bridge"]]
    return [{**p, "leisure": True} for p in picks][:K]  # engine output is already leisure-filtered


def run_persona(p):
    try:
        return _run_persona(p)
    except Exception as e:  # one bad persona shouldn't sink the whole run
        print(f"  {p[0]:8s} {p[2]:9s} FAILED: {e}", flush=True)
        return None


def _run_persona(p):
    home, favs, location, food = p
    point = geo.geocode(location)
    ids = [s["id"] for s in engine.resolve([{"name": n, "type": t} for n, t in favs])[0]]
    sig = {"signal.interests.entities": ",".join(ids)}
    out = {
        "LLM only": llm_only(favs, location),
        "Qloo, city-wide": raw_places({**sig, "filter.location.query": "London"}),
        "Qloo, nearby": raw_places({**sig, "filter.location": f"POINT({point[1]} {point[0]})", "filter.location.radius": 2000}),
        "TasteBridge": tastebridge(favs, location, food),
    }
    for picks in out.values():
        for x in picks:
            if x.get("lat") is not None:
                x["km"] = round(km(point, (x["lat"], x["lon"])), 2)
    print(f"  {home:8s} {location:9s} done", flush=True)
    return {"home": home, "location": location, "favourites": [n for n, _ in favs], "picks": out}


def summarise(rows):
    methods = list(rows[0]["picks"])
    summary = {}
    for m in methods:
        all_picks = [x for r in rows for x in r["picks"][m]]
        verified = [x for x in all_picks if not x.get("unverified")]
        dists = [x["km"] for x in verified if "km" in x]
        summary[m] = {
            "hotspot": 100 * sum(x.get("popularity", 0) >= HOTSPOT for x in verified) / max(len(verified), 1),
            "distance": sum(dists) / max(len(dists), 1),
            "chains": 100 * sum(engine.is_chain(x) for x in verified) / max(len(verified), 1),
            "valid": 100 * sum(1 for x in verified if x.get("leisure")) / max(len(all_picks), 1),
            "n": len(all_picks),
        }
    return summary


CHART_KEYS = ("hotspot", "distance", "valid")  # chains is near 0% for every method, so it stays table-only
METRICS = [
    ("hotspot", "Tourist hotspots", "% of picks", "lower is better", "{:.0f}%"),
    ("distance", "Distance from you", "mean km", "lower is better", "{:.1f} km"),
    ("chains", "Chain outlets", "% of picks", "lower is better", "{:.0f}%"),
    ("valid", "Real leisure places", "% verifiable in Qloo", "higher is better", "{:.0f}%"),
]


def chart(summary):
    """Small multiples, one per metric. TasteBridge is the one highlighted bar; baselines stay neutral."""
    methods = list(summary)
    metrics = [m for m in METRICS if m[0] in CHART_KEYS]
    W, gap = 1000, 18
    panel_w = (W - 48 - gap * (len(metrics) - 1)) / len(metrics)
    label_w, bar_h, bar_gap, top = 112, 24, 12, 64
    panel_h = top + len(methods) * (bar_h + bar_gap)
    H = 72 + panel_h + 18
    css = """
  .viz { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a887f; --grid:#e6e4de; --base:#b9b6ad; --accent:#2a78d6; }
  @media (prefers-color-scheme: dark) { .viz { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8e8c84; --grid:#33322f; --base:#5e5c56; --accent:#3987e5; } }
  .viz text { font-family: Inter, system-ui, -apple-system, 'Segoe UI', sans-serif; }
  .bg { fill: var(--surface); } .t1 { fill: var(--ink); font-size: 15px; font-weight: 600; }
  .t2 { fill: var(--ink2); font-size: 12.5px; } .t3 { fill: var(--muted); font-size: 11.5px; }
  .lab { fill: var(--ink2); font-size: 12.5px; } .lab.us { fill: var(--ink); font-weight: 600; }
  .val { fill: var(--ink2); font-size: 12px; } .val.us { fill: var(--ink); font-weight: 600; }
  .bar { fill: var(--base); } .bar.us { fill: var(--accent); } .axis { stroke: var(--grid); stroke-width: 1; }
"""
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" class="viz" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
             'aria-label="TasteBridge compared with LLM-only and raw Qloo baselines on three metrics">',
             f"<style>{css}</style>", f'<rect class="bg" width="{W}" height="{H}" rx="12"/>',
             f'<text class="t1" x="24" y="34">TasteBridge vs. simpler approaches, {N_ROWS} newcomer personas, {K} picks each</text>',
             '<text class="t3" x="24" y="54">Gray: baselines. Blue: TasteBridge. Hover a bar for its exact value.</text>']
    for i, (key, title, unit, better, fmt) in enumerate(metrics):
        px, py = 24 + i * (panel_w + gap), 72
        parts.append(f'<text class="t1" x="{px}" y="{py + 18}" style="font-size:14px">{title}</text>')
        parts.append(f'<text class="t3" x="{px}" y="{py + 36}">{unit} · {better}</text>')
        vmax = max(summary[m][key] for m in methods) or 1
        span = panel_w - label_w - 62
        x0 = px + label_w
        parts.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="{py + top - 6}" y2="{py + top + len(methods) * (bar_h + bar_gap) - bar_gap + 6}"/>')
        for j, m in enumerate(methods):
            v = summary[m][key]
            y = py + top + j * (bar_h + bar_gap)
            w = max(2.0, span * v / vmax)
            us = " us" if m == "TasteBridge" else ""
            parts.append(f'<text class="lab{us}" x="{x0 - 10}" y="{y + bar_h / 2 + 4}" text-anchor="end">{m}</text>')
            # 4px rounded data end, square at the baseline
            parts.append(f'<path class="bar{us}" d="M{x0},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v{bar_h - 8} a4,4 0 0 1 -4,4 h-{w - 4:.1f} z">'
                         f'<title>{m}: {fmt.format(v)}</title></path>')
            parts.append(f'<text class="val{us}" x="{x0 + w + 8:.1f}" y="{y + bar_h / 2 + 4}">{fmt.format(v)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Running {len(PERSONAS)} personas...")
    with ThreadPoolExecutor(3) as pool:  # Qloo's hackathon key rate-limits bursts
        rows = [r for r in pool.map(run_persona, PERSONAS) if r]
    global N_ROWS
    N_ROWS = len(rows)
    summary = summarise(rows)
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
    (OUT / "chart.svg").write_text(chart(summary))
    lines = ["| Method | " + " | ".join(t for _, t, *_ in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for m, s in summary.items():
        lines.append(f"| {m} | " + " | ".join(fmt.format(s[k]) for k, _, _, _, fmt in METRICS) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

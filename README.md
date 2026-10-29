# TasteBridge

**Helping international students make a new city feel like theirs.**

**Live demo: https://tastebridge-xofy.onrender.com** (free hosting: the first load after a quiet spell takes ~50 s)

Moving abroad is lonely. Research on international students links leisure to lower acculturative
stress and better integration, yet "things to do in London" lists all send you to the same tourist
spots. TasteBridge starts from what you already love back home (music, films, books, games) and uses
**Qloo's taste graph** to find:

- **A taste of home** nearby: home-cuisine places that people with your taste rate highly
- **Bridges**: local places loved by people who share your taste, explained by the favourite that led there
- **Taste Twin meetups**: a spot near the midpoint between you and a new friend that suits *both* of you,
  plus conversation starters grounded in your actual favourites

Alongside the picks you get **your taste, translated**: Qloo's descriptors for your taste (e.g. *Artistically
Rich, Lushly Animated*) and the artists, films and TV people like you love, plus a **map** of every pick
relative to home. While it works, the page shows **each step the agent takes, live**.

Six short questions collect favourites, location and this week's mood. Four mood choices, from
*Missing home a lot* to *Ready to explore*, set the mix. It is based on acculturation research (Berry):
people adapt best by *integrating*, keeping home culture while engaging with the new one. It also draws
on optimal-novelty theory (Berlyne): too familiar is dull, too foreign is stressful.

## Why it's better because of Qloo

A plain LLM recommends the famous places, and often places it can't back up: in our test below, 51 of the 80
places Nemotron named on its own could not be found in Qloo. Qloo connects taste *across domains* (Jay Chou →
a bubble-tea shop near Russell Square) and *locally* (radius, popularity and tag filters), but raw results are
still dominated by landmarks: an unfiltered query for "Jay Chou + Spirited Away" returns the Disney Store and
The Lion King. TasteBridge's pipeline on top of Qloo cuts tourist hotspots from 58–59% of picks to 5%, keeps
picks within ~1 km, and makes every pick a real, verifiable leisure place.

## How the ranking works (`app/engine.py`)

The pipeline borrows from sequential-recommendation research:

| Step | Idea |
|---|---|
| Two branches, loosely coupled | "Into it now" favourites (ordered, with recency decay) and "always loved" favourites (orderless) are ranked separately with reciprocal-rank fusion, then fused late so neither dominates |
| Denoising | Leave-one-out consistency: a favourite whose recommendations don't overlap with the others (a gift, a one-off, a wrong match) is down-weighted and shown struck through |
| Popularity debiasing | `filter.popularity.max` plus a score penalty that grows with the explore slider; chains are pushed down too |
| Attribution | Each place records which favourite contributed most, so every explanation cites a real cause; a cause-diversity re-rank (MMR) stops one favourite explaining everything |
| Group fusion | Taste Twin uses least-misery aggregation: a place scores as well as the person who'd like it less |

On top sits a tool-calling agent (`app/agent.py`): **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`)
on **Nebius Token Factory**, called through its OpenAI-compatible endpoint with the standard OpenAI SDK. It
curates the engine's plan, can re-run it (other area, mood, cuisine), look up Qloo tags or cross-domain taste,
and writes explanations that must cite the user's favourites (a banned-phrase list keeps out generic AI copy).
If the LLM is unavailable, the app still works with template explanations.

The engine's plan streams to the page first (about 5 s, as server-sent events), then the agent's tool calls
appear live and its curated version replaces the draft. Requests are rate-limited per visitor and per day so
the public demo can't exhaust the credits.

```
favourites ─► Qloo /search ─► per-favourite Qloo /v2/insights (near you, capped popularity)
          ─► two-branch fusion ─► denoise ─► debias ─► attribute ─► diversify
          ─► Nemotron agent (tools: build_plan, search_tags, expand_taste) ─► plan + explanations
```

## Does it actually help? A 20-persona comparison

We ran 20 synthetic newcomers (China, Korea, Japan, India, Nigeria, Ghana, Brazil, Mexico, Spain, Italy,
France, Turkey, Iran, Vietnam, Thailand, the US, Poland; each with 3 favourites and a London postcode)
through four ways of answering "where should I go?", 4 picks each (`scripts/evaluate.py`):

- **LLM only**: Nemotron asked directly, no tools; we then looked each named place up in Qloo
- **Qloo, city-wide**: one insights call with all favourites, filtered to "London"
- **Qloo, nearby**: the same call within 2 km of the user
- **TasteBridge**: our engine (no LLM step, so the result is deterministic)

![TasteBridge vs baselines](eval/chart.svg)

| Method | Tourist hotspots | Distance from you | Chain outlets | Real leisure places |
|---|---|---|---|---|
| LLM only | 62% | 2.5 km | 0% | 34% |
| Qloo, city-wide | 59% | 5.5 km | 0% | 81% |
| Qloo, nearby | 58% | 1.1 km | 0% | 81% |
| TasteBridge | 5% | 1.2 km | 0% | 100% |

- *Tourist hotspots* = Qloo popularity ≥ 0.9. Raw Qloo results are dominated by Big Ben, Piccadilly Circus
  and Tower Bridge; TasteBridge's 5% are all deliberate "taste of home" picks (a well-known home-food place is fine when you're homesick).
- *Real leisure places* = places we can verify in Qloo that aren't hotels, offices, schools etc. 51 of the
  80 places the LLM named could not be found in Qloo at all. TasteBridge scores 100% partly by construction:
  every pick comes from Qloo and passes its leisure filter.
- Distance: raw nearby Qloo is as close as TasteBridge (1.1 vs 1.2 km); the gain is in *what* is nearby.
- Caveats: synthetic personas, one run, no human ratings yet. Raw data in `eval/results.json`.

## Run locally

Team members: see [CONTRIBUTING.md](CONTRIBUTING.md) for the full setup, Git workflow and deployment guide (in Chinese).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add QLOO_API_KEY; NEBIUS_* is optional
python scripts/list_models.py   # with a Nebius key: pick a Nemotron model ID for NEBIUS_MODEL
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Deploy (Render)

Push to GitHub, then in Render choose **New → Blueprint** and select this repo (`render.yaml`).
Set `QLOO_API_KEY`, `NEBIUS_API_KEY` and `NEBIUS_MODEL` in the dashboard.

## Files

- `app/engine.py`: ranking pipeline
- `app/agent.py`: Nemotron tool-calling agent + no-LLM fallback
- `app/qloo.py`: Qloo client (search, tags, insights)
- `app/geo.py`: postcode → coordinates via postcodes.io
- `app/main.py`: FastAPI (`/api/plan`, `/api/twin`, and `/stream` variants), rate limiting
- `static/index.html`: UI (Leaflet map, live agent steps)
- `scripts/evaluate.py`: the 20-persona comparison below

## License

MIT

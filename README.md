# TasteBridge

**Helping international students make a new city feel like theirs.**

Moving abroad is lonely. Research on international students links leisure to lower acculturative
stress and better integration, yet "things to do in London" lists all send you to the same tourist
spots. TasteBridge starts from what you already love back home (music, films, books, games) and uses
**Qloo's taste graph** to find:

- **A taste of home** nearby: home-cuisine places that people with your taste rate highly
- **Bridges**: local places loved by people who share your taste, explained by the favourite that led there
- **Taste Twin meetups**: a spot near the midpoint between you and a new friend that suits *both* of you,
  plus conversation starters grounded in your actual favourites

A slider from *Homesick* to *Adventurous* sets the mix. It is based on acculturation research (Berry):
people adapt best by *integrating*, keeping home culture while engaging with the new one. It also draws
on optimal-novelty theory (Berlyne): too familiar is dull, too foreign is stressful.

## Why it's better because of Qloo

A plain LLM recommends the famous places. Qloo connects taste *across domains* (Jay Chou → a bubble-tea
shop near Russell Square), and *locally* (radius, popularity and tag filters). We tested this: an unfiltered
place query for "Jay Chou + Spirited Away" in London returns the Disney Store and The Lion King. TasteBridge's
pipeline returns independent spots within 2 km.

## How the ranking works (`app/engine.py`)

The pipeline borrows from sequential-recommendation research:

| Step | Idea |
|---|---|
| Two branches, loosely coupled | "Into it now" favourites (ordered, with recency decay) and "always loved" favourites (orderless) are ranked separately with reciprocal-rank fusion, then fused late so neither dominates |
| Denoising | Leave-one-out consistency: a favourite whose recommendations don't overlap with the others (a gift, a one-off, a wrong match) is down-weighted and shown struck through |
| Popularity debiasing | `filter.popularity.max` plus a score penalty that grows with the explore slider |
| Attribution | Each place records which favourite contributed most, so every explanation cites a real cause; a cause-diversity re-rank (MMR) stops one favourite explaining everything |
| Group fusion | Taste Twin uses least-misery aggregation: a place scores as well as the person who'd like it less |

On top sits a tool-calling agent (`app/agent.py`): an NVIDIA Nemotron model on **Nebius Token Factory**
curates the engine's plan, can re-run it (other area, mood, cuisine), look up Qloo tags or cross-domain taste,
and writes explanations that must cite the user's favourites (a banned-phrase list keeps out generic AI copy).
If the LLM is unavailable, the app still works with template explanations.

```
favourites ─► Qloo /search ─► per-favourite Qloo /v2/insights (near you, capped popularity)
          ─► two-branch fusion ─► denoise ─► debias ─► attribute ─► diversify
          ─► Nemotron agent (tools: build_plan, search_tags, expand_taste) ─► plan + explanations
```

## Run locally

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
- `app/main.py`: FastAPI (`/api/plan`, `/api/twin`)
- `static/index.html`: UI

## License

MIT

"""FastAPI server: serves the page plus /api/plan (one person) and /api/twin (meetup)."""
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from . import agent  # noqa: E402

STATIC = Path(__file__).parent.parent / "static"

app = FastAPI(title="TasteBridge")


class Favourite(BaseModel):
    name: str
    type: Optional[str] = None
    when: str = "always"  # "now" = into it right now, "always" = long-time favourite


class PlanRequest(BaseModel):
    favourites: List[Favourite] = Field(min_length=1, max_length=8)
    location: str
    explore: float = Field(0.5, ge=0, le=1)
    home_food: str = ""
    note: str = ""


class TwinRequest(BaseModel):
    me: List[Favourite] = Field(min_length=1, max_length=8)
    friend: List[Favourite] = Field(min_length=1, max_length=8)
    my_location: str
    friend_location: str


def _check_env():
    if not os.getenv("QLOO_API_KEY"):
        raise HTTPException(500, "Missing environment variable: QLOO_API_KEY")


def _run(fn, *args):
    _check_env()
    try:
        result = fn(*args)
    except Exception as e:
        raise HTTPException(502, f"Upstream error: {e}")
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "llm": agent.llm_enabled()}


@app.post("/api/plan")
def plan(req: PlanRequest):
    favs = [f.model_dump() for f in req.favourites]
    return _run(agent.plan, favs, req.location, req.explore, req.home_food, req.note)


@app.post("/api/twin")
def twin(req: TwinRequest):
    me = [f.model_dump() for f in req.me]
    friend = [f.model_dump() for f in req.friend]
    return _run(agent.twin, me, friend, req.my_location, req.friend_location)

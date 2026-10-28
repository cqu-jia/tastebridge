"""FastAPI server: the page, /api/plan and /api/twin (JSON), and /stream variants (server-sent events)."""
import json
import os
import queue
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from . import agent  # noqa: E402

STATIC = Path(__file__).parent.parent / "static"

# The demo is public and every request spends Nebius credits, so cap it per visitor and per day
PER_IP_PER_MINUTE = 6
PER_IP_PER_DAY = 60
GLOBAL_PER_DAY = 600

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


class RateLimiter:
    def __init__(self):
        self.lock = threading.Lock()
        self.hits = defaultdict(deque)  # ip -> timestamps in the last day
        self.day, self.today = time.strftime("%Y-%m-%d"), 0

    def check(self, ip: str):
        now = time.time()
        with self.lock:
            if time.strftime("%Y-%m-%d") != self.day:
                self.day, self.today = time.strftime("%Y-%m-%d"), 0
            q = self.hits[ip]
            while q and now - q[0] > 86400:
                q.popleft()
            recent = sum(1 for t in q if now - t < 60)
            if recent >= PER_IP_PER_MINUTE:
                raise HTTPException(429, "You're going fast! Please wait a minute and try again.")
            if len(q) >= PER_IP_PER_DAY or self.today >= GLOBAL_PER_DAY:
                raise HTTPException(429, "The demo has reached today's limit. Please come back tomorrow.")
            q.append(now)
            self.today += 1


limiter = RateLimiter()


def _client_ip(request: Request) -> str:
    # Render sits behind a proxy; the first X-Forwarded-For hop is the visitor
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _guard(request: Request):
    if not os.getenv("QLOO_API_KEY"):
        raise HTTPException(500, "Missing environment variable: QLOO_API_KEY")
    limiter.check(_client_ip(request))


def _run(fn, *args):
    try:
        result = fn(*args)
    except Exception as e:
        raise HTTPException(502, f"Upstream error: {e}")
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


def _stream(fn, *args):
    """Run fn(*args, emit=...) in a thread and relay its events as server-sent events."""
    events: "queue.Queue[Optional[dict]]" = queue.Queue()

    def work():
        try:
            result = fn(*args, emit=events.put)
            events.put({"type": "error", "message": result["error"]} if "error" in result
                       else {"type": "final", "data": result})
        except Exception as e:
            events.put({"type": "error", "message": f"Upstream error: {e}"})
        events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def gen():
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"  # stop proxies closing a quiet connection
                continue
            if event is None:
                return
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _favs(items: List[Favourite]) -> List[dict]:
    return [f.model_dump() for f in items]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "llm": agent.llm_enabled()}


@app.post("/api/plan")
def plan(req: PlanRequest, request: Request):
    _guard(request)
    return _run(agent.plan, _favs(req.favourites), req.location, req.explore, req.home_food, req.note)


@app.post("/api/twin")
def twin(req: TwinRequest, request: Request):
    _guard(request)
    return _run(agent.twin, _favs(req.me), _favs(req.friend), req.my_location, req.friend_location)


@app.post("/api/plan/stream")
def plan_stream(req: PlanRequest, request: Request):
    _guard(request)
    return _stream(agent.plan, _favs(req.favourites), req.location, req.explore, req.home_food, req.note)


@app.post("/api/twin/stream")
def twin_stream(req: TwinRequest, request: Request):
    _guard(request)
    return _stream(agent.twin, _favs(req.me), _favs(req.friend), req.my_location, req.friend_location)

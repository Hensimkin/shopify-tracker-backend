# main.py
from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ====== Storage config ======
STORAGE_FILE = os.environ.get("TRACKER_STORAGE_FILE", "tracker_storage.json")
_store_lock = threading.Lock()


def _ensure_storage_exists() -> None:
    """Create an empty JSON object file if missing."""
    if not os.path.exists(STORAGE_FILE):
        print(f"[INIT] Creating new storage file at {STORAGE_FILE}")
        with open(STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)


def _load_store() -> Dict[str, Any]:
    """Load the entire store as a dict {session_id: {session, itemsoncart, timeonpage, events}}."""
    _ensure_storage_exists()
    with open(STORAGE_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, dict):
                print("[WARN] Storage file content invalid, resetting to empty dict")
                return {}
            print(f"[LOAD] Loaded store with {len(data)} sessions")
            return data
        except json.JSONDecodeError:
            print("[ERROR] Corrupted JSON file, resetting store")
            return {}


def _atomic_write(obj: Any) -> None:
    """Write JSON atomically to avoid partial writes."""
    dir_name = os.path.dirname(os.path.abspath(STORAGE_FILE)) or "."
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, STORAGE_FILE)
    print(f"[WRITE] Persisted store with {len(obj)} sessions to {STORAGE_FILE}")


def _merge_session_records(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two session records:

    - events: append lists per page
    - timeonpage: sum per page
    - itemsoncart: OVERWRITE with new value if provided; otherwise keep old

    Record shape in storage:
    {
      "session": str,
      "itemsoncart": int,
      "timeonpage": {page: int},
      "events": {page: [event, ...]}
    }
    """
    merged: Dict[str, Any] = {
        "session": new.get("session") or old.get("session"),
        "timeonpage": {},
        "events": {},
        "itemsoncart": old.get("itemsoncart", 0),
    }

    # itemsoncart (overwrite if new present; allow zero)
    if "itemsoncart" in new and new["itemsoncart"] is not None:
        try:
            merged["itemsoncart"] = int(new["itemsoncart"])
        except (ValueError, TypeError):
            pass  # keep old

    # timeonpage (sum)
    old_tp = old.get("timeonpage", {}) or {}
    new_tp = new.get("timeonpage", {}) or {}
    for page in set(old_tp.keys()) | set(new_tp.keys()):
        old_val = int(old_tp.get(page, 0) or 0)
        new_val = int(new_tp.get(page, 0) or 0)
        merged["timeonpage"][page] = max(old_val + new_val, 0)

    # events (append)
    old_ev = old.get("events", {}) or {}
    new_ev = new.get("events", {}) or {}
    for page in set(old_ev.keys()) | set(new_ev.keys()):
        old_list = old_ev.get(page) or []
        new_list = new_ev.get(page) or []
        if not isinstance(old_list, list):
            old_list = []
        if not isinstance(new_list, list):
            new_list = []
        merged["events"][page] = [*old_list, *new_list]

    return merged


def upsert_session_record(session_id: str, record: Dict[str, Any]) -> None:
    """Insert or MERGE the record for a given session."""
    with _store_lock:
        print(f"[UPSERT] Upserting session {session_id}")
        store = _load_store()
        existing = store.get(session_id)
        if existing:
            print(f"[UPSERT] Session {session_id} exists — merging")
            record = _merge_session_records(existing, record)
        else:
            print(f"[UPSERT] Session {session_id} is new — inserting")

        store[session_id] = record
        _atomic_write(store)


def get_session_record(session_id: str) -> Dict[str, Any] | None:
    with _store_lock:
        store = _load_store()
        record = store.get(session_id)
        if record:
            print(f"[FETCH] Retrieved session {session_id}")
        else:
            print(f"[FETCH] Session {session_id} not found")
        return record


# ====== Pydantic models ======
class TrackerEvent(BaseModel):
    type: str
    ts: int
    url: str | None = None

    class Config:
        extra = "allow"


class TrackerState(BaseModel):
    session: str
    events: Dict[str, List[TrackerEvent]] = Field(default_factory=dict)
    timeonpage: Dict[str, int] = Field(default_factory=dict)
    itemsInCart: int = 0  # incoming camelCase from client

    @validator("timeonpage", pre=True)
    def validate_timeonpage(cls, v: Dict[str, Any]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k, val in (v or {}).items():
            try:
                num = int(val)
            except (ValueError, TypeError):
                num = 0
            out[k] = max(num, 0)
        return out


class TrackerPayload(BaseModel):
    reason: str
    ts: int
    session: str
    state: TrackerState


# ====== AI decision schema (flexible trigger/category) ======
class AIDecision(BaseModel):
    show_popup: bool
    message: str
    reason: str
    category: str = "none"  # "discount" | "bogo" | "info" | "none" (left open)
    trigger: str = "none"   # free text, e.g., "browsing_pants", "cart_idle"
    confidence: float = Field(ge=0, le=1, default=0.6)
    delay_ms: int = 0


# ====== FastAPI app ======
app = FastAPI(title="Shopify Tracker Ingest", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest")
async def ingest(payload: TrackerPayload, request: Request) -> Dict[str, Any]:
    client_host = request.client.host if request.client else "unknown"
    print(f"\n[INGEST] Received payload from {client_host}")
    print(f" Session: {payload.session}")
    print(f" Reason: {payload.reason}")
    print(f" Events pages: {list(payload.state.events.keys())}")
    print(f" TimeOnPage: {payload.state.timeonpage}")
    print(f" ItemsInCart: {payload.state.itemsInCart}")

    # Build the storage shape EXACTLY as requested:
    # - 'itemsoncart' (lowercase, single word)
    # - events/timeonpage as provided
    simplified = {
        "session": payload.session,
        "itemsoncart": int(payload.state.itemsInCart or 0),
        "timeonpage": payload.state.timeonpage,
        "events": {k: [e.model_dump() for e in v] for k, v in payload.state.events.items()},
    }

    try:
        upsert_session_record(payload.session, simplified)
    except Exception as e:
        print(f"[ERROR] Failed to persist session {payload.session}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to persist: {e}")

    print(f"[DONE] Stored session {payload.session}\n")
    return {
        "status": "ok",
        "stored_session": payload.session,
        "reason": payload.reason,
        "received_events_pages": len(simplified["events"]),
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    print(f"[API] Fetch request for session {session_id}")
    record = get_session_record(session_id)
    if not record:
        print(f"[API] Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found")
    print(f"[API] Returned session {session_id}")
    return record


@app.get("/healthz")
async def health() -> Dict[str, str]:
    print("[API] Health check requested")
    return {"status": "healthy"}


# ====== ASK AI helpers & endpoint ======
def _recent_events_slice(events: Dict[str, List[Dict[str, Any]]], max_pages: int = 4, max_per_page: int = 15):
    """Return a trimmed copy of events to keep tokens small."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for page in list(events.keys())[:max_pages]:
        lst = events.get(page, []) or []
        out[page] = lst[-max_per_page:]
    return out


def _guess_primary_topic_from_pages(pages: List[str]) -> str | None:
    """
    Heuristic: pick the most common token (>3 chars) from page names.
    E.g., ["Product:/men/pants/...", "Collection:/pants"] -> "pants"
    """
    from collections import Counter
    tokens: List[str] = []
    for p in pages:
        for tok in p.lower().replace(":", " ").replace("/", " ").split():
            tok = "".join(ch for ch in tok if ch.isalnum())
            if len(tok) >= 4:
                tokens.append(tok)
    if not tokens:
        return None
    common = Counter(tokens).most_common(1)[0][0]
    return common


def _summarize_session(record: Dict[str, Any]) -> Dict[str, Any]:
    events = record.get("events", {}) or {}
    timeonpage = record.get("timeonpage", {}) or {}
    total_events = sum(len(v or []) for v in events.values())
    total_time = sum(int(v or 0) for v in timeonpage.values())
    items = int(record.get("itemsoncart", 0) or 0)

    page_names = list(events.keys())
    primary_topic = _guess_primary_topic_from_pages(page_names)

    summary = {
        "session": record.get("session"),
        "totals": {
            "events": total_events,
            "time_ms": total_time,
            "items_in_cart": items,
        },
        "hints": {
            "primary_topic": primary_topic  # may be None
        },
        "recent_events_sample": _recent_events_slice(events),
        "timeonpage": timeonpage,
    }
    return summary


_SYSTEM_INSTRUCTIONS = """You are a conversion assistant for an online fashion store.
You will receive a compact summary of a shopper session (events, time on page, items_in_cart, and a primary_topic hint).
Your job:
1) Analyze behavior.
2) Decide if a popup should be shown NOW.
3) If yes, write a short helpful message. If not, explain why not.

Return ONLY strict JSON with this schema (no extra text):
{
  "show_popup": boolean,
  "message": string,     // ≤120 chars; friendly, concise; no all caps
  "reason": string,      // 1–2 sentences explaining the decision
  "category": string,    // e.g., "discount", "bogo", "info", "none"
  "trigger": string,     // free text, e.g., "browsing_pants", "viewed_cart", "idle_user", "generic_engagement"
  "confidence": number,  // 0..1
  "delay_ms": number     // recommended delay before showing popup
}

Guidance:
- Prefer NOT showing a popup if engagement is very low (e.g., <3 events AND <15s total time).
- If strongly browsing a product theme (e.g., primary_topic is "pants") and engagement is meaningful (>=5 events or >=20s): consider a relevant offer
  like "Get 40% off your 2nd item" or a category-agnostic BOGO ("Buy 1, get 1 on select styles").
- If there are items in the cart but no checkout behavior: consider a gentle info nudge ("Need any help? Free returns included.").
- Be helpful but not pushy. Do not invent prices or specific SKUs.
"""


@app.post("/askai")
async def ask_ai(request: Request):
    session_id = (await request.body()).decode().strip()  # raw text
    if not session_id:
        return {"ok": False, "error": "Missing session id in body"}

    record = get_session_record(session_id)
    if not record:
        return {"ok": False, "error": "Session not found", "session": session_id}

    # Build compact summary for the LLM
    summary = _summarize_session(record)

    # Compose messages for Groq chat
    messages = [
        {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": json.dumps(summary, ensure_ascii=False)},
    ]

    # Call Groqmodel="openai/gpt-oss-20b",
    try:
        completion = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=0.2,
            max_tokens=10000,
            messages=messages,
            response_format={"type": "json_object"},  # ask for strict JSON
        )
        raw = completion.choices[0].message.content
    except Exception as e:
        print(f"[ASKAI] Groq API error: {e}")
        # Conservative fallback
        fallback = AIDecision(
            show_popup=False,
            message="Still early to show anything.",
            reason="LLM unavailable; deferring popup to avoid disruption.",
            category="none",
            trigger="none",
            confidence=0.3,
            delay_ms=0,
        )
        return {"ok": True, "session": session_id, "decision": fallback.model_dump(), "totals": summary["totals"], "hints": summary["hints"]}

    # Parse & validate JSON
    try:
        ai_json = json.loads(raw)
        decision = AIDecision(**ai_json)
    except Exception as e:
        print(f"[ASKAI] Failed to parse/validate LLM JSON: {e}\nRAW:\n{raw}")

        # Heuristic fallback (category-agnostic & generic trigger)
        totals = summary["totals"]
        primary = summary["hints"].get("primary_topic")
        high_engagement = (totals["events"] >= 5) or (totals["time_ms"] >= 20000)
        has_cart = totals["items_in_cart"] > 0

        if not high_engagement and not has_cart:
            decision = AIDecision(
                show_popup=False,
                message="Still early to show anything.",
                reason="Low engagement so far; avoid interrupting.",
                category="none",
                trigger="idle_user",
                confidence=0.6,
                delay_ms=0,
            )
        else:
            # Pick a friendly, generic message
            if has_cart and not high_engagement:
                msg = "Need a hand with your cart? Free returns on all orders."
                category = "info"
                trigger = "cart_activity"
            else:
                # Category-agnostic light promo
                msg = "Get 40% off your 2nd item—want to check today’s picks?"
                category = "discount"
                trigger = f"browsing_{primary}" if primary else "generic_engagement"

            decision = AIDecision(
                show_popup=True,
                message=msg,
                reason="Fallback heuristic based on engagement/cart activity.",
                category=category,
                trigger=trigger,
                confidence=0.55,
                delay_ms=800,
            )

    print(decision.model_dump())
    # Return the structured decision to the frontend
    return {
        "ok": True,
        "session": session_id,
        "decision": decision.model_dump(),
        "totals": summary["totals"],
        "hints": summary["hints"],
    }

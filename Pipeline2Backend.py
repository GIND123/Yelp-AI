# Pipeline2Backend.py
# Parallel P/N + stronger judge prompt + global thread pool.
# ✅ Fully working version optimized for speed & throughput with gemini-2.5-flash-lite

import os
import re
import json
import requests
import threading
from urllib.parse import urlparse
from typing import Optional, Union, Any, Dict, List, Tuple
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from google import genai
from dotenv import load_dotenv


# ---------------------------
# ENV + CLIENTS
# ---------------------------
load_dotenv()

# Allow multiple Gemini keys (comma-separated)
GEMINI_API_KEYS_RAW = (
    os.environ.get("GEMINI_API_KEYS")
    or os.environ.get("GOOGLE_API_KEYS")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
)

YELP_API_KEY = os.environ.get("YELP_API_KEY")
YELP_AI_ENDPOINT = os.environ.get(
    "YELP_AI_ENDPOINT",
    "https://api.yelp.com/ai/chat/v2",
)

if not GEMINI_API_KEYS_RAW:
    raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY (or GEMINI_API_KEYS)")
if not YELP_API_KEY:
    raise RuntimeError("Missing YELP_API_KEY")

# ✅ Correct model from your quota: fastest non-streaming tier
MODEL_FAST = "gemini-2.5-flash-lite"

# Parse keys
GEMINI_KEYS = [k.strip() for k in GEMINI_API_KEYS_RAW.split(",") if k.strip()]
if not GEMINI_KEYS:
    raise RuntimeError("No valid Gemini keys after parsing")

# Instantiate clients
_GEMINI_CLIENTS = [genai.Client(api_key=k) for k in GEMINI_KEYS]
_client_cycle = cycle(_GEMINI_CLIENTS)
_client_lock = threading.Lock()

def _next_llm_client() -> genai.Client:
    with _client_lock:
        return next(_client_cycle)

# Global thread pool to reuse across requests
AGENT_POOL = ThreadPoolExecutor(max_workers=4)

YELP_BUSINESS_ENDPOINT = "https://api.yelp.com/v3/businesses/{business_id_or_alias}"
YELP_REVIEWS_ENDPOINT  = "https://api.yelp.com/v3/businesses/{business_id_or_alias}/reviews"


# ---------------------------
# FASTAPI APP
# ---------------------------
app = FastAPI(title="Yelp Pipeline 2 Backend", version="1.5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# PROMPTS
# ---------------------------
OPTIMIST_SYS = """
You are the Optimistic Agent. You receive context about a restaurant or hotel and several review snippets.
Focus on strengths, recurring positives, and reasons a typical guest might enjoy the place.
Highlight food quality, friendly/efficient service, value, vibe, convenience, and reliability.
Do not mention reviews or that you are an agent.

Output ONLY a valid JSON array of short points (strings), 3–6 items.
Example: ["Great pasta", "Warm service", "Cozy ambience"]
""".strip()

CRITIC_SYS = """
You are the Critical Agent. You receive context about a restaurant or hotel and several review snippets.
Focus on weaknesses, recurring complaints, risks, and situations where a guest could be disappointed.
Highlight inconsistent food, slow/rude service, cleanliness problems, cramped/noisy space, and poor value.
Do not mention reviews or that you are an agent.

Output ONLY a valid JSON array of short points (strings), 3–6 items.
Example: ["Long waits", "Inconsistent dishes", "Noisy dining room"]
""".strip()

JUDGE_SYS = """
You are the Judge Agent. You receive the business context plus an Optimistic analysis and a Critical analysis.

Do NOT split the difference. Decide a lean:
- If positives outweigh negatives → lean positive.
- If negatives outweigh positives → lean negative.
- Say "mixed" only if truly balanced.

Output ONLY a valid JSON array of short points (strings), 2–4 items:
1) Net verdict with lean.
2) Who it suits / best use-case.
3) Key caution or tip (optional).

Constraints:
- Base only on provided material.
- No invented statistics or prices.
- Do not mention Yelp, reviews, or agents.
""".strip()


# ---------------------------
# REQUEST SCHEMA
# ---------------------------
class AnalyzeRequest(BaseModel):
    business_url: str = Field(..., description="Full Yelp business URL")
    reviews_limit: int = Field(6, ge=1, le=20)
    ai_fallback: bool = True
    locale: Optional[str] = None

    model_config = {"extra": "ignore"}


# ---------------------------
# HELPERS
# ---------------------------
def _yelp_headers():
    return {
        "Authorization": f"Bearer {YELP_API_KEY}",
        "accept": "application/json",
    }


def run_agent(system_prompt: str, content: str) -> str:
    """
    Parallel-safe Gemini call using fastest available vision/text tier.
    Uses round-robin API key selection.
    """
    llm_client = _next_llm_client()

    resp = llm_client.models.generate_content(
        model=MODEL_FAST,
        contents=[system_prompt, content],
        config={"response_mime_type": "application/json"},
    )

    return (getattr(resp, "text", "") or "").strip()


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text.strip())


def _extract_json_array_substring(text: str) -> Optional[str]:
    s = text.find("[")
    e = text.rfind("]")
    if s != -1 and e != -1 and e > s:
        return text[s:e + 1]
    return None


def _sanitize_points(points: List[str], max_items: int) -> List[str]:
    out: List[str] = []
    for p in points:
        if not p:
            continue
        s = str(p).strip()
        s = s.strip(",")
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        if s:
            out.append(s)
    return out[:max_items]


def safe_points_parse(text: str, min_items: int = 2, max_items: int = 8) -> List[str]:
    if not text:
        return []

    cleaned = _strip_code_fences(text)

    # Try strict JSON
    try:
        arr = json.loads(cleaned)
        if isinstance(arr, list):
            pts = _sanitize_points(arr, max_items)
            if pts:
                return pts
    except Exception:
        pass

    # Extract embedded JSON array
    sub = _extract_json_array_substring(cleaned)
    if sub:
        try:
            arr = json.loads(sub)
            if isinstance(arr, list):
                pts = _sanitize_points(arr, max_items)
                if pts:
                    return pts
        except Exception:
            pass

    # Fallback: line split
    lines = re.split(r"(?:\r?\n)+", cleaned)
    pts: List[str] = []
    for ln in lines:
        ln = re.sub(r"^[-•\d\)\.]+\s*", "", ln.strip())
        if ln:
            pts.append(ln)

    if not pts and ";" in cleaned:
        pts = [p.strip() for p in cleaned.split(";") if p.strip()]

    pts = _sanitize_points(pts, max_items)
    return pts if len(pts) >= min_items else pts


def extract_business_id_or_alias_from_url(business_url: str) -> str:
    parsed = urlparse(business_url.strip())
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "biz":
        return parts[1]
    if re.match(r"^[A-Za-z0-9\-_]+$", business_url.strip()):
        return business_url.strip()
    raise ValueError("Unable to extract business alias/id from URL")


def get_business_details(business_id_or_alias: str, locale: Optional[str]) -> dict:
    r = requests.get(
        YELP_BUSINESS_ENDPOINT.format(business_id_or_alias=business_id_or_alias),
        headers=_yelp_headers(),
        params={"locale": locale} if locale else None,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def get_business_reviews_from_fusion(
    business_id_or_alias: str,
    limit: int,
    locale: Optional[str],
) -> list:

    r = requests.get(
        YELP_REVIEWS_ENDPOINT.format(business_id_or_alias=business_id_or_alias),
        headers=_yelp_headers(),
        params={"limit": limit, "sort_by": "yelp_sort", "locale": locale} if locale else {"limit": limit},
        timeout=30,
    )

    if r.status_code != 200:
        return []

    return (r.json() or {}).get("reviews", [])


def get_review_snippets_from_yelp_ai(business_name: str, city: str, state: str) -> str:

    location_str = ", ".join(p for p in [city, state] if p)
    payload = {
        "query": f"For {business_name} in {location_str}, summarize typical guest experiences "
                 f"as 3 short positives and 3 short negatives."
    }

    r = requests.post(
        YELP_AI_ENDPOINT,
        headers={
            "Authorization": f"Bearer {YELP_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=40,
    )

    if r.status_code != 200:
        raise HTTPException(502, f"Yelp AI fallback failed: {r.text[:300]}")

    return ((r.json() or {}).get("response") or {}).get("text", "") or ""


def normalize_business_payload(business: dict) -> dict:

    loc = business.get("location") or {}
    cats = [c.get("title") for c in (business.get("categories") or []) if c.get("title")]

    address = loc.get("formatted_address") or ", ".join(
        p for p in [
            loc.get("address1"),
            loc.get("address2"),
            loc.get("address3"),
            loc.get("city"),
            loc.get("state"),
            loc.get("zip_code"),
            loc.get("country"),
        ] if p
    )

    return {
        "name": business.get("name", "N/A"),
        "rating": business.get("rating", "N/A"),
        "price": business.get("price", "N/A"),
        "categories": cats,
        "address": address or "N/A",
        "url": business.get("url", "N/A"),
        "review_count": business.get("review_count", "N/A"),
    }


def build_context_from_reviews(business: dict, reviews: list) -> str:
    b = normalize_business_payload(business)
    out = f"""
Business:
Name: {b['name']}
Rating: {b['rating']}
Price: {b['price']}
Categories: {", ".join(b['categories'])}
Address: {b['address']}

Representative review snippets:
""".strip()

    for r in reviews[:10]:
        txt = (r.get("text") or "").replace("\n", " ").strip()
        if txt:
            out += f"\n- {r.get('rating')}★: {txt}"

    return out


def build_context_from_ai_summary(business: dict, summary: str) -> str:
    b = normalize_business_payload(business)
    return f"""
Business:
Name: {b['name']}
Rating: {b['rating']}
Price: {b['price']}
Categories: {", ".join(b['categories'])}
Address: {b['address']}

AI summary of typical positives/negatives:
{summary.strip()}
""".strip()


def run_multi_agent_debate(context: str) -> Tuple[List[str], List[str], List[str]]:

    # Run P + N in parallel
    futures = {
        AGENT_POOL.submit(run_agent, OPTIMIST_SYS, context): "P",
        AGENT_POOL.submit(run_agent, CRITIC_SYS, context): "N",
    }

    P_raw = N_raw = ""

    for fut in as_completed(futures):
        label = futures[fut]
        try:
            txt = fut.result()
        except Exception:
            txt = ""

        if label == "P":
            P_raw = txt
        else:
            N_raw = txt

    P = safe_points_parse(P_raw, min_items=3, max_items=6)
    N = safe_points_parse(N_raw, min_items=3, max_items=6)

    judge_input = f"""
{context}

Optimistic analysis: {json.dumps(P, ensure_ascii=False)}
Critical analysis: {json.dumps(N, ensure_ascii=False)}
Counts: positives={len(P)} negatives={len(N)}
""".strip()

    J_raw = AGENT_POOL.submit(run_agent, JUDGE_SYS, judge_input).result()
    J = safe_points_parse(J_raw, min_items=2, max_items=4)

    return P, N, J


def parse_request(payload: Union[str, Dict[str, Any]]) -> AnalyzeRequest:
    if isinstance(payload, str):
        return AnalyzeRequest(business_url=payload.strip())
    return AnalyzeRequest(**payload)


# ---------------------------
# ROUTES
# ---------------------------
@app.get("/")
def root():
    return {
        "service": "Yelp Pipeline 2 Backend",
        "docs": "/docs",
        "health": "/health",
        "endpoint": "/analyze-business",
        "body": "Send JSON {business_url} or raw text body with a Yelp URL",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-business")
def analyze_business(
    payload: Union[str, Dict[str, Any]] = Body(...),
):

    try:
        req = parse_request(payload)
    except ValidationError as e:
        raise HTTPException(422, e.errors())

    try:
        business_id = extract_business_id_or_alias_from_url(req.business_url)
    except Exception as e:
        raise HTTPException(400, str(e))

    # Parallel data fetch
    fbiz = AGENT_POOL.submit(get_business_details, business_id, req.locale)
    frev = AGENT_POOL.submit(
        get_business_reviews_from_fusion,
        business_id,
        req.reviews_limit,
        req.locale,
    )

    try:
        business = fbiz.result()
    except Exception as e:
        raise HTTPException(502, f"Business fetch failed: {e}")

    try:
        reviews = frev.result()
    except Exception:
        reviews = []

    context_source = "fusion_reviews"

    if reviews:
        context = build_context_from_reviews(business, reviews)
    else:
        if not req.ai_fallback:
            raise HTTPException(404, "Fusion reviews unavailable and ai_fallback=False")

        loc = business.get("location") or {}
        ai_txt = get_review_snippets_from_yelp_ai(
            business.get("name", ""),
            loc.get("city", ""),
            loc.get("state", ""),
        )

        context = build_context_from_ai_summary(business, ai_txt)
        context_source = "yelp_ai_summary"

    try:
        P, N, J = run_multi_agent_debate(context)
    except Exception as e:
        raise HTTPException(502, f"LLM debate failed: {str(e)[:300]}")

    return {
        "business_id": business_id,
        "business": normalize_business_payload(business),
        "context_source": context_source,
        "P": P,
        "N": N,
        "J": J,
    }


# ---------------------------
# LOCAL RUN
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )

# Pipeline2Backend.py
# Fixes: enforce consistent point-list outputs for P/N/J and robustly parse JSON arrays.

import os
import re
import json
import requests
from urllib.parse import urlparse
from typing import Optional, Union, Any, Dict, List

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from google import genai
from dotenv import load_dotenv

# ---------------------------
# Env + clients
# ---------------------------
load_dotenv()

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
YELP_API_KEY = os.environ.get("YELP_API_KEY")
YELP_AI_ENDPOINT = os.environ.get("YELP_AI_ENDPOINT", "https://api.yelp.com/ai/chat/v2")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment")
if not YELP_API_KEY:
    raise RuntimeError("Missing YELP_API_KEY in environment")

llm_client = genai.Client(api_key=GEMINI_API_KEY)

YELP_BUSINESS_ENDPOINT = "https://api.yelp.com/v3/businesses/{business_id_or_alias}"
YELP_REVIEWS_ENDPOINT  = "https://api.yelp.com/v3/businesses/{business_id_or_alias}/reviews"


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Yelp Pipeline 2 Backend", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Prompts (POINT LIST OUTPUT)
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
Weigh both sides fairly and produce a practical, unbiased verdict for a first-time visitor.

Output ONLY a valid JSON array of short points (strings), 2–4 items:
- overall balance
- who it suits / best use-case
- key caution (if any)

Constraints:
- Base judgment only on provided context and the two analyses.
- Do not invent statistics or exact prices/wait times.
- Do not mention Yelp, reviews, or agents.
Example: ["Mostly solid with a few consistency hiccups", "Best for casual diners who like lively rooms", "Go off-peak to avoid waits"]
""".strip()


# ---------------------------
# Request schema (JSON option)
# ---------------------------
class AnalyzeRequest(BaseModel):
    business_url: str = Field(..., description="Full Yelp business URL pasted by user")
    reviews_limit: int = Field(6, ge=1, le=20)
    ai_fallback: bool = True
    locale: Optional[str] = None

    model_config = {"extra": "ignore"}  # allow minimal JSON


# ---------------------------
# Helpers
# ---------------------------
def _yelp_headers():
    return {"Authorization": f"Bearer {YELP_API_KEY}", "accept": "application/json"}

def run_agent(system_prompt: str, content: str) -> str:
    """
    Calls Gemini. Tries to request JSON output type when supported.
    """
    try:
        # Some versions of google-genai support generation_config / response_mime_type.
        resp = llm_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, content],
            config={"response_mime_type": "application/json"},
        )
    except TypeError:
        # Fallback for older SDKs
        resp = llm_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, content],
        )

    return (getattr(resp, "text", "") or "").strip()

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text.strip())

def _extract_json_array_substring(text: str) -> Optional[str]:
    """
    Find the first '[' and last ']' and return that substring if plausible.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None

def _sanitize_points(points: List[str], max_items: int) -> List[str]:
    cleaned: List[str] = []
    for p in points:
        if p is None:
            continue
        s = str(p).strip()

        # Remove standalone bracket artifacts.
        if s in {"[", "]"}:
            continue

        # Trim trailing commas and surrounding quotes.
        s = s.strip().strip(",")
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()

        # Drop empties again.
        if s:
            cleaned.append(s)

    return cleaned[:max_items]

def safe_points_parse(text: str, min_items: int = 2, max_items: int = 8) -> List[str]:
    """
    Robustly parse an LLM response into a list of strings.

    Order:
    1) strict JSON loads (whole text)
    2) JSON loads of extracted array substring
    3) line/bullet splitting fallback
    """
    if not text:
        return []

    cleaned = _strip_code_fences(text)

    # 1) Strict JSON on full text
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            pts = _sanitize_points([str(x) for x in data], max_items)
            if pts:
                return pts
    except Exception:
        pass

    # 2) Try parsing an embedded JSON array substring
    arr_sub = _extract_json_array_substring(cleaned)
    if arr_sub:
        try:
            data = json.loads(arr_sub)
            if isinstance(data, list):
                pts = _sanitize_points([str(x) for x in data], max_items)
                if pts:
                    return pts
        except Exception:
            pass

    # 3) Fallback: split bullets/newlines/semicolons
    lines = re.split(r"(?:\r?\n)+", cleaned)
    pts: List[str] = []
    for ln in lines:
        ln = ln.strip()
        ln = re.sub(r"^[-•\d\)\.]+\s*", "", ln)  # remove bullets/numbers
        if ln and ln not in {"[", "]"}:
            pts.append(ln)

    if not pts and ";" in cleaned:
        pts = [p.strip() for p in cleaned.split(";") if p.strip()]

    pts = _sanitize_points(pts, max_items)

    # If still tiny, return what we have (don't invent)
    return pts if len(pts) >= min_items else pts

def extract_business_id_or_alias_from_url(business_url: str) -> str:
    parsed = urlparse(business_url.strip())
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "biz":
        return parts[1]
    if re.match(r"^[A-Za-z0-9\-_]+$", business_url.strip()):
        return business_url.strip()
    raise ValueError("Could not extract business alias/id from URL")

def get_business_details(business_id_or_alias: str, locale: Optional[str] = None) -> dict:
    url = YELP_BUSINESS_ENDPOINT.format(business_id_or_alias=business_id_or_alias)
    params = {"locale": locale} if locale else None
    r = requests.get(url, headers=_yelp_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_business_reviews_from_fusion(
    business_id_or_alias: str, limit: int = 6, locale: Optional[str] = None
) -> list:
    url = YELP_REVIEWS_ENDPOINT.format(business_id_or_alias=business_id_or_alias)
    params = {"limit": limit, "sort_by": "yelp_sort"}
    if locale:
        params["locale"] = locale

    r = requests.get(url, headers=_yelp_headers(), params=params, timeout=30)

    # If Fusion reviews are not permitted on plan, treat as empty.
    if r.status_code != 200:
        return []
    return (r.json() or {}).get("reviews", []) or []

def get_review_snippets_from_yelp_ai(business_name: str, city: str, state: str) -> str:
    location_str = ", ".join([p for p in [city, state] if p])
    query = (
        f"For {business_name} in {location_str}, summarize typical guest experiences "
        f"as 3 short positives and 3 short negatives focused on food, service, cleanliness, "
        f"crowding, reliability, and value."
    )
    payload = {"query": query}
    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    r = requests.post(YELP_AI_ENDPOINT, headers=headers, json=payload, timeout=40)

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Yelp AI fallback failed ({r.status_code}): {r.text[:300]}"
        )

    return ((r.json() or {}).get("response") or {}).get("text", "") or ""

def normalize_business_payload(business: dict) -> dict:
    loc = business.get("location") or {}
    categories = [c.get("title") for c in (business.get("categories") or []) if c.get("title")]

    address = loc.get("formatted_address")
    if not address:
        parts = [
            loc.get("address1"),
            loc.get("address2"),
            loc.get("address3"),
            loc.get("city"),
            loc.get("state"),
            loc.get("zip_code"),
            loc.get("country"),
        ]
        address = ", ".join([p for p in parts if p])

    return {
        "name": business.get("name") or "N/A",
        "rating": business.get("rating") if business.get("rating") is not None else "N/A",
        "price": business.get("price") or "N/A",
        "categories": categories,
        "address": address or "N/A",
        "url": business.get("url") or "N/A",
        "review_count": business.get("review_count") if business.get("review_count") is not None else "N/A",
    }

def build_context_from_reviews(business: dict, reviews: list) -> str:
    b = normalize_business_payload(business)
    context = f"""
Business:
Name: {b['name']}
Rating: {b['rating']}
Price: {b['price']}
Categories: {", ".join(b['categories'])}
Address: {b['address']}

Representative review snippets:
""".strip()

    for r in reviews[:10]:
        rating = r.get("rating")
        text = (r.get("text") or "").replace("\n", " ").strip()
        if text:
            context += f"\n- {rating}★: {text}"
    return context

def build_context_from_ai_summary(business: dict, ai_summary: str) -> str:
    b = normalize_business_payload(business)
    return f"""
Business:
Name: {b['name']}
Rating: {b['rating']}
Price: {b['price']}
Categories: {", ".join(b['categories'])}
Address: {b['address']}

AI summary of typical positives/negatives:
{ai_summary.strip()}
""".strip()

def run_multi_agent_debate(context: str):
    P_raw = run_agent(OPTIMIST_SYS, context)
    N_raw = run_agent(CRITIC_SYS, context)

    P = safe_points_parse(P_raw, min_items=3, max_items=6)
    N = safe_points_parse(N_raw, min_items=3, max_items=6)

    judge_input = f"""{context}

Optimistic analysis (points JSON):
{json.dumps(P, ensure_ascii=False)}

Critical analysis (points JSON):
{json.dumps(N, ensure_ascii=False)}
""".strip()

    J_raw = run_agent(JUDGE_SYS, judge_input)
    J = safe_points_parse(J_raw, min_items=2, max_items=4)

    return P, N, J

def parse_request(payload: Union[str, Dict[str, Any]]) -> AnalyzeRequest:
    if isinstance(payload, str):
        return AnalyzeRequest(business_url=payload.strip())
    return AnalyzeRequest(**payload)


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {
        "service": "Yelp Pipeline 2 Backend",
        "docs": "/docs",
        "health": "/health",
        "endpoint": "/analyze-business",
        "body": "Send JSON {business_url: <url>} or raw text body with url"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze-business")
def analyze_business(
    payload: Union[str, Dict[str, Any]] = Body(..., description="JSON or raw Yelp URL")
):
    # 1) Parse body robustly
    try:
        req = parse_request(payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    # 2) Extract alias/id
    try:
        business_id_or_alias = extract_business_id_or_alias_from_url(req.business_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3) Fetch business details
    try:
        business = get_business_details(business_id_or_alias, locale=req.locale)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Business details fetch failed: {e}")

    # 4) Try Fusion reviews (soft fail)
    try:
        reviews = get_business_reviews_from_fusion(
            business_id_or_alias, limit=req.reviews_limit, locale=req.locale
        )
    except Exception:
        reviews = []

    # 5) Build context
    context_source = "fusion_reviews"
    if reviews:
        context = build_context_from_reviews(business, reviews)
    else:
        if not req.ai_fallback:
            raise HTTPException(status_code=404, detail="Fusion reviews unavailable and ai_fallback=False")
        loc = business.get("location") or {}
        ai_summary = get_review_snippets_from_yelp_ai(
            business.get("name", ""),
            loc.get("city", ""),
            loc.get("state", "")
        )
        context = build_context_from_ai_summary(business, ai_summary)
        context_source = "yelp_ai_summary"

    # 6) Multi-agent debate
    try:
        P, N, J = run_multi_agent_debate(context)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini debate failed: {str(e)[:300]}")

    return {
        "business_id": business_id_or_alias,
        "business": normalize_business_payload(business),
        "context_source": context_source,
        "P": P,  # always list[str]
        "N": N,  # always list[str]
        "J": J,  # always list[str]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

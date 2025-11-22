# Pipeline2Backend.py
import os
import re
import json
import requests
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
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
YELP_REVIEWS_ENDPOINT = "https://api.yelp.com/v3/businesses/{business_id_or_alias}/reviews"


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Yelp Pipeline 2 Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Prompts (updated)
# ---------------------------

OPTIMIST_SYS = """
You are the Optimistic Agent. You receive context about a restaurant or hotel and several review snippets.
Focus on strengths, recurring positives, and reasons a typical guest might enjoy the place.
Highlight food quality, friendly/efficient service, value, vibe, convenience, and reliability.
Do not mention reviews or that you are an agent. Speak naturally.
Write 2–4 concise sentences.
""".strip()

CRITIC_SYS = """
You are the Critical Agent. You receive context about a restaurant or hotel and several review snippets.
Focus on weaknesses, recurring complaints, risks, and situations where a guest could be disappointed.
Highlight inconsistent food, slow/rude service, cleanliness problems, cramped/noisy space, and poor value.
Do not mention reviews or that you are an agent. Speak naturally.
Write 2–4 concise sentences.
""".strip()

# Judge now produces ONE unbiased verdict paragraph (no 3-line format).
JUDGE_SYS = """
You are the Judge Agent. You receive the business context plus an Optimistic analysis and a Critical analysis.
Weigh both sides fairly and produce a practical, unbiased verdict for a first-time visitor.

Output a single short paragraph (2–3 sentences):
- Sentence 1: balanced summary of overall experience.
- Sentence 2: who it suits / when it works best.
- Sentence 3 (optional): key caution if any.

Constraints:
- Base judgment only on provided context and the two analyses.
- Do not invent statistics or exact prices/wait times.
- Do not mention Yelp, reviews, or agents.
- Keep under 450 characters total.
""".strip()


# ---------------------------
# Request/Response schemas
# ---------------------------
class AnalyzeRequest(BaseModel):
    business_url: str = Field(..., description="Full Yelp business URL pasted by user")
    reviews_limit: int = Field(6, ge=1, le=20, description="How many Fusion reviews to request")
    ai_fallback: bool = Field(True, description="If Fusion reviews fail, use Yelp AI summary text")
    locale: str | None = Field(None, description="Optional Yelp locale, e.g., en_US")


# ---------------------------
# Helpers
# ---------------------------
def _yelp_headers():
    return {"Authorization": f"Bearer {YELP_API_KEY}", "accept": "application/json"}

def run_agent(system_prompt: str, content: str) -> str:
    resp = llm_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[system_prompt, content],
    )
    return (getattr(resp, "text", "") or "").strip()

def extract_business_id_or_alias_from_url(business_url: str) -> str:
    """
    Accepts URLs like:
      https://www.yelp.com/biz/college-park-diner-college-park
      https://www.yelp.com/biz/college-park-diner-college-park?foo=bar
    Returns the alias segment after /biz/
    """
    parsed = urlparse(business_url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "biz":
        return parts[1]
    # also allow user to paste just alias/id
    if re.match(r"^[A-Za-z0-9\-_]+$", business_url):
        return business_url
    raise ValueError("Could not extract business alias/id from URL")

def get_business_details(business_id_or_alias: str, locale: str | None = None) -> dict:
    url = YELP_BUSINESS_ENDPOINT.format(business_id_or_alias=business_id_or_alias)
    params = {"locale": locale} if locale else None
    r = requests.get(url, headers=_yelp_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_business_reviews_from_fusion(
    business_id_or_alias: str, limit: int = 6, locale: str | None = None
) -> list[dict]:
    url = YELP_REVIEWS_ENDPOINT.format(business_id_or_alias=business_id_or_alias)
    params = {"limit": limit, "sort_by": "yelp_sort"}
    if locale:
        params["locale"] = locale
    r = requests.get(url, headers=_yelp_headers(), params=params, timeout=30)
    if r.status_code != 200:
        return []
    return (r.json() or {}).get("reviews", []) or []

def get_review_snippets_from_yelp_ai(business_name: str, city: str, state: str) -> str:
    """
    Uses Yelp AI to get typical pros/cons when Fusion reviews aren't accessible.
    """
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
    r.raise_for_status()
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

def build_context_from_reviews(business: dict, reviews: list[dict]) -> str:
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
    context = f"""
Business:
Name: {b['name']}
Rating: {b['rating']}
Price: {b['price']}
Categories: {", ".join(b['categories'])}
Address: {b['address']}

AI summary of typical positives/negatives:
{ai_summary.strip()}
""".strip()
    return context

def run_multi_agent_debate(context: str) -> tuple[str, str, str]:
    p_text = run_agent(OPTIMIST_SYS, context)
    n_text = run_agent(CRITIC_SYS, context)

    judge_input = f"""{context}

Optimistic analysis:
{p_text}

Critical analysis:
{n_text}
""".strip()

    j_text = run_agent(JUDGE_SYS, judge_input)
    return p_text, n_text, j_text


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
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze-business")
def analyze_business(req: AnalyzeRequest):
    try:
        business_id_or_alias = extract_business_id_or_alias_from_url(req.business_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        business = get_business_details(business_id_or_alias, locale=req.locale)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch business details: {e}")

    reviews = get_business_reviews_from_fusion(
        business_id_or_alias, limit=req.reviews_limit, locale=req.locale
    )

    context_source = "fusion_reviews"
    if reviews:
        context = build_context_from_reviews(business, reviews)
    else:
        if not req.ai_fallback:
            raise HTTPException(status_code=404, detail="No Fusion reviews available and ai_fallback=False")
        loc = business.get("location") or {}
        ai_summary = get_review_snippets_from_yelp_ai(
            business.get("name", ""), loc.get("city", ""), loc.get("state", "")
        )
        context = build_context_from_ai_summary(business, ai_summary)
        context_source = "yelp_ai_summary"

    P, N, J = run_multi_agent_debate(context)

    out = {
        "business_id": business_id_or_alias,
        "business": normalize_business_payload(business),
        "context_source": context_source,
        "P": P,  # optimistic agent output
        "N": N,  # critical agent output
        "J": J,  # single unbiased verdict paragraph based on both
    }
    return out


# ---------------------------
# Local run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

import os
import json
import requests
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from google import genai
from google.genai import types
from dotenv import load_dotenv

# ---------------------------
# Env + clients
# ---------------------------
load_dotenv()

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
YELP_API_KEY = os.environ.get("YELP_API_KEY")

YELP_BUSINESS_ENDPOINT = "https://api.yelp.com/v3/businesses/{business_id}"
YELP_REVIEWS_ENDPOINT  = "https://api.yelp.com/v3/businesses/{business_id}/reviews"
YELP_AI_ENDPOINT       = os.environ.get("YELP_AI_ENDPOINT", "https://api.yelp.com/ai/chat/v2")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY")
if not YELP_API_KEY:
    raise RuntimeError("Missing YELP_API_KEY")

llm = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------
# Multi-agent prompts
# ---------------------------
OPTIMIST_SYS = """
You are an Optimistic Agent analyzing a business.
Base your view ONLY on the provided context.
Emphasize strengths, recurring positives, and who would enjoy it.
Write 2–4 concise sentences.
Do not mention reviews, Yelp, or that you're an agent.
"""

CRITIC_SYS = """
You are a Critical Agent analyzing a business.
Base your view ONLY on the provided context.
Emphasize weaknesses, recurring complaints, and who may dislike it.
Write 2–4 concise sentences.
Do not mention reviews, Yelp, or that you're an agent.
"""

JUDGE_SYS = """
You are a Judge Agent. You will be given:
(1) Business context
(2) An Optimistic take (P)
(3) A Critical take (N)

Task:
Write ONE independent verdict sentence that fairly weighs P versus N.
It must be realistic, practical, and conditional ("good if..., avoid if...").
Do not repeat the full P or N; synthesize.
No pros/cons headings.
Under 220 characters.
Do not mention reviews, Yelp, or agents.
"""

# ---------------------------
# Helper functions
# ---------------------------
def run_agent(system_prompt: str, content: str) -> str:
    resp = llm.models.generate_content(
        model="gemini-2.5-flash",
        contents=[system_prompt, content]
    )
    return (resp.text or "").strip()

def extract_business_id_from_url(business_url: str) -> str:
    parsed = urlparse(business_url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "biz":
        return parts[1]
    raise ValueError("Could not extract business id from URL")

def yelp_headers():
    return {
        "Authorization": f"Bearer {YELP_API_KEY}",
        "accept": "application/json"
    }

def get_business_details(business_id: str) -> dict:
    url = YELP_BUSINESS_ENDPOINT.format(business_id=business_id)
    r = requests.get(url, headers=yelp_headers(), timeout=40)
    r.raise_for_status()
    return r.json()

def get_business_reviews_from_fusion(business_id: str, limit: int = 3) -> list:
    # Yelp Fusion reviews endpoint returns up to 3 excerpts.
    url = YELP_REVIEWS_ENDPOINT.format(business_id=business_id)
    params = {"limit": limit, "sort_by": "yelp_sort"}
    r = requests.get(url, headers=yelp_headers(), params=params, timeout=40)
    if r.status_code != 200:
        return []
    return (r.json() or {}).get("reviews", []) or []

def get_review_snippets_from_yelp_ai(business_name: str, location_str: str) -> str:
    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    query = (
        f"Summarize customer experience for {business_name} in {location_str}. "
        f"Give 3 short positives and 3 short negatives focused on food/service/cleanliness/value."
    )
    payload = {"query": query}
    r = requests.post(YELP_AI_ENDPOINT, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return ((r.json() or {}).get("response") or {}).get("text", "") or ""

def build_context_from_reviews(business: dict, reviews: list) -> str:
    categories = ", ".join(c.get("title", "") for c in business.get("categories", []))
    loc = business.get("location", {}) or {}
    address = loc.get("formatted_address") or ", ".join(
        p for p in [loc.get("address1"), loc.get("city"), loc.get("state"), loc.get("zip_code")] if p
    )

    ctx = f"""Business:
Name: {business.get('name','')}
Rating: {business.get('rating')}
Price: {business.get('price','')}
Categories: {categories}
Address: {address}

Recent review excerpts:
"""
    for r in reviews:
        txt = (r.get("text") or "").replace("\n", " ").strip()
        ctx += f"- {r.get('rating')}★: {txt}\n"
    return ctx.strip()

def build_context_from_ai_summary(business: dict, ai_summary: str) -> str:
    categories = ", ".join(c.get("title", "") for c in business.get("categories", []))
    loc = business.get("location", {}) or {}
    address = loc.get("formatted_address") or ", ".join(
        p for p in [loc.get("address1"), loc.get("city"), loc.get("state"), loc.get("zip_code")] if p
    )

    ctx = f"""Business:
Name: {business.get('name','')}
Rating: {business.get('rating')}
Price: {business.get('price','')}
Categories: {categories}
Address: {address}

Typical positives/negatives summary:
{ai_summary}
"""
    return ctx.strip()

def slim_business_for_json(business: dict) -> dict:
    loc = business.get("location", {}) or {}
    categories = [c.get("title","") for c in (business.get("categories") or []) if c.get("title")]
    address = loc.get("formatted_address") or ", ".join(
        p for p in [loc.get("address1"), loc.get("city"), loc.get("state"), loc.get("zip_code"), loc.get("country")] if p
    )
    return {
        "name": business.get("name"),
        "rating": business.get("rating"),
        "price": business.get("price"),
        "categories": categories,
        "address": address,
        "url": business.get("url"),
        "review_count": business.get("review_count")
    }

def run_debate(context: str):
    P = run_agent(OPTIMIST_SYS, context)
    N = run_agent(CRITIC_SYS, context)
    judge_input = f"""{context}

Optimistic take (P):
{P}

Critical take (N):
{N}
"""
    J = run_agent(JUDGE_SYS, judge_input)
    return P, N, J

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Yelp Pipeline 2 Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.post("/debate-url")
def debate_url(business_url: str = Form(...), save_to_file: bool = Form(True)):
    try:
        business_id = extract_business_id_from_url(business_url.strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        business = get_business_details(business_id)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fusion business lookup failed: {e}")

    reviews = get_business_reviews_from_fusion(business_id)
    if reviews:
        context_source = "fusion_reviews"
        context = build_context_from_reviews(business, reviews)
    else:
        context_source = "yelp_ai_summary"
        loc = business.get("location", {}) or {}
        location_str = ", ".join([p for p in [loc.get("city"), loc.get("state")] if p]) or "this area"
        ai_summary = get_review_snippets_from_yelp_ai(business.get("name",""), location_str)
        context = build_context_from_ai_summary(business, ai_summary)

    P, N, J = run_debate(context)

    out = {
        "business_id": business_id,
        "business": slim_business_for_json(business),
        "context_source": context_source,
        "P": P,
        "N": N,
        "J": J
    }

    if save_to_file:
        with open("pipeline2_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    return out

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "Pipeline2Backend:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False
    )

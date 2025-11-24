import os
import re
import json
import requests
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import uvicorn
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

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Yelp AI Backend", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Guardrail prompt
# ---------------------------
GUARDRAIL_SYS = """
You are a safety + relevance gate for an app that ONLY helps users find restaurants and hotels.

Decide if the image is allowed for analysis.

ALLOWED images (examples):
- Food, drinks, desserts, groceries, plated dishes.
- Menus, storefronts, interiors/exteriors of restaurants, cafes, bars, hotels, resorts.
- Table settings, buffets, hotel rooms, lobbies, pools, gyms, amenities.
- People are okay ONLY if clearly in a dining/travel/venue context (e.g., group at a table, staff serving, people in hotel lobby).

DISALLOWED images:
- A single human face / selfie / portrait with no clear food or venue context.
- Any nudity, sexual content, or fetish content (even if partial/blurred).
- Minors in any sexualized context.
- Graphic violence, self-harm, medical gore.
- Illegal drugs being used/sold, instructions for use, or close-ups of drug paraphernalia.
- Weapons used threateningly or in violent context.
- Hate symbols or extremist propaganda.
- Anything totally unrelated to dining/travel/venues (cars, random objects, memes, pets, landscapes with no venue, screenshots of chats, etc.).
- Private/biometric close-ups that look like a face-scan or identity photo.

Output ONLY valid JSON with this exact shape:
{
  "allowed": true/false,
  "reason": "<short, user-facing reason focused on safety/relevance>",
  "category": "<one of: food_or_venue, face_only, adult_or_nudity, violence_or_gore, drugs_or_weapons, hate_or_extremism, unrelated, uncertain>"
}

Be conservative: if uncertain, set allowed=false and category="uncertain".
Keep reason under 200 characters.
""".strip()


# ---------------------------
# Helpers
# ---------------------------
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", (text or "").strip())

def _extract_json_obj_substring(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None

def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    sub = _extract_json_obj_substring(cleaned)
    if sub:
        try:
            obj = json.loads(sub)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None

def _truncate_to_sentence(text: str, max_len: int = 1000) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_period = truncated.rfind(".")
    last_excl = truncated.rfind("!")
    last_q = truncated.rfind("?")
    last_punc = max(last_period, last_excl, last_q)
    return truncated[: last_punc + 1] if last_punc != -1 else truncated

def _build_prompt(location: str, latitude: str, longitude: str, date: str, time: str) -> str:
    latlon_block = ""
    if latitude or longitude:
        lat = latitude or "N/A"
        lon = longitude or "N/A"
        latlon_block = f"Latitude: {lat}\nLongitude: {lon}\n"

    return (
        "Write exactly one Yelp search sentence.\n"
        f"Location: {location}\n"
        f"{latlon_block}\n"
        f"Date: {date}\n"
        f"Time: {time}\n"
        "Goal: find many relevant options, sorted by highest rating and review count.\n"
        "Use the image details + user intent. Be concrete about the item and dietary constraints.\n"
        "Rules:\n"
        "- One sentence only.\n"
        "- Mention location, date, time explicitly.\n"
        "- Ask for many options ordered by rating/popularity.\n"
        "- No meta, no mention of images/APIs/prompts.\n"
        "- Under 900 chars."
    )

def _guardrail_check_image(image_bytes: bytes, mime_type: str) -> Tuple[bool, str, str]:
    """
    Returns (allowed, reason, category).
    Blocks by default on parse/LLM failure.
    """
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                GUARDRAIL_SYS,
            ],
            config={"response_mime_type": "application/json"},
        )
        raw = (getattr(resp, "text", "") or "").strip()
        data = _safe_json_parse(raw) or {}
    except Exception:
        data = {}

    allowed = bool(data.get("allowed", False))
    category = str(data.get("category") or "uncertain").strip()
    reason = str(data.get("reason") or "").strip()

    if not reason:
        # conservative default
        reason = (
            "Image can’t be verified as food/restaurant/hotel related; "
            "to protect safety and relevance, it won’t be processed."
        )
        allowed = False
        category = "uncertain"

    return allowed, reason, category

def _gemini_image_to_query(
    image_bytes: bytes,
    mime_type: str,
    user_query: str,
    location: str,
    latitude: str,
    longitude: str,
    date: str,
    time: str,
) -> str:
    instruction = _build_prompt(location, latitude, longitude, date, time)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            instruction,
            f"User intent: {user_query}",
        ],
    )
    return _truncate_to_sentence(getattr(resp, "text", "") or "", 1000)

def _gemini_caption_to_query(
    user_query: str,
    location: str,
    latitude: str,
    longitude: str,
    date: str,
    time: str,
) -> str:
    instruction = _build_prompt(location, latitude, longitude, date, time)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            instruction,
            f"User intent: {user_query}",
        ],
    )
    return _truncate_to_sentence(getattr(resp, "text", "") or "", 1000)

def _call_yelp_ai(yelp_query: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {"query": yelp_query}
    r = requests.post(YELP_AI_ENDPOINT, headers=headers, json=payload, timeout=45)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

def _extract_results(data: Dict[str, Any], yelp_query: str) -> Dict[str, Any]:
    ai_text = (data.get("response") or {}).get("text", "") or ""
    results: Dict[str, Any] = {
        "chat_id": data.get("chat_id"),
        "query": yelp_query,
        "ai_response_text": ai_text,
        "businesses": [],
    }

    entities = data.get("entities") or []
    for entity in entities:
        for biz in (entity.get("businesses") or []):
            loc = biz.get("location") or {}
            coords = biz.get("coordinates") or {}
            summaries = biz.get("summaries") or {}
            contextual = biz.get("contextual_info") or {}
            photos = contextual.get("photos") or []
            business_hours = contextual.get("business_hours") or []
            openings = (biz.get("reservation_availability") or {}).get("openings") or []

            formatted_address = loc.get("formatted_address")
            if not formatted_address:
                parts = [loc.get("address1"), loc.get("address2"), loc.get("address3")]
                city_parts = [
                    loc.get("city"),
                    loc.get("state"),
                    loc.get("zip_code"),
                    loc.get("country"),
                ]
                formatted_address = (
                    ", ".join([p for p in parts if p] + [p for p in city_parts if p]) or "N/A"
                )

            first_photo_url = (
                photos[0].get("original_url")
                if photos and isinstance(photos[0], dict)
                else "N/A"
            )

            hours_list: List[Dict[str, Any]] = []
            for h in business_hours:
                day = h.get("day_of_week") or "N/A"
                slots = h.get("business_hours") or []
                slot_strs = []
                for s in slots:
                    ot = s.get("open_time")
                    ct = s.get("close_time")
                    if ot and ct:
                        slot_strs.append(f"{ot} to {ct}")
                hours_list.append({"day_of_week": day, "hours": slot_strs})

            opening_list: List[Dict[str, Any]] = []
            for op in openings:
                date_val = op.get("date") or "N/A"
                slots = op.get("slots") or []
                slot_list = []
                for sl in slots:
                    slot_list.append({
                        "time": sl.get("time") or "N/A",
                        "seating_areas": sl.get("seating_areas") or [],
                    })
                opening_list.append({"date": date_val, "slots": slot_list})

            biz_out = {
                "id": biz.get("id"),
                "name": biz.get("name") or "N/A",
                "address": formatted_address,
                "yelp_url": biz.get("url") or "N/A",
                "rating": biz.get("rating") if biz.get("rating") is not None else "N/A",
                "review_count": biz.get("review_count") if biz.get("review_count") is not None else "N/A",
                "price": biz.get("price") or "N/A",
                "latitude": coords.get("latitude") if coords.get("latitude") is not None else "N/A",
                "longitude": coords.get("longitude") if coords.get("longitude") is not None else "N/A",
                "short_summary": summaries.get("short")
                    or (contextual.get("summary") if isinstance(contextual, dict) else None)
                    or "N/A",
                "business_hours": hours_list,
                "photo_url": first_photo_url,
                "reservation_openings": opening_list,
                "phone": biz.get("phone") or "N/A",
            }

            results["businesses"].append(biz_out)

    def _sort_key(b):
        r = b["rating"]
        rc = b["review_count"]
        r_val = float(r) if isinstance(r, (int, float)) else -1.0
        rc_val = int(rc) if isinstance(rc, int) else -1
        return (r_val, rc_val)

    results["businesses"].sort(key=_sort_key, reverse=True)
    return results


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {"status": "running", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search-image")
async def search_image(
    image: UploadFile = File(...),
    user_query: str = Form(...),
    Location: str = Form(""),
    Latitude: str = Form(""),
    Longitude: str = Form(""),
    Date: str = Form("12/11/2025"),
    Time: str = Form("8pm"),
    save_to_file: bool = Form(False),
):
    try:
        image_bytes = await image.read()
        mime_type = image.content_type or "image/jpeg"

        # 0) Guardrail check before any other model use
        allowed, reason, category = _guardrail_check_image(image_bytes, mime_type)
        if not allowed:
            return JSONResponse(
                status_code=422,
                content={
                    "status": 422,
                    "message": reason,
                    "category": category,
                },
            )

        # 1) Generate Yelp query from image + intent
        yelp_query = _gemini_image_to_query(
            image_bytes=image_bytes,
            mime_type=mime_type,
            user_query=user_query,
            location=Location,
            latitude=Latitude,
            longitude=Longitude,
            date=Date,
            time=Time,
        )

        # 2) Call Yelp AI
        data = _call_yelp_ai(yelp_query)
        results = _extract_results(data, yelp_query)

        if save_to_file:
            with open("search_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-caption")
async def search_caption(
    user_query: str = Form(...),
    Location: str = Form(""),
    Latitude: str = Form(""),
    Longitude: str = Form(""),
    Date: str = Form("12/11/2025"),
    Time: str = Form("8pm"),
    save_to_file: bool = Form(False),
):
    try:
        yelp_query = _gemini_caption_to_query(
            user_query=user_query,
            location=Location,
            latitude=Latitude,
            longitude=Longitude,
            date=Date,
            time=Time,
        )

        data = _call_yelp_ai(yelp_query)
        results = _extract_results(data, yelp_query)

        if save_to_file:
            with open("search_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )

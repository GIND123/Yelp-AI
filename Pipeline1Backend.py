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

app = FastAPI(title="Yelp AI Backend", version="1.4.0")

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
You are a safety + relevance gate for an app that ONLY helps users find places to GET, EAT, or USE
food, drinks, groceries, desserts, and restaurant/hotel services.

You will see:
- An IMAGE
- A short USER INTENT (what they want to do with what’s in the image or in general)

Your job is to decide if this request is allowed for FOOD / GROCERIES / RESTAURANTS / HOTELS.

ALLOWED examples (set allowed=true):
- Any kind of food item: raw ingredients (bananas, flour, eggs, spices), packaged snacks, drinks,
  coffee cups, desserts, candy, groceries, etc.
- Single ingredients held in a hand (e.g., a banana in a hand) WHEN the user intent is about eating,
  buying, cooking, baking, decorating food, or ordering something similar.
- Food-related plants/flowers/herbs when the user intent is about using them in food, drinks, cakes,
  or restaurant decoration (e.g., “where can I get this flower as a cake topper?”).
- Requests for “trending”, “popular”, or “most-ordered” food items or places in a given city or
  neighborhood (e.g., “show me trending dessert spots like this vibe in NYC”).
- Menus, storefronts, interiors or exteriors of restaurants, cafes, bars, bakeries, groceries, hotels,
  resorts, or similar venues.
- Table setups, buffets, hotel rooms, lobbies, pools, gyms, or amenities clearly tied to a hotel/restaurant stay.
- People are fine IF the context is clearly dining, drinking, grocery shopping, or staying at a hotel.
- If the user intent is clearly about food/venues (including “trending food items”, “popular dishes”,
  or “hot spots”) and the image has NO unsafe content, you may allow it even if the picture itself is
  only loosely or aesthetically related (e.g., a pattern, abstract decor, generic street shot).

DISALLOWED examples (set allowed=false):
- A single human face / selfie / portrait with no clear food/grocery/venue context AND user intent is
  not about dining/travel.
- Any nudity or sexual content (even partial or suggestive).
- Minors in any sexualized or exploitative context.
- Graphic violence, self-harm, gore, or medical injury focus.
- Illegal drugs or paraphernalia, or instructions for use.
- Weapons in threatening or violent context.
- Hate symbols, extremist propaganda, or calls to violence.
- Objects/scenes with no reasonable link to food, groceries, restaurants, or hotels AND user intent is
  also unrelated (e.g., random car photo and the user asks about speed; a meme screenshot; pets only;
  landscapes with no venue and no food-related intent).
- Biometric close-ups (ID photos, passport scans) where the user intent is not about dining/travel/venues.

IMPORTANT LOGIC:
- Always consider BOTH the image and the user intent together.
- If the image shows a plausible food ingredient, grocery item, drink, dessert, edible decoration, or
  venue decor AND the user intent is about eating, buying, cooking, baking, decorating, ordering, or
  finding TRENDING / POPULAR items or places, treat it as ALLOWED, unless there is a safety violation.
- If the user intent is clearly food/dining/grocery/hotel-related (including requests for trending food)
  and the image contains no safety violations, lean toward allowed=true, category="food_or_venue", even
  when the visual relevance is weak.
- If you detect ANY nudity, sexual content, graphic violence, self-harm, illegal drugs, weapons in
  violent context, or hate/extremist symbols, set allowed=false and the matching category.

Output ONLY valid JSON with this exact shape:
{
  "allowed": true/false,
  "reason": "<short, user-facing reason focused on safety/relevance>",
  "category": "<one of: food_or_venue, face_only, adult_or_nudity, violence_or_gore,
               drugs_or_weapons, hate_or_extremism, unrelated, uncertain>"
}

Constraints:
- Be conservative for SAFETY categories (nudity, violence, drugs, weapons, hate). If any such content
  appears, set allowed=false and the matching category.
- If it is only a relevance issue (not obviously food/venue AND user intent not about food/travel) and
  you are unsure, set allowed=false and category="uncertain".
- Keep "reason" under 200 characters, written so the end user can understand.
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
        return text[start : end + 1]
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
    """
    Prompt now explicitly allows both:
    - queries for specific items (from the image)
    - and queries for TRENDING / POPULAR food items or places in that area.
    """
    latlon_block = ""
    if latitude or longitude:
        lat = latitude or "N/A"
        lon = longitude or "N/A"
        latlon_block = f"Latitude: {lat}\nLongitude: {lon}\n"

    return (
        "Write exactly ONE natural-language Yelp search sentence.\n"
        f"Location: {location}\n"
        f"{latlon_block}\n"
        f"Date: {date}\n"
        f"Time: {time}\n"
        "Goal: turn the user's intent (including image context) into a Yelp-style query that can:\n"
        "- find places selling or serving the specific item in the image, OR\n"
        "- find TRENDING / POPULAR food, desserts, drinks, or restaurants in that area that match the vibe,\n"
        "  IF the user is asking for trending or popular options.\n"
        "General rules:\n"
        "- Use clear, first-person language like a search request.\n"
        "- Mention location, date, and time explicitly.\n"
        "- Ask for MANY options, sorted by highest rating and review count / popularity.\n"
        "- If the user mentions 'trending', 'popular', 'most ordered', or 'hot right now', make that explicit.\n"
        "- No meta text, no mention of images, APIs, prompts, or models.\n"
        "- Output ONE sentence only, under 900 characters."
    )


def _guardrail_check_image(
    image_bytes: bytes,
    mime_type: str,
    user_intent: str,
) -> Tuple[bool, str, str]:
    """
    Returns (allowed, reason, category).
    Blocks by default on parse/LLM failure.
    """
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                GUARDRAIL_SYS,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                f"User intent: {user_intent}",
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
        # conservative default for unexplained failure
        reason = (
            "Image could not be reliably checked for safety/relevance; "
            "it will not be processed."
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
                    slot_list.append(
                        {
                            "time": sl.get("time") or "N/A",
                            "seating_areas": sl.get("seating_areas") or [],
                        }
                    )
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
    """
    Typical user_query examples this supports now:
    - "where can I get this with less sugar?"
    - "show me trending desserts like this near Times Square"
    - "find popular brunch spots with pancakes like this in Austin"
    """
    try:
        image_bytes = await image.read()
        mime_type = image.content_type or "image/jpeg"

        # 0) Guardrail check before any other model use (uses user_query as intent)
        allowed, reason, category = _guardrail_check_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            user_intent=user_query,
        )
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
    """
    Text-only version where the React app might send something like:
    - "trending ramen spots in Seattle"
    - "popular brunch places with pancakes near me"
    """
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

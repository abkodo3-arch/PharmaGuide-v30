import os, base64, json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, APIError, APIStatusError, APITimeoutError, RateLimitError, AuthenticationError

ROOT = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="PharmaGuide Medical OCR")

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "raw_text": {"type": "string"},
                    "matched_name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "strength": {"type": "string"},
                    "dosage_form": {"type": "string"},
                    "frequency": {"type": "string"},
                    "duration": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["raw_text", "matched_name", "confidence", "strength", "dosage_form", "frequency", "duration", "evidence"],
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["medications", "notes"],
}


def get_output_text(response):
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    # Defensive fallback for SDK response objects.
    try:
        return response.output[0].content[0].text
    except Exception:
        return ""


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "ai_ocr": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
    }


@app.post("/api/medical-ocr")
async def medical_ocr(image: UploadFile = File(...), medication_names: str = ""):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(503, "OPENAI_API_KEY is not configured.")

    data = await image.read()
    if not data:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image is too large. Use a clear cropped prescription under 12 MB.")

    mime = image.content_type or "image/jpeg"
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        mime = "image/jpeg"
    b64 = base64.b64encode(data).decode("ascii")

    names = [x.strip() for x in medication_names.split("|") if x.strip()]
    names = names[:2500]
    dictionary = "\n".join(f"- {n}" for n in names)

    prompt = f"""You are a medication-prescription handwriting recognition assistant for a pharmacist in Egypt.

Read the uploaded prescription visually. Identify handwritten medicine names and, only when visibly written, strength, dosage form, frequency and duration.

CRITICAL MATCHING RULES:
1. Return POSSIBLE medication matches, not a dispensing decision.
2. matched_name MUST be exactly one of the names in the supplied medication dictionary, or an empty string when no safe match exists.
3. Do not invent, autocomplete, or hallucinate a medicine that is not in the dictionary.
4. Separate raw_text (what the handwriting appears to say) from matched_name (the closest dictionary entry).
5. Confidence is 0 to 1. Use lower confidence when handwriting is ambiguous.
6. Never infer a dose/frequency/duration that is not visible. Use an empty string for missing fields.
7. Ignore patient-identifying information.
8. Prefer several plausible candidates only when genuinely plausible; otherwise return fewer candidates.

Medication dictionary:
{dictionary}
"""

    client = OpenAI(api_key=key, timeout=90.0, max_retries=1)
    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": "high"},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "prescription_reading",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
        )
    except AuthenticationError:
        raise HTTPException(502, "OpenAI rejected the API key. Create a new API key and set OPENAI_API_KEY again.")
    except RateLimitError as e:
        raise HTTPException(502, f"OpenAI rate/billing limit: {str(e)[:500]}")
    except APITimeoutError:
        raise HTTPException(504, "OpenAI OCR timed out. Try a smaller, cropped prescription image.")
    except APIStatusError as e:
        msg = getattr(e, "message", None) or str(e)
        raise HTTPException(502, f"OpenAI OCR error {getattr(e, 'status_code', '')}: {msg[:700]}")
    except APIError as e:
        raise HTTPException(502, f"OpenAI OCR API error: {str(e)[:700]}")
    except Exception as e:
        raise HTTPException(502, f"OCR request failed: {type(e).__name__}: {str(e)[:700]}")

    text = get_output_text(response)
    if not text:
        raise HTTPException(502, "OpenAI returned no OCR result.")

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(502, "OpenAI returned an unexpected OCR format.")

    allowed = {n.lower(): n for n in names}
    cleaned = []
    import difflib
    for m in result.get("medications", []):
        matched = (m.get("matched_name") or "").strip()
        if matched.lower() not in allowed and matched:
            cand = difflib.get_close_matches(matched, names, n=1, cutoff=0.84)
            matched = cand[0] if cand else ""
        if not matched:
            continue
        m["matched_name"] = matched
        try:
            m["confidence"] = max(0.0, min(1.0, float(m.get("confidence", 0))))
        except Exception:
            m["confidence"] = 0.0
        cleaned.append(m)

    result["medications"] = sorted(cleaned, key=lambda x: x["confidence"], reverse=True)[:12]
    result["notes"] = (result.get("notes") or [])[:8]
    return JSONResponse(result)


app.mount("/assets", StaticFiles(directory=ROOT), name="assets")


@app.get("/")
def home():
    return FileResponse(os.path.join(ROOT, "index.html"))

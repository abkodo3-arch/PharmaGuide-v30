# PharmaGuide v30 — Ready for Render

This package is prepared as a Render Web Service.

## Render settings
- Runtime: Python
- Build command: `pip install -r PharmaGuide_v24/requirements.txt`
- Start command: `uvicorn PharmaGuide_v24.server:app --host 0.0.0.0 --port $PORT`
- Environment variable: `OPENAI_API_KEY` (secret)
- Optional: `OPENAI_VISION_MODEL=gpt-4.1-mini`

After deployment, Render gives the app an `onrender.com` URL.

## Important data note
Daily Sales and Sales History are stored in browser localStorage, so they stay with the browser/device profile that uses the app. They are not a shared server database.

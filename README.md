# Shopify Tracker – Backend (FastAPI)

Backend service that receives client-side browsing events from a Shopify storefront, stores session aggregates, and (optionally) asks an LLM (Groq) for real-time suggestions such as “show a friendly popup if cart is large and there’s no checkout.”

* Framework: **FastAPI**
* Runtime: **Python 3.11+**
* Model provider: **Groq API**
* Storage: **Local JSON file** (simple flat-file DB)
* Hosting: local  / Render

---

## Features

* **Event ingest**: accept batched events from a browser script (page views, clicks, time-on-page, items in cart, etc.).
* **Session store**: store per-session data in a JSON file (safe for prototypes and small demos).
* **Ask AI**: call Groq to get structured UI recommendations (e.g., popup config) based on the session.
* **CORS-friendly**: configured so your Shopify theme/app can POST directly.

---

## Quickstart

### 1) Clone

```bash
git clone https://github.com/Hensimkin/shopify-tracker-backend.git
cd shopify-tracker-backend
```



### 2) Install deps

> Ensure `requirements.txt` includes: `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, `groq`, and anything else you use (`httpx`, etc.)

```bash
pip install -r requirements.txt
```

### 3) Environment variables

Create a `.env` in the project root:

```env
# Required for /askai
GROQ_API_KEY=your_groq_key_here

# Optional – where to persist sessions (default: tracker_storage.json)
TRACKER_STORAGE_FILE=tracker_storage.json

# Optional CORS – comma separated list (use your domains)
ALLOWED_ORIGINS=https://yourshop.myshopify.com,https://your-preview-url
```

### 4) Run locally

```bash
# Dev (auto-reload)
fastapi dev

# Or via uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

* API base: `http://127.0.0.1:8000`
* OpenAPI docs: `http://127.0.0.1:8000/docs`

---

## API


### Ingest Events

**POST** `/ingest`

Accepts a full snapshot of session state from the browser script.

**Body (JSON)**

```json
{
  "session": "af4bac4d-077a-4756-91d0-fa2570fd206f",
  "itemsInCart": 7,
  "timeOnPage": { "/products/xyz": 123456, "/cart": 9876 },
  "events": {
    "/": [
      { "type": "view", "ts": 1712345678901 },
      { "type": "click", "target": "#add-to-cart", "ts": 1712345680000 }
    ],
    "/cart": [
      { "type": "view", "ts": 1712345690000 }
    ]
  }
}
```

**Response**

```json
{ "ok": true, "saved": true }
```

### Ask AI (popup logic, etc.)

**POST** `/askai`

* **Request body**: plain text containing the session ID (not JSON).
  Example body: `af4bac4d-077a-4756-91d0-fa2570fd206f`

**cURL**

```bash
curl -X POST http://127.0.0.1:8000/askai \
  -H "Content-Type: text/plain" \
  --data "af4bac4d-077a-4756-91d0-fa2570fd206f"
```

**Response (example)**

```json
{
  "show_popup": true,
  "message": "Need help with your order? Free returns and easy exchanges are included.",
  "reason": "The shopper has added 7 items to cart and spent many hours browsing, but hasn't checked out. A friendly reminder about free returns can encourage completion.",
  "category": "info",
  "trigger": "items_in_cart",
  "confidence": 0.92,
  "delay_ms": 2000
}
```

---

## Data Model (on disk)

`TRACKER_STORAGE_FILE` (JSON object):

```json
{
  "af4bac4d-077a-4756-91d0-fa2570fd206f": {
    "session": "af4bac4d-077a-4756-91d0-fa2570fd206f",
    "itemsInCart": 7,
    "timeOnPage": { "/products/xyz": 123456, "/cart": 9876 },
    "events": {
      "/": [{ "type": "view", "ts": 1712345678901 }],
      "/cart": [{ "type": "view", "ts": 1712345690000 }]
    },
    "updatedAt": 1712345695000
  }
}
```

---

## Browser Snippet (shape expectation)

Your Shopify theme script should:

* Keep a stable `session` (UUID via `crypto.randomUUID()` fallback).
* Track `events`, `timeOnPage`, `itemsInCart`.
* Flush periodically or when reaching a threshold (e.g., every \~150 events).

Minimal example POST:

```js
await fetch("https://<your-backend>/events", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(state) // matches /events body
});
```

Ask AI:

```js
await fetch("https://<your-backend>/askai", {
  method: "POST",
  headers: { "Content-Type": "text/plain" },
  body: state.session
});
```

---

## Deployment

### Render

1. **Create a new Web Service** → **Build & deploy from GitHub**.
2. **Environment**: Python 3.11.
3. **Start command**:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
4. **Environment Variables** (Render Dashboard → Environment):

   * `GROQ_API_KEY` = your key
   * `TRACKER_STORAGE_FILE` = `tracker_storage.json` (optional)
   * `ALLOWED_ORIGINS` = your Shopify domain(s)



## Configuration & Tips

* **CORS**: Make sure `ALLOWED_ORIGINS` in `.env` (or your code) includes your Shopify domain(s).
* **.env in production**: Do **not** commit `.env`. Set variables in your host (Render Dashboard → Environment).
* **Storage**: On Render’s ephemeral FS, the JSON file is reset on redeploy. For persistence, mount a disk or switch to a database.
* **Prompt robustness**: Constrain Groq responses to valid JSON to avoid errors.

---

## Project Structure

```
shopify-tracker-backend/
├─ main.py                 # FastAPI app (routes: /events, /askai, /health)
├─ requirements.txt
├─ .env                    # local only (not committed)
├─ tracker_storage.json    # created on first run (or via env path)
└─ README.md
```



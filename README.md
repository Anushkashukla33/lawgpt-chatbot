# Chat Assistant (FastAPI)

A lightweight, runnable chatbot service with:
- Conversation memory per session
- Tone adaptation (professional, casual, playful)
- Follow-up suggestions per reply
- Optional knowledge fusion via Wikipedia
- Optional LLM integration if `OPENAI_API_KEY` is set
- Minimal web UI

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) To enable LLM responses, set your environment variable:
```bash
export OPENAI_API_KEY=your_key_here
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open the UI at `http://localhost:8000`.

## API
- `POST /chat` – JSON body `{ session_id?, user_message, user_name?, tone_preference? }`
- `GET /memory/{session_id}` – Inspect memory
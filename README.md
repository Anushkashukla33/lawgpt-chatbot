# Chat Assistant (FastAPI)

A lightweight, runnable chatbot service with:
- Conversation memory per session
- Tone adaptation (professional, casual, playful)
- Follow-up suggestions per reply
- Optional knowledge fusion via Wikipedia
- Optional LLM integration if `OPENAI_API_KEY` is set
- Minimal web UI
- RAG pipeline with PDF ingestion and Together (Mistral) model

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) To enable OpenAI LLM responses:
```bash
export OPENAI_API_KEY=your_key_here
```

To enable Together (RAG + Mistral):
```bash
export TOGETHER_API_KEY=your_together_key
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open the UI at `http://localhost:8000`.

## API
- `POST /chat` – JSON body `{ session_id?, user_message, user_name?, tone_preference? }`
- `GET /memory/{session_id}` – Inspect memory
- `POST /rag/ingest` – JSON body `{ "folder": "/absolute/or/workspace/path" }`
- `POST /rag/ask` – JSON body `{ "question": "...", "top_k": 5 }`

### Examples
```bash
# Ingest PDFs from a folder
curl -X POST http://localhost:8000/rag/ingest \
	-H 'Content-Type: application/json' \
	-d '{"folder":"/workspace/data/pdfs"}'

# Ask a RAG question
curl -X POST http://localhost:8000/rag/ask \
	-H 'Content-Type: application/json' \
	-d '{"question":"What is the warranty?","top_k":5}'
```
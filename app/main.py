import os
import time
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ChatRequest, ChatResponse, Source
from .schemas import RagIngestRequest, RagAskRequest
from .memory import memory_store
from .llm import generate_llm_reply
from .knowledge import wikipedia_search
from .utils import apply_tone_prefix, generate_suggestions
from .rag import get_rag


app = FastAPI(title="Chat Assistant")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
	return """
	<!DOCTYPE html>
	<html>
	<head>
		<meta charset='utf-8' />
		<meta name='viewport' content='width=device-width, initial-scale=1' />
		<title>Chat Assistant</title>
		<style>
			body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
			.container { max-width: 760px; margin: 0 auto; }
			.message { padding: 8px 12px; margin: 6px 0; border-radius: 8px; }
			.user { background: #eef6ff; }
			.assistant { background: #f7f7f7; }
			.small { color: #666; font-size: 12px; }
			.suggestions button { margin-right: 6px; margin-top: 6px; }
		</style>
	</head>
	<body>
		<div class='container'>
			<h2>Chat Assistant</h2>
			<div>
				<label>Session ID</label>
				<input id='session' style='width: 260px' />
				<button onclick='newSession()'>New</button>
			</div>
			<div style='margin-top:8px;'>
				<label>Name</label>
				<input id='name' placeholder='Optional' style='width:200px' />
				<label style='margin-left:10px;'>Tone</label>
				<select id='tone'>
					<option value='casual'>casual</option>
					<option value='professional'>professional</option>
					<option value='playful'>playful</option>
				</select>
			</div>
			<div id='chat' style='margin:16px 0;'></div>
			<div>
				<textarea id='input' rows='3' style='width:100%;'></textarea>
				<div style='margin-top:6px;'>
					<button onclick='send()'>Send</button>
					<span id='status' class='small'></span>
				</div>
				<div id='suggestions' class='suggestions'></div>
			</div>
		</div>
		<script>
		async function send(textOverride) {
			const session = document.getElementById('session').value || null;
			const name = document.getElementById('name').value || null;
			const tone = document.getElementById('tone').value || null;
			const input = document.getElementById('input');
			const text = textOverride || input.value.trim();
			if(!text) return;
			append('user', text);
			input.value = '';
			document.getElementById('status').innerText = 'Thinking...';
			try{
				const res = await fetch('/chat', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ session_id: session, user_message: text, user_name: name, tone_preference: tone })
				});
				if(!res.ok){ throw new Error('Request failed'); }
				const data = await res.json();
				document.getElementById('session').value = data.session_id;
				append('assistant', data.response);
				renderSuggestions(data.suggestions);
				document.getElementById('status').innerText = '';
			} catch(e) {
				document.getElementById('status').innerText = 'Error: ' + e.message;
			}
		}
		function append(role, text){
			const div = document.createElement('div');
			div.className = 'message ' + (role === 'user' ? 'user' : 'assistant');
			div.innerText = (role === 'user' ? 'You: ' : 'Assistant: ') + text;
			document.getElementById('chat').appendChild(div);
			document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
		}
		function renderSuggestions(list){
			const container = document.getElementById('suggestions');
			container.innerHTML = '';
			list.forEach(s => {
				const b = document.createElement('button');
				b.innerText = s;
				b.onclick = () => send(s);
				container.appendChild(b);
			});
		}
		function newSession(){ document.getElementById('session').value = ''; document.getElementById('chat').innerHTML = ''; }
		</script>
	</body>
	</html>
	"""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
	session_id = memory_store.get_or_create_session(req.session_id, req.user_name)
	memory_store.add_message(session_id, "user", req.user_message)

	# Knowledge fusion (simple heuristic)
	sources: List[Source] = []
	if len(req.user_message.split()) >= 4:
		wiki = wikipedia_search(req.user_message, limit=2)
		for w in wiki:
			sources.append(Source(title=w["title"], url=w["url"], snippet=w.get("snippet") or ""))

	# Tone and system prompt
	system_prefix = apply_tone_prefix(req.tone_preference, req.user_name)
	context_summary = memory_store.summarize(session_id)
	llm_messages = [
		{"role": "system", "content": system_prefix + "\nYou are an intelligent, friendly assistant. Keep answers accurate and concise."},
	]
	if context_summary:
		llm_messages.append({"role": "system", "content": f"Conversation so far (summary):\n{context_summary}"})
	if sources:
		snips = "\n\n".join([f"- {s.title}: {s.url}" for s in sources])
		llm_messages.append({"role": "system", "content": f"Relevant links you may cite when helpful:\n{snips}"})
	llm_messages.append({"role": "user", "content": req.user_message})

	answer = generate_llm_reply(llm_messages)
	if answer is None:
		# Fallback template-based reply
		if sources:
			src_lines = "\n".join([f"- {s.title} ({s.url})" for s in sources])
			answer = (
				"Here's a concise answer based on your message. I also found a few references you might explore:\n"
				+ src_lines
			)
		else:
			answer = "Got it! Here's a concise response. (Tip: set OPENAI_API_KEY to enable richer answers.)"

	memory_store.add_message(session_id, "assistant", answer)
	suggestions = generate_suggestions(answer)
	mem_sum = memory_store.summarize(session_id)
	return ChatResponse(session_id=session_id, response=answer, suggestions=suggestions, sources=sources, memory_summary=mem_sum)


@app.post("/rag/ingest")
def rag_ingest(body: RagIngestRequest):
	try:
		rag = get_rag()
	except RuntimeError as e:
		raise HTTPException(status_code=400, detail=str(e))
	count = rag.ingest_pdfs(body.folder)
	return {"ingested_chunks": count}


@app.post("/rag/ask")
def rag_ask(body: RagAskRequest):
	try:
		rag = get_rag()
	except RuntimeError as e:
		raise HTTPException(status_code=400, detail=str(e))
	answer, retrieved = rag.answer(body.question, top_k=body.top_k)
	sources = [
		{"text": t, "metadata": m, "score": s}
		for (t, m, s) in retrieved
	]
	return {"answer": answer, "sources": sources}


@app.get("/memory/{session_id}")
def memory(session_id: str):
	msgs = memory_store.get_messages(session_id)
	return {"session_id": session_id, "messages": [m.model_dump() for m in msgs]}
import time
import hashlib
from typing import Dict, List
from .schemas import Message


class ConversationMemory:
	def __init__(self) -> None:
		self._sessions: Dict[str, List[Message]] = {}

	def _generate_session_id(self, seed: str) -> str:
		return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

	def get_or_create_session(self, session_id: str | None, user_name: str | None) -> str:
		if session_id and session_id in self._sessions:
			return session_id
		seed = f"{user_name or 'anon'}-{time.time()}"
		new_id = self._generate_session_id(seed)
		self._sessions[new_id] = []
		return new_id

	def add_message(self, session_id: str, role: str, content: str) -> None:
		messages = self._sessions.setdefault(session_id, [])
		messages.append(Message(role=role, content=content, timestamp=time.time()))

	def get_messages(self, session_id: str) -> List[Message]:
		return list(self._sessions.get(session_id, []))

	def summarize(self, session_id: str, max_chars: int = 500) -> str:
		messages = self._sessions.get(session_id, [])
		if not messages:
			return ""
		parts: List[str] = []
		for m in messages[-10:]:
			prefix = "User" if m.role == "user" else "Assistant" if m.role == "assistant" else "System"
			parts.append(f"{prefix}: {m.content}")
		joined = " \n".join(parts)
		if len(joined) > max_chars:
			return joined[: max_chars - 3] + "..."
		return joined

	def clear(self, session_id: str) -> None:
		self._sessions.pop(session_id, None)


memory_store = ConversationMemory()
import os
from typing import List, Optional
from openai import OpenAI


def get_openai_client() -> Optional[OpenAI]:
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		return None
	return OpenAI(api_key=api_key)


def generate_llm_reply(messages: List[dict]) -> Optional[str]:
	client = get_openai_client()
	if client is None:
		return None
	try:
		completion = client.chat.completions.create(
			model="gpt-4o-mini",
			messages=messages,
			max_tokens=500,
			temperature=0.7,
		)
		return completion.choices[0].message.content
	except Exception:
		return None
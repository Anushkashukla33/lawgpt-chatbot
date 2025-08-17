from typing import List, Literal


Tone = Literal["professional", "casual", "playful"]

def apply_tone_prefix(tone: Tone | None, user_name: str | None) -> str:
	if tone == "professional":
		return f"Please respond concisely and formally. Address the user as {user_name or 'there'}."
	if tone == "playful":
		return f"Use a friendly, upbeat tone with occasional light humor. Address the user as {user_name or 'friend'}."
	# default casual
	return f"Use a warm, conversational tone. Address the user as {user_name or 'there'}."


def generate_suggestions(answer: str) -> List[str]:
	suggestions: List[str] = []
	if "code" in answer.lower() or "implement" in answer.lower():
		suggestions.append("Want a step-by-step breakdown or comments in the code?")
	suggestions.append("Should I save this session for later and name it?")
	suggestions.append("Want sources or deeper research on this topic?")
	return suggestions[:3]
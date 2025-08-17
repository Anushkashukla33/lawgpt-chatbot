from typing import List, Dict
import requests


def wikipedia_search(query: str, limit: int = 2) -> List[Dict[str, str]]:
	try:
		resp = requests.get(
			"https://en.wikipedia.org/w/api.php",
			params={
				"action": "query",
				"list": "search",
				"srsearch": query,
				"format": "json",
				"srlimit": limit,
			},
			timeout=8,
		)
		resp.raise_for_status()
		data = resp.json()
		results = []
		for item in data.get("query", {}).get("search", []):
			title = item.get("title", "")
			snippet = item.get("snippet", "")
			url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
			results.append({"title": title, "url": url, "snippet": snippet})
		return results
	except Exception:
		return []
import os
import re
import json
from typing import Dict, Any, Optional

# LangChain / LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI


def make_llm(model: str, temperature: float = 0.7):
    return ChatGoogleGenerativeAI(
        model=model, temperature=temperature, google_api_key=os.getenv("GEMINI_API_KEY")
    )


def _json_from_llm(resp: str) -> Dict[str, Any]:
    try:
        return json.loads(resp)
    except Exception:
        m = re.search(r"\{.*\}", resp, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

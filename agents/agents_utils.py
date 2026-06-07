import os
import re
import json
from typing import Dict, Any, Optional

# LangChain / LangGraph
from langchain_openai import ChatOpenAI


def make_llm(model: str, temperature: float = 0.7):
    return ChatOpenAI(
        model=model,
        base_url="https://ro-aaea614c7e4a437e97464a09b40e2adc.ecs.us-east-1.on.aws/v1/",
        api_key="router-client",
        temperature=temperature,
    )


def _json_from_llm(resp: str) -> Dict[str, Any]:
    try:
        return json.loads(resp)
    except Exception:
        m = re.search(r"\{.*\}", resp, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

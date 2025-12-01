#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any

# File parsing

# Output

# LangChain / LangGraph
from langchain_core.prompts import ChatPromptTemplate
from openai import BaseModel

from agents.base_agent import BaseAgent
from models import CVParserResponseModel, PipelineState

# ---------- prompts ----------


CV_FACTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a meticulous résumé analyst.
From the CV, extract facts that are actually stated (do not invent):
- titles (array)
- leadership_scope (teams, sizes, locations)
- domains (SaaS, AI, cloud, etc.)
- tech_stack (C#, .NET, Azure, React, Node.js, Python, etc.)
- accomplishments (concise bullets with metrics if present)
- patents_or_publications
Return valid JSON only. with this schema: {schema}""",
        ),
        ("human", "CV TEXT:\n{cv_text}"),
    ]
)


# ---------- nodes ----------
class CVParserAgent(BaseAgent):

    def __init__(self, model, debug=False):
        super().__init__(model, CV_FACTS_PROMPT, debug)

    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        # TODO: extract must requirements separately
        data = super()._invoke_llm(
            {
                "cv_text": state.cv_text,
                "schema": CVParserResponseModel.model_json_schema(),
            },
            node="CVParserAgent",
            response_type=CVParserResponseModel,
        )
        return {"extracted_cv_facts": data.model_dump()}


# ---------- end of file ----------

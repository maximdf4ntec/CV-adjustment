#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Dict, Any

# File parsing

# Output

# LangChain / LangGraph
from langchain_core.prompts import ChatPromptTemplate
from openai import BaseModel

from agents.base_agent import BaseAgent
from models import JobRequirementsResponseModel, PipelineState

# ---------- prompts ----------

REQUIREMENTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise job-requirements analyst.
Extract structured requirements from the job description:
- must_have_skills (array of strings)
- nice_to_have_skills (array of strings)
- responsibilities (array of strings)
- keywords (array of strings; include exact tech terms as they appear)
- seniority_tone ("hands-on", "leadership", "mixed")
Return valid JSON only. with this schema: {schema}  """,
        ),
        ("human", "JOB DESCRIPTION:\n{jd_text}"),
    ]
)

COVER_LETTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Write a concise enterprise-style cover letter:
- 120–180 words
- 3 short paragraphs
- Tone: executive, confident, factual
- Mirror JD language and highlight relevant achievements
- No buzzword stuffing.""",
        ),
        ("human", "JOB REQUIREMENTS JSON:\n{req_json}\n\nCV FACTS JSON:\n{cv_json}"),
    ]
)

OUTREACH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""Write a 50–80 word LinkedIn/InMail outreach blurb:
- Reference the role generically (no company name if unknown)
- Include 1–2 high-impact achievements from CV relevant to JD
- Soft call-to-action for next steps.
""",
        ),
        ("human", "JOB REQUIREMENTS JSON:\n{req_json}\n\nCV FACTS JSON:\n{cv_json}"),
    ]
)


# ---------- nodes ----------
class JobRequirementsAgent(BaseAgent):

    def __init__(self, model, debug=False):
        super().__init__(model, REQUIREMENTS_PROMPT, debug)

    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        # TODO: extract must requirements separately
        data = super()._invoke_llm(
            {
                "jd_text": state.jd_text,
                "schema": JobRequirementsResponseModel.model_json_schema(),
            },
            node="JobRequirementsAgent",
            response_type=JobRequirementsResponseModel,
        )
        return {"extracted_requirements": data.model_dump()}


def node_generate_cover_letter(state: PipelineState, llm) -> PipelineState:
    msg = COVER_LETTER_PROMPT.invoke(
        {
            "req_json": json.dumps(
                state.extracted_requirements, ensure_ascii=False, indent=2
            ),
            "cv_json": json.dumps(
                state.extracted_cv_facts, ensure_ascii=False, indent=2
            ),
        }
    )
    state.cover_letter_md = llm.invoke(msg.to_messages()).content.strip()
    return state


def node_generate_outreach(state: PipelineState, llm) -> PipelineState:
    msg = OUTREACH_PROMPT.invoke(
        {
            "req_json": json.dumps(
                state.extracted_requirements, ensure_ascii=False, indent=2
            ),
            "cv_json": json.dumps(
                state.extracted_cv_facts, ensure_ascii=False, indent=2
            ),
        }
    )
    state.outreach_blurb = llm.invoke(msg.to_messages()).content.strip()
    return state


# ---------- end of file ----------

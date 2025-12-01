#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Dict, Any

# File parsing

# Output

# LangChain / LangGraph
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from models import GapAnalysisResponseModel, PipelineState

# ---------- prompts ----------

GAPS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise job-requirements analyst.

Your task: identify exact alignment between job requirements and the candidate's CV facts. You must produce a structured comparison and identify gaps that prevent full alignment.

STRICT RULES (must follow all):

1. You MUST treat every must-have skill, responsibility, and keyword as an item to be evaluated individually.

2. For each requirement:
   - Perform both EXACT and SEMANTIC comparison with the CV facts.
   - If the CV supports the requirement → classify it as a strong_match.
   - If the CV does NOT support the requirement → classify it as a gap.

3. IMPORTANT:
   - If alignment_score < 100 → gaps MUST NOT be empty.
   - You MUST extract *at least one* gap when score < 100.
   - Each gap MUST have a corresponding user question (a simple question such as “Can you give an example of X?”).
   - gap questions MUST NOT be empty if alignment_score < 100 but no more than 15 questions. Group and deduplicate questions if needed.

4. Compute alignment_score using the required formula:
   
      alignment_score = floor((matched_requirements / total_requirements) * 100)

   - You MUST use this formula.
   - You MUST NOT set alignment_score = 100 if unmatched_requirements > 0.

5. HARD RULE:
   If alignment_score < 100:
        - gaps MUST NOT be empty
        - gap_questions MUST NOT be empty
        - remediation_suggestions MUST NOT be empty
   If alignment_score = 100:
        - gaps = []
        - gap_questions = []
        - remediation_suggestions = []

6. For each GAP:
   - reason: why CV facts do NOT support the requirement
   - gap question: "Can you provide evidence or examples of <requirement>?"

5. Scores:
   - alignment_score: integer 0–100 (percentage of must-have & keyword coverage)
   - strong_matches[*].match_score: integer 0–100

6. remediation_suggestions:
   - For each gap, provide a concrete, actionable suggestion for how the CV could be updated if the user confirms relevant experience.

7. Gap questions:
   - For each gap, formulate a simple, direct question to the user to confirm if they have relevant experience or knowledge to fill the gap.
   Important: questions must be straightforward and easy to answer with a yes/no or brief explanation.
   Important: try to cover all important keywords and must-have skills in the questions.
   Once all gaps are identified, ensure that each gap has a corresponding question.
   At the end, run deduplication to avoid asking the same question multiple times.
8. JSON rules:
   - Always return valid JSON.
   - No text outside the JSON.
   - All arrays must exist even if empty.
   - Follow this schema strictly.
OUTPUT JSON SCHEMA:: {schema}""",
        ),
        ("human", "JOB REQUIREMENTS JSON:\n{req_json}\n\nCV FACTS JSON:\n{cv_json}"),
    ]
)


# ---------- nodes ----------
class GapAnalysisAgent(BaseAgent):

    def __init__(self, model, debug=False):
        super().__init__(model, GAPS_PROMPT, debug)

    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        data = super()._invoke_llm(
            {
                "req_json": state.extracted_requirements.model_dump_json(indent=2),
                "cv_json": state.extracted_cv_facts.model_dump_json(indent=2),
                "schema": GapAnalysisResponseModel.model_json_schema(),
            },
            node="GapAnalysisAgent",
            response_type=GapAnalysisResponseModel,
        )
        return {"gaps_analysis": data.model_dump()}


# ---------- end of file ----------

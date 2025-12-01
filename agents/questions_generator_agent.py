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
from models import PipelineState, QuestionsResponseModel

# ---------- prompts ----------

GAPS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Compare job requirements with the candidate's CV facts.
Return JSON with:
- alignment_summary (short paragraph)
     - alignment_score (0-100)
- strong_matches - for each mached requirement include the requirement and 1-2 sentences how it supported + match score in 0 - 100 range(array)
- gaps - list of non-matched requirements(array)
     -gaps questions - for each gap, formulate a simple question to the user if he has sufficient knowledge to complete the gap(array)
- remediation_suggestions (array)""",
        ),
        ("human", "JOB REQUIREMENTS JSON:\n{req_json}\n\nCV FACTS JSON:\n{cv_json}"),
    ]
)


# ---------- nodes ----------
class QuestionsGeneratorAgent(BaseAgent):

    def __init__(self, model, debug=False):
        super().__init__(model, GAPS_PROMPT, debug)

    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        print(
            f"Gaps analysis: {state.gaps_analysis}, type: {type(state.gaps_analysis)} "
        )
        """ data_list = state.gaps_analysis.gaps_questions
        data = QuestionsResponseModel(
            generated_questions=[
                value for sub_dict in data_list for value in sub_dict.values()
            ]
        ) """
        data = QuestionsResponseModel(
            generated_questions=state.gaps_analysis.gaps_questions
        )
        return {"questions_response": data.model_dump()}


# ---------- end of file ----------

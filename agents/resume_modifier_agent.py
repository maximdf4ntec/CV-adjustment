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
from models import PipelineState, ResumeModifierResponseModel
from utils.utils import count_words_in_text

# ---------- prompts ----------
TAILORED_RESUME_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert résumé writer specialized in senior engineering leadership positions.

Your task: adjust the candidate's existing CV so it aligns better with the given job requirements, WITHOUT inventing any new facts.

STRICT RULES (must follow all):

1. **No hallucinations**
   - You MUST NOT add facts, achievements, skills, technologies, companies, titles, or dates that do not already appear in:
     - The original CV JSON
     - The user's answers JSON

2. **Preserve original CV**
   - Keep ALL original structure: section order, headings, bullet hierarchy.
   - Preserve ALL factual data: names, companies, dates, job titles, locations.
   - Do NOT rewrite the résumé from scratch.
   - Do NOT add sections that did not exist originally.

3. **Modify only where needed**
   - Only update bullets or phrases necessary to fill the identified gaps.
   - If a gap cannot be filled using ONLY the CV or user answers → do NOT fabricate; leave the gap unfilled.

4. **Conciseness and length limits**
   - Write in concise, senior-level executive language.
   - Aim to keep total token count similar to original CV (approx {original_cv_word_count} words). Maximum 3 pages when rendered.
   - Eliminate redundancy and overly long bullets.

5. **Grounding**
   - Every modification must map DIRECTLY to:
     - Job requirements JSON
     - Gaps analysis JSON
     - User answers JSON
   - Do NOT use external knowledge.

6. **Output Format (strict)**
Return ONLY valid JSON with this structure {schema}:

- "markdown" must contain ONLY the edited resume.
- "gaps_filled.completed" must be EMPTY string if gap couldn't be filled without invention.
- "llm_alignment_score" reflects how well the updated résumé matches the job using ONLY facts from CV + answers.
    Adhere strictly to these rules. Your goal is to enhance résumé alignment WITHOUT fabricating any information.
 """,
        ),
        (
            "human",
            "JOB REQUIREMENTS JSON:\n{req_json}\n\nCV Original JSON:\n{cv_json}\n Gaps Analysis JSON:\n{gaps_json} \n USER ANSWERS JSON:\n{answers_json}",
        ),
    ]
)


# ---------- nodes ----------
class ResumeModifierAgent(BaseAgent):
    # TODO: think how to learn from resume without response
    def __init__(self, model, debug=False):
        super().__init__(model, TAILORED_RESUME_PROMPT, debug)

    def invoke(self, state: PipelineState) -> Dict[str, Any]:

        # Check if the required data is actually in the state dictionary:
        if (
            state.extracted_requirements is None
            or state.cv_text is None
            or state.user_answers is None
            or state.gaps_analysis is None
        ):
            print("DEBUG ERROR: Missing required data in state!")
            return {"modified_resume": "Error: Missing data for resume modification."}
        data = super()._invoke_llm(
            {
                "req_json": state.extracted_requirements.model_dump_json(indent=2),
                "cv_json": f'{{"original_resume": "{state.cv_text}"}}',
                "answers_json": state.user_answers.model_dump_json(indent=2),
                "gaps_json": state.gaps_analysis.model_dump_json(indent=2),
                "original_cv_word_count": count_words_in_text(state.cv_text),
                "schema": ResumeModifierResponseModel.model_json_schema(),
            },
            node="ResumeModifierAgent",
            response_type=ResumeModifierResponseModel,
        )

        return {"modified_resume": data.model_dump()}


# ---------- end of file ----------

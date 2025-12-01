#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Dict, Any

# File parsing

# Output

# LangChain / LangGraph
from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import BaseAgent
from models import PipelineState
from utils.utils import count_words_in_markdown, count_words_in_text


# ---------- nodes ----------
class ResumeCandidateScorerAgent:
    def calculate_alignment_score(self, state, modified_resume):
        score = modified_resume.llm_alignment_score
        gaps_filled = modified_resume.gaps_filled
        gap_bonus = 0
        if gaps_filled:
            # count how many gaps were filled by counting non-empty and not "No" answers from user_answers
            total_gaps = sum(
                1 for answer in state.user_answers.answers.values() if answer
            )
            # TODO: fix later, this doesn't work well
            filled_gaps = min(total_gaps, len(gaps_filled))
            gap_bonus = int((filled_gaps / total_gaps) * 10)  # up to 10 points bonus
            score += gap_bonus
            score = min(score, 100)  # cap at 100
        original_words = count_words_in_text(state.cv_text)
        modified_words = count_words_in_markdown(modified_resume.markdown)
        length_increase = (modified_words - original_words) / original_words
        # Penalize for verbosity: subtract 1 point for every 2% increase in length
        score -= int(length_increase / 0.02)
        score = max(score, 0)  # floor at 0
        # TODO this is very bad practice, fix later
        # no more than 25% increase and at least 6 points from gaps filled to be useful
        modified_resume.useful = gap_bonus > 6 and length_increase <= 0.25
        return score

    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        modified_resume = state.modified_resume
        if modified_resume:
            score = self.calculate_alignment_score(state, modified_resume)
            if state.gaps_analysis.alignment_score < score and (
                state.best_resume_candidate is None
                or score > state.best_resume_candidate.alignment_score
            ):
                modified_resume.alignment_score = score
                state.best_resume_candidate = modified_resume
        return {
            "best_resume_candidate": (
                state.best_resume_candidate.model_dump()
                if state.best_resume_candidate
                else None
            )
        }


# ---------- end of file ----------

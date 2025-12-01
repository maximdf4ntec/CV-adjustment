#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tailor Resume with LangGraph
----------------------------
A no-UI Python program that:
- Reads your CV (DOCX/PDF/TXT/MD)
- Fetches a Job Description (from URL or local file)
- Uses LangGraph + LangChain + OpenAI to:
  * Extract job requirements & keywords
  * Extract CV facts (experience, stack, metrics)
  * Analyze gaps and propose targeted improvements
  * Generate a tailored resume (docx + md)
  * Generate a tailored cover letter (docx + md)
  * Generate a concise outreach blurb (txt)

Usage:
  export GEMINI_API_KEY=...

  python tailor_resume_langgraph.py \
      --cv /path/to/Maxim_Donde_Master_CV.docx \
      --jd_url "https://company.com/job/123" \
      --model gpt-4o \
      --out_dir ./output

  # or with a local JD file:
  python tailor_resume_langgraph.py \
      --cv /path/to/Maxim_Donde_Master_CV.docx \
      --jd_file ./samples/job_desc.txt \
      --model gpt-4o \
      --out_dir ./output
"""

import argparse
import os
import sys
from typing import Dict

from traitlets import Any
from agents.cv_parser_agent import CVParserAgent
from agents.gap_analysis_agent import GapAnalysisAgent
from agents.questions_generator_agent import QuestionsGeneratorAgent
from agents.resume_candidate_scorer_agent import ResumeCandidateScorerAgent
from agents.resume_modifier_agent import ResumeModifierAgent

from utils.utils import (
    fetch_job_description_from_url,
    normalize_text,
    read_text_from_file,
    write_output_files,
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from models import AnswersResponseModel, PipelineState
from agents.job_requirements_agent import JobRequirementsAgent


# ---------- graph logic helpers ----------
def check_for_human_answers(state: PipelineState) -> str:
    """Routing logic: Checks if the user has provided answers yet."""
    # This logic assumes the user runs the app again with answers in the input
    if state.user_answers:
        return "answers_provided"
    else:
        # If answers aren't there, the graph must stop temporarily
        return "wait_for_human"


""" 
    def update_best_candidate(state: PipelineState) -> Dict[str, Any]:
    # Routing logic: Checks if the generated resume is good enough.
    # TODO more advanced logic for generated resume to be implemented here
    modified_resume = state.modified_resume
    if modified_resume and isinstance(modified_resume, dict):
        score = modified_resume.get("llm_aligment_score", 0)
        if state.gaps_analysis.get("alignment_score", 0) < score and (
            state.best_resume_candidate is None
            or score > state.best_resume_candidate.get("llm_aligment_score", 0)
        ):
            state.best_resume_candidate = modified_resume
    return {"best_resume_candidate": state.best_resume_candidate}
"""


def check_generated_resume(state: PipelineState) -> str:
    best_generated_resume = state.best_resume_candidate
    result = (
        "max_attempts_reached"
        if state.generate_attempt_counter >= 3
        else "not_good_enough"
    )
    if best_generated_resume:
        is_useful = (
            best_generated_resume.useful
            if best_generated_resume.useful is not None
            else False
        )
        if is_useful:
            result = "good_enough"
        elif state.generate_attempt_counter >= 3:
            result = "max_attempts_reached"
        else:
            result = "not_good_enough"
    return result


def increment_attempt_counter(state: PipelineState) -> Dict[str, Any]:
    """Increments the attempt counter in the state metadata."""
    counter = state.generate_attempt_counter
    state.generate_attempt_counter = counter + 1
    print(f"Resume modification attempt #{state.generate_attempt_counter}")
    return {"generate_attempt_counter": state.generate_attempt_counter}


# ---------- CLI ----------
def build_graph(model: str):
    job_requirements_agent = JobRequirementsAgent(model)
    cv_parser_agent = CVParserAgent(model)
    gap_analysis_agent = GapAnalysisAgent(model)
    questions_generator_agent = QuestionsGeneratorAgent(model)
    resume_modifier_agent = ResumeModifierAgent(model, debug=True)
    resume_candidate_scorer_agent = ResumeCandidateScorerAgent()
    graph = StateGraph(PipelineState)
    graph.add_node("extract_requirements", job_requirements_agent.invoke)
    graph.add_node("extract_cv_facts", cv_parser_agent.invoke)
    graph.add_node("gaps_analysis", gap_analysis_agent.invoke)
    graph.add_node("questions_generator", questions_generator_agent.invoke)
    graph.add_node("resume_modifier", resume_modifier_agent.invoke)
    graph.add_node("increment_counter", increment_attempt_counter)
    graph.add_node("update_best_candidate", resume_candidate_scorer_agent.invoke)
    # graph.add_node("generate_resume", _resume)
    # -- graph.add_node("generate_cover_letter", _cover)
    # -- graph.add_node("generate_outreach", _outreach)

    graph.set_entry_point("extract_requirements")
    graph.add_edge("extract_requirements", "extract_cv_facts")
    graph.add_edge("extract_cv_facts", "gaps_analysis")
    # TODO add conditional edge here based on gaps_analysis results (if no gaps, skip questions)
    graph.add_edge("gaps_analysis", "questions_generator")
    graph.add_conditional_edges(
        "questions_generator",  # This node runs first to ask questions
        check_for_human_answers,  # The router logic
        {
            "answers_provided": "resume_modifier",  # If we have answers, go to modifier
            "wait_for_human": END,  # If not, STOP here (wait for next user input)
        },
    )
    graph.add_edge("resume_modifier", "update_best_candidate")
    graph.add_conditional_edges(
        "update_best_candidate",  # This node runs after resume modification
        check_generated_resume,  # The router logic
        {
            "good_enough": END,
            "not_good_enough": "increment_counter",  # Loop back for more Q&A
            "max_attempts_reached": END,
        },
    )
    graph.add_edge("increment_counter", "resume_modifier")
    # -- graph.add_edge("generate_cover_letter", END)
    # -- graph.add_edge("generate_outreach", END)
    return graph.compile(checkpointer=MemorySaver())


def main():
    parser = argparse.ArgumentParser(
        description="Tailor a CV to a job description using LangGraph."
    )
    parser.add_argument("--cv", required=True, help="Path to CV (.docx/.pdf/.txt/.md)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jd_url", help="URL to the job description")
    group.add_argument("--lnkd_id", help="Job description LinkedIn ID")
    parser.add_argument(
        "--model", default="gpt-4o", help="OpenAI model name (e.g., gpt-4o, gpt-4.1)"
    )
    parser.add_argument("--out_dir", default="./output", help="Output directory")
    args = parser.parse_args()
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Please set GEMINI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    cv_text = normalize_text(read_text_from_file(args.cv))
    if args.jd_url:
        jd_text = normalize_text(fetch_job_description_from_url(args.jd_url))
        meta = {"jd_source": "url", "jd_url": args.jd_url}
    elif args.lnkd_id:
        jd_text = normalize_text(
            fetch_job_description_from_url(
                f"https://www.linkedin.com/jobs/view/{args.lnkd_id}"
            )
        )
        meta = {"jd_source": "linkedin", "lnkd_id": args.lnkd_id}

    initial_state = PipelineState(
        cv_text=cv_text, jd_text=jd_text, lnkd_id=args.lnkd_id, meta=meta
    ).model_dump()
    graph = build_graph(model=args.model)
    questions_state_raw = graph.invoke(
        initial_state, {"configurable": {"thread_id": 1}}
    )
    questions_state = PipelineState(**questions_state_raw)
    print("\n--- Generated Questions for User ---")
    answers_from_user = {}
    if questions_state.questions_response.generated_questions:
        for question in questions_state.questions_response.generated_questions:
            print(f"Q: {question}")
            answer = input("A: ")
            answers_from_user[question] = (
                answer if answer.strip().lower() != "no" else None
            )
    else:
        print("No questions generated. The resume is good enough as is.")
        return
    user_input = {
        "user_answers": AnswersResponseModel(answers=answers_from_user).model_dump()
    }
    print("\n--- Invoking Phase 2 (Applying Answers) ---")
    final_state = graph.invoke(user_input, {"configurable": {"thread_id": 1}})
    outputs = write_output_files(PipelineState(**final_state), args.out_dir)
    print("Done. Outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

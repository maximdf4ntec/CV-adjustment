from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class QuestionsResponseModel(BaseModel):
    generated_questions: list[str]


class AnswersResponseModel(BaseModel):
    answers: Dict[str, Optional[str]]


class ResumeModifierResponseModel(BaseModel):
    markdown: str
    gaps_filled: list[str]
    llm_alignment_score: int
    alignment_score: Optional[int] = None  # added for internal use
    useful: Optional[bool] = None  # added for internal use


class JobRequirementsResponseModel(BaseModel):
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    responsibilities: list[str]
    keywords: list[str]
    seniority_tone: str


class CVParserResponseModel(BaseModel):
    titles: list[str]
    leadership_scope: list[str]
    domains: list[str]
    tech_stack: list[str]
    accomplishments: list[str]
    patents_or_publications: list[str]


class GapAnalysisResponseModel(BaseModel):
    alignment_summary: str
    alignment_score: int
    strong_matches: list[str]
    gaps: list[str]
    gaps_questions: list[str]
    remediation_suggestions: list[str]


# ---------- state ----------
class PipelineState(BaseModel):
    cv_text: str
    jd_text: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    extracted_requirements: Optional[JobRequirementsResponseModel] = None
    extracted_cv_facts: Optional[CVParserResponseModel] = None
    gaps_analysis: Optional[GapAnalysisResponseModel] = None
    questions_response: Optional[QuestionsResponseModel] = None
    user_answers: Optional[AnswersResponseModel] = None
    modified_resume: Optional[ResumeModifierResponseModel] = None
    # TODO best candiate should be initialized with the original resume data
    best_resume_candidate: Optional[ResumeModifierResponseModel] = None
    cover_letter: Optional[str] = None
    outreach_mail: Optional[str] = None
    generate_attempt_counter: int = 1
    lnkd_id: Optional[str] = None

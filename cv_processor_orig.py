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
import re
import sys
import json
import time
from typing import Dict, Any, Optional

# Text extraction
import requests
from bs4 import BeautifulSoup
import trafilatura

# File parsing
from docx import Document as DocxDocument
import subprocess
from pathlib import Path

# Output
from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH

# LangChain / LangGraph
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ---------- helpers: file & text ----------

def read_text_from_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()

def read_text_from_pdf(path: str) -> str:
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    except Exception:
        try:
            txt_path = path + ".txt"
            subprocess.run(["pdftotext", path, txt_path], check=True)
            return Path(txt_path).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF. Install pdfplumber or pdftotext. Error: {e}")

def read_text_from_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return read_text_from_docx(path)
    if ext == ".pdf":
        return read_text_from_pdf(path)
    if ext in {".txt", ".md"}:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file extension: {ext}")

def fetch_job_description_from_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if extracted and len(extracted.split()) > 50:
            return extracted.strip()
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---------- state ----------

class PipelineState(BaseModel):
    cv_text: str
    jd_text: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    extracted_requirements: Optional[Dict[str, Any]] = None
    extracted_cv_facts: Optional[Dict[str, Any]] = None
    gaps_analysis: Optional[Dict[str, Any]] = None
    tailored_resume_md: Optional[str] = None
    cover_letter_md: Optional[str] = None
    outreach_blurb: Optional[str] = None

# ---------- prompts ----------

REQUIREMENTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise job-requirements analyst.
Extract structured requirements from the job description:
- must_have_skills (array of strings)
- nice_to_have_skills (array of strings)
- responsibilities (array of strings)
- keywords (array of strings; include exact tech terms as they appear)
- seniority_tone ("hands-on", "leadership", "mixed")
Return valid JSON only."""),
    ("human", "JOB DESCRIPTION:\n{jd_text}")
])

CV_FACTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a meticulous résumé analyst.
From the CV, extract facts that are actually stated (do not invent):
- titles (array)
- leadership_scope (teams, sizes, locations)
- domains (SaaS, AI, cloud, etc.)
- tech_stack (C#, .NET, Azure, React, Node.js, Python, etc.)
- accomplishments (concise bullets with metrics if present)
- patents_or_publications
Return valid JSON only."""),
    ("human", "CV TEXT:\n{cv_text}")
])

GAPS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Compare job requirements with the candidate's CV facts.
Return JSON with:
- alignment_summary (short paragraph)
- strong_matches (array)
- gaps (array)
- remediation_suggestions (array)"""),
    ("human", "JOB REQUIREMENTS JSON:\n{req_json}\n\nCV FACTS JSON:\n{cv_json}")
])

TAILORED_RESUME_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert résumé writer for senior engineering leadership.
Create a tailored résumé (Markdown) that:
- mirrors the job's keywords and responsibilities
- keeps only true facts from the CV (no invention)
- emphasizes relevant achievements and tech
- stays concise (≈2 pages when rendered)
Sections:
1) Name + contacts (placeholder if not provided)
2) Professional Summary (3–4 lines aligned to JD)
3) Selected Achievements (4–6 bullets; metrics where possible)
4) Experience (reverse-chronological; 3–5 bullets per role relevant to JD)
5) Technical Skills (match JD terms)
6) Education, Patents (if any)
Return pure Markdown."""),
    ("human", "JOB REQUIREMENTS JSON:\n{req_json}\n\nCV FACTS JSON:\n{cv_json}")
])

COVER_LETTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Write a concise enterprise-style cover letter:
- 120–180 words
- 3 short paragraphs
- Tone: executive, confident, factual
- Mirror JD language and highlight relevant achievements
- No buzzword stuffing."""),
    ("human", "JOB REQUIREMENTS JSON:\n{req_json}\n\nCV FACTS JSON:\n{cv_json}")
])

OUTREACH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Write a 50–80 word LinkedIn/InMail outreach blurb:
- Reference the role generically (no company name if unknown)
- Include 1–2 high-impact achievements from CV relevant to JD
- Soft call-to-action for next steps."""),
    ("human", "JOB REQUIREMENTS JSON:\n{req_json}\n\nCV FACTS JSON:\n{cv_json}")
])

# ---------- nodes ----------

def make_llm(model: str, temperature: float = 0.7):
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=os.getenv("GEMINI_API_KEY"))

def _json_from_llm(resp: str) -> Dict[str, Any]:
    try:
        return json.loads(resp)
    except Exception:
        m = re.search(r"\{.*\}", resp, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def node_extract_requirements(state: PipelineState, llm) -> PipelineState:
    # TODO: extract must requirements separately
    msg = REQUIREMENTS_PROMPT.invoke({"jd_text": state.jd_text})
    data = _json_from_llm(llm.invoke(msg.to_messages()).content)
    state.extracted_requirements = data
    return state

def node_extract_cv_facts(state: PipelineState, llm) -> PipelineState:
    # TODO: R&D Director and Group Leader with 20+ years of experience building and scaling - this info is missing from CV facts
    # TODO: bridging technology and business needs - also missing
    msg = CV_FACTS_PROMPT.invoke({"cv_text": state.cv_text})
    data = _json_from_llm(llm.invoke(msg.to_messages()).content)
    state.extracted_cv_facts = data
    return state

def node_gaps_analysis(state: PipelineState, llm) -> PipelineState:
    # TODO: prepare a file with questions for gaps analysis
    msg = GAPS_PROMPT.invoke({
        "req_json": json.dumps(state.extracted_requirements, ensure_ascii=False, indent=2),
        "cv_json": json.dumps(state.extracted_cv_facts, ensure_ascii=False, indent=2),
    })
    data = _json_from_llm(llm.invoke(msg.to_messages()).content)
    state.gaps_analysis = data
    return state

def node_generate_tailored_resume(state: PipelineState, llm) -> PipelineState:
    # TODO: this works bad, the prompt should only complete missing parts based on gaps analysis
    msg = TAILORED_RESUME_PROMPT.invoke({
        "req_json": json.dumps(state.extracted_requirements, ensure_ascii=False, indent=2),
        "cv_json": json.dumps(state.extracted_cv_facts, ensure_ascii=False, indent=2),
    })
    state.tailored_resume_md = llm.invoke(msg.to_messages()).content.strip()
    return state

def node_generate_cover_letter(state: PipelineState, llm) -> PipelineState:
    msg = COVER_LETTER_PROMPT.invoke({
        "req_json": json.dumps(state.extracted_requirements, ensure_ascii=False, indent=2),
        "cv_json": json.dumps(state.extracted_cv_facts, ensure_ascii=False, indent=2),
    })
    state.cover_letter_md = llm.invoke(msg.to_messages()).content.strip()
    return state

def node_generate_outreach(state: PipelineState, llm) -> PipelineState:
    msg = OUTREACH_PROMPT.invoke({
        "req_json": json.dumps(state.extracted_requirements, ensure_ascii=False, indent=2),
        "cv_json": json.dumps(state.extracted_cv_facts, ensure_ascii=False, indent=2),
    })
    state.outreach_blurb = llm.invoke(msg.to_messages()).content.strip()
    return state

# ---------- output writers ----------

def md_to_docx(md_text: str, title: str, save_path: str):
    doc = DocxDocument()
    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in md_text.splitlines():
        if re.match(r"^\s*#{1,6}\s+", line):
            text = re.sub(r"^\s*#{1,6}\s+", "", line).strip()
            p = doc.add_paragraph()
            run = p.add_run(text); run.bold = True
        elif line.strip().startswith(("-", "*")):
            doc.add_paragraph(line.strip())
        else:
            doc.add_paragraph(line)
    doc.save(save_path)

def write_output_files(state: PipelineState, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # JSON artifacts
    (Path(out_dir)/f"requirements_{ts}.json").write_text(
        json.dumps(state.extracted_requirements, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(out_dir)/f"cv_facts_{ts}.json").write_text(
        json.dumps(state.extracted_cv_facts, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(out_dir)/f"gaps_{ts}.json").write_text(
        json.dumps(state.gaps_analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown + DOCX
    resume_md = Path(out_dir)/f"tailored_resume_{ts}.md"
    cover_md  = Path(out_dir)/f"cover_letter_{ts}.md"
    resume_md.write_text(state.tailored_resume_md or "", encoding="utf-8")
    cover_md.write_text(state.cover_letter_md or "", encoding="utf-8")

    resume_docx = Path(out_dir)/f"tailored_resume_{ts}.docx"
    cover_docx  = Path(out_dir)/f"cover_letter_{ts}.docx"
    md_to_docx(state.tailored_resume_md or "", "Tailored Resume", str(resume_docx))
    md_to_docx(state.cover_letter_md or "", "Cover Letter", str(cover_docx))

    # Outreach
    outreach = Path(out_dir)/f"outreach_{ts}.txt"
    outreach.write_text(state.outreach_blurb or "", encoding="utf-8")

    return {
        "resume_md": str(resume_md),
        "resume_docx": str(resume_docx),
        "cover_md": str(cover_md),
        "cover_docx": str(cover_docx),
        "outreach": str(outreach),
    }

# ---------- graph ----------

def build_graph(model: str):
    llm = make_llm(model=model)

    def _extract_requirements(d: Dict[str, Any]): 
        ps = PipelineState(**d.__dict__); ps = node_extract_requirements(ps, llm); return ps.model_dump()
        
    def _extract_cv_facts(d: Dict[str, Any]): 
        ps = PipelineState(**d.__dict__); ps = node_extract_cv_facts(ps, llm); return ps.model_dump()

    def _gaps(d: Dict[str, Any]): 
        ps = PipelineState(**d.__dict__); ps = node_gaps_analysis(ps, llm); return ps.model_dump()

    def _resume(d: Dict[str, Any]): 
        ps = PipelineState(**d.__dict__); ps = node_generate_tailored_resume(ps, llm); return ps.model_dump()

    def _cover(d: Dict[str, Any]): 
        ps = PipelineState(**d.__dict__); ps = node_generate_cover_letter(ps, llm); return ps.model_dump()

    def _outreach(d: Dict[str, Any]): 
        ps = PipelineState(**d.__dict__); ps = node_generate_outreach(ps, llm); return ps.model_dump()

    graph = StateGraph(PipelineState)
    graph.add_node("extract_requirements", _extract_requirements)
    graph.add_node("extract_cv_facts", _extract_cv_facts)
    graph.add_node("gaps_analysis", _gaps)
    graph.add_node("generate_resume", _resume)
    #-- graph.add_node("generate_cover_letter", _cover)
    # -- graph.add_node("generate_outreach", _outreach)

    graph.set_entry_point("extract_requirements")
    graph.add_edge("extract_requirements", "extract_cv_facts")
    graph.add_edge("extract_cv_facts", "gaps_analysis")
    graph.add_edge("gaps_analysis", "generate_resume")
    graph.add_edge("generate_resume", END)
    #-- graph.add_edge("generate_cover_letter", END)
    # -- graph.add_edge("generate_outreach", END)

    return graph.compile()

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Tailor a CV to a job description using LangGraph.")
    parser.add_argument("--cv", required=True, help="Path to CV (.docx/.pdf/.txt/.md)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jd_url", help="URL to the job description")
    group.add_argument("--jd_file", help="Path to local job description file (.txt/.md/.docx/.pdf)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name (e.g., gpt-4o, gpt-4.1)")
    parser.add_argument("--out_dir", default="./output", help="Output directory")
    args = parser.parse_args()
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Please set GEMINI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    cv_text = normalize_text(read_text_from_file(args.cv))
    if args.jd_url:
        jd_text = normalize_text(fetch_job_description_from_url(args.jd_url))
        meta = {"jd_source": "url", "jd_url": args.jd_url}
    else:
        jd_text = normalize_text(read_text_from_file(args.jd_file))
        meta = {"jd_source": "file", "jd_file": args.jd_file}

    state = PipelineState(cv_text=cv_text, jd_text=jd_text, meta=meta).model_dump()
    app = build_graph(model=args.model)
    final_state = app.invoke(state)

    outputs = write_output_files(PipelineState(**final_state), args.out_dir)
    print("Done. Outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import time
from typing import Dict, Any

# Text extraction
from pydantic import BaseModel
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

from models import ResumeModifierResponseModel

# LangChain / LangGraph


# ---------- helpers: file & text ----------
def count_words_in_markdown(markdown: str) -> int:
    # Convert markdown to plain text (approximate) by stripping markdown syntax
    # A simple regex can remove some syntax, but a full parser is better
    # For a more robust solution, you'd use a dedicated library that renders to plain text
    plain_text = re.sub(
        r"#+\s*|[*_`>()\[\]]", "", markdown
    )  # Simplified syntax removal
    words = plain_text.split()
    return len(words)


def count_words_in_text(text: str) -> int:
    words = text.split()
    return len(words)


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
            raise RuntimeError(
                f"Failed to read PDF. Install pdfplumber or pdftotext. Error: {e}"
            )


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
        extracted = trafilatura.extract(
            downloaded, include_comments=False, include_tables=False
        )
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


# ---------- output writers ----------


def md_to_docx(md_text: str, title: str, save_path: str):
    doc = DocxDocument()
    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in md_text.splitlines():
        if re.match(r"^\s*#{1,6}\s+", line):
            text = re.sub(r"^\s*#{1,6}\s+", "", line).strip()
            p = doc.add_paragraph()
            run = p.add_run(text)
            run.bold = True
        elif line.strip().startswith(("-", "*")):
            doc.add_paragraph(line.strip())
        else:
            doc.add_paragraph(line)
    doc.save(save_path)


def write_output_files(data: Dict[str, Any], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    lnkd_id = data.lnkd_id
    keys_to_dump = [
        "gaps_analysis",
        "best_resume_candidate",
        "cover_letter",
        "outreach_mail",
    ]
    for key, content in data:
        if key not in keys_to_dump:
            continue
        print(f"Writing {key} with {content} to output directory...")
        if isinstance(content, BaseModel):
            json_file = Path(out_dir) / f"{key}_{lnkd_id or ts}.json"
            json_file.write_text(
                content.model_dump_json(indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if isinstance(content, ResumeModifierResponseModel) and content.markdown:
                # md_file = Path(out_dir) / f"{key}_{ts}.md"
                # md_file.write_text(content or "", encoding="utf-8")
                md_to_docx(
                    content.markdown or "",
                    key.replace("_", " ").title(),
                    str(Path(out_dir) / f"{key}_{lnkd_id or ts}.docx"),
                )
    """    
    resume_md.write_text(state.tailored_resume_md or "", encoding="utf-8")
    cover_md.write_text(state.cover_letter_md or "", encoding="utf-8")

    resume_docx = Path(out_dir) / f"tailored_resume_{ts}.docx"
    cover_docx = Path(out_dir) / f"cover_letter_{ts}.docx"
    md_to_docx(state.tailored_resume_md or "", "Tailored Resume", str(resume_docx))
    md_to_docx(state.cover_letter_md or "", "Cover Letter", str(cover_docx))
    """
    return {}

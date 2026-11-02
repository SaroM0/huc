"""Draft exam YAMLs (questions + rubric) from the teacher's reference-solutions PDFs.

A solutions PDF holds every module of one week, each section headed `Taller S14M1` or
`Taller 4, M1`. `find_sections` slices it by those headers (text layer, no LLM);
`draft_exam` sends the section's rendered pages (math survives rendering, not text
extraction) to Claude, which returns the questions and criteria; `exam_yaml` writes the
draft for the teacher to review before `huc exam-load`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pymupdf
import yaml
from pydantic import BaseModel, Field

from huc.domain.models import Criterion, Exam, Question, QuestionType
from huc.domain.profile import CourseProfile
from huc.ingestion.loaders import Page, load_pdf
from huc.llm import build_messages, structured
from huc.prompts import rubric as prompts

# "Taller S14M1", "Taller S2L5", "Taller 4, L5" (week 1 uses "Taller 1|2|3, <module>")
MODULE_HEADER = re.compile(r"^[ \t]*Taller[ \t]+S?(\d+)[ \t]*,?[ \t]*([LMW]\d)[ \t]*$", re.MULTILINE)
# Week in the file name: soluciones_S14.pdf, solW14.pdf, SOLUCION_solW14.pdf
WEEK_IN_NAME = re.compile(r"[SW](\d+)")


@dataclass
class SolutionSection:
    module: str  # "M1"
    header: str  # as printed, e.g. "Taller S14M1"
    taller: int  # number in the header
    pages: list[int]  # 1-based pages the section spans
    text: str


def find_sections(path: Path) -> dict[str, SolutionSection]:
    """Module -> section of a solutions PDF, in document order."""
    with pymupdf.open(path) as doc:
        page_texts = [page.get_text() for page in doc]
    # Offsets of each page in the joined text, to map a section back to its pages
    starts, text, offset = [], "", 0
    for t in page_texts:
        starts.append(offset)
        text += t
        offset += len(t)

    def page_of(pos: int) -> int:
        return max(i for i, s in enumerate(starts) if s <= pos) + 1

    matches = list(MODULE_HEADER.finditer(text))
    sections: dict[str, SolutionSection] = {}
    for m, nxt in zip(matches, [*matches[1:], None], strict=True):
        module = m.group(2)
        if module in sections:  # a later mention of the same header is not a new section
            continue
        end = nxt.start() if nxt else len(text)
        first, last = page_of(m.start()), page_of(max(end - 1, m.start()))
        sections[module] = SolutionSection(
            module=module,
            header=m.group(0).strip(),
            taller=int(m.group(1)),
            pages=list(range(first, last + 1)),
            text=text[m.start() : end].strip(),
        )
    return sections


def week_of(path: Path, sections: dict[str, SolutionSection]) -> int | None:
    """Week from the file name, else the smallest taller number (week 1 numbers its
    tallers 1-3, later weeks use the week number)."""
    m = WEEK_IN_NAME.search(path.stem)
    if m:
        return int(m.group(1))
    return min((s.taller for s in sections.values()), default=None)


# -- structured output of the drafter --------------------------------------


class DraftLevel(BaseModel):
    points: int
    descriptor: str


class DraftCriterion(BaseModel):
    id: str
    description: str
    max_points: int = Field(ge=1)
    levels: list[DraftLevel]


class DraftQuestion(BaseModel):
    id: str
    number: str
    prompt: str
    reference_answer: str
    subject: str | None = None
    skill: str | None = None
    constraints: list[str] = Field(default_factory=list)
    criteria: list[DraftCriterion]


class RubricDraft(BaseModel):
    questions: list[DraftQuestion]


def _unique(ids: list[str]) -> list[str]:
    """Make ids unique by suffixing repeats: [a, a, b] -> [a, a_2, b]."""
    seen: dict[str, int] = {}
    out = []
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
        out.append(i if seen[i] == 1 else f"{i}_{seen[i]}")
    return out


def to_exam(draft: RubricDraft, profile: CourseProfile, exam_id: str, title: str) -> Exam:
    """Validate the model's draft: unique ids, known subjects, levels within the maximum."""
    subjects = {s.id for s in profile.subjects}
    questions = []
    for qid, dq in zip(_unique([q.id for q in draft.questions]), draft.questions, strict=True):
        criteria = [
            Criterion(
                id=cid,
                description=dc.description,
                max_points=dc.max_points,
                levels={
                    str(lv.points): lv.descriptor
                    for lv in sorted(dc.levels, key=lambda lv: -lv.points)
                    if 0 <= lv.points <= dc.max_points
                },
            )
            for cid, dc in zip(_unique([c.id for c in dq.criteria]), dq.criteria, strict=True)
        ]
        questions.append(
            Question(
                id=qid,
                number=dq.number,
                type=QuestionType.SHORT_ANSWER,
                prompt=dq.prompt,
                reference_answer=dq.reference_answer,
                subject=dq.subject if dq.subject in subjects else None,
                skill=dq.skill,
                constraints=dq.constraints,
                criteria=criteria,
            )
        )
    return Exam(id=exam_id, title=title, profile=profile, questions=questions)


async def draft_exam(
    model: Any,
    section: SolutionSection,
    pages: list[Page],
    profile: CourseProfile,
    *,
    exam_id: str,
    title: str,
) -> Exam:
    content: list[dict] = []
    for page in pages:
        content += [{"type": "text", "text": f"Page {page.number}:"}, page.to_content_block()]
    content.append(prompts.section_text_block(section.text))
    messages = build_messages(prompts.rubric_system(section.header, profile), content)
    draft = await structured(model, RubricDraft).ainvoke(messages)
    return to_exam(draft, profile, exam_id, title)


def section_pages(path: Path, section: SolutionSection, dpi: int, max_side: int) -> list[Page]:
    return load_pdf(path, dpi, max_side, page_numbers=section.pages).pages


# -- YAML ------------------------------------------------------------------


class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str)


# Key order of each question in the draft: what the reviewer reads first, then the rubric
QUESTION_KEYS = ["id", "number", "type", "subject", "skill", "constraints", "prompt",
                 "reference_answer", "criteria"]


def _num(x: float) -> float | int:
    return int(x) if float(x).is_integer() else x


def exam_yaml(exam: Exam, *, profile_id: str | None, source: str, model: str) -> str:
    """The exam as a YAML draft that `huc exam-load` accepts (profile by id, not inlined)."""
    data: dict = {"id": exam.id, "title": exam.title, "rubric_version": 1}
    if profile_id:
        data["profile"] = profile_id
    questions = []
    for q in exam.questions:
        d = q.model_dump(mode="json", exclude_defaults=True)
        for c in d.get("criteria", []):
            c["max_points"] = _num(c["max_points"])
        questions.append({k: d[k] for k in QUESTION_KEYS if k in d})
    data["questions"] = questions
    header = (
        f"# DRAFT rubric generated by `huc rubric-draft` ({model}, {date.today()})\n"
        f"# Source: {source}\n"
        "# Review prompts, reference answers and criteria before `huc exam-load`;\n"
        "# the teacher's solutions give no points, so the scale is the model's proposal.\n"
    )
    body = yaml.dump(data, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=100)
    return header + body

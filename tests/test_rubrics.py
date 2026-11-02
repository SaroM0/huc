"""Rubric drafting from solutions PDFs, and the frequent-errors report."""

from pathlib import Path

import pymupdf
import pytest
from conftest import FakeLLM, FakeOCR
from test_profile import PROFILES

from huc import rubrics
from huc.config import Settings
from huc.domain.models import CriterionScore, Grade, QuestionGrade
from huc.domain.profile import CourseProfile, ErrorType, GradeBand, GradingPolicy
from huc.report import build_report
from huc.services import HucService
from huc.storage.repository import FileRepository


def make_solutions_pdf(path: Path, sections: list[tuple[str, list[str]]]) -> Path:
    """One page per entry: (header, body lines); header "" continues the previous section."""
    doc = pymupdf.open()
    for header, lines in sections:
        page = doc.new_page()
        text = "\n".join(([header] if header else []) + lines)
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    return path


@pytest.fixture
def solutions(tmp_path) -> Path:
    return make_solutions_pdf(
        tmp_path / "soluciones_S14.pdf",
        [
            ("Taller S14L5", ["1. Problema L5", "Solucion."]),
            ("", ["2. sigue L5"]),
            ("Taller S14M1", ["1. Problema M1", "Solucion. Taller S14M1 citado"]),
        ],
    )


DRAFT = rubrics.RubricDraft(
    questions=[
        rubrics.DraftQuestion(
            id="p1",
            number="1",
            prompt="Demuestre que E es infinito.",
            reference_answer="Para cada k existe x_k ∈ E...",
            subject="calculo",
            skill="Demostrar",
            criteria=[
                rubrics.DraftCriterion(
                    id="idea",
                    description="Usa la densidad",
                    max_points=2,
                    levels=[
                        rubrics.DraftLevel(points=0, descriptor="Ausente"),
                        rubrics.DraftLevel(points=2, descriptor="Correcta"),
                        rubrics.DraftLevel(points=5, descriptor="fuera de rango"),
                    ],
                ),
                rubrics.DraftCriterion(
                    id="idea", description="Distintos", max_points=1, levels=[]
                ),
            ],
        ),
        rubrics.DraftQuestion(
            id="p1", number="2", prompt="b", reference_answer="r", subject="fisica",
            criteria=[rubrics.DraftCriterion(id="answer", description="Correcta", max_points=1,
                                             levels=[])],
        ),
    ]
)


def test_sections_are_sliced_by_module_header(solutions):
    sections = rubrics.find_sections(solutions)
    assert list(sections) == ["L5", "M1"]
    assert sections["L5"].pages == [1, 2] and "sigue L5" in sections["L5"].text
    # A header quoted inside a line does not start a section
    assert sections["M1"].pages == [3] and sections["M1"].header == "Taller S14M1"
    assert rubrics.week_of(solutions, sections) == 14


def test_week_falls_back_to_the_smallest_taller_number(tmp_path):
    pdf = make_solutions_pdf(
        tmp_path / "solucion.pdf", [("Taller 1, L5", ["x"]), ("Taller 2, M1", ["y"])]
    )
    sections = rubrics.find_sections(pdf)
    assert set(sections) == {"L5", "M1"} and rubrics.week_of(pdf, sections) == 1


def test_draft_is_sanitized():
    profile = CourseProfile.model_validate(
        {"subjects": [{"id": "calculo", "name": "Cálculo"}]}
    )
    exam = rubrics.to_exam(DRAFT, profile, "s14-m1", "Taller S14 M1")
    p1, p2 = exam.questions
    assert [p1.id, p2.id] == ["p1", "p1_2"]  # unique ids
    assert [c.id for c in p1.criteria] == ["idea", "idea_2"]
    assert p1.criteria[0].levels == {"2": "Correcta", "0": "Ausente"}  # out-of-range dropped
    assert p1.subject == "calculo" and p2.subject is None  # unknown subject dropped


async def test_drafts_load_as_exams(tmp_path, solutions):
    llm = FakeLLM()
    llm.canned[rubrics.RubricDraft] = DRAFT
    settings = Settings(data_dir=tmp_path / "data", profiles_dir=PROFILES)
    svc = HucService(FileRepository(settings.data_dir), llm=llm, settings=settings, ocr=FakeOCR())
    [(exam, text)] = await svc.draft_rubrics(solutions, ["m1"], profile_id="mat0004")

    assert exam.id == "s14-m1" and text.startswith("# DRAFT rubric")
    assert "profile: mat0004" in text
    # Only the M1 page goes to the model, rendered as an image
    system, user = llm.calls[0]["messages"]
    assert "`Taller S14M1`" in system.content[0]["text"]
    assert sum(b["type"] == "image" for b in user.content) == 1
    # The draft is a valid exam YAML: exam-load embeds the profile again
    loaded = svc.save_exam_yaml(text)
    assert loaded.profile.id == "mat0004" and loaded.questions[0].criteria[0].max_points == 2

    with pytest.raises(ValueError, match="W6"):
        await svc.draft_rubrics(solutions, ["W6"])


def test_report_counts_frequent_errors(exam):
    exam = exam.model_copy(
        update={
            "profile": CourseProfile(
                grading=GradingPolicy(error_types=[ErrorType(id="calculo", description="cálc")]),
                scale=[GradeBand(label="A", min_percent=50), GradeBand(label="R", min_percent=0)],
            )
        }
    )
    q = exam.questions[0]

    def grade(points: float, error: str | None) -> Grade:
        scores = [
            CriterionScore(
                criterion_id=c.id, points=points, justification=f"j{points}", confidence=1,
                error_type=error,
            )
            for c in q.criteria
        ]
        qg = QuestionGrade(question_id=q.id, scores=scores, graded_by="fake")
        return Grade(submission_id="s", exam_id=exam.id, rubric_version=1, questions=[qg])

    rep = build_report(exam, [], [grade(0, "calculo"), grade(0, "calculo"), grade(1, None)])
    assert rep.graded == 3 and rep.bands == {"A": 0, "R": 3}
    [err] = rep.errors
    assert err.error_type == "calculo" and err.count == 2 * len(q.criteria)
    assert err.description == "cálc" and err.examples == ["j0"]  # distinct examples
    assert rep.questions[0].errors[0].count == err.count
    assert rep.questions[0].mean_points == round(len(q.criteria) / 3, 2)

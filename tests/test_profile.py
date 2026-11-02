"""Course profiles: loading, prompt rendering and their effect on grading and export."""

from pathlib import Path

import pytest
import yaml
from conftest import EXAM_YAML, EXAMPLES, FakeLLM, FakeOCR
from test_graphs import ingest_and_grade

from huc.config import Settings
from huc.domain.models import CriterionScore, Exam
from huc.domain.profile import CourseProfile, ErrorType, GradeBand, GradingPolicy
from huc.llm import build_messages
from huc.prompts import grading as grading_prompts
from huc.prompts import ocr as ocr_prompts
from huc.prompts import review as review_prompts
from huc.services import HucService

PROFILES = Path(__file__).parent.parent / "profiles"


@pytest.fixture
def svc(repo, tmp_path) -> HucService:
    settings = Settings(data_dir=tmp_path / "data", profiles_dir=PROFILES)
    return HucService(repo, llm=FakeLLM(), settings=settings, ocr=FakeOCR())


def test_exam_yaml_embeds_the_named_profile(svc):
    exam = svc.load_exam_yaml(EXAMPLES / "taller_s14_m1.yaml")
    assert exam.profile.id == "mat0004"
    assert {s.id for s in exam.profile.subjects} == {"calculo", "algebra", "geometria"}
    assert exam.profile.error_type_ids() >= {"conceptual", "justificacion", "calculo"}
    # The copy is stored with the exam: grading never reads the profile file again.
    assert svc.repo.get_exam(exam.id).profile == exam.profile


def test_unknown_profile_and_subject_are_rejected(svc):
    data = yaml.safe_load(EXAM_YAML.read_text("utf-8"))
    with pytest.raises(ValueError, match="not found"):
        svc.save_exam_yaml(yaml.safe_dump({**data, "profile": "nope"}))
    data["questions"][0]["subject"] = "fisica"
    with pytest.raises(ValueError, match="fisica"):
        svc.save_exam_yaml(yaml.safe_dump(data))


def test_default_profile_keeps_the_generic_prompts():
    exam = Exam.model_validate(yaml.safe_load(EXAM_YAML.read_text("utf-8")))
    q = exam.questions[0]
    course, question = grading_prompts.grader_system(q, q.criteria[0], exam.profile)
    assert "## Course" not in course and "Set `error_type` to null" in course
    assert f"Set `criterion_id` to `{q.criteria[0].id}`" in question
    course, _ = review_prompts.reviewer_system(q, exam.profile)
    assert "in Spanish" in course and "how to improve it" in course
    assert "## Course notes" not in ocr_prompts.read_exam_system(exam)


def test_profile_reaches_every_prompt(svc):
    exam = svc.load_exam_yaml(EXAMPLES / "taller_s14_m1.yaml")
    q = exam.question("p1a")
    course, question = grading_prompts.grader_system(q, q.criteria[0], exam.profile)
    assert "MAT1104" in course  # subject context
    assert "UNA solución posible" in course  # grading rule
    assert "`justificacion`" in course  # error types
    assert "Skill assessed: Demostrar" in question

    course, _ = review_prompts.reviewer_system(q, exam.profile)
    assert "Siguientes pasos" in course  # GPS structure
    assert "No sabes la materia." in course  # forbidden comments
    assert "△ABC" in ocr_prompts.read_exam_system(exam)


def test_course_block_is_shared_by_a_subject_and_cached(svc):
    exam = svc.load_exam_yaml(EXAMPLES / "taller_s14_m1.yaml")
    p1a, p1b, p3 = exam.question("p1a"), exam.question("p1b"), exam.question("p3")
    a = grading_prompts.grader_system(p1a, p1a.criteria[0], exam.profile)
    b = grading_prompts.grader_system(p1b, p1b.criteria[1], exam.profile)
    c = grading_prompts.grader_system(p3, p3.criteria[0], exam.profile)
    assert a[0] == b[0] != c[0]
    system = build_messages(a, [{"type": "text", "text": "x"}])[0]
    assert [blk["cache_control"] for blk in system.content] == [{"type": "ephemeral"}] * 2


async def test_grading_follows_the_profile(repo, exam, scan_pdf):
    profile = CourseProfile(
        id="t",
        grading=GradingPolicy(
            error_types=[ErrorType(id="calculo", description="x")], verification_threshold=0.95
        ),
        scale=[GradeBand(label="A", min_percent=40), GradeBand(label="R", min_percent=0)],
    )
    exam = exam.model_copy(update={"profile": profile})
    repo.save_exam(exam)
    llm = FakeLLM(confidence=0.9)
    llm.canned[CriterionScore] = CriterionScore(
        criterion_id="x", points=1, justification="ok", confidence=0.9, error_type="calculo"
    )
    _, grade = await ingest_and_grade(repo, exam, scan_pdf, llm)
    scores = [s for qg in grade.questions for s in qg.scores]
    assert all(s.error_type == "calculo" for s in scores)
    # Profile threshold (0.95) overrides the settings default (0.7)
    assert grade.needs_verification

    llm.canned[CriterionScore] = llm.canned[CriterionScore].model_copy(
        update={"error_type": "invented"}
    )
    _, grade = await ingest_and_grade(repo, exam, scan_pdf, llm)
    assert all(s.error_type is None for qg in grade.questions for s in qg.scores)


async def test_export_has_percent_and_band(svc, exam, scan_pdf):
    exam = exam.model_copy(
        update={"profile": CourseProfile(scale=[GradeBand(label="A", min_percent=0)])}
    )
    svc.repo.save_exam(exam)
    await svc.ingest_file(exam.id, scan_pdf)
    await svc.grade_exam(exam.id)
    header, row = svc.export_csv(exam.id).splitlines()[:2]
    cols = dict(zip(header.split(","), row.split(","), strict=False))
    assert float(cols["percent"]) > 0 and cols["band"] == "A"

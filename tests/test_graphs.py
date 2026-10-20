import json
import subprocess
import sys

import pytest
from conftest import EXAMPLES, FakeLLM

from huc.domain.models import CriterionScore, SubmissionStatus
from huc.graphs.grading import build_grading_graph
from huc.graphs.ingestion import ExtractedAnswer, ExtractionResult, build_ingestion_graph

CONFIG = {"max_concurrency": 4}


async def ingest_and_grade(repo, exam, llm, file_path=EXAMPLES / "submission_lms.json"):
    ingest = build_ingestion_graph(repo=repo, llm=llm)
    out = await ingest.ainvoke({"exam_id": exam.id, "file_path": str(file_path)})
    submission = out["submission"]
    grading = build_grading_graph(repo=repo, llm=llm)
    result = await grading.ainvoke({"submission_id": submission.id}, config=CONFIG)
    return submission, result["grade"]


async def test_end_to_end_lms_submission(repo, exam):
    llm = FakeLLM(points=1.5)
    submission, grade = await ingest_and_grade(repo, exam, llm)

    assert exam.id == "bio-101-midterm-1"
    assert submission.student_id == "A00123"
    # q1 deterministic (1 pt) + q2: 3 criteria x 1.5, except `variation` clamped to 1
    assert grade.total == pytest.approx(1 + 1.5 + 1.5 + 1.0)
    assert [q.question_id for q in grade.questions] == ["q1", "q2"]
    q2_ids = [s.criterion_id for s in grade.questions[1].scores]
    assert q2_ids == ["products", "purpose", "variation"]
    assert not grade.needs_human_review
    # Persisted
    assert repo.get_grade(submission.id).total == pytest.approx(grade.total)
    assert repo.get_submission(submission.id).status is SubmissionStatus.GRADED
    # LMS ingestion and multiple choice never call the LLM: 3 graders + 1 reviewer
    assert llm.count("CriterionScore") == 3
    assert llm.count("ReviewVerdict") == 1
    assert llm.count("ExtractionResult") == 0


async def test_student_answer_is_in_user_message_only(repo, exam):
    llm = FakeLLM()
    await ingest_and_grade(repo, exam, llm)
    system, user = llm.calls[0]["messages"]
    assert system.content[0]["cache_control"] == {"type": "ephemeral"}
    assert "crossing-over" not in system.content[0]["text"]
    assert "<student_answer>" in user.content[0]["text"]


async def test_low_confidence_escalates_to_human(repo, exam):
    submission, grade = await ingest_and_grade(repo, exam, FakeLLM(confidence=0.4))
    assert grade.needs_human_review
    assert repo.get_submission(submission.id).status is SubmissionStatus.NEEDS_HUMAN_REVIEW


async def test_reviewer_can_escalate(repo, exam):
    submission, grade = await ingest_and_grade(repo, exam, FakeLLM(needs_human=True))
    assert grade.needs_human_review
    assert repo.get_submission(submission.id).status is SubmissionStatus.NEEDS_HUMAN_REVIEW


async def test_reviewer_adjustment_is_applied_and_clamped(repo, exam):
    adjusted = [
        CriterionScore(criterion_id="products", points=99, justification="x", confidence=0.95),
        CriterionScore(criterion_id="purpose", points=-3, justification="y", confidence=0.95),
        CriterionScore(criterion_id="not-in-rubric", points=5, justification="z", confidence=1),
    ]
    _, grade = await ingest_and_grade(repo, exam, FakeLLM(points=1, adjusted=adjusted))
    q2 = {s.criterion_id: s.points for s in grade.questions[1].scores}
    assert q2 == {"products": 2.0, "purpose": 0.0, "variation": 1.0}
    assert grade.total == pytest.approx(1 + 2 + 0 + 1)


async def test_empty_answer_skips_llm(repo, exam, tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"student_id": "A00999", "answers": {"q1": "C", "q2": "   "}}))
    llm = FakeLLM()
    _, grade = await ingest_and_grade(repo, exam, llm, file_path=path)

    assert grade.total == 0
    assert llm.calls == []
    assert all(s.justification == "Empty answer." for s in grade.questions[1].scores)
    assert grade.questions[1].review is None


async def test_scanned_pdf_goes_through_extractor(repo, exam, tmp_path):
    import pymupdf

    path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page().draw_rect(pymupdf.Rect(50, 50, 200, 200))  # no text -> treated as scanned
    doc.save(path)

    llm = FakeLLM()
    llm.canned[ExtractionResult] = ExtractionResult(
        student_id="A00777",
        answers=[
            ExtractedAnswer(question_id="q1", text="B", confidence=0.95, source_pages=[1]),
            ExtractedAnswer(question_id="ghost", text="dropped", confidence=1),
        ],
    )
    out = await build_ingestion_graph(repo=repo, llm=llm).ainvoke(
        {"exam_id": exam.id, "file_path": str(path)}
    )
    sub = out["submission"]
    assert sub.student_id == "A00777"
    assert [a.question_id for a in sub.answers] == ["q1"]
    assert sub.answers[0].extraction_confidence == 0.95
    assert repo.get_submission(sub.id).student_id == "A00777"
    image_block = llm.calls[0]["messages"][1].content[0]
    assert image_block["type"] == "image" and image_block["mime_type"] == "image/png"


def test_graphs_import_without_api_key(tmp_path):
    """Studio and tests import the module-level graphs; that must not need a key."""
    code = (
        "import json, importlib.util, pathlib\n"
        "cfg = json.loads(pathlib.Path('langgraph.json').read_text())\n"
        "for name, ref in cfg['graphs'].items():\n"
        "    path, attr = ref.split(':')\n"
        "    spec = importlib.util.spec_from_file_location(name, path)\n"
        "    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "    g = getattr(mod, attr)\n"
        "    print(name, sorted(g.get_graph().nodes))\n"
    )
    env = {"PATH": "", "HUC_DATA_DIR": str(tmp_path)}
    root = EXAMPLES.parent
    proc = subprocess.run(
        [sys.executable, "-c", code], check=False, cwd=root, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "grade_criterion" in proc.stdout and "extract" in proc.stdout


async def test_injected_settings_threshold_is_used(repo, exam):
    from huc.config import Settings

    llm = FakeLLM(confidence=0.8)  # above the default 0.7 threshold
    out = await build_ingestion_graph(repo=repo, llm=llm).ainvoke(
        {"exam_id": exam.id, "file_path": str(EXAMPLES / "submission_lms.json")}
    )
    strict = Settings(human_review_threshold=0.95)
    grading = build_grading_graph(repo, llm, settings=strict)
    result = await grading.ainvoke({"submission_id": out["submission"].id}, config=CONFIG)
    assert result["grade"].needs_human_review

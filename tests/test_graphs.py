import subprocess
import sys

import anthropic
import httpx
import pytest
from conftest import EXAMPLES, FEEDBACK, Q1_ANSWER, STUDENT, FakeLLM, FakeOCR

from huc.domain.models import CriterionScore, SourceKind, SubmissionStatus
from huc.graphs.grading import EMPTY_ANSWER_FEEDBACK, build_grading_graph
from huc.graphs.ingestion import build_ingestion_graph
from huc.ocr import OcrUnavailableError

CONFIG = {"max_concurrency": 4}


async def ingest(repo, exam, file_path, ocr=None, **extra):
    graph = build_ingestion_graph(repo=repo, ocr=ocr or FakeOCR())
    return await graph.ainvoke({"exam_id": exam.id, "file_path": str(file_path), **extra})


async def ingest_and_grade(repo, exam, file_path, llm, ocr=None, **kwargs):
    submission = (await ingest(repo, exam, file_path, ocr))["submission"]
    grading = build_grading_graph(repo=repo, llm=llm, **kwargs)
    result = await grading.ainvoke({"submission_id": submission.id}, config=CONFIG)
    return submission, result["grade"]


# -- ingestion -----------------------------------------------------------------


async def test_all_pages_are_read_in_one_call(repo, exam, scan_pdf):
    ocr = FakeOCR()
    out = await ingest(repo, exam, scan_pdf, ocr)

    # One OCR call with every page image, each labelled with its page number
    assert len(ocr.calls) == 1
    system, user = ocr.calls[0]
    images = [b for b in user.content if b["type"] == "image"]
    assert len(images) == 2 and images[0]["mime_type"] == "image/png"
    assert {"type": "text", "text": "Page 2:"} in user.content
    # The exam questions go in the cacheable system prompt
    assert system.content[0]["cache_control"] == {"type": "ephemeral"}
    assert "id=`q2`" in system.content[0]["text"]

    sub = out["submission"]
    assert sub.student_id == STUDENT  # from the file name: the scans hold no names
    assert sub.source_kind is SourceKind.PDF
    assert [a.question_id for a in sub.answers] == ["q1", "q2"]
    assert sub.answers[0].text == Q1_ANSWER
    assert sub.answers[0].extraction_confidence == 0.95
    assert repo.get_submission(sub.id) == sub


async def test_unknown_questions_dropped_and_student_id_override(repo, exam, scan_pdf):
    ocr = FakeOCR(reading={"answers": [
        {"question_id": "q1", "text": "x", "confidence": 0.5},
        {"question_id": "ghost", "text": "dropped", "confidence": 1},
    ]})
    out = await ingest(repo, exam, scan_pdf, ocr, student_id="A00777")
    sub = out["submission"]
    assert sub.student_id == "A00777"
    assert [a.question_id for a in sub.answers] == ["q1"]


async def test_image_ingestion(repo, exam, tmp_path):
    import pymupdf

    path = tmp_path / "Rojas_Diaz_Luis__photo.jpg"
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 400), False)
    pix.clear_with(255)
    pix.save(path)
    ocr = FakeOCR()
    out = await ingest(repo, exam, path, ocr)
    images = [b for b in ocr.calls[0][1].content if b["type"] == "image"]
    assert [i["mime_type"] for i in images] == ["image/jpeg"]
    assert out["submission"].source_kind is SourceKind.IMAGE
    assert out["submission"].student_id == "Rojas_Diaz_Luis"


class RejectedKeyOCR:
    model = "claude-haiku-5-5"

    def with_structured_output(self, schema, **kwargs):
        return self

    async def ainvoke(self, messages, *args, **kwargs):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.AuthenticationError(
            "invalid x-api-key", response=httpx.Response(401, request=request), body=None
        )


async def test_ocr_auth_error_is_clear(repo, exam, scan_pdf):
    with pytest.raises(OcrUnavailableError, match="ANTHROPIC_API_KEY"):
        await ingest(repo, exam, scan_pdf, RejectedKeyOCR())
    assert repo.list_submissions(exam.id) == []


async def test_unknown_exam_fails_before_ocr(repo, scan_pdf):
    ocr = FakeOCR()
    graph = build_ingestion_graph(repo=repo, ocr=ocr)
    with pytest.raises(FileNotFoundError):
        await graph.ainvoke({"exam_id": "nope", "file_path": str(scan_pdf)})
    assert ocr.calls == []


# -- grading -------------------------------------------------------------------


async def test_end_to_end_ingest_and_grade(repo, exam, scan_pdf):
    llm = FakeLLM(points=1.5)
    submission, grade = await ingest_and_grade(repo, exam, scan_pdf, llm)

    assert exam.id == "bio-101-midterm-1"
    # Every criterion gets 1.5, clamped to its maximum: q1 = 1.5+1.5+1, q2 = 1.5+1
    assert grade.total == pytest.approx(4 + 2.5)
    assert [q.question_id for q in grade.questions] == ["q1", "q2"]
    assert [s.criterion_id for s in grade.questions[0].scores] == [
        "products", "purpose", "variation"
    ]
    assert not grade.needs_verification and grade.verification_reasons == []
    assert [q.feedback for q in grade.questions] == [FEEDBACK, FEEDBACK]
    assert repo.get_grade(submission.id).total == pytest.approx(grade.total)
    assert repo.get_submission(submission.id).status is SubmissionStatus.GRADED
    # 5 criteria + 2 reviews (the reading is the OCR model's single call)
    assert llm.count("CriterionScore") == 5
    assert llm.count("ReviewVerdict") == 2


async def test_student_answer_is_in_user_message_only(repo, exam, scan_pdf):
    llm = FakeLLM()
    await ingest_and_grade(repo, exam, scan_pdf, llm)
    grader_call = next(c for c in llm.calls if c["schema"] == "CriterionScore")
    system, user = grader_call["messages"]
    assert system.content[0]["cache_control"] == {"type": "ephemeral"}
    assert "crossing-over" not in system.content[0]["text"]
    assert "<student_answer>" in user.content[0]["text"]


async def test_low_extraction_confidence_caps_grade_confidence(repo, exam, scan_pdf):
    ocr = FakeOCR(reading={"answers": [
        {"question_id": "q1", "text": "[illegible] [illegible] mitosis", "confidence": 0.3},
    ]})
    _, grade = await ingest_and_grade(repo, exam, scan_pdf, FakeLLM(confidence=0.95), ocr)
    assert all(s.confidence <= 0.3 for s in grade.questions[0].scores)
    assert grade.needs_verification
    assert "q1/products: confidence 0.30 < 0.7" in grade.verification_reasons


async def test_low_confidence_is_graded_and_labelled(repo, exam, scan_pdf):
    submission, grade = await ingest_and_grade(repo, exam, scan_pdf, FakeLLM(confidence=0.4))
    assert grade.needs_verification
    assert len(grade.verification_reasons) == 5  # one per criterion
    assert grade.total > 0 and all(q.feedback for q in grade.questions)
    assert repo.get_submission(submission.id).status is SubmissionStatus.GRADED


async def test_reviewer_can_flag_for_verification(repo, exam, scan_pdf):
    llm = FakeLLM(needs_verification=True)
    submission, grade = await ingest_and_grade(repo, exam, scan_pdf, llm)
    assert grade.needs_verification
    assert grade.verification_reasons == ["q1: flagged by the reviewer", "q2: flagged by the reviewer"]
    assert repo.get_submission(submission.id).status is SubmissionStatus.GRADED


async def test_reviewer_adjustment_is_applied_and_clamped(repo, exam, scan_pdf):
    adjusted = [
        CriterionScore(criterion_id="products", points=99, justification="x", confidence=0.95),
        CriterionScore(criterion_id="purpose", points=-3, justification="y", confidence=0.95),
        CriterionScore(criterion_id="not-in-rubric", points=5, justification="z", confidence=1),
    ]
    _, grade = await ingest_and_grade(repo, exam, scan_pdf, FakeLLM(points=1, adjusted=adjusted))
    q1 = {s.criterion_id: s.points for s in grade.questions[0].scores}
    assert q1 == {"products": 2.0, "purpose": 0.0, "variation": 1.0}


async def test_empty_answers_skip_llm(repo, exam, scan_pdf):
    llm = FakeLLM()
    ocr = FakeOCR(reading={"answers": [{"question_id": "q1", "text": "  ", "confidence": 1}]})
    _, grade = await ingest_and_grade(repo, exam, scan_pdf, llm, ocr)

    assert grade.total == 0
    assert llm.count("CriterionScore") == llm.count("ReviewVerdict") == 0
    assert all(s.justification == "Empty answer." for q in grade.questions for s in q.scores)
    assert all(q.review is None for q in grade.questions)
    assert all(q.feedback == EMPTY_ANSWER_FEEDBACK for q in grade.questions)


async def test_injected_settings_threshold_is_used(repo, exam, scan_pdf):
    from huc.config import Settings

    llm = FakeLLM(confidence=0.8)  # above the default 0.7 threshold
    _, grade = await ingest_and_grade(
        repo, exam, scan_pdf, llm, settings=Settings(verification_threshold=0.95)
    )
    assert grade.needs_verification


# -- Studio --------------------------------------------------------------------


def test_graphs_import_without_network_or_api_key(tmp_path):
    """Studio imports the module-level graphs; that must need neither network nor a key."""
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
    # An unroutable API URL and an invalid key: any call at import time would fail.
    env = {
        "PATH": "", "HUC_DATA_DIR": str(tmp_path), "ANTHROPIC_API_KEY": "sk-invalid",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code], check=False, cwd=EXAMPLES.parent, env=env,
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for node in ("load_pages", "read_exam", "grade_criterion", "review_question"):
        assert node in proc.stdout
    assert "grade_multiple_choice" not in proc.stdout


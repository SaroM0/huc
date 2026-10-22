import asyncio
import subprocess
import sys
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from conftest import EXAMPLES, Q1_ANSWER, FakeLLM, FakeOCR, make_scan_pdf

from huc.domain.models import CriterionScore, SourceKind, SubmissionStatus
from huc.graphs.grading import build_grading_graph
from huc.graphs.ingestion import SegmentationResult, build_ingestion_graph
from huc.ocr import OcrUnavailableError

CONFIG = {"max_concurrency": 4}


async def ingest(repo, exam, file_path, llm, ocr=None, **extra):
    graph = build_ingestion_graph(repo=repo, llm=llm, ocr=ocr or FakeOCR())
    return await graph.ainvoke({"exam_id": exam.id, "file_path": str(file_path), **extra})


async def ingest_and_grade(repo, exam, file_path, llm, **kwargs):
    submission = (await ingest(repo, exam, file_path, llm))["submission"]
    grading = build_grading_graph(repo=repo, llm=llm, **kwargs)
    result = await grading.ainvoke({"submission_id": submission.id}, config=CONFIG)
    return submission, result["grade"]


# -- ingestion -----------------------------------------------------------------


async def test_scanned_pdf_is_ocrd_per_page_then_segmented(repo, exam, scan_pdf):
    llm, ocr = FakeLLM(), FakeOCR()
    out = await ingest(repo, exam, scan_pdf, llm, ocr)

    # OCR: one local call per page, each with that page's image
    assert len(ocr.calls) == 2
    image = ocr.calls[0][1].content[0]
    assert image["type"] == "image" and image["mime_type"] == "image/png"
    assert [t["page"] for t in out["transcripts"]] == [1, 2]
    assert "ID: A00123" in out["transcripts"][0]["text"]

    # Segmentation: one Claude call, text only (no image leaves the machine)
    assert llm.count("SegmentationResult") == 1
    user = llm.calls[0]["messages"][1]
    assert all(block["type"] == "text" for block in user.content)
    assert '<page n="2">' in user.content[0]["text"]
    # The exam goes in the cacheable system prompt, the transcript in the user message
    assert "crossing-over" not in llm.calls[0]["messages"][0].content[0]["text"]

    sub = out["submission"]
    assert sub.student_id == "A00123"
    assert sub.source_kind is SourceKind.PDF
    assert [a.question_id for a in sub.answers] == ["q1", "q2"]
    assert sub.answers[0].text == Q1_ANSWER
    assert sub.answers[0].extraction_confidence == 0.95
    assert repo.get_submission(sub.id) == sub


async def test_unknown_questions_dropped_and_student_id_override(repo, exam, scan_pdf):
    llm = FakeLLM()
    llm.canned[SegmentationResult] = SegmentationResult.model_validate(
        {"student_id": None, "answers": [
            {"question_id": "q1", "text": "x", "confidence": 0.5},
            {"question_id": "ghost", "text": "dropped", "confidence": 1},
        ]}
    )
    out = await ingest(repo, exam, scan_pdf, llm, student_id="A00777")
    sub = out["submission"]
    assert sub.student_id == "A00777"
    assert [a.question_id for a in sub.answers] == ["q1"]


async def test_unidentified_student_fails(repo, exam, scan_pdf):
    llm = FakeLLM()
    llm.canned[SegmentationResult] = SegmentationResult(student_id=None, answers=[])
    with pytest.raises(ValueError, match="Could not identify the student"):
        await ingest(repo, exam, scan_pdf, llm)


async def test_image_ingestion(repo, exam, tmp_path):
    import pymupdf

    path = tmp_path / "photo.jpg"
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 400), False)
    pix.clear_with(255)
    pix.save(path)
    ocr = FakeOCR()
    out = await ingest(repo, exam, path, FakeLLM(), ocr)
    assert len(ocr.calls) == 1
    assert ocr.calls[0][1].content[0]["mime_type"] == "image/jpeg"
    assert out["submission"].source_kind is SourceKind.IMAGE


async def test_ocr_pages_run_in_parallel_and_keep_order(repo, exam, tmp_path):
    class SlowFirstPages:
        """Page N is the N-th call; earlier pages answer later (reverse completion order)."""

        def __init__(self) -> None:
            self.started = self.running = self.peak = 0

        async def ainvoke(self, messages, *args, **kwargs):
            self.started += 1
            page = self.started
            self.running += 1
            self.peak = max(self.peak, self.running)
            await asyncio.sleep(0.01 * (4 - page))
            self.running -= 1
            return SimpleNamespace(content=f"texto {page}")

    ocr = SlowFirstPages()
    out = await ingest(repo, exam, make_scan_pdf(tmp_path / "s.pdf", pages=3), FakeLLM(), ocr)
    assert ocr.peak == 3  # all pages in flight at once
    assert [(t["page"], t["text"]) for t in out["transcripts"]] == [
        (1, "texto 1"), (2, "texto 2"), (3, "texto 3")
    ]


class RejectedKeyOCR:
    model = "claude-haiku-5-5"

    async def ainvoke(self, messages, *args, **kwargs):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.AuthenticationError(
            "invalid x-api-key", response=httpx.Response(401, request=request), body=None
        )


async def test_ocr_auth_error_is_clear(repo, exam, scan_pdf):
    llm = FakeLLM()
    with pytest.raises(OcrUnavailableError, match="ANTHROPIC_API_KEY"):
        await ingest(repo, exam, scan_pdf, llm, RejectedKeyOCR())
    assert llm.calls == []  # never reached the segmenter
    assert repo.list_submissions(exam.id) == []


async def test_unknown_exam_fails_before_ocr(repo, scan_pdf):
    ocr = FakeOCR()
    graph = build_ingestion_graph(repo=repo, llm=FakeLLM(), ocr=ocr)
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
    assert not grade.needs_human_review
    assert repo.get_grade(submission.id).total == pytest.approx(grade.total)
    assert repo.get_submission(submission.id).status is SubmissionStatus.GRADED
    # 1 segmentation + 5 criteria + 2 reviews
    assert llm.count("SegmentationResult") == 1
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
    llm = FakeLLM(confidence=0.95)
    llm.canned[SegmentationResult] = SegmentationResult.model_validate(
        {"student_id": "A1", "answers": [
            {"question_id": "q1", "text": "[illegible] [illegible] mitosis", "confidence": 0.3},
        ]}
    )
    _, grade = await ingest_and_grade(repo, exam, scan_pdf, llm)
    assert all(s.confidence <= 0.3 for s in grade.questions[0].scores)
    assert grade.needs_human_review


async def test_low_confidence_escalates_to_human(repo, exam, scan_pdf):
    submission, grade = await ingest_and_grade(repo, exam, scan_pdf, FakeLLM(confidence=0.4))
    assert grade.needs_human_review
    assert repo.get_submission(submission.id).status is SubmissionStatus.NEEDS_HUMAN_REVIEW


async def test_reviewer_can_escalate(repo, exam, scan_pdf):
    submission, grade = await ingest_and_grade(repo, exam, scan_pdf, FakeLLM(needs_human=True))
    assert grade.needs_human_review
    assert repo.get_submission(submission.id).status is SubmissionStatus.NEEDS_HUMAN_REVIEW


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
    llm.canned[SegmentationResult] = SegmentationResult.model_validate(
        {"student_id": "A00999", "answers": [{"question_id": "q1", "text": "  ", "confidence": 1}]}
    )
    _, grade = await ingest_and_grade(repo, exam, scan_pdf, llm)

    assert grade.total == 0
    assert llm.count("CriterionScore") == llm.count("ReviewVerdict") == 0
    assert all(s.justification == "Empty answer." for q in grade.questions for s in q.scores)
    assert all(q.review is None for q in grade.questions)


async def test_injected_settings_threshold_is_used(repo, exam, scan_pdf):
    from huc.config import Settings

    llm = FakeLLM(confidence=0.8)  # above the default 0.7 threshold
    _, grade = await ingest_and_grade(
        repo, exam, scan_pdf, llm, settings=Settings(human_review_threshold=0.95)
    )
    assert grade.needs_human_review


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
    for node in ("load_pages", "ocr_pages", "segment", "grade_criterion", "review_question"):
        assert node in proc.stdout
    assert "grade_multiple_choice" not in proc.stdout


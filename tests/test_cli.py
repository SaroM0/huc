"""CLI tests. The service gets fake Claude and OCR models: no network, no API key."""

import anthropic
import httpx
import pytest
from conftest import EXAM_YAML, FEEDBACK, PAGE_TEXTS, STUDENT, FakeLLM, FakeOCR, make_scan_pdf
from typer.testing import CliRunner

from huc import cli
from huc.config import Settings
from huc.services import HucService
from huc.storage.repository import FileRepository

runner = CliRunner()


@pytest.fixture
def service(tmp_path, monkeypatch) -> HucService:
    data = tmp_path / "data"
    svc = HucService(
        FileRepository(data), llm=FakeLLM(), settings=Settings(data_dir=data), ocr=FakeOCR()
    )
    monkeypatch.setattr(cli, "get_service", lambda: svc)
    return svc


def run(*args):
    return runner.invoke(cli.app, [str(a) for a in args])


def _load_exam(service) -> str:
    r = run("exam-load", EXAM_YAML)
    assert r.exit_code == 0, r.output
    return service.repo.list_exams()[0].id


def test_exam_load(service):
    exam_id = _load_exam(service)
    assert f"Exam loaded: {exam_id}" in run("exam-load", EXAM_YAML).output


def test_exam_load_missing_file(service, tmp_path):
    assert run("exam-load", tmp_path / "missing.yaml").exit_code == 1


def test_ocr_prints_each_page_without_grading(service, scan_pdf):
    r = run("ocr", scan_pdf)
    assert r.exit_code == 0, r.output
    assert "--- page 1 (" in r.output and "--- page 2 (" in r.output
    assert PAGE_TEXTS[0] in r.output and PAGE_TEXTS[1] in r.output
    assert len(service.ocr.calls) == 2
    assert service.repo.list_exams() == []  # nothing saved


def test_ocr_errors(service, tmp_path, monkeypatch):
    assert run("ocr", tmp_path / "missing.pdf").exit_code == 1
    (tmp_path / "notes.txt").write_text("x")
    assert run("ocr", tmp_path / "notes.txt").exit_code == 1

    class Down:
        async def ainvoke(self, *a, **k):
            raise anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            )

    down = HucService(service.repo, settings=service.settings, ocr=Down())
    monkeypatch.setattr(cli, "get_service", lambda: down)
    r = run("ocr", make_scan_pdf(tmp_path / "s.pdf", pages=1))
    assert r.exit_code == 1
    assert "Cannot reach the Anthropic API" in r.output


def test_ingest_file_and_dir(service, tmp_path, scan_pdf):
    exam_id = _load_exam(service)
    r = run("ingest", exam_id, scan_pdf)
    assert r.exit_code == 0, r.output
    assert f"student={STUDENT}  answers=2" in r.output

    folder = tmp_path / "batch"
    folder.mkdir()
    make_scan_pdf(folder / "Soto_Vera_Eva__a.pdf")
    make_scan_pdf(folder / "Leon_Paz_Ivan__b.PDF")
    (folder / "notes.txt").write_text("ignored")
    (folder / "old.json").write_text("{}")  # LMS exports are no longer supported
    r = run("ingest", exam_id, folder)
    assert r.exit_code == 0, r.output
    assert "Ingested 2 submission(s)." in r.output
    students = sorted(s.student_id for s in service.repo.list_submissions(exam_id))
    assert students == ["Leon_Paz_Ivan", STUDENT, "Soto_Vera_Eva"]


def test_ingest_unknown_exam_fails(service, scan_pdf):
    assert run("ingest", "nope", scan_pdf).exit_code == 1
    assert service.ocr.calls == []


def test_grade_show_export(service, tmp_path, scan_pdf):
    exam_id = _load_exam(service)
    run("ingest", exam_id, scan_pdf)
    sub_id = service.repo.list_submissions(exam_id)[0].id

    r = run("grade", exam_id)
    assert r.exit_code == 0, r.output
    assert "Graded 1 submission(s)." in r.output
    # Nothing left pending.
    assert "Graded 0 submission(s)." in run("grade", exam_id).output

    r = run("show", sub_id)
    assert r.exit_code == 0, r.output
    assert "Total: 6.50 / 8" in r.output
    assert '> "2 células diploides iguales"' in r.output
    assert f"Feedback: {FEEDBACK}" in r.output
    assert "needs teacher verification" not in r.output

    out = tmp_path / "grades.csv"
    assert run("export", exam_id, "--out", out).exit_code == 0
    assert out.read_text().splitlines()[0] == (
        "student_id,status,q1,q2,total,needs_verification,verification_reasons,"
        "q1_feedback,q2_feedback"
    )


def test_low_confidence_is_labelled(service, monkeypatch, scan_pdf):
    unsure = HucService(
        service.repo, llm=FakeLLM(confidence=0.4), settings=service.settings, ocr=FakeOCR()
    )
    monkeypatch.setattr(cli, "get_service", lambda: unsure)
    exam_id = _load_exam(unsure)
    run("ingest", exam_id, scan_pdf)
    r = run("grade", exam_id)
    assert "[needs teacher verification]" in r.output
    sub_id = unsure.repo.list_submissions(exam_id)[0].id
    r = run("show", sub_id)
    assert "status=graded" in r.output
    assert "? q1/products: confidence 0.40 < 0.7" in r.output


def test_show_unknown_or_ungraded(service, scan_pdf):
    assert run("show", "nope").exit_code == 1
    exam_id = _load_exam(service)
    run("ingest", exam_id, scan_pdf)
    sub_id = service.repo.list_submissions(exam_id)[0].id
    assert run("show", sub_id).exit_code == 1


class BrokenLLM:
    def with_structured_output(self, schema, **kwargs):
        raise RuntimeError("LLM down")


def test_grade_failure_marks_submission_failed(service, monkeypatch, scan_pdf):
    exam_id = _load_exam(service)
    run("ingest", exam_id, scan_pdf)
    sub_id = service.repo.list_submissions(exam_id)[0].id

    broken = HucService(service.repo, llm=BrokenLLM(), settings=service.settings, ocr=FakeOCR())
    monkeypatch.setattr(cli, "get_service", lambda: broken)
    assert run("grade", exam_id).exit_code != 0
    assert service.repo.get_submission(sub_id).status == "failed"

    # Failed submissions are retried by the next `grade` run.
    monkeypatch.setattr(cli, "get_service", lambda: service)
    assert "Graded 1 submission(s)." in run("grade", exam_id).output

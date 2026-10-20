"""CLI tests. The service is injected with a fake LLM: no network, no API key."""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from huc import cli
from huc.config import Settings
from huc.services import HucService
from huc.storage.repository import FileRepository

EXAMPLES = Path(__file__).parent.parent / "examples"
EXAM_YAML = EXAMPLES / "exam_example.yaml"
LMS_JSON = EXAMPLES / "submission_lms.json"

runner = CliRunner()


class FakeStructuredLLM:
    def __init__(self, schema: type) -> None:
        self.schema = schema

    async def ainvoke(self, messages, *args, **kwargs):
        name = self.schema.__name__
        if name == "CriterionScore":
            return self.schema(
                criterion_id="ignored", points=1.0, justification="Mentions crossing-over.",
                evidence=["hay crossing-over"], confidence=0.9,
            )
        if name == "ReviewVerdict":
            return self.schema(approved=True)
        raise AssertionError(f"unexpected schema {name}")


class FakeLLM:
    """Minimal chat-model stand-in: `.with_structured_output(schema).ainvoke(messages)`."""

    def with_structured_output(self, schema, **kwargs):
        return FakeStructuredLLM(schema)


@pytest.fixture
def service(tmp_path, monkeypatch) -> HucService:
    data = tmp_path / "data"
    svc = HucService(FileRepository(data), llm=FakeLLM(), settings=Settings(data_dir=data))
    monkeypatch.setattr(cli, "get_service", lambda: svc)
    return svc


def run(*args: str):
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


def test_ingest_file_and_dir(service, tmp_path):
    exam_id = _load_exam(service)
    assert run("ingest", exam_id, LMS_JSON).exit_code == 0

    folder = tmp_path / "batch"
    folder.mkdir()
    shutil.copy(LMS_JSON, folder / "a.json")
    shutil.copy(LMS_JSON, folder / "b.json")
    (folder / "notes.txt").write_text("ignored")
    r = run("ingest", exam_id, folder)
    assert r.exit_code == 0, r.output
    assert "Ingested 2 submission(s)." in r.output
    assert len(service.repo.list_submissions(exam_id)) == 3


def test_ingest_unknown_exam_fails(service):
    assert run("ingest", "nope", LMS_JSON).exit_code == 1


def test_grade_show_finalize_export(service, tmp_path):
    exam_id = _load_exam(service)
    run("ingest", exam_id, LMS_JSON)
    sub_id = service.repo.list_submissions(exam_id)[0].id

    r = run("grade", exam_id)
    assert r.exit_code == 0, r.output
    assert "Graded 1 submission(s)." in r.output
    # Nothing left pending.
    assert "Graded 0 submission(s)." in run("grade", exam_id).output

    r = run("show", sub_id)
    assert r.exit_code == 0, r.output
    assert "Mentions crossing-over." in r.output
    assert '> "hay crossing-over"' in r.output

    assert run("finalize", sub_id).exit_code == 0
    assert service.repo.get_submission(sub_id).status == "finalized"

    out = tmp_path / "grades.csv"
    assert run("export", exam_id, "--out", out).exit_code == 0
    assert out.read_text().splitlines()[0] == "student_id,status,q1,q2,total"


def test_show_unknown_or_ungraded(service):
    assert run("show", "nope").exit_code == 1
    exam_id = _load_exam(service)
    run("ingest", exam_id, LMS_JSON)
    sub_id = service.repo.list_submissions(exam_id)[0].id
    r = run("show", sub_id)
    assert r.exit_code == 1
    assert run("finalize", sub_id).exit_code == 1


class BrokenLLM:
    def with_structured_output(self, schema, **kwargs):
        raise RuntimeError("LLM down")


def test_grade_failure_marks_submission_failed(service, monkeypatch):
    exam_id = _load_exam(service)
    run("ingest", exam_id, LMS_JSON)
    sub_id = service.repo.list_submissions(exam_id)[0].id

    broken = HucService(service.repo, llm=BrokenLLM(), settings=service.settings)
    monkeypatch.setattr(cli, "get_service", lambda: broken)
    assert run("grade", exam_id).exit_code != 0
    assert service.repo.get_submission(sub_id).status == "failed"

    # Failed submissions are retried by the next `grade` run.
    monkeypatch.setattr(cli, "get_service", lambda: service)
    assert "Graded 1 submission(s)." in run("grade", exam_id).output

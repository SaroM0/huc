"""HTTP API tests. The LLM is a fake: no network, no API key."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from huc.api.app import create_app
from huc.config import Settings
from huc.services import HucService
from huc.storage.repository import FileRepository

EXAMPLES = Path(__file__).parent.parent / "examples"
EXAM_YAML = EXAMPLES / "exam_example.yaml"
LMS_JSON = EXAMPLES / "submission_lms.json"


class FakeStructuredLLM:
    def __init__(self, parent: "FakeLLM", schema: type) -> None:
        self.parent, self.schema = parent, schema

    async def ainvoke(self, messages, *args, **kwargs):
        name = self.schema.__name__
        self.parent.calls.append(name)
        if name == "CriterionScore":
            return self.schema(
                criterion_id="ignored", points=1.0, justification="Correct and complete.",
                evidence=["4 haploid cells"], confidence=0.9,
            )
        if name == "ReviewVerdict":
            return self.schema(approved=True)
        raise AssertionError(f"unexpected schema {name}")


class FakeLLM:
    """Minimal chat-model stand-in: `.with_structured_output(schema).ainvoke(messages)`."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def with_structured_output(self, schema, **kwargs):
        return FakeStructuredLLM(self, schema)


@pytest.fixture
def service(tmp_path) -> HucService:
    settings = Settings(data_dir=tmp_path)
    return HucService(FileRepository(tmp_path), llm=FakeLLM(), settings=settings)


@pytest.fixture
def client(service) -> TestClient:
    return TestClient(create_app(service))


def _create_exam(client: TestClient) -> str:
    r = client.post("/exams", files={"file": ("exam.yaml", EXAM_YAML.read_bytes())})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _upload_lms(client: TestClient, exam_id: str) -> dict:
    r = client.post(
        f"/exams/{exam_id}/submissions",
        files={"file": ("submission.json", LMS_JSON.read_bytes(), "application/json")},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_import_does_not_require_api_key():
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    proc = subprocess.run(
        [sys.executable, "-c", "import huc.api.app"],
        check=False, env=env, capture_output=True, text=True, cwd=Path(__file__).parent.parent,
    )
    assert proc.returncode == 0, proc.stderr


def test_create_and_list_exams(client):
    exam_id = _create_exam(client)
    assert [e["id"] for e in client.get("/exams").json()] == [exam_id]


def test_invalid_exam_yaml_is_422(client):
    r = client.post("/exams", files={"file": ("exam.yaml", b"title: missing questions")})
    assert r.status_code == 422


def test_upload_and_get_submission(client):
    exam_id = _create_exam(client)
    sub = _upload_lms(client, exam_id)

    assert sub["status"] == "ingested"
    assert client.get(f"/submissions/{sub['id']}").json()["id"] == sub["id"]
    assert [s["id"] for s in client.get(f"/exams/{exam_id}/submissions").json()] == [sub["id"]]


def test_unknown_ids_are_404(client):
    assert client.get("/submissions/nope").status_code == 404
    assert client.get("/submissions/nope/grade").status_code == 404
    assert client.get("/exams/nope/submissions").status_code == 404
    assert client.post("/exams/nope/grade").status_code == 404
    assert client.get("/exams/nope/export.csv").status_code == 404
    assert client.post("/submissions/nope/finalize").status_code == 404


def test_grade_finalize_export(client):
    exam_id = _create_exam(client)
    sub = _upload_lms(client, exam_id)

    # Finalizing before grading is a client error.
    assert client.post(f"/submissions/{sub['id']}/finalize").status_code == 422

    # TestClient runs background tasks before returning.
    r = client.post(f"/exams/{exam_id}/grade")
    assert r.status_code == 202

    grade = client.get(f"/submissions/{sub['id']}/grade").json()
    assert grade["submission_id"] == sub["id"]
    assert client.get(f"/submissions/{sub['id']}").json()["status"] == "graded"

    assert client.post(f"/submissions/{sub['id']}/finalize").status_code == 204
    assert client.get(f"/submissions/{sub['id']}").json()["status"] == "finalized"

    csv_text = client.get(f"/exams/{exam_id}/export.csv").text
    assert csv_text.splitlines()[0] == "student_id,status,q1,q2,total"
    assert "A00123,finalized" in csv_text

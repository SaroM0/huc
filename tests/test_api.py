"""HTTP API tests. Claude and OCR models are fakes: no network, no API key."""

import os
import subprocess
import sys
from pathlib import Path

import anthropic
import httpx
import pytest
from conftest import EXAM_YAML, STUDENT, FakeLLM, FakeOCR
from fastapi.testclient import TestClient

from huc.api.app import create_app
from huc.config import Settings
from huc.services import HucService
from huc.storage.repository import FileRepository


@pytest.fixture
def service(tmp_path) -> HucService:
    settings = Settings(data_dir=tmp_path)
    return HucService(FileRepository(tmp_path), llm=FakeLLM(), settings=settings, ocr=FakeOCR())


@pytest.fixture
def client(service) -> TestClient:
    return TestClient(create_app(service))


def _create_exam(client: TestClient) -> str:
    r = client.post("/exams", files={"file": ("exam.yaml", EXAM_YAML.read_bytes())})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _upload_scan(client: TestClient, exam_id: str, pdf: Path, **params) -> httpx.Response:
    return client.post(
        f"/exams/{exam_id}/submissions",
        params=params,
        files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
    )


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


def test_upload_and_get_submission(client, service, scan_pdf):
    exam_id = _create_exam(client)
    r = _upload_scan(client, exam_id, scan_pdf, student_id="S-1")
    assert r.status_code == 200, r.text
    sub = r.json()

    assert sub["status"] == "ingested"
    assert sub["student_id"] == "S-1"
    assert sub["source_kind"] == "pdf"
    assert len(service.ocr.calls) == 1  # all pages in one reading call
    assert client.get(f"/submissions/{sub['id']}").json()["id"] == sub["id"]
    assert [s["id"] for s in client.get(f"/exams/{exam_id}/submissions").json()] == [sub["id"]]


def test_unsupported_upload_is_422(client):
    exam_id = _create_exam(client)
    r = client.post(
        f"/exams/{exam_id}/submissions", files={"file": ("answers.json", b"{}", "application/json")}
    )
    assert r.status_code == 422


def test_ocr_api_down_is_503(tmp_path, scan_pdf):
    class Down:
        def with_structured_output(self, *a, **k):
            return self

        async def ainvoke(self, *a, **k):
            raise anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            )

    svc = HucService(FileRepository(tmp_path), settings=Settings(data_dir=tmp_path), ocr=Down())
    client = TestClient(create_app(svc))
    r = _upload_scan(client, _create_exam(client), scan_pdf)
    assert r.status_code == 503
    assert "Cannot reach the Anthropic API" in r.json()["detail"]


def test_unknown_ids_are_404(client):
    assert client.get("/submissions/nope").status_code == 404
    assert client.get("/submissions/nope/grade").status_code == 404
    assert client.get("/exams/nope/submissions").status_code == 404
    assert client.post("/exams/nope/grade").status_code == 404
    assert client.get("/exams/nope/export.csv").status_code == 404


def test_student_comes_from_uploaded_file_name(client, scan_pdf):
    sub = _upload_scan(client, _create_exam(client), scan_pdf).json()
    assert sub["student_id"] == STUDENT  # not the random name of the stored upload


def test_grade_and_export(client, scan_pdf):
    exam_id = _create_exam(client)
    sub = _upload_scan(client, exam_id, scan_pdf).json()

    # TestClient runs background tasks before returning.
    assert client.post(f"/exams/{exam_id}/grade").status_code == 202

    grade = client.get(f"/submissions/{sub['id']}/grade").json()
    assert grade["submission_id"] == sub["id"]
    assert grade["needs_verification"] is False
    assert all(q["feedback"] for q in grade["questions"])
    assert client.get(f"/submissions/{sub['id']}").json()["status"] == "graded"

    csv_text = client.get(f"/exams/{exam_id}/export.csv").text
    assert csv_text.splitlines()[0].startswith("student_id,status,q1,q2,total,needs_verification")
    assert f"{STUDENT},graded" in csv_text

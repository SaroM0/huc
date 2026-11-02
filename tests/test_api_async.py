"""Exercise the API in the event loop without the TestClient thread portal."""
import httpx
from conftest import EXAM_YAML, FakeLLM, FakeOCR

from huc.api.app import create_app
from huc.config import Settings
from huc.services import HucService
from huc.storage.repository import FileRepository


async def test_async_api_exam_and_grade_history(tmp_path):
    service = HucService(FileRepository(tmp_path), settings=Settings(data_dir=tmp_path))
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/exams", files={"file": ("exam.yaml", b"""id: sample
title: Sample
questions:
  - id: q
    number: '1'
    type: short_answer
    prompt: Question
    criteria:
      - {id: c, description: Correct, max_points: 1}
""")})
        assert response.status_code == 200
        response = await client.get("/exams")
        assert response.json()[0]["id"] == "sample"
        response = await client.get("/submissions/unknown/grade-history")
        assert response.status_code == 404


async def test_async_api_upload_grade_and_history(tmp_path, scan_pdf):
    service = HucService(FileRepository(tmp_path), llm=FakeLLM(), ocr=FakeOCR(),
                         settings=Settings(data_dir=tmp_path))
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/exams", files={"file": ("exam.yaml", EXAM_YAML.read_bytes())})
        exam_id = response.json()["id"]
        response = await client.post(f"/exams/{exam_id}/submissions", params={"student_id": "test"},
                                     files={"file": ("scan.pdf", scan_pdf.read_bytes())})
        assert response.status_code == 200
        submission_id = response.json()["id"]
        response = await client.post(f"/exams/{exam_id}/grade")
        assert response.status_code == 202
        response = await client.get(f"/submissions/{submission_id}/grade")
        assert response.status_code == 200
        assert all(question["feedback"] for question in response.json()["questions"])
        response = await client.get(f"/submissions/{submission_id}/grade-history")
        assert len(response.json()) == 1

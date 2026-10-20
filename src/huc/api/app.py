"""HTTP API. Run with: uvicorn huc.api.app:app --reload"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from huc.domain.models import Exam, Grade, Submission
from huc.services import HucService


@lru_cache
def _default_service() -> HucService:
    # Built on first request, so importing this module needs no API key.
    return HucService.default()


def create_app(service: HucService | None = None) -> FastAPI:
    """Build the FastAPI app. Pass `service` to inject a custom one (e.g. in tests)."""
    app = FastAPI(
        title="HUC - exam grading",
        description="Ingest student exams and grade them with LLM agents per rubric criterion.",
    )

    def get_service() -> HucService:
        return service or _default_service()

    svc_dep = Depends(get_service)

    @app.exception_handler(FileNotFoundError)
    async def not_found(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Not found: {exc.filename or exc}"})

    @app.exception_handler(ValueError)
    async def invalid(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.post("/exams", response_model=Exam, summary="Create an exam from a YAML file")
    async def create_exam(file: UploadFile, svc: HucService = svc_dep) -> Exam:
        return svc.save_exam_yaml(await file.read())

    @app.get("/exams", response_model=list[Exam], summary="List exams")
    def list_exams(svc: HucService = svc_dep) -> list[Exam]:
        return svc.repo.list_exams()

    @app.post(
        "/exams/{exam_id}/submissions",
        response_model=Submission,
        summary="Upload and ingest a submission (PDF, image or LMS JSON)",
    )
    async def upload_submission(
        exam_id: str, file: UploadFile, student_id: str | None = None, svc: HucService = svc_dep
    ) -> Submission:
        svc.repo.get_exam(exam_id)
        uploads = svc.settings.data_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "upload").suffix
        # TODO: move uploads to object storage (S3/GCS) instead of the local data dir.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=uploads) as tmp:
            tmp.write(await file.read())
        return await svc.ingest_file(exam_id, Path(tmp.name), student_id)

    @app.get(
        "/exams/{exam_id}/submissions",
        response_model=list[Submission],
        summary="List the submissions of an exam",
    )
    def list_submissions(exam_id: str, svc: HucService = svc_dep) -> list[Submission]:
        svc.repo.get_exam(exam_id)
        return svc.repo.list_submissions(exam_id)

    @app.post(
        "/exams/{exam_id}/grade",
        status_code=202,
        summary="Grade all pending submissions of an exam (in the background)",
    )
    def grade_exam(exam_id: str, background: BackgroundTasks, svc: HucService = svc_dep) -> dict:
        svc.repo.get_exam(exam_id)
        # TODO: use a real job queue (Arq/Celery/RQ) instead of BackgroundTasks in production.
        background.add_task(svc.grade_exam, exam_id)
        return {"status": "queued"}

    @app.get("/submissions/{submission_id}", response_model=Submission, summary="Get a submission")
    def get_submission(submission_id: str, svc: HucService = svc_dep) -> Submission:
        return svc.repo.get_submission(submission_id)

    @app.get(
        "/submissions/{submission_id}/grade",
        response_model=Grade,
        summary="Get the grade of a submission",
    )
    def get_grade(submission_id: str, svc: HucService = svc_dep) -> Grade:
        grade = svc.repo.get_grade(submission_id)
        if grade is None:
            raise HTTPException(404, "Not graded yet")
        return grade

    @app.post(
        "/submissions/{submission_id}/finalize",
        status_code=204,
        summary="Finalize a grade after teacher review",
    )
    def finalize(submission_id: str, svc: HucService = svc_dep) -> None:
        svc.finalize(submission_id)

    @app.get(
        "/exams/{exam_id}/export.csv",
        response_class=PlainTextResponse,
        summary="Export the grades of an exam as CSV",
    )
    def export_csv(exam_id: str, svc: HucService = svc_dep) -> str:
        return svc.export_csv(exam_id)

    return app


app = create_app()

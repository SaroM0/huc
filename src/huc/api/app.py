"""HTTP API. Run with: uvicorn huc.api.app:app --reload"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from huc.domain.models import Exam, Grade, Submission
from huc.domain.student_profile import StudentProfile
from huc.domain.workshop import WorkshopEntry, WorkshopSession
from huc.ingestion import loaders
from huc.ocr import OcrUnavailableError
from huc.report import ExamReport
from huc.services import HucService


class WorkshopCreate(BaseModel):
    student_id: str = Field(min_length=1)



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

    async def get_service() -> HucService:
        return service or _default_service()

    svc_dep = Depends(get_service)

    @app.exception_handler(FileNotFoundError)
    async def not_found(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Not found: {exc.filename or exc}"})

    @app.exception_handler(ValueError)
    async def invalid(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(OcrUnavailableError)
    async def ocr_unavailable(_: Request, exc: OcrUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.post("/exams", response_model=Exam, summary="Create an exam from a YAML file")
    async def create_exam(file: UploadFile, svc: HucService = svc_dep) -> Exam:
        return svc.save_exam_yaml(await file.read())

    @app.get(
        "/exams/{exam_id}/report",
        response_model=ExamReport,
        summary="Frequent errors and score summary of the graded submissions",
    )
    async def exam_report(exam_id: str, svc: HucService = svc_dep) -> ExamReport:
        return svc.report(exam_id)

    @app.get("/exams", response_model=list[Exam], summary="List exams")
    async def list_exams(svc: HucService = svc_dep) -> list[Exam]:
        return svc.repo.list_exams()

    @app.post(
        "/exams/{exam_id}/submissions",
        response_model=Submission,
        summary="Upload and ingest a handwritten submission (scanned PDF, PNG or JPG)",
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
        # The stored file has a random name: take the student from the uploaded one.
        student_id = student_id or loaders.student_id_from_path(Path(file.filename or tmp.name))
        return await svc.ingest_file(exam_id, Path(tmp.name), student_id)

    @app.get(
        "/exams/{exam_id}/submissions",
        response_model=list[Submission],
        summary="List the submissions of an exam",
    )
    async def list_submissions(exam_id: str, svc: HucService = svc_dep) -> list[Submission]:
        svc.repo.get_exam(exam_id)
        return svc.repo.list_submissions(exam_id)

    @app.post(
        "/exams/{exam_id}/grade",
        status_code=202,
        summary="Grade all pending submissions of an exam (in the background)",
    )
    async def grade_exam(exam_id: str, background: BackgroundTasks, svc: HucService = svc_dep) -> dict:
        svc.repo.get_exam(exam_id)
        # TODO: use a real job queue (Arq/Celery/RQ) instead of BackgroundTasks in production.
        background.add_task(svc.grade_exam, exam_id)
        return {"status": "queued"}

    @app.get("/submissions/{submission_id}", response_model=Submission, summary="Get a submission")
    async def get_submission(submission_id: str, svc: HucService = svc_dep) -> Submission:
        return svc.repo.get_submission(submission_id)

    @app.get(
        "/submissions/{submission_id}/grade",
        response_model=Grade,
        summary="Get the grade of a submission",
    )
    async def get_grade(submission_id: str, svc: HucService = svc_dep) -> Grade:
        grade = svc.repo.get_grade(submission_id)
        if grade is None:
            raise HTTPException(404, "Not graded yet")
        return grade

    @app.get("/submissions/{submission_id}/grade-history", response_model=list[Grade])
    async def grade_history(submission_id: str, svc: HucService = svc_dep) -> list[Grade]:
        svc.repo.get_submission(submission_id)
        return svc.repo.list_grade_history(submission_id)

    @app.get(
        "/exams/{exam_id}/export.csv",
        response_class=PlainTextResponse,
        summary="Export the grades of an exam as CSV",
    )
    async def export_csv(exam_id: str, include_flagged_feedback: bool = False,
                         svc: HucService = svc_dep) -> str:
        return svc.export_csv(exam_id, include_flagged_feedback=include_flagged_feedback)

    @app.post(
        "/exams/{exam_id}/profile",
        status_code=202,
        summary="Rate the submissions of an exam on the profile dimensions (in the background)",
    )
    async def profile_exam(exam_id: str, background: BackgroundTasks, force: bool = False,
                           svc: HucService = svc_dep) -> dict:
        if not svc.repo.get_exam(exam_id).profile.profiling.dimensions:
            raise ValueError(f"Exam {exam_id} has no profiling dimensions")
        background.add_task(svc.profile_exam, exam_id, force=force)
        return {"status": "queued"}

    @app.get("/students", response_model=list[str], summary="List profiled students")
    async def list_students(svc: HucService = svc_dep) -> list[str]:
        return svc.profiled_students()

    @app.get(
        "/students/{student_id}/profile",
        response_model=StudentProfile,
        summary="Student profile: baseline, current level and trend per dimension",
    )
    async def student_profile(student_id: str, svc: HucService = svc_dep) -> StudentProfile:
        return svc.student_profile(student_id)

    @app.get(
        "/profiles/export.csv",
        response_class=PlainTextResponse,
        summary="Export every student profile as CSV",
    )
    async def export_profiles(svc: HucService = svc_dep) -> str:
        return svc.export_profiles_csv()

    @app.post("/exams/{exam_id}/workshops", response_model=WorkshopSession, status_code=201)
    async def create_workshop(exam_id: str, body: WorkshopCreate,
                              svc: HucService = svc_dep) -> WorkshopSession:
        return svc.create_workshop(exam_id, body.student_id)

    @app.get("/exams/{exam_id}/workshops", response_model=list[WorkshopSession])
    async def list_workshops(exam_id: str, svc: HucService = svc_dep) -> list[WorkshopSession]:
        svc.repo.get_exam(exam_id)
        return svc.repo.list_workshops(exam_id)

    @app.get("/workshops/{session_id}", response_model=WorkshopSession)
    async def get_workshop(session_id: str, svc: HucService = svc_dep) -> WorkshopSession:
        return svc.repo.get_workshop(session_id)

    @app.post("/workshops/{session_id}/events", response_model=WorkshopSession, status_code=201)
    async def record_event(session_id: str, body: WorkshopEntry,
                           svc: HucService = svc_dep) -> WorkshopSession:
        return svc.record_workshop_event(session_id, body)

    return app


app = create_app()

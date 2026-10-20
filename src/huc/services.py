"""High-level use cases shared by the CLI and the API.

This is a thin layer over the LangGraph graphs: it builds them once, feeds them
inputs and handles the few things that are not part of a graph (loading exams,
finalizing, exporting).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import yaml

from huc.config import Settings, get_settings
from huc.domain.models import Exam, Grade, Submission, SubmissionStatus
from huc.graphs.grading import build_grading_graph
from huc.graphs.ingestion import build_ingestion_graph
from huc.storage.repository import FileRepository, Repository

# File types picked up when ingesting a whole directory.
INGESTIBLE_SUFFIXES = {".pdf", ".json", ".png", ".jpg", ".jpeg"}

# Submissions in these states are (re)graded by `grade_exam`.
PENDING_STATUSES = (SubmissionStatus.INGESTED, SubmissionStatus.FAILED)


class HucService:
    def __init__(self, repo: Repository, llm=None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.repo = repo
        self.ingestion_graph = build_ingestion_graph(repo=repo, llm=llm, settings=self.settings)
        self.grading_graph = build_grading_graph(repo=repo, llm=llm, settings=self.settings)

    @classmethod
    def default(cls) -> HucService:
        settings = get_settings()
        return cls(FileRepository(settings.data_dir), settings=settings)

    # -- exams --------------------------------------------------------------

    def load_exam_yaml(self, path: Path) -> Exam:
        return self.save_exam_yaml(Path(path).read_text("utf-8"))

    def save_exam_yaml(self, text: str | bytes) -> Exam:
        """Validate an exam definition (YAML) and persist it."""
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid exam YAML: {e}") from e
        exam = Exam.model_validate(data)
        self.repo.save_exam(exam)
        return exam

    # -- ingestion ----------------------------------------------------------

    async def ingest_file(
        self, exam_id: str, path: Path, student_id: str | None = None
    ) -> Submission:
        self.repo.get_exam(exam_id)  # fail fast if the exam does not exist
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result = await self.ingestion_graph.ainvoke(
            {"exam_id": exam_id, "file_path": str(path), "student_id": student_id}
        )
        return result["submission"]

    async def ingest_dir(self, exam_id: str, directory: Path) -> list[Submission]:
        files = sorted(
            p for p in Path(directory).iterdir()
            if p.is_file() and p.suffix.lower() in INGESTIBLE_SUFFIXES
        )
        return [await self.ingest_file(exam_id, f) for f in files]

    # -- grading ------------------------------------------------------------

    async def grade(self, submission_id: str) -> Grade:
        submission = self.repo.get_submission(submission_id)
        try:
            result = await self.grading_graph.ainvoke(
                {"submission_id": submission_id},
                config={"max_concurrency": self.settings.max_concurrency},
            )
        except Exception:
            submission = self.repo.get_submission(submission_id)
            submission.status = SubmissionStatus.FAILED
            self.repo.save_submission(submission)
            raise
        return result["grade"]

    async def grade_exam(self, exam_id: str) -> list[Grade]:
        """Grade every pending submission of an exam, one after another.

        Each submission already fans out per criterion inside the graph, so
        submissions are processed sequentially to keep concurrency bounded.
        """
        self.repo.get_exam(exam_id)
        pending = [
            s for s in self.repo.list_submissions(exam_id) if s.status in PENDING_STATUSES
        ]
        return [await self.grade(s.id) for s in pending]

    # -- human review & export ----------------------------------------------

    def finalize(self, submission_id: str) -> Submission:
        """Called by the teacher after reviewing (or accepting) the grade."""
        submission = self.repo.get_submission(submission_id)
        if self.repo.get_grade(submission_id) is None:
            raise ValueError(f"Submission {submission_id} has no grade yet")
        submission.status = SubmissionStatus.FINALIZED
        self.repo.save_submission(submission)
        return submission

    def export_csv(self, exam_id: str) -> str:
        exam = self.repo.get_exam(exam_id)
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(
            ["student_id", "status", *[f"q{q.number}" for q in exam.questions], "total"]
        )
        for s in self.repo.list_submissions(exam_id):
            grade = self.repo.get_grade(s.id)
            per_q = {qg.question_id: qg.points for qg in grade.questions} if grade else {}
            writer.writerow([
                s.student_id,
                s.status,
                *[per_q.get(q.id, "") for q in exam.questions],
                grade.total if grade else "",
            ])
        return out.getvalue()

"""High-level use cases shared by the CLI and the API.

This is a thin layer over the LangGraph graphs: it builds them once, feeds them
inputs and handles the few things that are not part of a graph (loading exams,
exporting).
"""

from __future__ import annotations

import csv
import io
from collections.abc import AsyncIterator
from pathlib import Path

import yaml

from huc import rubrics
from huc.config import Settings, get_settings
from huc.domain.models import Exam, Grade, Submission, SubmissionStatus
from huc.domain.profile import CourseProfile
from huc.graphs.grading import build_grading_graph
from huc.graphs.ingestion import build_ingestion_graph
from huc.ingestion import loaders
from huc.llm import get_chat_model
from huc.ocr import get_ocr_model, transcribe_page
from huc.report import ExamReport, build_report
from huc.storage.repository import FileRepository, Repository

# File types picked up when ingesting a whole directory.
INGESTIBLE_SUFFIXES = loaders.SUPPORTED_SUFFIXES

# Submissions in these states are (re)graded by `grade_exam`.
PENDING_STATUSES = (SubmissionStatus.INGESTED, SubmissionStatus.FAILED)


class HucService:
    def __init__(
        self, repo: Repository, llm=None, settings: Settings | None = None, ocr=None
    ) -> None:
        """`llm`: Claude-like chat model for grading and rubric drafting (None -> real
        models); `ocr`: OCR vision model that reads the scans (None -> Claude Haiku)."""
        self.settings = settings or get_settings()
        self.repo = repo
        self.llm = llm
        self.ocr = ocr
        self.ingestion_graph = build_ingestion_graph(repo=repo, settings=self.settings, ocr=ocr)
        self.grading_graph = build_grading_graph(repo=repo, llm=llm, settings=self.settings)

    @classmethod
    def default(cls) -> HucService:
        settings = get_settings()
        return cls(FileRepository(settings.data_dir), settings=settings)

    # -- exams --------------------------------------------------------------

    def load_exam_yaml(self, path: Path) -> Exam:
        return self.save_exam_yaml(Path(path).read_text("utf-8"))

    def save_exam_yaml(self, text: str | bytes) -> Exam:
        """Validate an exam definition (YAML) and persist it.

        `profile: <id>` is replaced by the profile file `<profiles_dir>/<id>.yaml`, so the
        stored exam carries a copy of it (later edits to the file need an exam reload).
        """
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid exam YAML: {e}") from e
        if isinstance(data, dict) and isinstance(data.get("profile"), str):
            data["profile"] = self.load_profile(data["profile"])
        exam = Exam.model_validate(data)
        if not exam.questions:
            raise ValueError("An exam must contain at least one question")
        ids = [q.id for q in exam.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Question ids must be unique")
        for question in exam.questions:
            criteria_ids = [c.id for c in question.criteria]
            if not criteria_ids or len(criteria_ids) != len(set(criteria_ids)):
                raise ValueError(f"Question {question.id} needs criteria with unique ids")
            for criterion in question.criteria:
                if criterion.max_points <= 0:
                    raise ValueError(f"Criterion {criterion.id} must have positive max_points")
                try:
                    levels = [float(level) for level in criterion.levels]
                except ValueError as e:
                    raise ValueError(f"Criterion {criterion.id} has nonnumeric levels") from e
                if any(not 0 <= level <= criterion.max_points for level in levels):
                    raise ValueError(f"Criterion {criterion.id} has levels outside its range")
        unknown = {q.subject for q in exam.questions if q.subject} - {
            s.id for s in exam.profile.subjects
        }
        if unknown:
            raise ValueError(f"Subjects not in profile {exam.profile.id!r}: {sorted(unknown)}")
        self.repo.save_exam(exam)
        return exam

    async def draft_rubrics(
        self,
        solutions_pdf: Path,
        modules: list[str] | None = None,
        *,
        profile_id: str | None = None,
        week: int | None = None,
    ) -> list[tuple[Exam, str]]:
        """Draft one exam YAML per module section of a reference-solutions PDF (all
        sections if `modules` is None). Returns (exam, YAML text); nothing is stored."""
        path = Path(solutions_pdf)
        if not path.is_file():
            raise FileNotFoundError(path)
        sections = rubrics.find_sections(path)
        if not sections:
            raise ValueError(f"No 'Taller <week><module>' sections found in {path.name}")
        wanted = [m.upper() for m in modules] if modules else list(sections)
        missing = [m for m in wanted if m not in sections]
        if missing:
            raise ValueError(f"Modules {missing} not in {path.name} (found: {list(sections)})")
        week = week or rubrics.week_of(path, sections)
        profile = self.load_profile(profile_id) if profile_id else CourseProfile()
        model_name = self.settings.rubric_model
        model = self.llm if self.llm is not None else get_chat_model(model_name)
        drafts = []
        for module in wanted:
            section = sections[module]
            exam_id = f"s{week}-{module.lower()}" if week else module.lower()
            pages = rubrics.section_pages(
                path, section, self.settings.pdf_dpi, self.settings.ocr_max_side_px
            )
            exam = await rubrics.draft_exam(
                model, section, pages, profile,
                exam_id=exam_id, title=f"Taller S{week or '?'} {module}",
            )
            source = f"{path.name}, section '{section.header}', pages {section.pages}"
            drafts.append(
                (exam, rubrics.exam_yaml(exam, profile_id=profile_id, source=source,
                                         model=model_name))
            )
        return drafts

    def load_profile(self, profile_id: str) -> CourseProfile:
        path = self.settings.profiles_dir / f"{profile_id}.yaml"
        if not path.is_file():
            raise ValueError(f"Profile {profile_id!r} not found ({path})")
        try:
            data = yaml.safe_load(path.read_text("utf-8"))
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid profile YAML {path}: {e}") from e
        return CourseProfile.model_validate({"id": profile_id, **(data or {})})

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

    async def ocr_file(self, path: Path) -> AsyncIterator[tuple[int, str]]:
        """Plain per-page OCR (no grading, nothing saved): yield (page number, text)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        doc = loaders.load_document(
            path, dpi=self.settings.pdf_dpi, max_side=self.settings.ocr_max_side_px
        )
        model = self.ocr if self.ocr is not None else get_ocr_model(self.settings)
        for page in doc.pages:
            yield page.number, await transcribe_page(model, page)

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

    # -- report / export ----------------------------------------------------

    def report(self, exam_id: str) -> ExamReport:
        """Frequent errors and score summary of an exam's graded submissions."""
        exam = self.repo.get_exam(exam_id)
        submissions = self.repo.list_submissions(exam_id)
        grades = [g for s in submissions if (g := self.repo.get_grade(s.id)) is not None]
        if any(g.exam_snapshot is None for g in grades):
            raise ValueError("Cannot aggregate grades without an exam snapshot")
        if any(g.exam_snapshot is not None and g.exam_snapshot != exam for g in grades):
            raise ValueError(
                "Grades were produced using different exam or rubric snapshots; "
                "do not combine them in one report"
            )
        return build_report(exam, submissions, grades)

    def export_csv(self, exam_id: str, *, include_flagged_feedback: bool = False) -> str:
        """One row per submission: points, error types and feedback per question, total,
        percentage, the profile's band and the verification label."""
        exam = self.repo.get_exam(exam_id)
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([
            "student_id",
            "status",
            *[f"q{q.number}" for q in exam.questions],
            "total",
            "percent",
            "band",
            "needs_verification",
            "verification_reasons",
            "rubric_snapshot_mismatch",
            *[f"q{q.number}_errors" for q in exam.questions],
            *[f"q{q.number}_feedback" for q in exam.questions],
            *[f"q{q.number}_feedback_needs_review" for q in exam.questions],
        ])
        for s in self.repo.list_submissions(exam_id):
            grade = self.repo.get_grade(s.id)
            per_q = {qg.question_id: qg for qg in grade.questions} if grade else {}
            legacy_flag = bool(grade and grade.needs_verification and not any(
                qg.feedback_needs_review for qg in grade.questions
            ))

            review_status = {qid: legacy_flag or qg.requires_feedback_review()
                             for qid, qg in per_q.items()}

            grade_exam = grade.exam_snapshot if grade and grade.exam_snapshot else exam
            grade_max = sum(q.max_points for q in grade_exam.questions)
            percent = round(100 * grade.total / grade_max, 1) if grade and grade_max else None
            band = grade_exam.profile.band(percent) if percent is not None else None
            writer.writerow([
                s.student_id,
                s.status,
                *[per_q[q.id].points if q.id in per_q else "" for q in exam.questions],
                grade.total if grade else "",
                "" if percent is None else percent,
                band.label if band else "",
                grade.needs_verification if grade else "",
                "; ".join(grade.verification_reasons) if grade else "",
                grade.exam_snapshot != exam if grade and grade.exam_snapshot else "unknown",
                *[
                    "; ".join(sc.error_type for sc in per_q[q.id].scores if sc.error_type)
                    if q.id in per_q
                    else ""
                    for q in exam.questions
                ],
                *[
                    per_q[q.id].feedback
                    if q.id in per_q and (include_flagged_feedback or not review_status[q.id])
                    else ""
                    for q in exam.questions
                ],
                *[review_status[q.id] if q.id in per_q else ""
                  for q in exam.questions],
            ])
        return out.getvalue()

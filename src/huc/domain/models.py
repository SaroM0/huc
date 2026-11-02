"""Domain model. The contract shared by ingestion, grading, storage and the API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from huc.domain.profile import CourseProfile


def _id() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Exam definition (written by the teacher)
# ---------------------------------------------------------------------------


class QuestionType(StrEnum):
    """All questions are open-ended (handwritten); every type is graded per rubric criterion."""

    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    CODE = "code"


class Criterion(BaseModel):
    """A rubric criterion. Each criterion is graded by a separate agent call."""

    id: str
    description: str
    max_points: float = Field(gt=0, allow_inf_nan=False)
    # Per-level descriptors, e.g. {"2": "complete and correct", "1": "partial", "0": "missing"}
    levels: dict[str, str] = Field(default_factory=dict)


class Question(BaseModel):
    id: str
    number: str  # visible label: "1", "2a"...
    type: QuestionType
    prompt: str
    reference_answer: str | None = None
    criteria: list[Criterion] = Field(default_factory=list)
    # Id of a subject of the exam's profile (its context is given to the grader)
    subject: str | None = None
    # Skill the question assesses, e.g. "Demostrar por contrapositivo"
    skill: str | None = None
    # Restrictions the answer must respect, e.g. "Usar solo los axiomas de cuerpo"
    constraints: list[str] = Field(default_factory=list)

    @property
    def max_points(self) -> float:
        return sum(c.max_points for c in self.criteria)


class Exam(BaseModel):
    id: str = Field(default_factory=_id)
    title: str
    course: str | None = None
    rubric_version: int = 1
    # Course parameters (context, grading/feedback policy, scale); see domain/profile.py
    profile: CourseProfile = Field(default_factory=CourseProfile)
    questions: list[Question]

    def question(self, question_id: str) -> Question:
        return next(q for q in self.questions if q.id == question_id)


# ---------------------------------------------------------------------------
# Student submission (output of ingestion)
# ---------------------------------------------------------------------------


class SourceKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"


class Answer(BaseModel):
    question_id: str
    text: str
    # Reading confidence (0-1): transcription + question assignment, estimated by the OCR model
    extraction_confidence: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    source_pages: list[int] = Field(default_factory=list)
    reading_issues: list[str] = Field(default_factory=list)


class SubmissionStatus(StrEnum):
    INGESTED = "ingested"
    GRADING = "grading"
    GRADED = "graded"
    FAILED = "failed"


class Submission(BaseModel):
    id: str = Field(default_factory=_id)
    exam_id: str
    student_id: str
    source_kind: SourceKind
    source_path: str | None = None
    answers: list[Answer] = Field(default_factory=list)
    reading_issues: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    status: SubmissionStatus = SubmissionStatus.INGESTED
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Grading results
# ---------------------------------------------------------------------------


class CriterionScore(BaseModel):
    """Structured output of the grader agent for one criterion."""

    criterion_id: str
    points: float = Field(allow_inf_nan=False)
    justification: str
    evidence: list[str] = Field(
        default_factory=list, description="Verbatim quotes from the answer supporting the score"
    )
    confidence: float = Field(ge=0, le=1)
    error_type: str | None = Field(
        default=None, description="Id of the main error type of the profile; null if none"
    )


class ReviewVerdict(BaseModel):
    """Structured output of the reviewer agent over all scores of one answer."""

    approved: bool
    issues: list[str] = Field(default_factory=list)
    adjusted_scores: list[CriterionScore] = Field(default_factory=list)
    needs_verification: bool = Field(
        default=False, description="Real ambiguity a teacher should check"
    )
    feedback: str = Field(
        default="", description="Feedback for the student on this answer, in Spanish"
    )


class FeedbackAudit(BaseModel):
    """Separate model call's fallible audit of the mathematical feedback."""

    needs_verification: bool
    issues: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class QuestionGrade(BaseModel):
    question_id: str
    scores: list[CriterionScore]
    proposed_scores: list[CriterionScore] = Field(default_factory=list)
    review: ReviewVerdict | None = None
    feedback_audit: FeedbackAudit | None = None
    feedback: str = ""
    feedback_needs_review: bool = False
    feedback_review_reasons: list[str] = Field(default_factory=list)
    graded_by: str  # model name

    def requires_feedback_review(self) -> bool:
        """Conservatively handle old records and runs without a feedback audit."""
        return (self.feedback_needs_review or self.feedback_audit is None
                or self.feedback_audit.needs_verification or bool(self.feedback_audit.issues)
                or bool(self.review and self.review.needs_verification))

    @property
    def points(self) -> float:
        return sum(s.points for s in self.scores)


class Grade(BaseModel):
    id: str = Field(default_factory=_id)
    submission_id: str
    exam_id: str
    rubric_version: int
    questions: list[QuestionGrade]
    # Every exam is graded; low confidence only labels the grade for teacher verification.
    needs_verification: bool = False
    verification_reasons: list[str] = Field(default_factory=list)
    # Snapshots make a run inspectable even if the exam or submission changes later.
    exam_snapshot: Exam | None = None
    submission_snapshot: Submission | None = None
    provenance: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)

    @property
    def total(self) -> float:
        return sum(q.points for q in self.questions)

"""Domain model. The contract shared by ingestion, grading, storage and the API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


def _id() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Exam definition (written by the teacher)
# ---------------------------------------------------------------------------


class QuestionType(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    CODE = "code"


class Criterion(BaseModel):
    """A rubric criterion. Each criterion is graded by a separate agent call."""

    id: str
    description: str
    max_points: float
    # Per-level descriptors, e.g. {"2": "complete and correct", "1": "partial", "0": "missing"}
    levels: dict[str, str] = Field(default_factory=dict)


class Question(BaseModel):
    id: str
    number: str  # visible label: "1", "2a"...
    type: QuestionType
    prompt: str
    reference_answer: str | None = None
    # Multiple choice only: graded deterministically, without an LLM
    correct_option: str | None = None
    criteria: list[Criterion] = Field(default_factory=list)

    @property
    def max_points(self) -> float:
        return sum(c.max_points for c in self.criteria)


class Exam(BaseModel):
    id: str = Field(default_factory=_id)
    title: str
    course: str | None = None
    rubric_version: int = 1
    questions: list[Question]

    def question(self, question_id: str) -> Question:
        return next(q for q in self.questions if q.id == question_id)


# ---------------------------------------------------------------------------
# Student submission (output of ingestion)
# ---------------------------------------------------------------------------


class SourceKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"
    LMS_JSON = "lms_json"


class Answer(BaseModel):
    question_id: str
    text: str
    # Transcription/segmentation confidence (1.0 when the answer was born digital)
    extraction_confidence: float = 1.0
    source_pages: list[int] = Field(default_factory=list)


class SubmissionStatus(StrEnum):
    INGESTED = "ingested"
    GRADING = "grading"
    GRADED = "graded"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    FINALIZED = "finalized"
    FAILED = "failed"


class Submission(BaseModel):
    id: str = Field(default_factory=_id)
    exam_id: str
    student_id: str
    source_kind: SourceKind
    source_path: str | None = None
    answers: list[Answer] = Field(default_factory=list)
    status: SubmissionStatus = SubmissionStatus.INGESTED
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Grading results
# ---------------------------------------------------------------------------


class CriterionScore(BaseModel):
    """Structured output of the grader agent for one criterion."""

    criterion_id: str
    points: float
    justification: str
    evidence: list[str] = Field(
        default_factory=list, description="Verbatim quotes from the answer supporting the score"
    )
    confidence: float = Field(ge=0, le=1)


class ReviewVerdict(BaseModel):
    """Structured output of the reviewer agent over all scores of one answer."""

    approved: bool
    issues: list[str] = Field(default_factory=list)
    adjusted_scores: list[CriterionScore] = Field(default_factory=list)
    needs_human: bool = False


class QuestionGrade(BaseModel):
    question_id: str
    scores: list[CriterionScore]
    review: ReviewVerdict | None = None
    graded_by: str  # model name, "deterministic" or "human:<id>"

    @property
    def points(self) -> float:
        return sum(s.points for s in self.scores)


class Grade(BaseModel):
    id: str = Field(default_factory=_id)
    submission_id: str
    exam_id: str
    rubric_version: int
    questions: list[QuestionGrade]
    needs_human_review: bool = False
    created_at: datetime = Field(default_factory=_now)

    @property
    def total(self) -> float:
        return sum(q.points for q in self.questions)

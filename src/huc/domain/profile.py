"""Course profile: the course-level grading parameters (context, policy, feedback, scale).

An exam YAML names a profile (`profile: mat0004`, a file in `HUC_PROFILES_DIR`) or inlines
one. When the exam is loaded the profile is copied into the stored Exam, so a grade is
always reproducible from the exam alone. Every field is optional: the default profile
reproduces the generic behaviour (no course context, 2-4 sentence feedback in Spanish).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Subject(BaseModel):
    """A strand of the course (e.g. Cálculo). Questions point to it with `subject: <id>`."""

    id: str
    name: str
    # What the grader must know about it: topics, conventions, results students may use.
    context: str = ""


class ErrorType(BaseModel):
    """A class of mistake the grader labels each score with (`CriterionScore.error_type`)."""

    id: str
    description: str


class ReadingPolicy(BaseModel):
    """Extra rules for the OCR/reader model (e.g. course notation)."""

    notes: list[str] = Field(default_factory=list)


class GradingPolicy(BaseModel):
    # Added to the grader and reviewer prompts, in order.
    rules: list[str] = Field(default_factory=list)
    # Empty: scores carry no error type.
    error_types: list[ErrorType] = Field(default_factory=list)
    # Overrides HUC_VERIFICATION_THRESHOLD for exams with this profile.
    verification_threshold: float | None = Field(default=None, ge=0, le=1)


class FeedbackStep(BaseModel):
    name: str
    question: str


DEFAULT_STRUCTURE = [
    FeedbackStep(name="Logros", question="what the answer gets right"),
    FeedbackStep(name="Errores", question="what is missing or wrong"),
    FeedbackStep(name="Mejora", question="how to improve it"),
]


class FeedbackPolicy(BaseModel):
    language: str = "Spanish"
    # How the feedback addresses the student.
    address: str = "tú"
    sentences: str = "2-4"
    # The feedback covers these steps, in order.
    structure: list[FeedbackStep] = Field(default_factory=lambda: list(DEFAULT_STRUCTURE))
    principles: list[str] = Field(default_factory=list)
    good_examples: list[str] = Field(default_factory=list)
    # Comments the feedback must never resemble.
    forbidden: list[str] = Field(default_factory=list)
    # Feedback for unanswered questions (no reviewer call).
    empty_answer: str = "No respondiste esta pregunta."


class RubricPolicy(BaseModel):
    """How `huc rubric-draft` turns a reference solution into questions and criteria."""

    # Points of each question (sub-parts are separate questions); None: the model decides.
    points_per_question: float | None = Field(default=None, gt=0)
    max_criteria: int = Field(default=3, ge=1)
    guidelines: list[str] = Field(default_factory=list)


class GradeBand(BaseModel):
    """A label for a score percentage, e.g. D (Destacado) from 75 %."""

    label: str
    min_percent: float = Field(ge=0, le=100)
    description: str = ""


class CourseProfile(BaseModel):
    id: str = "default"
    name: str = ""
    # Course-wide context for every grader call (who the students are, what is valued).
    description: str = ""
    subjects: list[Subject] = Field(default_factory=list)
    reading: ReadingPolicy = Field(default_factory=ReadingPolicy)
    grading: GradingPolicy = Field(default_factory=GradingPolicy)
    feedback: FeedbackPolicy = Field(default_factory=FeedbackPolicy)
    rubric: RubricPolicy = Field(default_factory=RubricPolicy)
    # Bands by score percentage, any order. Empty: no band in the export.
    scale: list[GradeBand] = Field(default_factory=list)

    def subject(self, subject_id: str | None) -> Subject | None:
        return next((s for s in self.subjects if s.id == subject_id), None)

    def band(self, percent: float) -> GradeBand | None:
        reached = [b for b in self.scale if percent >= b.min_percent]
        return max(reached, key=lambda b: b.min_percent, default=None)

    def error_type_ids(self) -> set[str]:
        return {e.id for e in self.grading.error_types}

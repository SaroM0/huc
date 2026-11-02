"""Exam report: frequent errors and score summary over every graded submission.

Implements the TA guide's "registrar errores frecuentes": the error types the graders
assigned, counted per question and overall, with example justifications to comment on in
class. Pure function over stored data (no LLM).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from huc.domain.models import Exam, Grade, Submission

# Example justifications kept per error type
MAX_EXAMPLES = 3


class ErrorCount(BaseModel):
    error_type: str
    description: str = ""
    count: int
    examples: list[str] = Field(default_factory=list, description="Grader justifications")


class CriterionStats(BaseModel):
    criterion_id: str
    mean_points: float
    max_points: float


class QuestionStats(BaseModel):
    question_id: str
    number: str
    graded: int
    mean_points: float
    max_points: float
    errors: list[ErrorCount]
    criteria: list[CriterionStats]


class ExamReport(BaseModel):
    exam_id: str
    title: str
    submissions: int
    graded: int
    mean_percent: float | None
    bands: dict[str, int] = Field(description="Graded submissions per profile band")
    needs_verification: int
    errors: list[ErrorCount] = Field(description="Error types over the whole exam")
    questions: list[QuestionStats]


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def build_report(exam: Exam, submissions: list[Submission], grades: list[Grade]) -> ExamReport:
    """`grades`: the stored grades of the exam's submissions (ungraded ones absent)."""
    descriptions = {e.id: e.description for e in exam.profile.grading.error_types}
    max_total = sum(q.max_points for q in exam.questions)

    def counts(pairs: list[tuple[str, str]]) -> list[ErrorCount]:
        """(error_type, justification) pairs -> counts, most frequent first."""
        examples: dict[str, list[str]] = defaultdict(list)
        for etype, why in pairs:
            if why and why not in examples[etype] and len(examples[etype]) < MAX_EXAMPLES:
                examples[etype].append(why)
        return [
            ErrorCount(
                error_type=etype,
                description=descriptions.get(etype, ""),
                count=n,
                examples=examples[etype],
            )
            for etype, n in Counter(e for e, _ in pairs).most_common()
        ]

    all_errors: list[tuple[str, str]] = []
    questions = []
    for q in exam.questions:
        graded = [qg for g in grades for qg in g.questions if qg.question_id == q.id]
        pairs = [(s.error_type, s.justification) for qg in graded for s in qg.scores if s.error_type]
        all_errors += pairs
        by_criterion = defaultdict(list)
        for qg in graded:
            for s in qg.scores:
                by_criterion[s.criterion_id].append(s.points)
        questions.append(
            QuestionStats(
                question_id=q.id,
                number=q.number,
                graded=len(graded),
                mean_points=_mean([qg.points for qg in graded]),
                max_points=q.max_points,
                errors=counts(pairs),
                criteria=[
                    CriterionStats(
                        criterion_id=c.id,
                        mean_points=_mean(by_criterion[c.id]),
                        max_points=c.max_points,
                    )
                    for c in q.criteria
                ],
            )
        )

    percents = [100 * g.total / max_total for g in grades] if max_total else []
    bands = Counter(
        band.label for p in percents if (band := exam.profile.band(p)) is not None
    )
    return ExamReport(
        exam_id=exam.id,
        title=exam.title,
        submissions=len(submissions),
        graded=len(grades),
        mean_percent=round(sum(percents) / len(percents), 1) if percents else None,
        # Profile band order (highest first), including empty bands
        bands={
            b.label: bands.get(b.label, 0)
            for b in sorted(exam.profile.scale, key=lambda b: -b.min_percent)
        },
        needs_verification=sum(g.needs_verification for g in grades),
        errors=counts(all_errors),
        questions=questions,
    )

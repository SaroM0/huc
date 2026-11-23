"""Longitudinal student profile (see docs/PERFIL_ESTUDIANTE.md).

Every submission is rated, blind, on the profile's course-wide dimensions and stored as a
`SubmissionProfile` (append-only). `compute_profile` folds a student's stored ratings into a
`StudentProfile`: the baseline is their first submission, each later one is compared with
it. The comparison is deterministic code, never the model.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from huc.domain.models import ExamSequence, _id, _now

# ---------------------------------------------------------------------------
# Profiler output (structured output of the model, one call per submission)
# ---------------------------------------------------------------------------


class DimensionRating(BaseModel):
    dimension_id: str
    observable: bool = Field(
        description="false when the submission gives no evidence about this dimension"
    )
    level: int | None = Field(default=None, description="A level of the scale; null if not observable")
    evidence: list[str] = Field(
        default_factory=list, description="Verbatim quotes from the answers supporting the level"
    )
    rationale: str = Field(default="", description="Why this level, in Spanish")
    confidence: float = Field(ge=0, le=1)


class ProfilerOutput(BaseModel):
    ratings: list[DimensionRating]


# ---------------------------------------------------------------------------
# Stored record
# ---------------------------------------------------------------------------


class DimensionObservation(BaseModel):
    dimension_id: str
    level: int | None = None  # None: not observable in this submission
    evidence: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(default=0, ge=0, le=1)
    # Checks the code made on the model output (e.g. evidence not in the transcription)
    issues: list[str] = Field(default_factory=list)
    # Grader error types of this dimension counted in the submission's grade (None: no grade)
    error_count: int | None = None

    @property
    def observable(self) -> bool:
        return self.level is not None


class SubmissionProfile(BaseModel):
    """One profiling run of one submission. Runs are kept; the latest one counts."""

    id: str = Field(default_factory=_id)
    submission_id: str
    exam_id: str
    student_id: str
    exam_title: str = ""
    sequence: ExamSequence | None = None
    submitted_at: datetime  # Submission.created_at: order fallback without a sequence
    profiling_version: int
    dimensions_hash: str
    observations: list[DimensionObservation]
    provenance: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)

    @property
    def label(self) -> str:
        return self.sequence.label if self.sequence else self.exam_id

    @property
    def order_key(self) -> tuple:
        # Exams without a sequence go after the sequenced ones, by submission time.
        if self.sequence is not None:
            return (0, self.sequence.key, self.submitted_at)
        return (1, (), self.submitted_at)

    def observation(self, dimension_id: str) -> DimensionObservation | None:
        return next((o for o in self.observations if o.dimension_id == dimension_id), None)


# ---------------------------------------------------------------------------
# Computed profile
# ---------------------------------------------------------------------------


class TrendStatus(StrEnum):
    IMPROVED = "mejora"
    STABLE = "estable"
    REGRESSED = "retroceso"
    BASELINE_ONLY = "solo_linea_base"  # observed once: nothing to compare yet
    NO_BASELINE = "sin_linea_base"  # never observable so far


class ProfilePoint(BaseModel):
    submission_id: str
    exam_id: str
    label: str
    level: int | None
    delta: int | None  # level - baseline; None when either is missing
    confidence: float
    low_confidence: bool
    issues: list[str] = Field(default_factory=list)
    error_count: int | None = None


class DimensionTrend(BaseModel):
    dimension_id: str
    name: str
    baseline_level: int | None = None
    baseline_label: str | None = None
    # The first submission did not show this dimension: the baseline is a later one.
    baseline_deferred: bool = False
    current_level: int | None = None
    current_label: str | None = None
    delta: int | None = None
    status: TrendStatus
    # The current observation is below the confidence threshold or failed a check.
    low_confidence: bool = False
    series: list[ProfilePoint]


class StudentProfile(BaseModel):
    student_id: str
    profiling_version: int | None = None
    dimensions_hash: str | None = None
    first_submission: str | None = None  # label of the baseline submission
    submissions: int = 0
    dimensions: list[DimensionTrend] = Field(default_factory=list)
    # Runs not used: rated with other dimensions than the current ones.
    excluded_runs: int = 0
    warnings: list[str] = Field(default_factory=list)


def latest_runs(runs: list[SubmissionProfile]) -> list[SubmissionProfile]:
    """The newest run of each submission."""
    latest: dict[str, SubmissionProfile] = {}
    for run in sorted(runs, key=lambda r: r.created_at):
        latest[run.submission_id] = run
    return list(latest.values())


def compute_profile(
    student_id: str,
    runs: list[SubmissionProfile],
    dimensions: list[tuple[str, str]],
    *,
    min_change: float,
    confidence_threshold: float,
) -> StudentProfile:
    """Fold a student's profiling runs into a profile.

    `runs`: their stored runs (any order, any version); only the newest run per submission
    whose `dimensions_hash` matches the newest run overall is used. `dimensions`:
    (id, name) of the current dimensions, in report order.
    """
    runs = latest_runs([r for r in runs if r.student_id == student_id])
    if not runs:
        return StudentProfile(student_id=student_id)
    current_hash = max(runs, key=lambda r: r.created_at).dimensions_hash
    used = sorted((r for r in runs if r.dimensions_hash == current_hash), key=lambda r: r.order_key)
    profile = StudentProfile(
        student_id=student_id,
        profiling_version=used[-1].profiling_version,
        dimensions_hash=current_hash,
        first_submission=used[0].label,
        submissions=len(used),
        excluded_runs=len(runs) - len(used),
    )
    if profile.excluded_runs:
        profile.warnings.append(
            f"{profile.excluded_runs} submission(s) rated with other dimensions; re-run profiling"
        )
    if any(r.sequence is None for r in used):
        profile.warnings.append("some exams have no sequence; ordered by submission time")
    labels = [r.label for r in used]
    if len(set(labels)) != len(labels):
        profile.warnings.append("several submissions for the same exam")

    for dimension_id, name in dimensions:
        series: list[ProfilePoint] = []
        baseline: tuple[int, str] | None = None
        for run in used:
            obs = run.observation(dimension_id)
            level = obs.level if obs else None
            if baseline is None and level is not None:
                baseline = (level, run.label)
            confidence = obs.confidence if obs else 0
            series.append(ProfilePoint(
                submission_id=run.submission_id,
                exam_id=run.exam_id,
                label=run.label,
                level=level,
                delta=None if level is None or baseline is None else level - baseline[0],
                confidence=confidence,
                low_confidence=level is not None and (
                    confidence < confidence_threshold or bool(obs and obs.issues)),
                issues=obs.issues if obs else [],
                error_count=obs.error_count if obs else None,
            ))
        trend = DimensionTrend(dimension_id=dimension_id, name=name, series=series,
                               status=TrendStatus.NO_BASELINE)
        if baseline is not None:
            trend.baseline_level, trend.baseline_label = baseline
            trend.baseline_deferred = series[0].level is None
            observed = [p for p in series if p.level is not None]
            current = observed[-1]
            assert current.level is not None
            trend.current_level, trend.current_label = current.level, current.label
            trend.low_confidence = current.low_confidence
            if len(observed) == 1:
                trend.status = TrendStatus.BASELINE_ONLY
            else:
                delta = current.level - trend.baseline_level
                trend.delta = delta
                if delta >= min_change:
                    trend.status = TrendStatus.IMPROVED
                elif delta <= -min_change:
                    trend.status = TrendStatus.REGRESSED
                else:
                    trend.status = TrendStatus.STABLE
        profile.dimensions.append(trend)
    return profile

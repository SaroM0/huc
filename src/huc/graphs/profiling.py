"""Profiling graph: Submission -> persisted SubmissionProfile (student profile ratings).

    START -> load -> rate (one LLM call per submission) -> save -> END

The profiler sees the questions and the transcribed answers only: never the grade or the
student's earlier ratings. The grade is read in `save` just to count the grader's error
types per dimension (a derived signal, computed in code).

Input:  {"submission_id": str}
Output: state with "profile" (persisted SubmissionProfile).
"""

import hashlib
import json
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from huc.config import Settings, get_settings
from huc.domain.models import Answer, Exam, Grade, Submission
from huc.domain.profile import ProfilingPolicy
from huc.domain.student_profile import (
    DimensionObservation,
    DimensionRating,
    ProfilerOutput,
    SubmissionProfile,
)
from huc.llm import build_messages, get_chat_model, structured
from huc.prompts import profiling as prompts
from huc.storage.repository import FileRepository, Repository


class ProfilingInput(TypedDict):
    submission_id: str


class ProfilingState(ProfilingInput, total=False):
    submission: Submission
    exam: Exam
    output: ProfilerOutput | None
    profile: SubmissionProfile


def _answers(submission: Submission) -> dict[str, Answer]:
    return {a.question_id: a for a in submission.answers}


def sanitize_rating(
    rating: DimensionRating | None, policy: ProfilingPolicy, answers: list[Answer]
) -> DimensionObservation:
    """Never trust the model blindly: snap the level to the scale, check that the quotes are
    in the transcription and cap the confidence at the reading confidence."""
    if rating is None:
        return DimensionObservation(dimension_id="", issues=["no rating returned by the model"])
    issues: list[str] = []
    level = rating.level if rating.observable else None
    levels = policy.level_values()
    if level is not None and level not in levels:
        raw = level
        snapped = min(levels, key=lambda v: (abs(v - raw), v))
        issues.append(f"level {raw} outside the scale, set to {snapped}")
        level = snapped
    if rating.observable and rating.level is None:
        issues.append("observable without a level; treated as not observable")
    evidence = [q for q in rating.evidence if q.strip()]
    if level is not None:
        texts = [a.text for a in answers]
        if not evidence:
            issues.append("no evidence quoted")
        elif any(not any(q in t for t in texts) for q in evidence):
            issues.append("evidence not in transcription")
    reading = min((a.extraction_confidence for a in answers if a.text.strip()), default=1.0)
    return DimensionObservation(
        dimension_id=rating.dimension_id,
        level=level,
        evidence=evidence,
        rationale=rating.rationale,
        confidence=min(rating.confidence, reading),
        issues=issues,
    )


def error_counts(policy: ProfilingPolicy, grade: Grade | None) -> dict[str, int | None]:
    """Per dimension, how many criterion scores of the grade carry one of its error types."""
    if grade is None:
        return {d.id: None for d in policy.dimensions}
    types = [s.error_type for qg in grade.questions for s in qg.scores if s.error_type]
    return {d.id: sum(t in d.error_types for t in types) if d.error_types else None
            for d in policy.dimensions}


def build_profiling_graph(
    repo: Repository | None = None,
    llm: BaseChatModel | Any | None = None,
    *,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """`repo=None` -> FileRepository(settings.data_dir); `llm=None` -> the Claude
    `profiler_model`; `settings=None` -> get_settings()."""

    def cfg() -> Settings:
        return settings or get_settings()

    def get_repo() -> Repository:
        return repo if repo is not None else FileRepository(cfg().data_dir)

    def model() -> Any:
        return llm if llm is not None else get_chat_model(cfg().profiler_model)

    def model_name() -> str:
        if llm is None:
            return cfg().profiler_model
        return str(getattr(llm, "model", None) or type(llm).__name__)

    async def load(state: ProfilingState) -> dict:
        r = get_repo()
        submission = r.get_submission(state["submission_id"])
        exam = r.get_exam(submission.exam_id)
        if not exam.profile.profiling.dimensions:
            raise ValueError(f"Profile {exam.profile.id!r} of exam {exam.id} has no "
                             "profiling dimensions")
        return {"submission": submission, "exam": exam}

    async def rate(state: ProfilingState) -> dict:
        exam, submission = state["exam"], state["submission"]
        if not any(a.text.strip() for a in submission.answers):
            return {"output": None}  # nothing to rate: every dimension is not observable
        messages = build_messages(
            prompts.profiler_system(exam),
            prompts.answers_content(exam, _answers(submission)),
        )
        return {"output": await structured(model(), ProfilerOutput).ainvoke(messages)}

    async def save(state: ProfilingState) -> dict:
        exam, submission, output = state["exam"], state["submission"], state.get("output")
        policy = exam.profile.profiling
        r = get_repo()
        counts = error_counts(policy, r.get_grade(submission.id))
        ratings = {rt.dimension_id: rt for rt in (output.ratings if output else [])}
        observations = []
        for d in policy.dimensions:
            if output is None:
                obs = DimensionObservation(dimension_id=d.id, rationale="Entrega sin respuestas.")
            else:
                obs = sanitize_rating(ratings.get(d.id), policy, submission.answers)
                obs.dimension_id = d.id
            obs.error_count = counts[d.id]
            observations.append(obs)
        system = prompts.profiler_system(exam)
        profile = SubmissionProfile(
            submission_id=submission.id,
            exam_id=exam.id,
            student_id=submission.student_id,
            exam_title=exam.title,
            sequence=exam.sequence,
            submitted_at=submission.created_at,
            profiling_version=policy.version,
            dimensions_hash=policy.dimensions_hash(),
            observations=observations,
            provenance={
                "profiler_model": model_name(),
                "prompt_sha256": hashlib.sha256(
                    json.dumps(system, ensure_ascii=False).encode()).hexdigest(),
                "llm_called": output is not None,
                "unknown_dimensions": sorted(set(ratings) - {d.id for d in policy.dimensions}),
                "ocr": submission.provenance,
            },
        )
        r.save_submission_profile(profile)
        return {"profile": profile}

    builder = StateGraph(ProfilingState, input_schema=ProfilingInput)
    builder.add_node("load", load)
    builder.add_node("rate", rate)
    builder.add_node("save", save)
    builder.add_edge(START, "load")
    builder.add_edge("load", "rate")
    builder.add_edge("rate", "save")
    builder.add_edge("save", END)
    return builder.compile(name="profiling")


graph = build_profiling_graph()

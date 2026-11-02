"""Grading graph: Submission -> persisted Grade.

    START -> load_submission ── grade_criterion × N  (Send)    (one LLM call per criterion)
          -> collect_scores ──── review_question × M  (Send)    (one LLM call per question)
          -> finalize -> END                                    (build + persist Grade, set status)

Every submission is graded and gets feedback per question. Low confidence or a reviewer
flag only labels the grade `needs_verification` (with the reasons); it never blocks it.

Input:  {"submission_id": str}
Output: state with "grade" (persisted); submission status set to GRADED.
Limit parallel LLM calls with config={"max_concurrency": settings.max_concurrency}.
Setting the status to FAILED on exceptions is left to the caller.
"""

import hashlib
import json
import operator
import re
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from huc.config import Settings, get_settings
from huc.domain.models import (
    Answer,
    Criterion,
    CriterionScore,
    Exam,
    FeedbackAudit,
    Grade,
    Question,
    QuestionGrade,
    ReviewVerdict,
    Submission,
    SubmissionStatus,
)
from huc.domain.profile import CourseProfile
from huc.llm import build_messages, get_chat_model, structured
from huc.prompts import grading as grading_prompts
from huc.prompts import review as review_prompts
from huc.storage.repository import FileRepository, Repository

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class GradingInput(TypedDict):
    submission_id: str


class ScoredCriterion(TypedDict):
    question_id: str
    score: CriterionScore


class GradingState(GradingInput, total=False):
    submission: Submission
    exam: Exam
    # Fan-in channels: parallel nodes append to these lists
    criterion_scores: Annotated[list[ScoredCriterion], operator.add]
    question_grades: Annotated[list[QuestionGrade], operator.add]
    grade: Grade


class CriterionTask(TypedDict):
    """Payload sent to one `grade_criterion` run."""

    question: Question
    criterion: Criterion
    answer: Answer
    profile: CourseProfile


class ReviewTask(TypedDict):
    """Payload sent to one `review_question` run."""

    question: Question
    answer: Answer
    scores: list[CriterionScore]
    profile: CourseProfile


# ---------------------------------------------------------------------------
# Pure helpers (no LLM, no I/O)
# ---------------------------------------------------------------------------


def _clamp(points: float, max_points: float) -> float:
    return min(max(points, 0.0), max_points)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def linear_consistency_warnings(text: str) -> list[str]:
    """Flag conflicting simple equalities for inspection; never decide a grade.

    This deliberately covers only simple integer forms such as `2x = 4` and `x = 4`.
    Equalities in different cases or corrections can be legitimately different, so this
    heuristic is a warning, not a mathematical proof of a student's error.
    """
    pattern = re.compile(r"(?<![\w.])([+-]?\d*)\s*([a-z])\s*=\s*([+-]?\d+)(?!\w|\.\d|\s*/\s*\d)")
    by_variable: dict[str, list[tuple[Fraction, str]]] = {}
    for match in pattern.finditer(text):
        coefficient, variable, constant = match.groups()
        multiplier = -1 if coefficient == "-" else 1 if coefficient in ("", "+") else int(coefficient)
        if multiplier == 0:
            continue
        by_variable.setdefault(variable, []).append((Fraction(int(constant), multiplier), match.group()))
    return [f"potentially conflicting equalities: {items[0][1]} and {other[1]}"
            for items in by_variable.values() for other in items[1:]
            if other[0] != items[0][0]][:3]


def _answers_by_question(exam: Exam, submission: Submission) -> dict[str, Answer]:
    """Preserve duplicated OCR fragments and mark ambiguous assignments for review."""
    answers = {}
    for q in exam.questions:
        fragments = [answer for answer in submission.answers if answer.question_id == q.id]
        if not fragments:
            answers[q.id] = Answer(question_id=q.id, text="", extraction_confidence=0,
                                   reading_issues=["OCR omitted this question"])
        elif len(fragments) == 1:
            answers[q.id] = fragments[0]
        else:
            answers[q.id] = Answer(
                question_id=q.id,
                text="\n\n".join(a.text for a in fragments),
                extraction_confidence=min(a.extraction_confidence for a in fragments),
                source_pages=sorted({page for a in fragments for page in a.source_pages}),
                reading_issues=list(dict.fromkeys(
                    ["OCR returned multiple answers for this question"]
                    + [issue for a in fragments for issue in a.reading_issues]
                )),
            )
    return answers


def sanitize_score(
    score: CriterionScore,
    criterion: Criterion,
    answer: Answer,
    error_types: set[str] = frozenset(),
) -> CriterionScore:
    """Never trust the model blindly: force the criterion id, clamp points, cap confidence,
    drop error types the profile does not define."""
    return score.model_copy(
        update={
            "criterion_id": criterion.id,
            "points": _clamp(score.points, criterion.max_points),
            # A doubtful transcription limits how confident the grade can be.
            "confidence": min(score.confidence, answer.extraction_confidence),
            "error_type": score.error_type if score.error_type in error_types else None,
        }
    )


def apply_verdict(
    question: Question,
    answer: Answer,
    scores: list[CriterionScore],
    verdict: ReviewVerdict,
    error_types: set[str] = frozenset(),
) -> list[CriterionScore]:
    """Replace the scores the reviewer adjusted, enforcing the rubric limits."""
    criteria = {c.id: c for c in question.criteria}
    adjusted = {s.criterion_id: s for s in verdict.adjusted_scores if s.criterion_id in criteria}
    return [
        sanitize_score(adjusted[s.criterion_id], criteria[s.criterion_id], answer, error_types)
        if s.criterion_id in adjusted
        else s
        for s in scores
    ]


def verification_reasons(question_grade: QuestionGrade, threshold: float) -> list[str]:
    """Why a teacher should check this question's grade (empty list: nothing to check)."""
    qid = question_grade.question_id
    reasons = [
        f"{qid}/{s.criterion_id}: confidence {s.confidence:.2f} < {threshold:g}"
        for s in question_grade.scores
        if s.confidence < threshold
    ]
    if question_grade.review and question_grade.review.needs_verification:
        reasons.append(f"{qid}: flagged by the reviewer")
    if question_grade.feedback_audit:
        audit = question_grade.feedback_audit
        if audit.needs_verification or audit.issues:
            reasons.extend(f"{qid}: feedback audit: {issue}" for issue in audit.issues)
            if audit.needs_verification and not audit.issues:
                reasons.append(f"{qid}: feedback flagged by auditor")
        if audit.confidence < threshold:
            reasons.append(f"{qid}: feedback audit confidence {audit.confidence:.2f} < {threshold:g}")
    elif question_grade.review:
        reasons.append(f"{qid}: feedback audit not performed")
    if question_grade.review and not question_grade.review.feedback.strip():
        reasons.append(f"{qid}: feedback is empty")
    if question_grade.review and not question_grade.review.approved and not question_grade.review.adjusted_scores:
        reasons.append(f"{qid}: review did not approve scores and gave no adjustments")
    return reasons


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_grading_graph(
    repo: Repository | None = None,
    llm: BaseChatModel | Any | None = None,
    *,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Build the grading graph.

    `repo=None` -> FileRepository(settings.data_dir); `llm=None` -> lazy Claude models per
    role (grader/reviewer); `settings=None` -> get_settings().
    """

    def cfg() -> Settings:
        return settings or get_settings()

    def get_repo() -> Repository:
        return repo if repo is not None else FileRepository(cfg().data_dir)

    def model_for(role: str) -> Any:
        if llm is not None:
            return llm
        return get_chat_model(getattr(cfg(), f"{role}_model"))

    def grader_name() -> str:
        if llm is None:
            return cfg().grader_model
        return str(getattr(llm, "model", None) or type(llm).__name__)

    # -- nodes ---------------------------------------------------------------

    async def load_submission(state: GradingState) -> dict:
        r = get_repo()
        submission = r.get_submission(state["submission_id"])
        exam = r.get_exam(submission.exam_id)
        submission.status = SubmissionStatus.GRADING
        r.save_submission(submission)
        return {"submission": submission, "exam": exam}

    async def fan_out_criteria(state: GradingState) -> list[Send] | str:
        exam = state["exam"]
        answers = _answers_by_question(exam, state["submission"])
        sends = [
            Send(
                "grade_criterion",
                {"question": q, "criterion": c, "answer": answers[q.id], "profile": exam.profile},
            )
            for q in exam.questions
            for c in q.criteria
        ]
        return sends or "collect_scores"

    async def grade_criterion(task: CriterionTask) -> dict:
        question, criterion, answer = task["question"], task["criterion"], task["answer"]
        profile = task["profile"]
        if not answer.text.strip():
            score = CriterionScore(
                criterion_id=criterion.id,
                points=0,
                justification="Empty answer.",
                confidence=answer.extraction_confidence,
            )
        else:
            messages = build_messages(
                grading_prompts.grader_system(question, criterion, profile),
                [grading_prompts.student_answer_block(answer.text),
                 grading_prompts.reading_context_block(answer)],
            )
            raw = await structured(model_for("grader"), CriterionScore).ainvoke(messages)
            score = sanitize_score(raw, criterion, answer, profile.error_type_ids())
        return {"criterion_scores": [{"question_id": question.id, "score": score}]}

    async def collect_scores(state: GradingState) -> dict:
        """Fan-in barrier: waits for every grading branch before reviews start."""
        return {}

    async def fan_out_reviews(state: GradingState) -> list[Send] | str:
        exam = state["exam"]
        answers = _answers_by_question(exam, state["submission"])
        # Keyed by (question, criterion): if a thread is re-run, the latest score wins.
        latest = {
            (sc["question_id"], sc["score"].criterion_id): sc["score"]
            for sc in state.get("criterion_scores", [])
        }
        sends = [
            Send(
                "review_question",
                {
                    "question": q,
                    "answer": answers[q.id],
                    # Rubric order, independent of the order the parallel branches finished
                    "scores": [latest[(q.id, c.id)] for c in q.criteria if (q.id, c.id) in latest],
                    "profile": exam.profile,
                },
            )
            for q in exam.questions
        ]
        return sends or "finalize"

    async def review_question(task: ReviewTask) -> dict:
        question, answer, scores = task["question"], task["answer"], task["scores"]
        profile = task["profile"]
        verdict: ReviewVerdict | None = None
        audit: FeedbackAudit | None = None
        feedback = "" if answer.text.strip() else (
            "No se pudo confirmar una respuesta en la transcripción; revisa el documento original."
        )
        proposed_scores = list(scores)
        if answer.text.strip() and scores:
            messages = build_messages(
                review_prompts.reviewer_system(question, profile),
                [
                    grading_prompts.student_answer_block(answer.text),
                    grading_prompts.reading_context_block(answer),
                    review_prompts.scores_block(scores),
                ],
            )
            verdict = await structured(model_for("reviewer"), ReviewVerdict).ainvoke(messages)
            scores = apply_verdict(question, answer, scores, verdict, profile.error_type_ids())
            feedback = verdict.feedback
            threshold = profile.grading.verification_threshold
            if threshold is None:
                threshold = cfg().verification_threshold
            if (answer.extraction_confidence < threshold or answer.reading_issues
                    or "[illegible]" in answer.text.lower()):
                feedback = (
                    "La lectura de tu respuesta es incierta; este comentario puede omitir "
                    "partes de tu trabajo y requiere revisar el documento original.\n\n" + feedback
                )
            if cfg().feedback_audit_enabled:
                audit_messages = build_messages(
                    review_prompts.feedback_auditor_system(question, profile),
                    [grading_prompts.student_answer_block(answer.text),
                     grading_prompts.reading_context_block(answer),
                     review_prompts.scores_block(scores),
                     {"type": "text", "text": f"<feedback>\n{feedback}\n</feedback>"}],
                )
                audit = await structured(model_for("feedback_auditor"), FeedbackAudit).ainvoke(
                    audit_messages
                )
        qg = QuestionGrade(
            question_id=question.id,
            scores=scores,
            proposed_scores=proposed_scores,
            review=verdict,
            feedback_audit=audit,
            feedback=feedback,
            graded_by=grader_name(),
        )
        return {"question_grades": [qg]}

    async def finalize(state: GradingState) -> dict:
        exam, submission = state["exam"], state["submission"]
        threshold = exam.profile.grading.verification_threshold
        if threshold is None:
            threshold = cfg().verification_threshold
        by_question = {qg.question_id: qg for qg in state.get("question_grades", [])}
        questions = [by_question[q.id] for q in exam.questions if q.id in by_question]
        reasons = [reason for qg in questions for reason in verification_reasons(qg, threshold)]
        answers = _answers_by_question(exam, submission)
        reasons.extend(submission.reading_issues)
        for qg in questions:
            answer = answers[qg.question_id]
            question = exam.question(qg.question_id)
            criteria = {criterion.id: criterion for criterion in question.criteria}
            reasons.extend(f"{qg.question_id}: {issue}" for issue in answer.reading_issues)
            reasons.extend(f"{qg.question_id}: {issue}"
                           for issue in linear_consistency_warnings(answer.text))
            if "[illegible]" in answer.text.lower():
                reasons.append(f"{qg.question_id}: transcription contains illegible material")
            if not answer.text.strip():
                reasons.append(f"{qg.question_id}: blank or missing transcription requires checking")
            for score in [*qg.proposed_scores, *qg.scores]:
                criterion = criteria.get(score.criterion_id)
                if criterion and criterion.levels:
                    try:
                        allowed = {float(value) for value in criterion.levels}
                    except ValueError:
                        allowed = set()
                    if score.points not in allowed:
                        reasons.append(
                            f"{qg.question_id}/{score.criterion_id}: "
                            f"score {score.points:g} is not a defined rubric level"
                        )
                if answer.text.strip() and not score.evidence:
                    reasons.append(f"{qg.question_id}/{score.criterion_id}: no evidence quoted")
                if any(not quote.strip() or quote not in answer.text for quote in score.evidence):
                    reasons.append(f"{qg.question_id}/{score.criterion_id}: evidence not in transcription")
        reasons = list(dict.fromkeys(reasons))
        for qg in questions:
            qg.feedback_review_reasons = [
                reason for reason in reasons
                if reason.startswith((f"{qg.question_id}/", f"{qg.question_id}:"))
            ]
            qg.feedback_needs_review = bool(qg.feedback_review_reasons or submission.reading_issues)
        prompt_content = {
            q.id: {
                "grader": {c.id: grading_prompts.grader_system(q, c, exam.profile) for c in q.criteria},
                "reviewer": review_prompts.reviewer_system(q, exam.profile),
                "feedback_auditor": review_prompts.feedback_auditor_system(q, exam.profile),
            } for q in exam.questions
        }
        prompt_hash = hashlib.sha256(json.dumps(prompt_content, sort_keys=True,
                                               ensure_ascii=False).encode()).hexdigest()
        grade = Grade(
            submission_id=submission.id,
            exam_id=exam.id,
            rubric_version=exam.rubric_version,
            questions=questions,
            needs_verification=bool(reasons),
            verification_reasons=reasons,
            exam_snapshot=exam.model_copy(deep=True),
            submission_snapshot=submission.model_copy(deep=True),
            provenance={"grader_model": grader_name(), "reviewer_model":
                        cfg().reviewer_model if llm is None else grader_name(),
                        "ocr": submission.provenance, "prompt_sha256": prompt_hash,
                        "feedback_audit_enabled": cfg().feedback_audit_enabled,
                        "feedback_auditor_model": cfg().feedback_auditor_model if llm is None else grader_name(),
                        "prompts": prompt_content, "verification_threshold": threshold,
                        "llm_max_tokens": cfg().llm_max_tokens,
                        "llm_thinking": cfg().llm_thinking,
                        "packages": {name: _package_version(name) for name in
                                     ("huc", "langgraph", "langchain-anthropic", "anthropic")}},
        )
        r = get_repo()
        r.save_grade(grade)
        submission.status = SubmissionStatus.GRADED
        r.save_submission(submission)
        return {"grade": grade, "submission": submission}

    # -- wiring --------------------------------------------------------------

    builder = StateGraph(GradingState, input_schema=GradingInput)
    builder.add_node("load_submission", load_submission)
    builder.add_node("grade_criterion", grade_criterion)
    builder.add_node("collect_scores", collect_scores)
    builder.add_node("review_question", review_question)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "load_submission")
    builder.add_conditional_edges(
        "load_submission", fan_out_criteria, ["grade_criterion", "collect_scores"]
    )
    builder.add_edge("grade_criterion", "collect_scores")
    builder.add_conditional_edges(
        "collect_scores", fan_out_reviews, ["review_question", "finalize"]
    )
    builder.add_edge("review_question", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(name="grading")


graph = build_grading_graph()

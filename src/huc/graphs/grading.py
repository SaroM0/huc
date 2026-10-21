"""Grading graph: Submission -> persisted Grade.

    START -> load_submission ── grade_criterion × N  (Send)    (one LLM call per criterion)
          -> collect_scores ──── review_question × M  (Send)    (one LLM call per question)
          -> finalize -> END                                    (build + persist Grade, set status)

Input:  {"submission_id": str}
Output: state with "grade" (persisted); submission status set to GRADED or NEEDS_HUMAN_REVIEW.
Limit parallel LLM calls with config={"max_concurrency": settings.max_concurrency}.
Setting the status to FAILED on exceptions is left to the caller.
"""

import operator
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
    Grade,
    Question,
    QuestionGrade,
    ReviewVerdict,
    Submission,
    SubmissionStatus,
)
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


class ReviewTask(TypedDict):
    """Payload sent to one `review_question` run."""

    question: Question
    answer: Answer
    scores: list[CriterionScore]


# ---------------------------------------------------------------------------
# Pure helpers (no LLM, no I/O)
# ---------------------------------------------------------------------------


def _clamp(points: float, max_points: float) -> float:
    return min(max(points, 0.0), max_points)


def _answers_by_question(exam: Exam, submission: Submission) -> dict[str, Answer]:
    """Every exam question gets an Answer; missing ones become empty answers."""
    given = {a.question_id: a for a in submission.answers}
    return {q.id: given.get(q.id) or Answer(question_id=q.id, text="") for q in exam.questions}


def sanitize_score(
    score: CriterionScore, criterion: Criterion, answer: Answer
) -> CriterionScore:
    """Never trust the model blindly: force the criterion id, clamp points, cap confidence."""
    return score.model_copy(
        update={
            "criterion_id": criterion.id,
            "points": _clamp(score.points, criterion.max_points),
            # A doubtful transcription limits how confident the grade can be.
            "confidence": min(score.confidence, answer.extraction_confidence),
        }
    )


def apply_verdict(
    question: Question, answer: Answer, scores: list[CriterionScore], verdict: ReviewVerdict
) -> list[CriterionScore]:
    """Replace the scores the reviewer adjusted, enforcing the rubric limits."""
    criteria = {c.id: c for c in question.criteria}
    adjusted = {s.criterion_id: s for s in verdict.adjusted_scores if s.criterion_id in criteria}
    return [
        sanitize_score(adjusted[s.criterion_id], criteria[s.criterion_id], answer)
        if s.criterion_id in adjusted
        else s
        for s in scores
    ]


def needs_human(question_grade: QuestionGrade, threshold: float) -> bool:
    low_confidence = any(s.confidence < threshold for s in question_grade.scores)
    return low_confidence or bool(question_grade.review and question_grade.review.needs_human)


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

    def load_submission(state: GradingState) -> dict:
        r = get_repo()
        submission = r.get_submission(state["submission_id"])
        exam = r.get_exam(submission.exam_id)
        submission.status = SubmissionStatus.GRADING
        r.save_submission(submission)
        return {"submission": submission, "exam": exam}

    def fan_out_criteria(state: GradingState) -> list[Send] | str:
        exam = state["exam"]
        answers = _answers_by_question(exam, state["submission"])
        sends = [
            Send("grade_criterion", {"question": q, "criterion": c, "answer": answers[q.id]})
            for q in exam.questions
            for c in q.criteria
        ]
        return sends or "collect_scores"

    async def grade_criterion(task: CriterionTask) -> dict:
        question, criterion, answer = task["question"], task["criterion"], task["answer"]
        if not answer.text.strip():
            score = CriterionScore(
                criterion_id=criterion.id,
                points=0,
                justification="Empty answer.",
                confidence=answer.extraction_confidence,
            )
        else:
            messages = build_messages(
                grading_prompts.grader_system(question, criterion),
                [grading_prompts.student_answer_block(answer.text)],
            )
            raw = await structured(model_for("grader"), CriterionScore).ainvoke(messages)
            score = sanitize_score(raw, criterion, answer)
        return {"criterion_scores": [{"question_id": question.id, "score": score}]}

    def collect_scores(state: GradingState) -> dict:
        """Fan-in barrier: waits for every grading branch before reviews start."""
        return {}

    def fan_out_reviews(state: GradingState) -> list[Send] | str:
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
                },
            )
            for q in exam.questions
        ]
        return sends or "finalize"

    async def review_question(task: ReviewTask) -> dict:
        question, answer, scores = task["question"], task["answer"], task["scores"]
        verdict: ReviewVerdict | None = None
        if answer.text.strip() and scores:
            messages = build_messages(
                review_prompts.reviewer_system(question),
                [
                    grading_prompts.student_answer_block(answer.text),
                    review_prompts.scores_block(scores),
                ],
            )
            verdict = await structured(model_for("reviewer"), ReviewVerdict).ainvoke(messages)
            scores = apply_verdict(question, answer, scores, verdict)
        qg = QuestionGrade(
            question_id=question.id, scores=scores, review=verdict, graded_by=grader_name()
        )
        return {"question_grades": [qg]}

    def finalize(state: GradingState) -> dict:
        exam, submission = state["exam"], state["submission"]
        threshold = cfg().human_review_threshold
        by_question = {qg.question_id: qg for qg in state.get("question_grades", [])}
        questions = [by_question[q.id] for q in exam.questions if q.id in by_question]
        grade = Grade(
            submission_id=submission.id,
            exam_id=exam.id,
            rubric_version=exam.rubric_version,
            questions=questions,
            needs_human_review=any(needs_human(qg, threshold) for qg in questions),
        )
        r = get_repo()
        r.save_grade(grade)
        submission.status = (
            SubmissionStatus.NEEDS_HUMAN_REVIEW
            if grade.needs_human_review
            else SubmissionStatus.GRADED
        )
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

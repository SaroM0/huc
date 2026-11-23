"""Student profile: dimensions in the course profile, the blind profiler graph, the
deterministic baseline comparison, and the service/CLI/API around them."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from conftest import EXAMPLES, FakeLLM, FakeOCR
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from huc import cli
from huc.api.app import create_app
from huc.config import Settings
from huc.domain.models import (
    Answer,
    CriterionScore,
    ExamSequence,
    Grade,
    QuestionGrade,
    SourceKind,
    Submission,
)
from huc.domain.profile import CourseProfile, ProfilingPolicy
from huc.domain.student_profile import (
    DimensionObservation,
    DimensionRating,
    ProfilerOutput,
    SubmissionProfile,
    TrendStatus,
    compute_profile,
)
from huc.graphs.profiling import sanitize_rating
from huc.services import HucService

PROFILES = Path(__file__).parent.parent / "profiles"
DIMENSIONS = ["justificacion", "estructura_logica", "dominio_conceptual", "generalidad",
              "precision_simbolica", "comunicacion", "procedimiento"]
ANSWER = "Sea x en R. Por densidad existe e en E con |e - x| < 1/4, por lo tanto E es infinito."
QUOTE = "por lo tanto E es infinito"


@pytest.fixture
def svc(repo, tmp_path, monkeypatch) -> HucService:
    settings = Settings(data_dir=tmp_path / "data", profiles_dir=PROFILES)
    service = HucService(repo, llm=FakeLLM(), settings=settings, ocr=FakeOCR())
    monkeypatch.setattr(cli, "get_service", lambda: service)
    return service


def load_exam(svc, exam_id: str, week: int, module: str = "M1"):
    """The S14 M1 exam (profile mat0004) under another id and position in the course."""
    data = yaml.safe_load((EXAMPLES / "taller_s14_m1.yaml").read_text("utf-8"))
    data.update(id=exam_id, title=f"Taller S{week} {module}",
                sequence={"week": week, "module": module})
    return svc.save_exam_yaml(yaml.safe_dump(data, allow_unicode=True))


def add_submission(svc, exam, student="EST-001", text=ANSWER, created_at=None):
    answers = [Answer(question_id=q.id, text=text if i == 0 else "", extraction_confidence=0.9)
               for i, q in enumerate(exam.questions)]
    submission = Submission(exam_id=exam.id, student_id=student, source_kind=SourceKind.PDF,
                            answers=answers, created_at=created_at or datetime.now(UTC))
    svc.repo.save_submission(submission)
    return submission


def ratings(levels: dict[str, int | None], confidence=0.9, quote=QUOTE) -> ProfilerOutput:
    return ProfilerOutput(ratings=[
        DimensionRating(dimension_id=d, observable=levels.get(d) is not None,
                        level=levels.get(d), evidence=[quote] if levels.get(d) else [],
                        rationale="ok", confidence=confidence)
        for d in DIMENSIONS
    ])


def run(svc, *levels_by_run):
    """Ratings the fake profiler returns, one call after another."""
    outputs = iter(levels_by_run)
    original = svc.llm._respond

    def respond(schema):
        return next(outputs) if schema is ProfilerOutput else original(schema)

    svc.llm._respond = respond


# -- configuration ---------------------------------------------------------------


def test_course_profile_defines_the_seven_dimensions(svc):
    exam = load_exam(svc, "s14-m1", 14)
    policy = exam.profile.profiling
    assert [d.id for d in policy.dimensions] == DIMENSIONS
    assert policy.baseline == "first_submission" and policy.level_values() == [1, 2, 3, 4]
    # Every dimension has an anchor per level and a boundary with the others.
    assert all(set(d.anchors) == {"1", "2", "3", "4"} and d.not_this for d in policy.dimensions)
    assert exam.sequence == ExamSequence(week=14, module="M1")


def test_invalid_profiling_config_is_rejected():
    with pytest.raises(ValueError, match="unknown error types"):
        CourseProfile.model_validate({"profiling": {"dimensions": [
            {"id": "a", "name": "A", "description": "x", "error_types": ["nope"]}]}})
    with pytest.raises(ValueError, match="anchors outside"):
        ProfilingPolicy.model_validate({"dimensions": [
            {"id": "a", "name": "A", "description": "x", "anchors": {"7": "?"}}]})
    with pytest.raises(ValueError, match="unique"):
        ProfilingPolicy.model_validate({"dimensions": [
            {"id": "a", "name": "A", "description": "x"},
            {"id": "a", "name": "B", "description": "y"}]})


def test_sequence_orders_days_and_modules_within_a_week():
    order = ["L5", "L6", "M1", "M3", "W5"]
    keys = [ExamSequence(week=3, module=m).key for m in reversed(order)]
    assert [k[2] for k in sorted(keys)] == order
    assert ExamSequence(week=2, module="W6").key < ExamSequence(week=3, module="L5").key


def test_dimension_hash_ignores_error_types_but_not_anchors():
    base = {"dimensions": [{"id": "a", "name": "A", "description": "x", "anchors": {"1": "p"}}]}
    policy = ProfilingPolicy.model_validate(base)
    linked = ProfilingPolicy.model_validate(
        {"dimensions": [{**base["dimensions"][0], "error_types": ["calculo"]}]})
    reworded = ProfilingPolicy.model_validate(
        {"dimensions": [{**base["dimensions"][0], "anchors": {"1": "q"}}]})
    assert policy.dimensions_hash() == linked.dimensions_hash()
    assert policy.dimensions_hash() != reworded.dimensions_hash()


# -- profiler output checks ------------------------------------------------------


def test_sanitize_snaps_level_checks_evidence_and_caps_confidence():
    policy = ProfilingPolicy()
    answers = [Answer(question_id="q", text=ANSWER, extraction_confidence=0.6)]
    obs = sanitize_rating(DimensionRating(dimension_id="d", observable=True, level=9,
                                          evidence=["inventado"], confidence=0.95),
                          policy, answers)
    assert obs.level == 4 and obs.confidence == 0.6
    assert any("outside the scale" in i for i in obs.issues)
    assert "evidence not in transcription" in obs.issues

    hidden = sanitize_rating(DimensionRating(dimension_id="d", observable=False, level=2,
                                             confidence=0.9), policy, answers)
    assert hidden.level is None and not hidden.issues


# -- the graph -------------------------------------------------------------------


async def test_profiler_is_blind_to_grade_and_history(svc):
    exam = load_exam(svc, "s14-m1", 14)
    submission = add_submission(svc, exam)
    # A grade with recognisable content and an earlier profile: neither may reach the model.
    svc.repo.save_grade(Grade(submission_id=submission.id, exam_id=exam.id, rubric_version=2,
                              questions=[QuestionGrade(
                                  question_id="p1a", graded_by="x", feedback="FEEDBACK-SECRETO",
                                  scores=[CriterionScore(criterion_id="construction", points=1.5,
                                                         justification="JUSTIF-SECRETA",
                                                         confidence=0.9,
                                                         error_type="justificacion")])]))
    run(svc, ratings({"justificacion": 2}), ratings({"justificacion": 3}))
    await svc.profile_submission(submission.id)
    run_ = await svc.profile_submission(submission.id)

    calls = [c for c in svc.llm.calls if c["schema"] == "ProfilerOutput"]
    assert len(calls) == 2
    prompt = str([m.content for m in calls[-1]["messages"]])
    for leaked in ("FEEDBACK-SECRETO", "JUSTIF-SECRETA", "1.5"):
        assert leaked not in prompt
    assert ANSWER in prompt and "estructura_logica" in prompt
    # The grade is still used in code: error types counted per dimension.
    assert run_.observation("justificacion").error_count == 1
    assert run_.observation("comunicacion").error_count is None  # no error types linked
    assert run_.observation("procedimiento").error_count == 0
    # Both runs are kept; the profile uses the newest.
    assert len(svc.repo.list_submission_profiles("EST-001")) == 2
    assert svc.student_profile("EST-001").dimensions[0].current_level == 3


async def test_blank_submission_is_not_sent_to_the_model(svc):
    exam = load_exam(svc, "s14-m1", 14)
    submission = add_submission(svc, exam, text="")
    run_ = await svc.profile_submission(submission.id)
    assert svc.llm.count("ProfilerOutput") == 0
    assert all(o.level is None for o in run_.observations)
    assert run_.provenance["llm_called"] is False


async def test_missing_rating_becomes_not_observable_with_issue(svc):
    exam = load_exam(svc, "s14-m1", 14)
    submission = add_submission(svc, exam)
    output = ratings({"justificacion": 3})
    output.ratings = output.ratings[:1] + [DimensionRating(
        dimension_id="inventada", observable=True, level=2, confidence=0.9)]
    run(svc, output)
    run_ = await svc.profile_submission(submission.id)
    assert [o.dimension_id for o in run_.observations] == DIMENSIONS
    assert run_.observation("comunicacion").issues == ["no rating returned by the model"]
    assert run_.provenance["unknown_dimensions"] == ["inventada"]


async def test_exam_without_dimensions_is_rejected(svc, exam):
    with pytest.raises(ValueError, match="no profiling dimensions"):
        await svc.profile_exam(exam.id)


# -- baseline and trend ----------------------------------------------------------


async def test_baseline_is_the_first_submission_in_course_order(svc):
    s14 = load_exam(svc, "s14-m1", 14)
    s03 = load_exam(svc, "s03-m1", 3)
    s08 = load_exam(svc, "s08-l5", 8, "L5")
    # Profiled out of course order: the sequence decides the baseline, not run order.
    run(svc,
        ratings({"justificacion": 3, "comunicacion": 2, "procedimiento": 3}),  # S14
        ratings({"justificacion": 1, "comunicacion": 2}),  # S03: procedimiento not observable
        ratings({"justificacion": 2, "comunicacion": 3, "procedimiento": 4}))  # S08
    for exam in (s14, s03, s08):
        await svc.profile_submission(add_submission(svc, exam).id)

    prof = svc.student_profile("EST-001")
    assert prof.first_submission == "S03 M1" and prof.submissions == 3
    trends = {d.dimension_id: d for d in prof.dimensions}
    assert [p.label for p in trends["justificacion"].series] == ["S03 M1", "S08 L5", "S14 M1"]

    just = trends["justificacion"]
    assert (just.baseline_level, just.current_level, just.delta) == (1, 3, 2)
    assert just.status is TrendStatus.IMPROVED and not just.baseline_deferred
    assert [p.delta for p in just.series] == [0, 1, 2]

    assert trends["comunicacion"].status is TrendStatus.STABLE  # 2 -> 2 (3 in between)

    proc = trends["procedimiento"]  # not observable in the first submission
    assert proc.baseline_deferred and proc.baseline_label == "S08 L5"
    assert (proc.delta, proc.status) == (-1, TrendStatus.REGRESSED)

    assert trends["generalidad"].status is TrendStatus.NO_BASELINE


def make_run(label_week, levels, *, hash_="h", confidence=0.9, created=0, issues=()):
    return SubmissionProfile(
        submission_id=f"sub-{label_week}", exam_id=f"e{label_week}", student_id="EST",
        sequence=ExamSequence(week=label_week), submitted_at=datetime(2026, 3, 1, tzinfo=UTC),
        profiling_version=1, dimensions_hash=hash_,
        created_at=datetime(2026, 6, 1, tzinfo=UTC) + timedelta(minutes=created),
        observations=[DimensionObservation(dimension_id="d", level=lv, confidence=confidence,
                                           issues=list(issues)) for lv in [levels]],
    )


def profile_of(runs, min_change=1):
    return compute_profile("EST", runs, [("d", "D")], min_change=min_change,
                           confidence_threshold=0.7)


def test_trend_rules():
    only = profile_of([make_run(1, 2)]).dimensions[0]
    assert only.status is TrendStatus.BASELINE_ONLY and only.delta is None

    # The current level is the latest observable one, even if the last submission hid it.
    hidden_last = profile_of([make_run(1, 1), make_run(2, 3), make_run(3, None)]).dimensions[0]
    assert (hidden_last.current_label, hidden_last.status) == ("S02", TrendStatus.IMPROVED)

    small = profile_of([make_run(1, 2), make_run(2, 3)], min_change=2).dimensions[0]
    assert small.status is TrendStatus.STABLE

    shaky = profile_of([make_run(1, 2), make_run(2, 3, confidence=0.4)]).dimensions[0]
    assert shaky.low_confidence and shaky.status is TrendStatus.IMPROVED
    flagged = profile_of([make_run(1, 2), make_run(2, 3, issues=["no evidence quoted"])])
    assert flagged.dimensions[0].low_confidence


def test_runs_with_other_dimensions_are_not_compared():
    prof = profile_of([make_run(1, 1, hash_="old"), make_run(2, 2, hash_="new", created=1),
                       make_run(3, 4, hash_="new", created=2)])
    assert prof.excluded_runs == 1 and prof.first_submission == "S02"
    assert prof.dimensions[0].delta == 2
    assert any("other dimensions" in w for w in prof.warnings)


# -- service, CLI, API -----------------------------------------------------------


async def test_profile_exam_skips_rated_submissions_unless_forced(svc):
    exam = load_exam(svc, "s14-m1", 14)
    add_submission(svc, exam, "EST-001")
    add_submission(svc, exam, "EST-002")
    run(svc, *[ratings({"justificacion": 2})] * 4)
    assert len(await svc.profile_exam(exam.id)) == 2
    assert await svc.profile_exam(exam.id) == []
    assert len(await svc.profile_exam(exam.id, force=True)) == 2
    assert len(svc.repo.list_submission_profiles()) == 4
    assert svc.profiled_students() == ["EST-001", "EST-002"]


def test_cli_and_csv(svc, tmp_path):
    s03, s14 = load_exam(svc, "s03-m1", 3), load_exam(svc, "s14-m1", 14)
    add_submission(svc, s03)
    add_submission(svc, s14)
    run(svc, ratings({"justificacion": 1}), ratings({"justificacion": 3}))
    runner = CliRunner()
    assert "Profiled 1" in runner.invoke(cli.app, ["profile-run", "s03-m1"]).output
    assert "Profiled 1" in runner.invoke(cli.app, ["profile-run", "s14-m1"]).output

    out = runner.invoke(cli.app, ["profile", "EST-001"]).output
    assert "baseline: S03 M1" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("justificacion"))
    assert "+2" in line and "mejora" in line
    assert "sin_linea_base" in out
    assert "EST-001" in runner.invoke(cli.app, ["profile"]).output
    assert runner.invoke(cli.app, ["profile", "NADIE"]).exit_code == 1

    csv_path = tmp_path / "p.csv"
    runner.invoke(cli.app, ["profile-export", "--out", str(csv_path)])
    rows = csv_path.read_text("utf-8").splitlines()
    assert len(rows) == 1 + 2 * len(DIMENSIONS)
    assert "EST-001,justificacion" in rows[2] and ",3,1,S03 M1,False,2,mejora," in rows[2]


def test_api_profile_endpoints(svc):
    exam = load_exam(svc, "s14-m1", 14)
    add_submission(svc, exam)
    run(svc, ratings({"justificacion": 2}))
    client = TestClient(create_app(svc))
    assert client.post(f"/exams/{exam.id}/profile").status_code == 202
    assert client.get("/students").json() == ["EST-001"]
    body = client.get("/students/EST-001/profile").json()
    assert body["first_submission"] == "S14 M1"
    assert body["dimensions"][0]["status"] == "solo_linea_base"
    assert client.get("/students/NADIE/profile").status_code == 404
    assert "justificacion" in client.get("/profiles/export.csv").text

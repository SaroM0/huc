import csv
import io

import pytest

from huc.domain.models import CriterionScore, Grade, QuestionGrade, ReviewVerdict, Submission
from huc.graphs.grading import (
    _answers_by_question,
    linear_consistency_warnings,
    verification_reasons,
)
from huc.services import HucService


def test_missing_ocr_answer_has_zero_confidence_and_reason(exam):
    from huc.domain.models import SourceKind

    submission = Submission(exam_id=exam.id, student_id="synthetic", source_kind=SourceKind.IMAGE)
    answer = _answers_by_question(exam, submission)[exam.questions[0].id]
    assert answer.text == ""
    assert answer.extraction_confidence == 0
    assert answer.reading_issues == ["OCR omitted this question"]


def test_duplicate_ocr_fragments_are_preserved_and_flagged(exam):
    from huc.domain.models import Answer, SourceKind

    qid = exam.questions[0].id
    submission = Submission(exam_id=exam.id, student_id="synthetic", source_kind=SourceKind.IMAGE,
                            answers=[Answer(question_id=qid, text="primer paso", source_pages=[1]),
                                     Answer(question_id=qid, text="conclusión", source_pages=[2],
                                            extraction_confidence=0.5)])
    answer = _answers_by_question(exam, submission)[qid]
    assert answer.text == "primer paso\n\nconclusión"
    assert answer.source_pages == [1, 2]
    assert answer.extraction_confidence == 0.5
    assert any("multiple answers" in issue for issue in answer.reading_issues)


def test_empty_feedback_and_unresolved_review_are_flagged():
    score = CriterionScore(criterion_id="c", points=1, justification="x", confidence=1)
    review = ReviewVerdict(approved=False, feedback=" ")
    qg = QuestionGrade(question_id="q", scores=[score], review=review, graded_by="test")
    reasons = verification_reasons(qg, 0.7)
    assert any("feedback is empty" in reason for reason in reasons)
    assert any("did not approve" in reason for reason in reasons)


def test_linear_consistency_warning_is_narrow_and_non_grading():
    assert linear_consistency_warnings("2x = 4; por tanto x = 4")
    assert linear_consistency_warnings("2x = 4; por tanto x = 4.")
    assert not linear_consistency_warnings("2x = 8; por tanto x = 4")
    assert not linear_consistency_warnings("x^2 = 4")
    assert not linear_consistency_warnings("2x = 1; x = 1/2")
    assert not linear_consistency_warnings("2x = 1; x = 1 / 2")


def test_grade_history_preserves_prior_runs(repo, exam):
    first = Grade(submission_id="s", exam_id=exam.id, rubric_version=1, questions=[],
                  exam_snapshot=exam)
    second = Grade(submission_id="s", exam_id=exam.id, rubric_version=2, questions=[])
    repo.save_grade(first)
    repo.save_grade(second)
    assert repo.get_grade("s").id == second.id
    assert {g.id for g in repo.list_grade_history("s")} == {first.id, second.id}
    assert next(g for g in repo.list_grade_history("s") if g.id == first.id).exam_snapshot == exam
    with pytest.raises(FileExistsError):
        repo.save_grade(first)


def test_reject_duplicate_criteria(repo):
    text = """id: e
title: Test
questions:
  - id: q
    number: '1'
    type: short_answer
    prompt: test
    criteria:
      - {id: c, description: one, max_points: 1}
      - {id: c, description: two, max_points: 1}
"""
    with pytest.raises(ValueError, match="unique ids"):
        HucService(repo).save_exam_yaml(text)


def test_export_quarantines_flagged_and_legacy_feedback(repo, exam):
    from huc.domain.models import FeedbackAudit, SourceKind

    submission = Submission(exam_id=exam.id, student_id="synthetic", source_kind=SourceKind.IMAGE)
    repo.save_submission(submission)
    q1, q2 = exam.questions
    grade = Grade(submission_id=submission.id, exam_id=exam.id, rubric_version=1, questions=[
        QuestionGrade(question_id=q1.id, scores=[], graded_by="fake", feedback="bad feedback",
                      feedback_needs_review=True,
                      feedback_audit=FeedbackAudit(needs_verification=True, confidence=1)),
        QuestionGrade(question_id=q2.id, scores=[], graded_by="old", feedback="legacy draft"),
    ])
    repo.save_grade(grade)
    service = HucService(repo)
    row = next(csv.DictReader(io.StringIO(service.export_csv(exam.id))))
    assert row["q1_feedback"] == row["q2_feedback"] == ""
    assert row["q1_feedback_needs_review"] == row["q2_feedback_needs_review"] == "True"
    raw = next(csv.DictReader(io.StringIO(service.export_csv(exam.id, include_flagged_feedback=True))))
    assert raw["q1_feedback"] == "bad feedback"
    assert raw["q2_feedback"] == "legacy draft"


def test_export_quarantines_old_grade_level_flags(repo, exam):
    from huc.domain.models import FeedbackAudit, SourceKind

    submission = Submission(exam_id=exam.id, student_id="synthetic", source_kind=SourceKind.IMAGE)
    repo.save_submission(submission)
    qid = exam.questions[0].id
    grade = Grade(submission_id=submission.id, exam_id=exam.id, rubric_version=1,
                  needs_verification=True, verification_reasons=["old warning"], questions=[
        QuestionGrade(question_id=qid, scores=[], graded_by="fake", feedback="draft",
                      feedback_audit=FeedbackAudit(needs_verification=False, confidence=1)),
    ])
    repo.save_grade(grade)
    row = next(csv.DictReader(io.StringIO(HucService(repo).export_csv(exam.id))))
    assert row["q1_feedback"] == ""
    assert row["q1_feedback_needs_review"] == "True"


def test_report_rejects_mixed_rubric_snapshots_and_export_uses_original_max(repo, exam):
    from huc.domain.models import SourceKind

    submission = Submission(exam_id=exam.id, student_id="synthetic", source_kind=SourceKind.IMAGE)
    repo.save_submission(submission)
    snapshot = exam.model_copy(deep=True)
    snapshot.questions[0].criteria[0].max_points *= 2
    grade = Grade(submission_id=submission.id, exam_id=exam.id, rubric_version=1,
                  exam_snapshot=snapshot, questions=[QuestionGrade(
                      question_id=exam.questions[0].id,
                      scores=[CriterionScore(criterion_id="c", points=2,
                                             justification="", confidence=1)], graded_by="fake")])
    repo.save_grade(grade)
    service = HucService(repo)
    with pytest.raises(ValueError, match="different exam or rubric snapshots"):
        service.report(exam.id)
    row = next(csv.DictReader(io.StringIO(service.export_csv(exam.id))))
    expected = round(200 / sum(q.max_points for q in snapshot.questions), 1)
    assert float(row["percent"]) == expected
    assert row["rubric_snapshot_mismatch"] == "True"


def test_report_rejects_legacy_grade_without_snapshot(repo, exam):
    from huc.domain.models import SourceKind

    submission = Submission(exam_id=exam.id, student_id="synthetic", source_kind=SourceKind.IMAGE)
    repo.save_submission(submission)
    repo.save_grade(Grade(submission_id=submission.id, exam_id=exam.id, rubric_version=1,
                          questions=[]))
    with pytest.raises(ValueError, match="without an exam snapshot"):
        HucService(repo).report(exam.id)


def test_nonfinite_scores_and_out_of_range_reading_confidence_are_rejected():
    from pydantic import ValidationError

    from huc.domain.models import Answer, Criterion

    with pytest.raises(ValidationError):
        Criterion(id="c", description="invalid", max_points=float("nan"))
    with pytest.raises(ValidationError):
        CriterionScore(criterion_id="c", points=float("inf"), justification="x", confidence=1)
    with pytest.raises(ValidationError):
        Answer(question_id="q", text="x", extraction_confidence=2)

import pytest

from huc.domain.models import Grade, SourceKind, Submission


def test_roundtrip(repo, exam):
    assert repo.get_exam(exam.id) == exam
    assert [e.id for e in repo.list_exams()] == [exam.id]

    sub = Submission(exam_id=exam.id, student_id="A1", source_kind=SourceKind.LMS_JSON)
    other = Submission(exam_id="other", student_id="A2", source_kind=SourceKind.PDF)
    repo.save_submission(sub)
    repo.save_submission(other)
    assert repo.get_submission(sub.id) == sub
    assert [s.id for s in repo.list_submissions(exam.id)] == [sub.id]

    assert repo.get_grade(sub.id) is None
    grade = Grade(submission_id=sub.id, exam_id=exam.id, rubric_version=1, questions=[])
    repo.save_grade(grade)
    assert repo.get_grade(sub.id) == grade


def test_missing_entities_raise(repo):
    with pytest.raises(FileNotFoundError):
        repo.get_exam("nope")
    with pytest.raises(FileNotFoundError):
        repo.get_submission("nope")

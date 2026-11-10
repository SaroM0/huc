from concurrent.futures import ThreadPoolExecutor

import pytest
from httpx import ASGITransport, AsyncClient

from huc.api.app import create_app
from huc.domain.models import Exam
from huc.domain.workshop import WorkshopEntry
from huc.services import HucService
from huc.storage.repository import FileRepository


@pytest.fixture
def service(tmp_path):
    repo = FileRepository(tmp_path)
    repo.save_exam(Exam(id="exam", title="Workshop", questions=[]))
    return HucService(repo)


def test_complete_process_survives_reload(service):
    session = service.create_workshop("exam", "student")
    entries = [
        {"stage": "initial_solution", "text": "Initial proof"},
        {"stage": "initial_diagnosis", "diagnosis": {"solo": "multistructural",
                                                   "toulmin": {"warrant": "missing"}}},
        {"stage": "socratic_response", "questions": ["What justifies this step?"]},
        {"stage": "student_question", "text": "Does the definition apply?"},
        {"stage": "socratic_response", "questions": ["What are its hypotheses?"]},
        {"stage": "reflection", "text": "I omitted a hypothesis."},
        {"stage": "final_solution", "text": "Revised proof"},
        {"stage": "final_diagnosis", "diagnosis": {"solo": "relational"}},
    ]
    for entry in entries:
        result = service.record_workshop_event(session.id, WorkshopEntry(actor="researcher", **entry))
    reloaded = FileRepository(service.repo._root).get_workshop(session.id)
    assert reloaded == result
    assert [e.sequence for e in reloaded.events] == list(range(1, 9))
    assert reloaded.events[0].text == "Initial proof"
    with pytest.raises(ValueError, match="already recorded"):
        service.record_workshop_event(session.id, WorkshopEntry(actor="student", **entries[0]))


def test_concurrent_notes_are_not_lost(service):
    session = service.create_workshop("exam", "student")
    def append(i):
        service.record_workshop_event(session.id, WorkshopEntry(
            actor="teacher", stage="teacher_note", text=str(i)))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append, range(12)))
    assert len(service.repo.get_workshop(session.id).events) == 12


async def test_api_validates_order_and_keeps_snapshot(service):
    client = AsyncClient(transport=ASGITransport(app=create_app(service)), base_url="http://test")
    response = await client.post("/exams/exam/workshops", json={"student_id": "student"})
    assert response.status_code == 201
    sid = response.json()["id"]
    url = f"/workshops/{sid}/events"
    assert (await client.post(url, json={"stage": "final_solution", "actor": "student",
                                        "text": "proof"})).status_code == 422
    assert (await client.post(url, json={"stage": "initial_solution", "actor": "student",
                                        "text": "first"})).status_code == 201
    service.repo.save_exam(Exam(id="exam", title="Changed", questions=[]))
    result = (await client.get(f"/workshops/{sid}")).json()
    assert result["exam_snapshot"]["title"] == "Workshop"
    assert result["events"][0]["text"] == "first"
    assert (await client.get("/workshops/missing")).status_code == 404
    await client.aclose()


def test_snapshot_and_student_ownership(service):
    from huc.domain.models import Answer, SourceKind, Submission
    submission = Submission(exam_id="exam", student_id="student", source_kind=SourceKind.PDF,
                            answers=[Answer(question_id="q1", text="original")])
    service.repo.save_submission(submission)
    session = service.create_workshop("exam", "student")
    service.record_workshop_event(session.id, WorkshopEntry(
        stage="initial_solution", actor="student", submission_id=submission.id))
    submission.answers[0].text = "changed"
    service.repo.save_submission(submission)
    event = service.repo.get_workshop(session.id).events[0]
    assert event.submission_snapshot.answers[0].text == "original"
    other = service.create_workshop("exam", "other")
    with pytest.raises(ValueError, match="does not belong"):
        service.record_workshop_event(other.id, WorkshopEntry(
            stage="initial_solution", actor="student", submission_id=submission.id))


def test_question_limits(service):
    session = service.create_workshop("exam", "student")
    for data in [
        {"stage": "initial_solution", "text": "proof"},
        {"stage": "initial_diagnosis", "diagnosis": {"observations": "gap"}},
    ]:
        service.record_workshop_event(session.id, WorkshopEntry(actor="teacher", **data))
    with pytest.raises(ValueError, match="up to 3"):
        service.record_workshop_event(session.id, WorkshopEntry(
            stage="socratic_response", actor="ai", questions=["why?"] * 4))
    service.record_workshop_event(session.id, WorkshopEntry(
        stage="socratic_response", actor="ai", questions=["why?"]))
    with pytest.raises(ValueError, match="new student question"):
        service.record_workshop_event(session.id, WorkshopEntry(
            stage="socratic_response", actor="ai", questions=["why again?"]))

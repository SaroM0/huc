"""Records of a student's complete workshop process, without inferred learning claims."""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from huc.domain.models import Exam, Grade, Submission, _id, _now


class WorkshopStage(StrEnum):
    INITIAL_SOLUTION = "initial_solution"
    INITIAL_DIAGNOSIS = "initial_diagnosis"
    STUDENT_QUESTION = "student_question"
    SOCRATIC_RESPONSE = "socratic_response"
    REFLECTION = "reflection"
    FINAL_SOLUTION = "final_solution"
    FINAL_DIAGNOSIS = "final_diagnosis"
    TEACHER_NOTE = "teacher_note"


class WorkshopEntry(BaseModel):
    """Input record. Provenance can retain exact prompts, models and configuration."""
    model_config = ConfigDict(extra="forbid")
    stage: WorkshopStage
    actor: str = Field(min_length=1)
    text: str = ""
    question_id: str | None = None
    submission_id: str | None = None
    grade_id: str | None = None
    questions: list[str] = Field(default_factory=list)
    diagnosis: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)


class WorkshopEvent(WorkshopEntry):
    id: str = Field(default_factory=_id)
    sequence: int
    created_at: datetime = Field(default_factory=_now)
    submission_snapshot: Submission | None = None
    grade_snapshot: Grade | None = None


class WorkshopSession(BaseModel):
    id: str = Field(default_factory=_id)
    exam_id: str
    student_id: str = Field(min_length=1)
    exam_snapshot: Exam
    created_at: datetime = Field(default_factory=_now)
    events: list[WorkshopEvent] = Field(default_factory=list)

    def append(self, event: WorkshopEvent) -> None:
        stages = [item.stage for item in self.events]
        stage = event.stage
        if event.question_id and event.question_id not in {q.id for q in self.exam_snapshot.questions}:
            raise ValueError("Question does not belong to the session exam")
        once = {WorkshopStage.INITIAL_SOLUTION, WorkshopStage.INITIAL_DIAGNOSIS,
                WorkshopStage.FINAL_SOLUTION, WorkshopStage.FINAL_DIAGNOSIS}
        if stage in once and stage in stages:
            raise ValueError(f"Stage {stage} already recorded")
        prerequisites = {
            WorkshopStage.INITIAL_DIAGNOSIS: WorkshopStage.INITIAL_SOLUTION,
            WorkshopStage.STUDENT_QUESTION: WorkshopStage.INITIAL_SOLUTION,
            WorkshopStage.SOCRATIC_RESPONSE: WorkshopStage.INITIAL_DIAGNOSIS,
            WorkshopStage.REFLECTION: WorkshopStage.INITIAL_SOLUTION,
            WorkshopStage.FINAL_SOLUTION: WorkshopStage.INITIAL_DIAGNOSIS,
            WorkshopStage.FINAL_DIAGNOSIS: WorkshopStage.FINAL_SOLUTION,
        }
        if stage in prerequisites and prerequisites[stage] not in stages:
            raise ValueError(f"Stage {stage} requires {prerequisites[stage]}")
        if WorkshopStage.FINAL_SOLUTION in stages and stage in {
            WorkshopStage.STUDENT_QUESTION, WorkshopStage.SOCRATIC_RESPONSE,
            WorkshopStage.REFLECTION,
        }:
            raise ValueError("Student interaction is closed after the final solution")
        if stage in {WorkshopStage.INITIAL_SOLUTION, WorkshopStage.FINAL_SOLUTION}:
            if not event.text.strip() and event.submission_snapshot is None:
                raise ValueError("A solution needs text or a submission")
        elif stage in {WorkshopStage.INITIAL_DIAGNOSIS, WorkshopStage.FINAL_DIAGNOSIS}:
            if not event.diagnosis:
                raise ValueError("A diagnosis needs structured observations")
        elif stage == WorkshopStage.SOCRATIC_RESPONSE:
            previous = [item for item in self.events if item.stage == stage]
            limit = 3 if not previous else 2
            if len(previous) >= 2 or not 1 <= len(event.questions) <= limit:
                raise ValueError("Allow up to 3 initial questions and 2 follow-up questions")
            if any(not question.strip() for question in event.questions):
                raise ValueError("Socratic questions cannot be blank")
            if previous and not any(
                item.stage == WorkshopStage.STUDENT_QUESTION
                and item.sequence > previous[-1].sequence for item in self.events
            ):
                raise ValueError("A follow-up response requires a new student question")
        elif not event.text.strip():
            raise ValueError("This event needs text")
        event.sequence = len(self.events) + 1
        self.events.append(event)

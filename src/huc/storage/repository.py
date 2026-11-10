"""Persistence.

`Repository` is the port; `FileRepository` is a simple JSON-file adapter for
development. For production, implement e.g. `PostgresRepository` with the same interface.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from huc.domain.models import Exam, Grade, Submission
from huc.domain.workshop import WorkshopEvent, WorkshopSession

M = TypeVar("M", bound=BaseModel)


class Repository(Protocol):
    def create_workshop(self, session: WorkshopSession) -> None: ...
    def get_workshop(self, session_id: str) -> WorkshopSession: ...
    def list_workshops(self, exam_id: str) -> list[WorkshopSession]: ...
    def append_workshop_event(self, session_id: str, event: WorkshopEvent) -> WorkshopSession: ...
    def save_exam(self, exam: Exam) -> None: ...
    def get_exam(self, exam_id: str) -> Exam: ...
    def list_exams(self) -> list[Exam]: ...
    def save_submission(self, s: Submission) -> None: ...
    def get_submission(self, submission_id: str) -> Submission: ...
    def list_submissions(self, exam_id: str) -> list[Submission]: ...
    def save_grade(self, g: Grade) -> None: ...
    def get_grade(self, submission_id: str) -> Grade | None: ...
    def list_grade_history(self, submission_id: str) -> list[Grade]: ...


class FileRepository:
    """Stores each entity as `<root>/<kind>/<id>.json`. Getters raise FileNotFoundError."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        for sub in ("exams", "submissions", "grades", "workshops"):
            (self._root / sub).mkdir(parents=True, exist_ok=True)

    def _write(self, kind: str, key: str, model: BaseModel) -> None:
        (self._root / kind / f"{key}.json").write_text(model.model_dump_json(indent=2), "utf-8")

    def _read(self, kind: str, key: str, cls: type[M]) -> M:
        return cls.model_validate_json((self._root / kind / f"{key}.json").read_text("utf-8"))

    def _all(self, kind: str, cls: type[M]) -> list[M]:
        paths = sorted((self._root / kind).glob("*.json"))
        return [cls.model_validate_json(p.read_text("utf-8")) for p in paths]

    def save_exam(self, exam: Exam) -> None:
        self._write("exams", exam.id, exam)

    def get_exam(self, exam_id: str) -> Exam:
        return self._read("exams", exam_id, Exam)

    def list_exams(self) -> list[Exam]:
        return self._all("exams", Exam)

    def save_submission(self, s: Submission) -> None:
        self._write("submissions", s.id, s)

    def get_submission(self, submission_id: str) -> Submission:
        return self._read("submissions", submission_id, Submission)

    def list_submissions(self, exam_id: str) -> list[Submission]:
        return [s for s in self._all("submissions", Submission) if s.exam_id == exam_id]

    def save_grade(self, g: Grade) -> None:
        # Keep each run as well as the latest-result pointer used by existing clients.
        history = self._root / "grade_history" / g.submission_id
        history.mkdir(parents=True, exist_ok=True)
        with (history / f"{g.id}.json").open("x", encoding="utf-8") as output:
            output.write(g.model_dump_json(indent=2))
        self._write("grades", g.submission_id, g)

    def get_grade(self, submission_id: str) -> Grade | None:
        path = self._root / "grades" / f"{submission_id}.json"
        return Grade.model_validate_json(path.read_text("utf-8")) if path.exists() else None

    def list_grade_history(self, submission_id: str) -> list[Grade]:
        directory = self._root / "grade_history" / submission_id
        grades = [Grade.model_validate_json(p.read_text("utf-8")) for p in directory.glob("*.json")]
        return sorted(grades, key=lambda grade: grade.created_at)

    def create_workshop(self, session: WorkshopSession) -> None:
        with (self._root / "workshops" / f"{session.id}.json").open("x", encoding="utf-8") as out:
            out.write(session.model_dump_json(indent=2))

    def get_workshop(self, session_id: str) -> WorkshopSession:
        return self._read("workshops", session_id, WorkshopSession)

    def list_workshops(self, exam_id: str) -> list[WorkshopSession]:
        return [s for s in self._all("workshops", WorkshopSession) if s.exam_id == exam_id]

    def append_workshop_event(self, session_id: str, event: WorkshopEvent) -> WorkshopSession:
        self.get_workshop(session_id)
        directory = self._root / "workshops"
        with (directory / f"{session_id}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            session = self.get_workshop(session_id)
            session.append(event)
            name = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                                 delete=False) as out:
                    name = out.name
                    out.write(session.model_dump_json(indent=2))
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(name, directory / f"{session_id}.json")
            finally:
                if name and os.path.exists(name):
                    os.unlink(name)
            return session

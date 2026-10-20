"""Persistence.

`Repository` is the port; `FileRepository` is a simple JSON-file adapter for
development. For production, implement e.g. `PostgresRepository` with the same interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from huc.domain.models import Exam, Grade, Submission

M = TypeVar("M", bound=BaseModel)


class Repository(Protocol):
    def save_exam(self, exam: Exam) -> None: ...
    def get_exam(self, exam_id: str) -> Exam: ...
    def list_exams(self) -> list[Exam]: ...
    def save_submission(self, s: Submission) -> None: ...
    def get_submission(self, submission_id: str) -> Submission: ...
    def list_submissions(self, exam_id: str) -> list[Submission]: ...
    def save_grade(self, g: Grade) -> None: ...
    def get_grade(self, submission_id: str) -> Grade | None: ...


class FileRepository:
    """Stores each entity as `<root>/<kind>/<id>.json`. Getters raise FileNotFoundError."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        for sub in ("exams", "submissions", "grades"):
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
        # One grade per submission; regrading overwrites (history is a job for a real DB).
        self._write("grades", g.submission_id, g)

    def get_grade(self, submission_id: str) -> Grade | None:
        path = self._root / "grades" / f"{submission_id}.json"
        return Grade.model_validate_json(path.read_text("utf-8")) if path.exists() else None

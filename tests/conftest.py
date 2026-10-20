"""Shared test fixtures: a tiny fake chat model and a temp repository with the example exam."""

from pathlib import Path

import pytest
import yaml

from huc.domain.models import CriterionScore, Exam, ReviewVerdict
from huc.storage.repository import FileRepository

EXAMPLES = Path(__file__).parent.parent / "examples"


class FakeLLM:
    """Minimal stand-in for a LangChain chat model.

    `with_structured_output(schema)` returns a runnable-like object whose
    `ainvoke`/`invoke` return canned Pydantic instances and record the call.
    """

    model = "fake-model"

    def __init__(
        self,
        points: float = 1.5,
        confidence: float = 0.9,
        needs_human: bool = False,
        adjusted: list[CriterionScore] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.points, self.confidence, self.needs_human = points, confidence, needs_human
        self.adjusted = adjusted or []
        self.canned: dict[type, object] = {}  # extra schemas, e.g. ExtractionResult

    def with_structured_output(self, schema, **_):
        fake = self

        class _Bound:
            async def ainvoke(self, messages, *args, **kwargs):
                return self.invoke(messages)

            def invoke(self, messages, *args, **kwargs):
                fake.calls.append({"schema": schema.__name__, "messages": messages})
                return fake._respond(schema)

        return _Bound()

    def _respond(self, schema):
        if schema in self.canned:
            return self.canned[schema]
        if schema is CriterionScore:
            return CriterionScore(
                criterion_id="ignored",
                points=self.points,
                justification="ok",
                evidence=["2 identical diploid cells"],
                confidence=self.confidence,
            )
        if schema is ReviewVerdict:
            return ReviewVerdict(
                approved=not self.adjusted,
                needs_human=self.needs_human,
                adjusted_scores=self.adjusted,
            )
        raise AssertionError(f"unexpected schema {schema}")

    def count(self, schema_name: str) -> int:
        return [c["schema"] for c in self.calls].count(schema_name)


@pytest.fixture
def repo(tmp_path) -> FileRepository:
    return FileRepository(tmp_path / "data")


@pytest.fixture
def exam(repo) -> Exam:
    exam = Exam.model_validate(yaml.safe_load((EXAMPLES / "exam_example.yaml").read_text("utf-8")))
    repo.save_exam(exam)
    return exam

"""Shared test fixtures: fake Claude and OCR models, a temp repository with the example exam,
and a generated scanned PDF. No network, no API key."""

from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
import yaml

from huc.domain.models import CriterionScore, Exam, ReviewVerdict
from huc.ocr import ExamReading
from huc.storage.repository import FileRepository

EXAMPLES = Path(__file__).parent.parent / "examples"
EXAM_YAML = EXAMPLES / "exam_example.yaml"

Q1_ANSWER = (
    "La mitosis da 2 células diploides iguales, sirve para crecer. "
    "La meiosis da 4 células haploides y hay crossing-over."
)
Q2_ANSWER = "La fotosíntesis ocurre en los cloroplastos: CO2 + agua + luz -> glucosa y oxígeno."

# What the fake OCR transcribes on each page of the generated scan (`huc ocr`).
PAGE_TEXTS = [f"1. {Q1_ANSWER}", f"2. {Q2_ANSWER}"]

FEEDBACK = "Bien explicado; menciona también la recombinación."

# What the fake OCR returns by default when reading a whole exam.
DEFAULT_READING = {
    "answers": [
        {"question_id": "q1", "text": Q1_ANSWER, "confidence": 0.95, "source_pages": [1]},
        {"question_id": "q2", "text": Q2_ANSWER, "confidence": 0.9, "source_pages": [2]},
    ],
}


class FakeLLM:
    """Minimal stand-in for a LangChain chat model (grader and reviewer roles).

    `with_structured_output(schema)` returns a runnable-like object whose
    `ainvoke`/`invoke` return canned Pydantic instances and record the call.
    """

    model = "fake-model"

    def __init__(
        self,
        points: float = 1.5,
        confidence: float = 0.9,
        needs_verification: bool = False,
        adjusted: list[CriterionScore] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.points, self.confidence = points, confidence
        self.needs_verification = needs_verification
        self.adjusted = adjusted or []
        self.canned: dict[type, object] = {}  # per-schema overrides

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
                evidence=["2 células diploides iguales"],
                confidence=self.confidence,
            )
        if schema is ReviewVerdict:
            return ReviewVerdict(
                approved=not self.adjusted,
                needs_verification=self.needs_verification,
                adjusted_scores=self.adjusted,
                feedback=FEEDBACK,
            )
        raise AssertionError(f"unexpected schema {schema}")

    def count(self, schema_name: str) -> int:
        return [c["schema"] for c in self.calls].count(schema_name)


class FakeOCR:
    """Stand-in for the Claude OCR model.

    - `with_structured_output(ExamReading).ainvoke(messages)` (whole exam) returns `reading`.
    - `ainvoke(messages)` (one page) returns the next canned page text.
    Both record the messages in `calls`.
    """

    model = "fake-ocr"

    def __init__(
        self, texts: list[str] | None = None, reading: ExamReading | dict | None = None
    ) -> None:
        self.texts = list(texts or PAGE_TEXTS)
        self.reading = ExamReading.model_validate(reading or DEFAULT_READING)
        self.calls: list[list] = []

    def with_structured_output(self, schema, **_):
        assert schema is ExamReading
        fake = self

        class _Bound:
            async def ainvoke(self, messages, *args, **kwargs):
                fake.calls.append(messages)
                return fake.reading

        return _Bound()

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(content=self.texts[(len(self.calls) - 1) % len(self.texts)])


def make_scan_pdf(path: Path, pages: int = 2) -> Path:
    """A PDF with no text layer, like a scanner produces."""
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page().draw_rect(pymupdf.Rect(50, 50, 200, 200))
    doc.save(path)
    return path


@pytest.fixture
def repo(tmp_path) -> FileRepository:
    return FileRepository(tmp_path / "data")


@pytest.fixture
def exam(repo) -> Exam:
    exam = Exam.model_validate(yaml.safe_load(EXAM_YAML.read_text("utf-8")))
    repo.save_exam(exam)
    return exam


# Real submissions are named `<student>__<original name>`.
STUDENT = "Perez_Gomez_Ana"


@pytest.fixture
def scan_pdf(tmp_path) -> Path:
    return make_scan_pdf(tmp_path / f"{STUDENT}__scan.pdf")

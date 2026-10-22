"""Ingestion graph: handwritten exam scan (PDF / PNG / JPG) -> persisted Submission.

    START -> load_pages   (rasterize PDF pages / load the photo)
          -> read_exam    (Claude Haiku vision: all pages in ONE call -> answer per question)
          -> save -> END

Page images are sent to the Anthropic API. The scans hold only answers; the student
comes from `student_id` or from the file name (`<student>__<anything>.pdf`).

Input:  {"exam_id": str, "file_path": str, "student_id": str | None}
Output: state with "submission" (already saved).
"""

from pathlib import Path
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from huc.config import Settings, get_settings
from huc.domain.models import Answer, SourceKind, Submission
from huc.ingestion import loaders
from huc.ocr import get_ocr_model, read_exam
from huc.storage.repository import FileRepository, Repository


class IngestionInput(TypedDict, total=False):
    exam_id: str
    file_path: str
    student_id: str | None


class IngestionState(IngestionInput, total=False):
    source_kind: SourceKind
    pages: list[loaders.Page]
    submission: Submission


def build_ingestion_graph(
    repo: Repository | None = None,
    *,
    settings: Settings | None = None,
    ocr: BaseChatModel | Any | None = None,
) -> CompiledStateGraph:
    """Build the ingestion graph.

    `repo=None` -> FileRepository(settings.data_dir); `ocr=None` -> lazy Claude OCR model
    (`settings.ocr_model`); `settings=None` -> get_settings(). Building the graph
    connects to nothing.
    """

    def cfg() -> Settings:
        return settings or get_settings()

    def get_repo() -> Repository:
        return repo if repo is not None else FileRepository(cfg().data_dir)

    # -- nodes ---------------------------------------------------------------

    def load_pages(state: IngestionState) -> dict:
        get_repo().get_exam(state["exam_id"])  # fail before the OCR call if the exam is unknown
        settings = cfg()
        doc = loaders.load_document(
            Path(state["file_path"]), dpi=settings.pdf_dpi, max_side=settings.ocr_max_side_px
        )
        return {"source_kind": doc.kind, "pages": doc.pages}

    async def read(state: IngestionState) -> dict:
        exam = get_repo().get_exam(state["exam_id"])
        path = Path(state["file_path"])
        model = ocr if ocr is not None else get_ocr_model(cfg())
        reading = await read_exam(model, state["pages"], exam)

        valid_ids = {q.id for q in exam.questions}
        submission = Submission(
            exam_id=exam.id,
            student_id=state.get("student_id") or loaders.student_id_from_path(path),
            source_kind=state["source_kind"],
            source_path=str(path),
            answers=[
                Answer(
                    question_id=a.question_id,
                    text=a.text,
                    extraction_confidence=a.confidence,
                    source_pages=a.source_pages,
                )
                for a in reading.answers
                if a.question_id in valid_ids
            ],
        )
        return {"submission": submission}

    def save(state: IngestionState) -> dict:
        get_repo().save_submission(state["submission"])
        return {}

    # -- wiring --------------------------------------------------------------

    builder = StateGraph(IngestionState, input_schema=IngestionInput)
    builder.add_node("load_pages", load_pages)
    builder.add_node("read_exam", read)
    builder.add_node("save", save)
    builder.add_edge(START, "load_pages")
    builder.add_edge("load_pages", "read_exam")
    builder.add_edge("read_exam", "save")
    builder.add_edge("save", END)
    return builder.compile(name="ingestion")


graph = build_ingestion_graph()

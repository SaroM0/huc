"""Ingestion graph: handwritten exam scan (PDF / PNG / JPG) -> persisted Submission.

    START -> load_pages   (rasterize PDF pages / load the photo)
          -> ocr_pages    (local Ollama vision model, one page after another)
          -> segment      (Claude, text only: split the transcript by question)
          -> save -> END

Page images never leave the machine: only the OCR text is sent to Claude.

Input:  {"exam_id": str, "file_path": str, "student_id": str | None}
Output: state with "transcripts" (per-page OCR text) and "submission" (already saved).
"""

from pathlib import Path
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from huc.config import Settings, get_settings
from huc.domain.models import Answer, SourceKind, Submission
from huc.ingestion import loaders
from huc.llm import build_messages, get_chat_model, structured
from huc.ocr import get_ocr_model, transcribe_page
from huc.prompts import segmentation as prompts
from huc.storage.repository import FileRepository, Repository


class IngestionInput(TypedDict, total=False):
    exam_id: str
    file_path: str
    student_id: str | None


class PageTranscript(TypedDict):
    page: int  # 1-based page number
    text: str


class IngestionState(IngestionInput, total=False):
    source_kind: SourceKind
    pages: list[loaders.Page]
    transcripts: list[PageTranscript]
    submission: Submission


# Structured output of the segmenter agent
class SegmentedAnswer(BaseModel):
    question_id: str
    text: str
    confidence: float = Field(ge=0, le=1)
    source_pages: list[int] = Field(default_factory=list)


class SegmentationResult(BaseModel):
    student_id: str | None = None
    answers: list[SegmentedAnswer]


def build_ingestion_graph(
    repo: Repository | None = None,
    llm: BaseChatModel | Any | None = None,
    *,
    settings: Settings | None = None,
    ocr: BaseChatModel | Any | None = None,
) -> CompiledStateGraph:
    """Build the ingestion graph.

    `repo=None` -> FileRepository(settings.data_dir); `llm=None` -> lazy Claude segmenter
    model; `ocr=None` -> lazy Ollama vision model (`settings.ocr_model`); `settings=None` ->
    get_settings(). Building the graph connects to nothing.
    """

    def cfg() -> Settings:
        return settings or get_settings()

    def get_repo() -> Repository:
        return repo if repo is not None else FileRepository(cfg().data_dir)

    # -- nodes ---------------------------------------------------------------

    def load_pages(state: IngestionState) -> dict:
        get_repo().get_exam(state["exam_id"])  # fail before the slow OCR if the exam is unknown
        doc = loaders.load_document(Path(state["file_path"]), dpi=cfg().pdf_dpi)
        return {"source_kind": doc.kind, "pages": doc.pages}

    async def ocr_pages(state: IngestionState) -> dict:
        """Sequential on purpose: a CPU-only Ollama cannot run pages in parallel anyway."""
        model = ocr if ocr is not None else get_ocr_model(cfg())
        transcripts: list[PageTranscript] = [
            {"page": page.number, "text": await transcribe_page(model, page)}
            for page in state["pages"]
        ]
        return {"transcripts": transcripts}

    async def segment(state: IngestionState) -> dict:
        exam = get_repo().get_exam(state["exam_id"])
        path = Path(state["file_path"])
        model = llm if llm is not None else get_chat_model(cfg().segmenter_model)
        messages = build_messages(
            prompts.segmentation_system(exam),
            [
                prompts.transcript_block(state["transcripts"]),
                {"type": "text", "text": prompts.SEGMENTATION_USER},
            ],
        )
        result: SegmentationResult = await structured(model, SegmentationResult).ainvoke(messages)

        student_id = state.get("student_id") or result.student_id
        if not student_id:
            raise ValueError(f"Could not identify the student in {path.name}; pass student_id")
        valid_ids = {q.id for q in exam.questions}
        submission = Submission(
            exam_id=exam.id,
            student_id=student_id,
            source_kind=state["source_kind"],
            source_path=str(path),
            answers=[
                Answer(
                    question_id=a.question_id,
                    text=a.text,
                    extraction_confidence=a.confidence,
                    source_pages=a.source_pages,
                )
                for a in result.answers
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
    builder.add_node("ocr_pages", ocr_pages)
    builder.add_node("segment", segment)
    builder.add_node("save", save)
    builder.add_edge(START, "load_pages")
    builder.add_edge("load_pages", "ocr_pages")
    builder.add_edge("ocr_pages", "segment")
    builder.add_edge("segment", "save")
    builder.add_edge("save", END)
    return builder.compile(name="ingestion")


graph = build_ingestion_graph()

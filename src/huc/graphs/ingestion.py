"""Ingestion graph: file -> persisted Submission.

    START -> (route by file type)
               ├─ LMS JSON  -> parse_lms  (no LLM, extraction confidence = 1.0)
               └─ PDF/image -> extract    (vision LLM transcribes + segments by question)
          -> save -> END

Input:  {"exam_id": str, "file_path": str, "student_id": str | None}
Output: state with "submission" (already saved in the repository).
"""

from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from huc.config import Settings, get_settings
from huc.domain.models import Answer, SourceKind, Submission
from huc.ingestion import loaders
from huc.llm import build_messages, get_chat_model, structured
from huc.prompts import extraction as prompts
from huc.storage.repository import FileRepository, Repository


class IngestionInput(TypedDict, total=False):
    exam_id: str
    file_path: str
    student_id: str | None


class IngestionState(IngestionInput, total=False):
    submission: Submission


# Structured output of the extractor agent
class ExtractedAnswer(BaseModel):
    question_id: str
    text: str
    confidence: float = Field(ge=0, le=1)
    source_pages: list[int] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    student_id: str | None = None
    answers: list[ExtractedAnswer]


def build_ingestion_graph(
    repo: Repository | None = None,
    llm: BaseChatModel | Any | None = None,
    *,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Build the ingestion graph.

    `repo=None` -> FileRepository(settings.data_dir); `llm=None` -> lazy Claude extractor
    model; `settings=None` -> get_settings().
    """

    def cfg() -> Settings:
        return settings or get_settings()

    def get_repo() -> Repository:
        return repo if repo is not None else FileRepository(cfg().data_dir)

    def route(state: IngestionState) -> Literal["parse_lms", "extract"]:
        kind = loaders.detect_kind(Path(state["file_path"]))
        return "parse_lms" if kind is SourceKind.LMS_JSON else "extract"

    def parse_lms(state: IngestionState) -> dict:
        """Digital answers skip the LLM entirely."""
        exam = get_repo().get_exam(state["exam_id"])
        path = Path(state["file_path"])
        data = loaders.load_lms_json(path)
        valid_ids = {q.id for q in exam.questions}
        submission = Submission(
            exam_id=exam.id,
            student_id=state.get("student_id") or data["student_id"],
            source_kind=SourceKind.LMS_JSON,
            source_path=str(path),
            answers=[
                Answer(question_id=qid, text=str(text))
                for qid, text in data["answers"].items()
                if qid in valid_ids
            ],
        )
        return {"submission": submission}

    async def extract(state: IngestionState) -> dict:
        exam = get_repo().get_exam(state["exam_id"])
        path = Path(state["file_path"])
        doc = loaders.load_document(path)

        model = llm if llm is not None else get_chat_model(cfg().extractor_model)
        content = [p.to_content_block() for p in doc.pages]
        content.append({"type": "text", "text": prompts.EXTRACTION_USER})
        messages = build_messages(prompts.extraction_system(exam), content)
        result: ExtractionResult = await structured(model, ExtractionResult).ainvoke(messages)

        student_id = state.get("student_id") or result.student_id
        if not student_id:
            raise ValueError(f"Could not identify the student in {path.name}")
        valid_ids = {q.id for q in exam.questions}
        submission = Submission(
            exam_id=exam.id,
            student_id=student_id,
            source_kind=doc.kind,
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

    builder = StateGraph(IngestionState, input_schema=IngestionInput)
    builder.add_node("parse_lms", parse_lms)
    builder.add_node("extract", extract)
    builder.add_node("save", save)
    builder.add_conditional_edges(START, route)
    builder.add_edge("parse_lms", "save")
    builder.add_edge("extract", "save")
    builder.add_edge("save", END)
    return builder.compile(name="ingestion")


graph = build_ingestion_graph()

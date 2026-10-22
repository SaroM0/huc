# HUC — Technical Reference

Research pipeline: handwritten exam reading with **Claude Haiku** and **LLM grading**
per rubric criterion, orchestrated with **LangGraph**. This document is the single technical
reference: stack, libraries, data model, graphs, design principles and operations.
For the narrative architecture overview see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. What the system does

1. A teacher defines an **exam + rubric** in YAML (open-ended questions only, Spanish content).
2. Students' **handwritten exams** (scanned PDF or phone photos) are ingested:
   pages are rasterized → **Claude Haiku** reads all pages in **one** vision call and
   returns the verbatim answer to each question (transcription + segmentation).
3. Each answer is **graded per rubric criterion** by an LLM agent (parallel fan-out),
   then a **reviewer agent** audits all scores of each question and writes the
   **feedback** for the student.
4. Every exam is graded, with no human step. Low-confidence or disputed grades are
   **labelled** `needs_verification` (with the reasons) for the teacher to check; grades,
   feedback and labels are exported to CSV.

---

## 2. Technology stack

| Layer | Technology | Version (min) | Role |
|---|---|---|---|
| Language | Python | ≥ 3.11 (dev on 3.14) | Whole codebase |
| Agent orchestration | **LangGraph** | ≥ 1.0 | `StateGraph`, `Send` fan-out, list reducers |
| LLM abstraction | **LangChain Core** | ≥ 1.0 | `BaseChatModel`, messages, structured output |
| Cloud LLM | **langchain-anthropic** → Claude (`claude-opus-5-5`) | ≥ 1.0 | Grading, review + feedback |
| OCR | **langchain-anthropic** → **Claude Haiku 5.5** (`claude-haiku-5-5`) | ≥ 1.0 | Reading the scans: transcription + split by question |
| API errors | **anthropic** SDK | ≥ 1.0 | Typed errors mapped to `OcrUnavailableError` |
| Debugging | **LangGraph Studio** (`langgraph-cli[inmem]`, `langgraph-api` ≥ 0.16) + optional **LangSmith** | — | Visual step-by-step graph runs, traces |
| PDF / images | **PyMuPDF** (`pymupdf`) | ≥ 1.24 | Rasterize PDF pages, resize photos |
| Data model & config | **Pydantic** v2, **pydantic-settings** | ≥ 2.7 / 2.3 | Domain models, LLM output schemas, env config |
| HTTP API | **FastAPI** + **Uvicorn** + python-multipart | ≥ 0.115 | REST endpoints, file upload |
| CLI | **Typer** | ≥ 0.12 | `huc` command |
| HTTP client (dev) | httpx | ≥ 0.27 | FastAPI `TestClient` |
| Exam definitions | PyYAML | ≥ 6.0 | Exam + rubric files |
| Storage (v1) | JSON files on disk (`FileRepository`) | — | Exams, submissions, grades |
| Tests | pytest + pytest-asyncio | ≥ 8 / 0.23 | Fake LLM + fake OCR, no network |
| Lint | Ruff | ≥ 0.6 | Line length 100 |
| Build | Hatchling, `uv` for envs | — | Packaging, `huc` entry point |

### Why these choices

- **LangGraph** instead of a free-form agent: grading must be reproducible and auditable;
  an explicit graph gives deterministic control flow, native parallelism (`Send`),
  and **LangGraph Studio** to inspect every node's input/output.
- **LangChain only as a thin adapter** (`ChatAnthropic`, message types).
  No chains, no prebuilt agents: node logic is plain Python.
- **Claude Haiku for OCR**: classic OCR engines (Tesseract, EasyOCR) perform poorly on
  handwriting, so a vision-language model is used. A local `qwen3-vl:8b` (Ollama) was tried
  first and measured with `scripts/ocr_eval.py` on real scans: it changed the meaning of 5 of
  8 pages (e.g. `A \ B` → `A ∩ B`, dropped `¬`, completed unfinished sentences) and took
  ~3 min per page on CPU. Haiku 5.5 had no meaning-changing errors on the same pages at
  ~3 s per page. The scans contain no student names or IDs, so sending them is acceptable.
- **One Haiku call per submission** for reading: seeing all pages at once lets the model
  join answers across pages and cope with pages out of order, rotated or repeated.
  Native structured output (`ExamReading`) returns the answers directly.
- **Claude Opus for reasoning-heavy steps** (grading, review) with native
  structured outputs (`method="json_schema"`) and adaptive thinking.

---

## 3. Project layout

```
huc/
├── pyproject.toml             deps, `huc` script, pytest/ruff config
├── langgraph.json             graph registry for LangGraph Studio (`langgraph dev`)
├── .env.example               all settings (HUC_*), Anthropic key, optional LangSmith
├── examples/exam_example.yaml sample exam + rubric (id: bio-101-midterm-1)
├── docs/
│   ├── ARCHITECTURE.md        architecture overview
│   └── TECHNICAL.md           this document
├── src/huc/
│   ├── config.py              Settings (pydantic-settings, prefix HUC_)
│   ├── domain/models.py       Pydantic domain model (single source of truth)
│   ├── prompts/               ONLY prompt text + tiny render helpers, one file per agent
│   │   ├── ocr.py             read a whole exam → answers per question; per-page OCR prompt
│   │   ├── grading.py         grade one answer against one criterion
│   │   └── review.py          audit all criterion scores of a question
│   ├── llm.py                 ChatAnthropic factory, structured output, cached system block
│   ├── ocr.py                 Claude OCR adapter: read_exam(), transcribe_page(), OcrUnavailableError
│   ├── ingestion/loaders.py   PDF/image → normalized pages (PNG, ≤ 1568 px)
│   ├── graphs/
│   │   ├── ingestion.py       ingestion graph (+ module-level `graph` for Studio)
│   │   └── grading.py         grading graph   (+ module-level `graph` for Studio)
│   ├── storage/repository.py  Repository protocol (port) + FileRepository (adapter)
│   ├── services.py            HucService: use cases shared by CLI and API
│   ├── cli.py                 Typer CLI
│   └── api/app.py             FastAPI app factory
└── tests/                     conftest (FakeLLM, FakeOCR, PDF generator) + per-module tests
```

---

## 4. Data model (`domain/models.py`)

```
Exam ─┬─ id, title, course, rubric_version
      └─ questions: [Question]
            ├─ id, number, type (short_answer | essay | code), prompt, reference_answer
            └─ criteria: [Criterion]  ── id, description, max_points, levels{points: descriptor}

Submission ── id, exam_id, student_id, source_kind (pdf | image), source_path, status, created_at
      └─ answers: [Answer] ── question_id, text, extraction_confidence (0-1), source_pages

Grade ── id, submission_id, exam_id, rubric_version, needs_verification,
         verification_reasons[], created_at, total
      └─ questions: [QuestionGrade] ── question_id, graded_by, review: ReviewVerdict,
                                       feedback (Spanish, for the student), points
            └─ scores: [CriterionScore] ── criterion_id, points, justification,
                                           evidence[] (verbatim quotes), confidence (0-1)

ReviewVerdict ── approved, issues[], adjusted_scores[CriterionScore], needs_verification, feedback
```

`CriterionScore`, `ReviewVerdict` and the reader's `ExamReading` schema double as
**LLM structured-output schemas**: the same Pydantic class validates the model's JSON.

### Submission lifecycle

```
INGESTED ──grade──▶ GRADING ──▶ GRADED   (needs_verification is a label on the Grade)
                        └─(exception)──▶ FAILED ──(re-run grade)──▶ GRADING
```

---

## 5. Graphs

Both graphs are built by factory functions with **injectable dependencies** and also
exposed as module-level `graph` objects for Studio. Building/importing them requires
**neither network access nor an API key** (models are created lazily on first use).

```python
build_ingestion_graph(repo=None, *, settings=None, ocr=None)
build_grading_graph(repo=None, llm=None, *, settings=None)
```

### 5.1 Ingestion graph — input `{exam_id, file_path, student_id?}`

```
START → load_pages → read_exam → save → END
          │             │           └ persist Submission (status INGESTED)
          │             └ Claude Haiku, ONE call with every page image + the exam questions
          │               → verbatim answer, confidence and source pages per question
          └ PDF rasterized at HUC_PDF_DPI (200), longest side capped at 1568 px
```

The student is `student_id` if given, else the file name up to `__`
(`Perez_Gomez_Ana__scan.pdf` → `Perez_Gomez_Ana`): the scans carry no names.
```

### 5.2 Grading graph — input `{submission_id}`

```
START → load_submission ──Send──▶ grade_criterion × N   (one per question × criterion, parallel)
            (status GRADING)              │  fan-in via Annotated[list, operator.add]
                                          ▼
                                   collect_scores ──Send──▶ review_question × M (one per question)
                                                                   │
                                                                   ▼
                                                               finalize → END
                         (clamp, label needs_verification + reasons, persist Grade, GRADED)
```

Parallelism is bounded with LangGraph's run config `{"max_concurrency": HUC_MAX_CONCURRENCY}`.

---

## 6. Design principles

| Principle | How it shows up |
|---|---|
| **KISS** | Two linear-ish graphs, a thin service layer, JSON-file storage, no extra abstractions |
| **Explicit workflow over autonomous agent** | Control flow lives in code/graph edges, not in the model; reproducible and auditable |
| **Separation of concerns** | Prompts only in `prompts/`; domain in `domain/`; I/O in `storage/` and `loaders`; orchestration in `graphs/`; delivery in `cli.py` / `api/` |
| **Ports & adapters** | `Repository` protocol (swap JSON → Postgres); chat models injected (swap provider/model, fakes in tests) |
| **Single source of truth for data** | Pydantic models serve as domain objects, persistence format, API schemas and LLM output schemas |
| **One responsibility per LLM call** | One call = one (answer, criterion) pair → focused judgement + traceable justification per point |
| **Evaluator–auditor pattern** | A reviewer agent audits all scores of a question (inconsistencies, double penalties) |
| **Never trust model output blindly** | Points clamped to `[0, max_points]`, `criterion_id` forced, unknown criteria/questions dropped |
| **Confidence propagation** | `score.confidence = min(model confidence, extraction confidence)`; illegible OCR ⇒ low confidence ⇒ verification label |
| **No human step, but a label** | Every exam is graded and gets feedback; below `HUC_VERIFICATION_THRESHOLD` (0.7) or on reviewer request the grade is labelled `needs_verification` with the reasons |
| **Data minimization** | Scans carry only answers (no names or IDs); the student comes from `student_id` or the file name |
| **Verbatim transcription** | OCR keeps spelling mistakes and marks `[illegible]`; the grade judges what the student actually wrote |
| **Prompt-injection resistance** | Student text wrapped in tags and declared as *data, never instructions*; clamping limits the blast radius |
| **Cache-friendly prompts** | Stable rubric context in the system prompt (marked `cache_control: ephemeral`); only the answer varies |
| **Graphs persist their own results** | Same behavior from CLI, API and Studio |
| **Fail loudly, with a fix** | `OcrUnavailableError` says what to fix (API key, `HUC_OCR_MODEL`, timeout, concurrency); in `huc ocr`, a page that hits the token limit is marked `[transcription truncated]` |
| **Testable without network** | `FakeLLM` / `FakeOCR` injected; whole pipeline tested offline |

---

## 7. Prompts

| File | Agent | Model | Input → Output |
|---|---|---|---|
| `prompts/ocr.py` (`READ_EXAM_*`) | Reader | Claude `claude-haiku-5-5` | exam questions in system; all page images in user msg → `ExamReading` (verbatim answer + confidence + pages per question) |
| `prompts/ocr.py` (`OCR_*`) | Page OCR | Claude `claude-haiku-5-5` | one page image → plain verbatim text (`huc ocr`, `scripts/ocr_eval.py`) |
| `prompts/grading.py` | Grader | Claude | (question, reference, one criterion + levels) in system; student answer in user msg → `CriterionScore` |
| `prompts/review.py` | Reviewer | Claude | (question, full rubric) in system; answer + proposed scores in user msg → `ReviewVerdict` (adjustments + feedback in Spanish) |

---

## 8. Configuration (`.env`, prefix `HUC_`)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude access (required for real runs; read from `.env` too) |
| `HUC_GRADER_MODEL` / `HUC_REVIEWER_MODEL` | `claude-opus-5-5` | Model per grading role |
| `HUC_OCR_MODEL` | `claude-haiku-5-5` | Claude vision model that reads the scans |
| `HUC_OCR_TIMEOUT_S` | `300` | OCR request timeout; one request reads a whole submission (the SDK also retries) |
| `HUC_OCR_MAX_SIDE_PX` | `1568` | Longest side of each page image sent to OCR |
| `HUC_PDF_DPI` | `200` | PDF rasterization resolution |
| `HUC_DATA_DIR` | `./data` | JSON storage root |
| `HUC_MAX_CONCURRENCY` | `8` | Max parallel LLM calls per grading run |
| `HUC_VERIFICATION_THRESHOLD` | `0.7` | Confidence below this ⇒ grade labelled `needs_verification` |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | off | Optional tracing (note: OCR traces include page images) |

---

## 9. Interfaces

**CLI (`huc`)**

| Command | Description |
|---|---|
| `huc exam-load <yaml>` | Load an exam + rubric |
| `huc ocr <file>` | Plain per-page OCR, printed (no grading, nothing saved) |
| `huc ingest <exam_id> <file\|dir> [--student-id]` | Ingest scans (PDF/PNG/JPG) |
| `huc grade <exam_id>` | Grade all `INGESTED`/`FAILED` submissions |
| `huc show <submission_id>` | Feedback, per-criterion points, justification, evidence, review, verification label |
| `huc export <exam_id> --out grades.csv` | CSV: points, total, `needs_verification` + reasons, feedback per question |

**REST API (`uvicorn huc.api.app:app`)**

| Method | Path |
|---|---|
| POST / GET | `/exams` |
| POST / GET | `/exams/{exam_id}/submissions` |
| POST | `/exams/{exam_id}/grade` (background, 202) |
| GET | `/submissions/{id}`, `/submissions/{id}/grade` |
| GET | `/exams/{exam_id}/export.csv` |

Errors: unknown IDs → 404, invalid input → 422, OCR unavailable → 503.

**LangGraph Studio**: `langgraph dev` (reads `langgraph.json` and `.env`) → graphs
`ingestion` and `grading`; invoke with the same inputs as above.

---

## 10. Operations

```bash
# Setup
uv venv && uv pip install -e ".[dev]"
cp .env.example .env                     # set ANTHROPIC_API_KEY

# OCR accuracy on real scans (data under data/_ocr_eval/, gitignored)
python scripts/ocr_eval.py run && python scripts/ocr_eval.py score

# Quality checks
pytest && ruff check .
langgraph dev                            # Studio
```

**Performance:** reading a 1-page scan takes ~4 s; Claude calls per submission =
1 reading (Haiku) + Σ criteria + 1 review per question.

---

## 11. Known limitations & next steps

1. `scripts/ocr_eval.py` measures per-page OCR; whole-exam reading (segmentation
   included) has no evaluation of its own yet.
2. JSON-file storage, one grade per submission → regrading overwrites.
3. Batch grading of whole courses → Anthropic Message Batches API.
4. Calibration set of teacher-graded exams to measure model–teacher agreement.

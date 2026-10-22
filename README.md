# HUC - Exam grading with LLM agents

Ingests handwritten student exams in Spanish (scanned PDFs or photos), transcribes
them with **Claude Haiku** as the OCR model, and grades them with one Claude agent per
rubric criterion, a reviewer agent per question, and escalation to human review.
Page images are sent to the Anthropic API for OCR; the scans contain only answers (no
student names or IDs). Built on [LangGraph](https://langchain-ai.github.io/langgraph/).
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the full technical reference in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env        # set ANTHROPIC_API_KEY
```

All settings are environment variables with the `HUC_` prefix (Claude model per role,
OCR model, `HUC_DATA_DIR`, `HUC_MAX_CONCURRENCY`, `HUC_HUMAN_REVIEW_THRESHOLD`, ...);
see `.env.example`.

### OCR (Claude Haiku)

Each page image is transcribed by `claude-haiku-5-5` (`HUC_OCR_MODEL`), one API call per
page, with the pages of a file processed in parallel (up to `HUC_MAX_CONCURRENCY`). A page
takes a few seconds. PDFs are rasterized at `HUC_PDF_DPI` (default 200) and every image is
capped at `HUC_OCR_MAX_SIDE_PX` (default 1568 px, the largest size Claude uses without
downscaling). Rate limits, overload and transient errors are retried by the SDK; if a page
still fails, raise `HUC_OCR_TIMEOUT_S` (default 120 s) or lower `HUC_MAX_CONCURRENCY`.

Check the OCR alone, without segmentation or grading calls:

```bash
huc ocr scans/ana-perez.pdf     # prints the transcription of each page and its time
```

To measure OCR accuracy against reference transcriptions (CER/WER per page), use
`scripts/ocr_eval.py` (`export`, `run`, `score`); its data lives in the gitignored
`data/_ocr_eval/`.

Illegible words come out as `[illegible]`; crossed-out text is dropped.

## CLI

```bash
huc exam-load examples/exam_example.yaml                     # exam id: bio-101-midterm-1
huc ocr scans/ana-perez.pdf                                  # OCR only, prints each page
huc ingest bio-101-midterm-1 scans/ana-perez.pdf             # one file (--student-id to force it)
huc ingest bio-101-midterm-1 ./scans/                        # every PDF/PNG/JPG in a folder
huc grade bio-101-midterm-1                                  # grade all pending submissions
huc show SUBMISSION_ID                                       # per-criterion justification + evidence
huc finalize SUBMISSION_ID                                   # teacher accepts the grade
huc export bio-101-midterm-1 --out grades.csv
```

`ingest` fails if the student cannot be identified in the exam (ID or name);
pass `--student-id` in that case. `grade` picks up submissions in `ingested` or
`failed` status, so re-running it retries failures. Use `huc --help` / `huc <command> --help` for details.

## API

```bash
uvicorn huc.api.app:app --reload     # docs at http://localhost:8000/docs
```

| Method | Path | Description |
|---|---|---|
| POST | `/exams` | Upload an exam YAML (multipart `file`) |
| GET | `/exams` | List exams |
| POST | `/exams/{exam_id}/submissions` | Upload and ingest a scanned PDF / PNG / JPG (`file`, optional `?student_id=`). Synchronous: OCR takes minutes per page |
| GET | `/exams/{exam_id}/submissions` | List submissions of an exam |
| POST | `/exams/{exam_id}/grade` | Grade pending submissions in the background (202) |
| GET | `/submissions/{id}` | Get a submission (status, answers) |
| GET | `/submissions/{id}/grade` | Get the grade |
| POST | `/submissions/{id}/finalize` | Finalize after teacher review (204) |
| GET | `/exams/{exam_id}/export.csv` | Export grades as CSV |

Unknown IDs return 404; invalid input (bad YAML, finalizing an ungraded
submission, unsupported file type) returns 422; the OCR model unavailable (invalid
API key, unknown model, API unreachable or overloaded) returns 503.

## LangGraph Studio

`langgraph.json` at the repo root registers both graphs (`ingestion` and
`grading`), so you can run, inspect and debug them visually:

```bash
langgraph dev        # installed with the dev extras (langgraph-cli[inmem])
```

This opens LangGraph Studio in the browser. Invoke the graphs with the same
input as the code does:

- `ingestion`: `{"exam_id": "...", "file_path": "/abs/path/to/scan.pdf"}` (the state
  shows the per-page OCR `transcripts`)
- `grading`: `{"submission_id": "..."}`

The graphs read and write the same `HUC_DATA_DIR`, so load an exam first
(`huc exam-load ...`). `langgraph dev` reads `.env`.

OCR calls go through LangChain's `ChatAnthropic`, so they are traced like the other Claude calls.
Optional LangSmith tracing (for Studio and for CLI/API runs):

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=huc
```

## Project layout

```
src/huc/
  domain/models.py        Exam, Question, Criterion, Submission, Grade, ...
  ingestion/loaders.py    scanned PDF / photo -> page images
  ocr.py                  OCR adapter: Claude Haiku vision model, one page per call
  prompts/                prompt templates (ocr, segmenter, grader, reviewer)
  graphs/ingestion.py     ingestion graph: file -> OCR -> segment -> Submission
  graphs/grading.py       grading graph: fan-out per criterion -> review -> finalize
  storage/repository.py   Repository (port) + FileRepository (JSON files)
  config.py               Settings (env vars, HUC_ prefix)
  services.py             use cases shared by CLI and API (thin layer over the graphs)
  cli.py                  Typer CLI
  api/app.py              FastAPI app
langgraph.json            graph registry for `langgraph dev` / Studio
examples/                 example exam YAML (open-ended questions, Spanish content)
tests/                    fake LLM, no network
```

## Tests

```bash
pytest
ruff check .
```

Tests inject fake Claude and OCR models and generate scans in a temp dir, so they need
neither network access nor an API key.

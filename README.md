# HUC - Exam grading with LLM agents

Ingests student exams (scanned or digital PDF, images, LMS JSON exports) and
grades them with one LLM agent per rubric criterion, a reviewer agent per
question, and escalation to human review. Built on [LangGraph](https://langchain-ai.github.io/langgraph/).
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env        # set ANTHROPIC_API_KEY
```

All settings are environment variables with the `HUC_` prefix (models per role,
`HUC_DATA_DIR`, `HUC_MAX_CONCURRENCY`, `HUC_HUMAN_REVIEW_THRESHOLD`); see `.env.example`.

## CLI

```bash
huc exam-load examples/exam_example.yaml                     # exam id: bio-101-midterm-1
huc ingest bio-101-midterm-1 examples/submission_lms.json    # one file
huc ingest bio-101-midterm-1 ./scans/                        # every PDF/image/JSON in a folder
huc grade bio-101-midterm-1                                  # grade all pending submissions
huc show SUBMISSION_ID                                       # per-criterion justification + evidence
huc finalize SUBMISSION_ID                                   # teacher accepts the grade
huc export bio-101-midterm-1 --out grades.csv
```

`grade` picks up submissions in `ingested` or `failed` status, so re-running it
retries failures. Use `huc --help` / `huc <command> --help` for details.

## API

```bash
uvicorn huc.api.app:app --reload     # docs at http://localhost:8000/docs
```

| Method | Path | Description |
|---|---|---|
| POST | `/exams` | Upload an exam YAML (multipart `file`) |
| GET | `/exams` | List exams |
| POST | `/exams/{exam_id}/submissions` | Upload and ingest a submission (`file`, optional `?student_id=`) |
| GET | `/exams/{exam_id}/submissions` | List submissions of an exam |
| POST | `/exams/{exam_id}/grade` | Grade pending submissions in the background (202) |
| GET | `/submissions/{id}` | Get a submission (status, answers) |
| GET | `/submissions/{id}/grade` | Get the grade |
| POST | `/submissions/{id}/finalize` | Finalize after teacher review (204) |
| GET | `/exams/{exam_id}/export.csv` | Export grades as CSV |

Unknown IDs return 404; invalid input (bad YAML, finalizing an ungraded
submission, unsupported file type) returns 422.

## LangGraph Studio

`langgraph.json` at the repo root registers both graphs (`ingestion` and
`grading`), so you can run, inspect and debug them visually:

```bash
langgraph dev        # installed with the dev extras (langgraph-cli[inmem])
```

This opens LangGraph Studio in the browser. Invoke the graphs with the same
input as the code does:

- `ingestion`: `{"exam_id": "...", "file_path": "examples/submission_lms.json"}`
- `grading`: `{"submission_id": "..."}`

The graphs read and write the same `HUC_DATA_DIR`, so load an exam first
(`huc exam-load ...`). `langgraph dev` reads `.env`.

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
  ingestion/loaders.py    PDF / image / LMS JSON -> pages
  prompts/                prompt templates (extractor, grader, reviewer)
  graphs/ingestion.py     ingestion graph: file -> Submission
  graphs/grading.py       grading graph: fan-out per criterion -> review -> finalize
  storage/repository.py   Repository (port) + FileRepository (JSON files)
  config.py               Settings (env vars, HUC_ prefix)
  services.py             use cases shared by CLI and API (thin layer over the graphs)
  cli.py                  Typer CLI
  api/app.py              FastAPI app
langgraph.json            graph registry for `langgraph dev` / Studio
examples/                 example exam YAML and LMS submission
tests/                    fake LLM, no network
```

## Tests

```bash
pytest
ruff check .
```

Tests inject a fake chat model, so they need neither network access nor an API key.

# HUC - Exam grading with LLM agents

Research pipeline. Ingests handwritten student exams in Spanish (scanned PDFs or photos),
reads them with **Claude Haiku** (all pages in one call, split into one answer per
question), and grades them with one Claude agent per rubric criterion and a reviewer agent
per question that also writes the feedback. Every exam is graded; low-confidence grades
are labelled **needs teacher verification**, with the reasons.
An additional configurable model audits the mathematical feedback. It can share the
grader's failure modes and does not certify correctness.

The feedback and score are research outputs, not verified mathematical facts. The pipeline
flags omitted or blank OCR answers, missing or nonliteral evidence, empty feedback and
unresolved reviews for inspection. A flag does not prevent generating a result, and lack of
a flag does not prove correctness. A reference answer is optional; course rules and rubric
criteria should describe the mathematical outcome rather than mandate a single method.
See [the synthetic research evaluation](research/synthetic/README.md) for repeatability
experiments and a feedback assessment rubric; it does not measure learning impact.
Page images are sent to the Anthropic API for OCR; the scans contain only answers (no
student names or IDs). Built on [LangGraph](https://langchain-ai.github.io/langgraph/).
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the full technical reference in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env        # set ANTHROPIC_API_KEY
```

All run settings are environment variables with the `HUC_` prefix (Claude model per role,
OCR model, `HUC_DATA_DIR`, `HUC_PROFILES_DIR`, `HUC_MAX_CONCURRENCY`,
`HUC_VERIFICATION_THRESHOLD`, ...); see `.env.example`.

### Course profiles

Course parameters live in `profiles/<id>.yaml` and an exam YAML selects one with
`profile: <id>`: course and subject context, grading rules, error types, feedback policy
(language, structure, tone, good and forbidden comments) and the grade bands of the export.
Questions can name a `subject`, the `skill` they assess and `constraints` the answer must
respect. `profiles/mat0004.yaml` holds what the MAT0004 teacher gave us; see
`examples/taller_s14_m1.yaml` and [docs/TECHNICAL.md §8.1](docs/TECHNICAL.md).

### Reading the scans (Claude Haiku)

`claude-haiku-5-5` (`HUC_OCR_MODEL`) receives every page image of a submission plus the
exam questions in **one** API call and returns, per question, the verbatim answer, a
confidence (0-1) and the source pages. Seeing all pages at once lets it join answers that
continue on another page and cope with pages out of order, rotated or repeated. PDFs are
rasterized at `HUC_PDF_DPI` (default 200) and every image is capped at
`HUC_OCR_MAX_SIDE_PX` (default 1568 px, the largest size Claude uses without downscaling).
Transient errors are retried by the SDK; if a long submission times out, raise
`HUC_OCR_TIMEOUT_S` (default 300 s).

The scans carry no names: the student ID is the file name up to `__`
(`Perez_Gomez_Ana__Taller_3.pdf` → `Perez_Gomez_Ana`), or `--student-id`.

Plain per-page OCR, without grading:

```bash
huc ocr scans/ana-perez.pdf     # prints the transcription of each page and its time
```

To measure OCR accuracy against reference transcriptions (CER/WER per page), use
`scripts/ocr_eval.py` (`export`, `run`, `score`); its data lives in the gitignored
`data/_ocr_eval/`.

Illegible words come out as `[illegible]`; crossed-out text is dropped.

## CLI

```bash
huc rubric-draft data/course/talleres/S14/soluciones_S14.pdf -m M1 --profile mat0004
                                                             # draft data/rubrics/s14-m1.yaml
huc exam-load examples/exam_example.yaml                     # exam id: bio-101-midterm-1
huc ocr scans/ana-perez.pdf                                  # OCR only, prints each page
huc ingest bio-101-midterm-1 scans/ana-perez.pdf             # one file (--student-id to force it)
huc ingest bio-101-midterm-1 ./scans/                        # every PDF/PNG/JPG in a folder
huc grade bio-101-midterm-1                                  # grade all pending submissions
huc show SUBMISSION_ID                                       # feedback, justification, evidence
huc report bio-101-midterm-1                                 # frequent errors, means, bands (--json)
huc export bio-101-midterm-1 --out grades.csv                # points, %, band, errors, feedback, verification
```

`rubric-draft` slices a reference-solutions PDF by its `Taller S14M1` / `Taller 4, M1`
headers and has Claude (`HUC_RUBRIC_MODEL`) turn each module's section (rendered pages)
into questions and criteria, following the profile's `rubric` policy. Drafts are not
loaded: review them, then `exam-load`. Without `-m` it drafts every module of the file.

`grade` picks up submissions in `ingested` or
`failed` status, so re-running it retries failures. Use `huc --help` / `huc <command> --help` for details.

Every grading run stores its exam and submission snapshots, initial and reviewed scores,
prompt text and hash, configured models and package versions. The local file repository
keeps previous runs under `data/grade_history/<submission-id>/`; `GET
/submissions/{id}/grade-history` returns them. These files are local research records,
not tamper-proof or a replacement for controlled archival storage.

CSV exports omit feedback flagged for review (and legacy or unaudited feedback) by default;
per-question columns identify withheld drafts. Researchers can inspect all drafts via
`huc show`, the grade API or `huc export --include-flagged-feedback`. An unflagged draft
is still not a validated mathematical result. Reports reject mixed or missing exam snapshots, and
CSV percentages use the snapshot that actually produced each grade.

## API

```bash
uvicorn huc.api.app:app --reload     # docs at http://localhost:8000/docs
```

| Method | Path | Description |
|---|---|---|
| POST | `/exams` | Upload an exam YAML (multipart `file`) |
| GET | `/exams` | List exams |
| POST | `/exams/{exam_id}/submissions` | Upload and ingest a scanned PDF / PNG / JPG (`file`, optional `?student_id=`, else taken from the uploaded file name) |
| GET | `/exams/{exam_id}/submissions` | List submissions of an exam |
| POST | `/exams/{exam_id}/grade` | Grade pending submissions in the background (202) |
| GET | `/submissions/{id}` | Get a submission (status, answers) |
| GET | `/submissions/{id}/grade` | Get the grade |
| GET | `/exams/{exam_id}/report` | Frequent errors (per error type, with examples), means per question/criterion, bands |
| GET | `/exams/{exam_id}/export.csv` | Export grades as CSV |

Unknown IDs return 404; invalid input (bad YAML, unsupported file type) returns 422; the OCR model unavailable (invalid
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
  shows the `submission` with its answers)
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
  domain/profile.py       CourseProfile: course-level grading parameters
  ingestion/loaders.py    scanned PDF / photo -> page images
  ocr.py                  OCR adapter: Claude Haiku reads a whole exam (and plain page OCR)
  prompts/                prompt templates (ocr/reader, grader, reviewer)
  graphs/ingestion.py     ingestion graph: file -> read_exam (Haiku) -> Submission
  graphs/grading.py       grading graph: fan-out per criterion -> review -> finalize
  storage/repository.py   Repository (port) + FileRepository (JSON files)
  config.py               Settings (env vars, HUC_ prefix)
  rubrics.py              solutions PDF -> sections -> drafted exam YAML (Claude)
  report.py               frequent errors and score summary of an exam
  services.py             use cases shared by CLI and API (thin layer over the graphs)
  cli.py                  Typer CLI
  api/app.py              FastAPI app
langgraph.json            graph registry for `langgraph dev` / Studio
examples/                 example exam YAMLs (generic; real taller S14 M1 with profile mat0004)
profiles/                 course profiles (context, grading/feedback policy, scale)
data/course/              course material: workshop statements + solutions per week, class notes,
                          TA lists (gitignored: copyrighted and personal data; see its README.md)
tests/                    fake LLM, no network
```

## Tests

```bash
pytest
ruff check .
```

Tests inject fake Claude and OCR models and generate scans in a temp dir, so they need
neither network access nor an API key.

## Workshop process records

A workshop session preserves one student's full process for an exam independently of
its latest grade. Create it with `POST /exams/{exam_id}/workshops` and JSON
`{"student_id": "student-01"}`. List sessions at `GET /exams/{exam_id}/workshops`;
retrieve the complete chronological record at `GET /workshops/{session_id}`.

Append a record with `POST /workshops/{session_id}/events`, for example:

```json
{"stage": "initial_solution", "actor": "student", "text": "My initial proof..."}
```

Stages are `initial_solution`, `initial_diagnosis`, `student_question`,
`socratic_response`, `reflection`, `final_solution`, `final_diagnosis`, and
`teacher_note`. Each event retains server-assigned ID, timestamp and sequence. Records
are appended through the API; there are no update/delete endpoints. Local JSON storage
is not tamper-proof. Writes use a local filesystem lock and atomic replacement.

Use `diagnosis` for structured Toulmin/SOLO observations and supporting evidence,
`questions` for the exact socratic questions shown, `text` for questions, solutions,
reflections or notes, and `provenance` for exact prompts, models and configuration.
`actor` identifies who supplied the record; it is caller-provided, not authentication.
`question_id` optionally identifies an exam question. The initial/final diagnoses are
session-level records whose structured content can cover each question.

Solutions can include `submission_id` instead of text to retain a snapshot of an existing
OCR submission. An optional `grade_id` with that submission captures a particular grading
run from its history. The service checks student/exam ownership and grade rubric consistency.
The exam definition is copied when the session is created. Subsequent edits or regrading
do not change saved event snapshots.

Initial diagnosis requires the initial solution. Final solution requires the initial
diagnosis; final diagnosis requires the final solution. Each of those stages is recorded
once. Socratic responses require the initial diagnosis: at most three questions in the
first response, and two in a second response after a new student question. Student
interaction closes when the final solution is recorded. Teacher notes remain available.

This feature records supplied observations and interactions; it does not yet generate
Toulmin/SOLO diagnoses or socratic questions, and does not infer learning from changes
between initial and final work. The existing ingestion and grading commands remain usable
when preparing submissions and grades to attach to a session.

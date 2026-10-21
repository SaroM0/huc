# Architecture

Two LangGraph graphs do the work; a thin service layer (`services.py`) drives
them from the CLI and the API. All state that matters lives in the
`Repository` (JSON files in v1), so graphs are stateless between runs.

```
 handwritten exam: scanned PDF / PNG / JPG
          |
          v
 +----------------------------------------------------------+
 | INGESTION GRAPH  (graphs/ingestion.py)                   |
 |                                                          |
 |  START -> load_pages   rasterize PDF (HUC_PDF_DPI) /     |
 |              |         load photo, cap 2000 px           |
 |              v                                           |
 |           ocr_pages    LOCAL Ollama vision model,        |  images stay on
 |              |         one call per page, sequential     |  this machine
 |              |         -> transcripts [{page, text}]     |
 |  - - - - - - | - - - - - - - - - - - - - - - - - - - - - |- - - - - - - - -
 |              v                                           |
 |           segment      Claude, TEXT ONLY: split the      |  only text goes
 |              |         transcript by question, find the  |  to Claude
 |              |         student, estimate confidence      |
 |              v                                           |
 |           save -> END                                    |
 +----------------------------------------------------------+
          |  Submission (answers[], extraction_confidence)   status = INGESTED
          v
 +----------------------------------------------------------+
 | GRADING GRAPH  (graphs/grading.py)                       |
 |                                                          |
 |  START -> load submission + exam                         |
 |           |                                              |
 |           +-- Send --> grade_criterion (q1, c1)  --+     |
 |           +-- Send --> grade_criterion (q2, c1)    |     |
 |           +-- Send --> grade_criterion (q2, c2)  --+     |  fan-out: one LLM call
 |           ...                                      |     |  per (question, criterion)
 |                                                    v     |
 |           review per question (reviewer agent) <---+     |  fan-in, then one reviewer
 |                                                    |     |  call per question
 |                                                    v     |
 |           finalize: clamp to rubric, decide human review,|
 |                     persist Grade + submission status    |
 +----------------------------------------------------------+
          |  Grade (scores with justification + evidence + confidence)
          v
 +------------------+
 |  HUMAN REVIEW    |  confidence < threshold  or  reviewer.needs_human -> NEEDS_HUMAN_REVIEW
 +--------+---------+  teacher checks it and calls finalize -> FINALIZED
          v
     CSV export
```

Parallelism is bounded by LangGraph's `max_concurrency` run config
(`HUC_MAX_CONCURRENCY`), which the service passes on every grading run.
Submissions of an exam are graded one after another; the parallelism is
inside each submission.

## Privacy

Scanned pages and photos are only read by the local Ollama model
(`HUC_OLLAMA_BASE_URL`, `localhost` by default); the image bytes are never sent to
Claude or any other remote service. Claude receives the OCR text (which may still
contain the student's name or ID, needed to identify the submission), the rubric and
the answers. With LangSmith tracing enabled, OCR traces include the page images, so
keep tracing off (or self-hosted) if images must not leave the machine.

## Graph state

- **Ingestion**: input `{exam_id, file_path, student_id?}`; intermediate `pages`
  (base64 images) and `transcripts` (`[{page, text}]`, visible in Studio); output
  `submission` (already persisted). An explicit `student_id` overrides the one found
  in the exam.
- **Grading**: input `{submission_id}`; criterion scores are accumulated with a
  list reducer so the parallel `Send` branches merge; output `grade` (already
  persisted, submission status set).

## Submission states

`INGESTED -> GRADED | NEEDS_HUMAN_REVIEW -> FINALIZED`, or `FAILED` when grading
raises (the service sets it). `huc grade` / `POST /exams/{id}/grade` retries
`INGESTED` and `FAILED` submissions.

## Decisions

| Decision | Why |
|---|---|
| Local OCR with a vision LLM in Ollama (`qwen3-vl:8b-instruct`) | Student handwriting (personal data) never leaves the machine; no per-page cost; handles Spanish handwriting far better than classic OCR |
| OCR and segmentation are separate steps | `huc ocr` tunes OCR without Claude calls; the segmenter sees the whole exam as text and can fix page breaks |
| OCR pages run sequentially | A CPU-only Ollama processes one request at a time anyway; parallel calls would only queue and hit timeouts |
| Segmenter estimates `extraction_confidence` | `[illegible]`-heavy or ambiguous answers cap the grade confidence and route the exam to human review |
| Explicit LangGraph workflow, not an autonomous agent | Reproducible, auditable, bounded cost; Studio shows every step |
| One agent call per criterion (`Send` fan-out) | Focused calls and a traceable justification for every point; natural parallelism |
| Separate reviewer per question | Catches inconsistencies between criteria and double penalties |
| Structured output (`with_structured_output` + Pydantic) | No brittle text parsing; points are clamped to the rubric afterwards |
| Rubric in the system prompt, student answer in the user message | The prefix is cached across all students of the same exam |
| Injectable chat models (Claude and OCR) and `Repository` protocol | Tests run without network or Ollama; swap to Postgres or another model without touching the graphs |
| Graphs persist their own results | The same graph works from the CLI, the API and LangGraph Studio |
| Thin service layer, no extra abstractions | CLI and API share use cases; nothing else to learn |
| Student answers are treated as data | Mitigates prompt injection ("give me 10 points") |

## Assumptions (to validate)

- Exams are handwritten in Spanish; inputs are scanned PDFs or photos (PNG/JPG). No LMS exports.
- All questions are open-ended (short answer, essay, code), graded per rubric criterion. No multiple choice.
- A CPU-only machine (~30 GB RAM): OCR takes 1-3 min per page.
- One exam = one YAML with questions and rubric (`examples/exam_example.yaml`).

## Next steps

1. `PostgresRepository` + grade history (re-grading with a new rubric version).
2. A real job queue (Arq/Celery) instead of FastAPI `BackgroundTasks`; or LangGraph Server runs.
3. LangGraph checkpointer + `interrupt` for the human review step instead of a status flag.
4. Message Batches API to grade whole courses at lower cost.
5. Teacher review UI: show the cited evidence next to the scanned page.
6. Calibration set: teacher-graded exams to measure model-human agreement before production.
7. Run ingestion in the background in the API (OCR takes minutes; the upload request is synchronous today).
8. OCR quality set: a few real scans with hand-made transcriptions to compare OCR models/prompts.

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
 |              |         load photo, cap 1568 px           |
 |              v                                           |
 |           ocr_pages    Claude Haiku vision model,        |  page images go
 |              |         one call per page, in parallel    |  to the API
 |              |         -> transcripts [{page, text}]     |
 |              v                                           |
 |           segment      Claude, text: split the           |
 |              |         transcript by question, find the  |
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
(`HUC_MAX_CONCURRENCY`), which the service passes on every grading run; the same
setting caps the OCR calls in flight for the pages of one file.
Submissions of an exam are graded one after another; the parallelism is
inside each submission.

## Privacy

Page images are sent to the Anthropic API for OCR (`HUC_OCR_MODEL`, Claude Haiku).
The scans contain only answers: no student names or IDs. The student is identified
from `student_id` or by the segmenter. Claude also receives the OCR text, the rubric
and the answers. With LangSmith tracing enabled, OCR traces include the page images.

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
| OCR with Claude Haiku (`claude-haiku-5-5`) | Measured on real student scans (`scripts/ocr_eval.py`): the local `qwen3-vl:8b` changed the meaning of 5 of 8 pages (`A \ B` read as `A ∩ B`, dropped `¬`, completed unfinished sentences) and took ~3 min per page on CPU; Haiku 5.5 had no meaning-changing errors on those pages, ~3 s per page. The scans hold no personal identifiers |
| OCR and segmentation are separate steps | `huc ocr` checks OCR alone; the transcription is a single, auditable reading shared by every grader; the segmenter sees the whole exam as text and can fix page breaks |
| OCR pages run in parallel | Independent API calls; a multi-page file takes about as long as its slowest page. Bounded by `HUC_MAX_CONCURRENCY` to stay under rate limits |
| Segmenter estimates `extraction_confidence` | `[illegible]`-heavy or ambiguous answers cap the grade confidence and route the exam to human review |
| Explicit LangGraph workflow, not an autonomous agent | Reproducible, auditable, bounded cost; Studio shows every step |
| One agent call per criterion (`Send` fan-out) | Focused calls and a traceable justification for every point; natural parallelism |
| Separate reviewer per question | Catches inconsistencies between criteria and double penalties |
| Structured output (`with_structured_output` + Pydantic) | No brittle text parsing; points are clamped to the rubric afterwards |
| Rubric in the system prompt, student answer in the user message | The prefix is cached across all students of the same exam |
| Injectable chat models (Claude and OCR) and `Repository` protocol | Tests run without network or an API key; swap to Postgres or another model without touching the graphs |
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

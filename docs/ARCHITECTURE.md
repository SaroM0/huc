# Architecture

Two LangGraph graphs do the work; a thin service layer (`services.py`) drives
them from the CLI and the API. All state that matters lives in the
`Repository` (JSON files in v1), so graphs are stateless between runs.

```
 PDF / image / LMS JSON
          |
          v
 +----------------------------------------------------------+
 | INGESTION GRAPH  (graphs/ingestion.py)                   |
 |                                                          |
 |  START -> load file --+-- LMS JSON --------> build ------+--> persist -> END
 |                       |   (no LLM, confidence = 1.0)     |
 |                       +-- PDF / image -> extract (vision)|
 |                           transcribe + split by question |
 +----------------------------------------------------------+
          |  Submission (answers[], extraction_confidence)   status = INGESTED
          v
 +----------------------------------------------------------+
 | GRADING GRAPH  (graphs/grading.py)                       |
 |                                                          |
 |  START -> load submission + exam                         |
 |           |  multiple choice -> deterministic score      |
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
     CSV export / LMS
```

Parallelism is bounded by LangGraph's `max_concurrency` run config
(`HUC_MAX_CONCURRENCY`), which the service passes on every grading run.
Submissions of an exam are graded one after another; the parallelism is
inside each submission.

## Graph state

- **Ingestion**: input `{exam_id, file_path, student_id?}`; output `submission`
  (already persisted).
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
| Explicit LangGraph workflow, not an autonomous agent | Reproducible, auditable, bounded cost; Studio shows every step |
| One agent call per criterion (`Send` fan-out) | Focused calls and a traceable justification for every point; natural parallelism |
| Separate reviewer per question | Catches inconsistencies between criteria and double penalties |
| Structured output (`with_structured_output` + Pydantic) | No brittle text parsing; points are clamped to the rubric afterwards |
| Rubric in the system prompt, student answer in the user message | The prefix is cached across all students of the same exam |
| Injectable chat model and `Repository` protocol | Tests run without network; swap to Postgres or another model without touching the graphs |
| Graphs persist their own results | The same graph works from the CLI, the API and LangGraph Studio |
| Thin service layer, no extra abstractions | CLI and API share use cases; nothing else to learn |
| Student answers are treated as data | Mitigates prompt injection ("give me 10 points") |

## Assumptions (to validate)

- Input formats: PDF (digital or scanned), PNG/JPG and LMS JSON.
- Question types: multiple choice, short answer, essay and code (code uses the same rubric flow).
- One exam = one YAML with questions and rubric (`examples/exam_example.yaml`).

## Next steps

1. `PostgresRepository` + grade history (re-grading with a new rubric version).
2. A real job queue (Arq/Celery) instead of FastAPI `BackgroundTasks`; or LangGraph Server runs.
3. LangGraph checkpointer + `interrupt` for the human review step instead of a status flag.
4. Message Batches API to grade whole courses at lower cost.
5. Teacher review UI: show the cited evidence next to the scanned page.
6. Calibration set: teacher-graded exams to measure model-human agreement before production.
7. Long essays: split many-page PDFs before extraction.

# Architecture

A research pipeline, not a production grading system. Two LangGraph graphs do the work; a thin service layer (`services.py`) drives
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
 |           read_exam    Claude Haiku vision model,        |  page images go
 |              |         ONE call with all pages + the     |  to the API
 |              |         questions -> verbatim answer,     |
 |              |         confidence, pages per question    |
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
 |             adjusts scores, writes feedback        |     |  call per question
 |                                                    v     |
 |           finalize: clamp to rubric, label               |
 |                     needs_verification, persist Grade    |
 +----------------------------------------------------------+
          |  Grade (scores + justification + evidence + confidence, feedback per question,
          |         needs_verification + reasons)                status = GRADED
          v
     CSV export   (confidence < threshold or reviewer flag -> "needs teacher verification")
```

Parallelism is bounded by LangGraph's `max_concurrency` run config
(`HUC_MAX_CONCURRENCY`), which the service passes on every grading run.
Submissions of an exam are graded one after another; the parallelism is
inside each submission.

## Privacy

Page images are sent to the Anthropic API for OCR (`HUC_OCR_MODEL`, Claude Haiku).
The scans contain only answers: no student names or IDs. The student is identified
from `student_id` or the file name (`<student>__<anything>.pdf`). Claude also receives
the rubric and the transcribed answers. With LangSmith tracing enabled, OCR traces include the page images.

## Graph state

- **Ingestion**: input `{exam_id, file_path, student_id?}`; intermediate `pages`
  (base64 images); output `submission` (already persisted, answers visible in Studio).
  An explicit `student_id` overrides the one taken from the file name.
- **Grading**: input `{submission_id}`; criterion scores are accumulated with a
  list reducer so the parallel `Send` branches merge; output `grade` (already
  persisted, submission status set).

## Submission states

`INGESTED -> GRADING -> GRADED`, or `FAILED` when grading raises (the service sets
it). There is no human step: "needs teacher verification" is a label on the `Grade`
(`needs_verification` + `verification_reasons`), not a state that blocks it. `huc grade` / `POST /exams/{id}/grade` retries
`INGESTED` and `FAILED` submissions.

## Decisions

| Decision | Why |
|---|---|
| OCR with Claude Haiku (`claude-haiku-5-5`) | Measured on real student scans (`scripts/ocr_eval.py`): the local `qwen3-vl:8b` changed the meaning of 5 of 8 pages (`A \ B` read as `A ∩ B`, dropped `¬`, completed unfinished sentences) and took ~3 min per page on CPU; Haiku 5.5 had no meaning-changing errors on those pages, ~3 s per page. The scans hold no personal identifiers |
| OCR and segmentation in one Haiku call per submission | Seeing every page at once, the model joins answers across pages and handles pages out of order, rotated or repeated; one call instead of N pages + an Opus segmentation. `huc ocr` keeps plain per-page OCR for `scripts/ocr_eval.py` |
| Reader estimates `extraction_confidence` | `[illegible]`-heavy or ambiguous answers cap the grade confidence and label the grade for verification |
| No teacher step; a verification label instead | Research pipeline: every exam must be graded with feedback; the label and its reasons say which grades a teacher should check |
| Reviewer writes the feedback | It already sees the answer, the whole rubric and the final scores, so feedback costs no extra call and agrees with the grade |
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
- One exam = one YAML with questions and rubric (`examples/exam_example.yaml`).

## Next steps

1. Evaluate whole-exam reading (answer text + question assignment), not only per-page OCR.
2. Calibration set: teacher-graded exams to measure model-teacher agreement.
3. Message Batches API to grade whole courses at lower cost.

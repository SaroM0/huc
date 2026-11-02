"""Command-line interface: huc rubric-draft | exam-load | ocr | ingest | grade | show |
report | export."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from huc.domain.models import Grade
from huc.ocr import OcrUnavailableError
from huc.services import HucService

VERIFY_LABEL = "  [needs teacher verification]"

app = typer.Typer(help="Exam ingestion and grading with LLM agents.", no_args_is_help=True)


def get_service() -> HucService:
    """Service factory. Tests monkeypatch this to inject a fake LLM / temp repo."""
    return HucService.default()


def _fail(message: str) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


@app.command("rubric-draft")
def rubric_draft(
    solutions: Annotated[Path, typer.Argument(help="Reference-solutions PDF of one week.")],
    module: Annotated[
        list[str] | None, typer.Option("--module", "-m", help="Module, e.g. M1 (repeatable).")
    ] = None,
    profile: Annotated[str | None, typer.Option(help="Course profile id, e.g. mat0004.")] = None,
    week: Annotated[int | None, typer.Option(help="Week (default: from the file).")] = None,
    out: Annotated[Path, typer.Option(help="Directory for the YAML drafts.")] = Path(
        "data/rubrics"
    ),
) -> None:
    """Draft exam YAMLs (questions + rubric) from the teacher's solutions, one per module.

    The drafts are not loaded: review them, then `huc exam-load <draft>`.
    """
    try:
        drafts = asyncio.run(
            get_service().draft_rubrics(solutions, module, profile_id=profile, week=week)
        )
    except (FileNotFoundError, ValueError) as e:
        _fail(str(e))
    out.mkdir(parents=True, exist_ok=True)
    for exam, text in drafts:
        path = out / f"{exam.id}.yaml"
        path.write_text(text, "utf-8")
        points = sum(q.max_points for q in exam.questions)
        typer.echo(f"{path}  ({len(exam.questions)} questions, {points:g} points)")


@app.command("exam-load")
def exam_load(path: Annotated[Path, typer.Argument(help="Exam + rubric YAML file.")]) -> None:
    """Load an exam and its rubric from a YAML file."""
    try:
        exam = get_service().load_exam_yaml(path)
    except (FileNotFoundError, ValueError) as e:
        _fail(str(e))
    typer.echo(f"Exam loaded: {exam.id} ({exam.title})")


@app.command()
def ocr(path: Annotated[Path, typer.Argument(help="A scanned PDF or a photo (PNG/JPG).")]) -> None:
    """Transcribe each page of a file separately and print it (no grading, nothing saved)."""

    async def run() -> None:
        start = time.monotonic()
        async for number, text in get_service().ocr_file(path):
            elapsed = time.monotonic() - start
            typer.echo(f"--- page {number} ({elapsed:.0f}s) ---\n{text}\n")
            start = time.monotonic()

    try:
        asyncio.run(run())
    except (FileNotFoundError, ValueError, OcrUnavailableError) as e:
        _fail(str(e))


@app.command()
def ingest(
    exam_id: Annotated[str, typer.Argument(help="Exam ID.")],
    path: Annotated[Path, typer.Argument(help="A scanned PDF / PNG / JPG, or a directory.")],
    student_id: Annotated[str | None, typer.Option(help="Student ID (single file only).")] = None,
) -> None:
    """Ingest a submission file or every supported file in a directory."""
    svc = get_service()
    try:
        if path.is_dir():
            subs = asyncio.run(svc.ingest_dir(exam_id, path))
        else:
            subs = [asyncio.run(svc.ingest_file(exam_id, path, student_id))]
    except (FileNotFoundError, ValueError, OcrUnavailableError) as e:
        _fail(str(e))
    for s in subs:
        typer.echo(f"{s.id}  student={s.student_id}  answers={len(s.answers)}")
    typer.echo(f"Ingested {len(subs)} submission(s).")


@app.command()
def grade(exam_id: Annotated[str, typer.Argument(help="Exam ID.")]) -> None:
    """Grade every pending (ingested or failed) submission of an exam."""
    try:
        grades = asyncio.run(get_service().grade_exam(exam_id))
    except (FileNotFoundError, ValueError) as e:
        _fail(str(e))
    for g in grades:
        flag = VERIFY_LABEL if g.needs_verification else ""
        typer.echo(f"{g.submission_id}  total={g.total:.2f}{flag}")
    typer.echo(f"Graded {len(grades)} submission(s).")


@app.command()
def show(submission_id: Annotated[str, typer.Argument(help="Submission ID.")]) -> None:
    """Show a grade with feedback and per-criterion justification and evidence."""
    svc = get_service()
    try:
        submission = svc.repo.get_submission(submission_id)
    except FileNotFoundError:
        _fail(f"submission {submission_id} not found")
    grade_ = svc.repo.get_grade(submission_id)
    if grade_ is None:
        _fail(f"submission {submission_id} has no grade yet (status: {submission.status})")
    exam = svc.repo.get_exam(submission.exam_id)
    typer.echo(_format_grade(grade_, exam, submission.student_id, submission.status))


def _format_grade(grade_: Grade, exam, student_id: str, status: str) -> str:
    max_total = sum(q.max_points for q in exam.questions)
    lines = [
        f"Submission {grade_.submission_id}  student={student_id}  status={status}",
        f"Exam {exam.id} ({exam.title})  rubric v{grade_.rubric_version}",
        f"Total: {grade_.total:.2f} / {max_total:g}"
        + (VERIFY_LABEL if grade_.needs_verification else ""),
    ]
    lines += [f"  ? {reason}" for reason in grade_.verification_reasons]
    questions = {q.id: q for q in exam.questions}
    for qg in grade_.questions:
        q = questions.get(qg.question_id)
        label = f"Q{q.number}" if q else qg.question_id
        max_q = f" / {q.max_points:g}" if q else ""
        lines += ["", f"{label}: {qg.points:.2f}{max_q}  (graded by {qg.graded_by})"]
        if qg.feedback:
            review_label = " [requires review; research draft]" if qg.requires_feedback_review() else " [unflagged draft]"
            lines.append(f"  Feedback{review_label}: {qg.feedback}")
        for s in qg.scores:
            lines.append(
                f"  - {s.criterion_id}: {s.points:g}  (confidence {s.confidence:.2f})"
                + (f"  [{s.error_type}]" if s.error_type else "")
            )
            lines.append(f"      {s.justification}")
            for quote in s.evidence:
                lines.append(f'      > "{quote}"')
        if qg.review is not None:
            verdict = "approved" if qg.review.approved else "not approved"
            verify = ", needs verification" if qg.review.needs_verification else ""
            lines.append(f"  Review: {verdict}{verify}")
            for issue in qg.review.issues:
                lines.append(f"    ! {issue}")
    return "\n".join(lines)


@app.command()
def report(
    exam_id: Annotated[str, typer.Argument(help="Exam ID.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Frequent errors and score summary over the graded submissions of an exam."""
    try:
        rep = get_service().report(exam_id)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
    if as_json:
        typer.echo(rep.model_dump_json(indent=2))
        return
    mean = f"{rep.mean_percent:g} %" if rep.mean_percent is not None else "-"
    lines = [
        f"Exam {rep.exam_id} ({rep.title}): {rep.graded}/{rep.submissions} graded, "
        + f"mean {mean}, {rep.needs_verification} need verification"
    ]
    if rep.bands:
        lines.append("Bands: " + ", ".join(f"{b}={n}" for b, n in rep.bands.items()))
    lines += _format_errors(rep.errors, "Frequent errors")
    for q in rep.questions:
        lines += ["", f"Q{q.number}: mean {q.mean_points:g} / {q.max_points:g} ({q.graded} graded)"]
        lines += [
            f"  - {c.criterion_id}: {c.mean_points:g} / {c.max_points:g}" for c in q.criteria
        ]
        lines += _format_errors(q.errors, "  Errors", indent="  ")
    typer.echo("\n".join(lines))


def _format_errors(errors, title: str, indent: str = "") -> list[str]:
    if not errors:
        return []
    lines = [f"{title}:"]
    for e in errors:
        desc = f" ({e.description})" if e.description else ""
        lines.append(f"{indent}  {e.error_type}: {e.count}{desc}")
        lines += [f'{indent}      > "{ex}"' for ex in e.examples]
    return lines


@app.command()
def export(
    exam_id: Annotated[str, typer.Argument(help="Exam ID.")],
    out: Annotated[Path, typer.Option(help="Output CSV path.")] = Path("grades.csv"),
    include_flagged_feedback: Annotated[bool, typer.Option(
        "--include-flagged-feedback", help="Include flagged research drafts in the CSV."
    )] = False,
) -> None:
    """Export the grades of an exam as CSV."""
    try:
        csv_text = get_service().export_csv(exam_id, include_flagged_feedback=include_flagged_feedback)
    except FileNotFoundError:
        _fail(f"exam {exam_id} not found")
    out.write_text(csv_text, "utf-8")
    typer.echo(f"Exported to {out}")


if __name__ == "__main__":
    app()

"""Command-line interface: huc exam-load | ocr | ingest | grade | show | export."""

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
            lines.append(f"  Feedback: {qg.feedback}")
        for s in qg.scores:
            lines.append(
                f"  - {s.criterion_id}: {s.points:g}  (confidence {s.confidence:.2f})"
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
def export(
    exam_id: Annotated[str, typer.Argument(help="Exam ID.")],
    out: Annotated[Path, typer.Option(help="Output CSV path.")] = Path("grades.csv"),
) -> None:
    """Export the grades of an exam as CSV."""
    try:
        csv_text = get_service().export_csv(exam_id)
    except FileNotFoundError:
        _fail(f"exam {exam_id} not found")
    out.write_text(csv_text, "utf-8")
    typer.echo(f"Exported to {out}")


if __name__ == "__main__":
    app()

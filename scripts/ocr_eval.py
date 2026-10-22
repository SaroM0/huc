"""Evaluate the OCR (production adapter, `huc.ocr`) against reference transcriptions.

Every submission file under the data dir (skipping SOLUCION_* and the eval dir itself) is
loaded with the production loader, so the model sees exactly the page images it would see
in `huc ingest`. Outputs contain student work, so they live under data/ (gitignored).

    python scripts/ocr_eval.py export              # page images + empty ground-truth files
    python scripts/ocr_eval.py run                 # OCR every page (resumable)
    python scripts/ocr_eval.py score               # metrics vs ground truth -> report.md / .csv

A run is named after HUC_OCR_MODEL and HUC_OCR_MAX_SIDE_PX (e.g. claude-haiku-5-5__1568px),
so runs with different settings sit side by side and can be scored separately.
Older runs from the former local model (qwen3-vl) stay under runs/ and can still be scored.

Layout (under data/_ocr_eval/):
    pages/<doc>/pN.<ext>          image sent to the model
    ground_truth/<doc>/pN.txt     reference transcription (see GROUND_TRUTH.md)
    runs/<run>/<doc>/pN.txt       model output; runs/<run>/timings.json, config.json
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Annotated

import typer

from huc.config import get_settings
from huc.ingestion.loaders import SUPPORTED_SUFFIXES, load_document
from huc.ocr import get_ocr_model, transcribe_page
from huc.prompts import ocr as prompts

app = typer.Typer(add_completion=False, no_args_is_help=True)

EVAL_DIRNAME = "_ocr_eval"
EXT = {"image/png": "png", "image/jpeg": "jpg"}


def _eval_dir() -> Path:
    return get_settings().data_dir / EVAL_DIRNAME


def _submission_files() -> list[Path]:
    data = get_settings().data_dir
    return sorted(
        p for p in data.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith("SOLUCION_")
        and EVAL_DIRNAME not in p.parts
    )


def _doc_id(path: Path) -> str:
    """`Semana_03_Modulo_M2__Martinez_Hoffmann_Ignacio__Taller_Mat_S3M2_1_`."""
    return f"{path.parent.name}__{path.stem}"


def _run_name() -> str:
    settings = get_settings()
    model = settings.ocr_model.replace(":", "_").replace("/", "_")
    return f"{model}__{settings.ocr_max_side_px}px"


def _run_dir(run: str) -> Path:
    return _eval_dir() / "runs" / run


# --- Text normalization and metrics -------------------------------------------------------

_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ", "0123456789+-n")
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₙₖⱼ", "0123456789nkj")
_SUP_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ]+")
_SUB_RUN = re.compile("[₀₁₂₃₄₅₆₇₈₉ₙₖⱼ]+")
# Equivalent spellings the model and the reference may choose differently.
_EQUIV = {
    "−": "-", "–": "-", "—": "-", "‐": "-",
    "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "´": "'",
    "⟹": "⇒", "=>": "⇒", "⟸": "⇐", "<=>": "⇔", "⟺": "⇔", "<->": "↔",
    "->": "→", "⟶": "→", "<-": "←",
    "<=": "≤", ">=": "≥", "!=": "≠", "×": "*", "·": "*", "\\cdot": "*", "∡": "∠",
    "ℝ": "R", "ℤ": "Z", "ℕ": "N", "ℚ": "Q",
    "①": "(1)", "②": "(2)", "③": "(3)", "④": "(4)", "⑤": "(5)",
}
# Reference-only markers that the model is not asked to produce.
_GT_TAGS = re.compile(r"\[(figura|diagrama|dibujo|tabla)[^\]]*\]", re.IGNORECASE)
ILLEGIBLE = re.compile(r"\[illegible\]|\[ilegible\]", re.IGNORECASE)


def normalize(text: str) -> str:
    text = _SUP_RUN.sub(lambda m: "^" + m.group().translate(_SUP), text)
    text = _SUB_RUN.sub(lambda m: "_" + m.group().translate(_SUB), text)
    text = unicodedata.normalize("NFC", text)
    text = _GT_TAGS.sub(" ", text)
    for a, b in _EQUIV.items():
        text = text.replace(a, b)
    text = text.replace("^{", "^(").replace("_{", "_(").replace("}", ")").replace("{", "(")
    return re.sub(r"\s+", " ", text.lower()).strip()


def levenshtein(a: list | str, b: list | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def page_metrics(hyp: str, ref: str) -> dict:
    h, r = normalize(hyp), normalize(ref)
    hs, rs = h.replace(" ", ""), r.replace(" ", "")
    hw, rw = h.split(), r.split()
    return {
        "ref_chars": len(rs),
        "hyp_chars": len(hs),
        "cer": levenshtein(h, r) / max(len(r), 1),
        "cer_nospace": levenshtein(hs, rs) / max(len(rs), 1),
        "wer": levenshtein(hw, rw) / max(len(rw), 1),
        "illegible": len(ILLEGIBLE.findall(hyp)),
        "len_ratio": len(hs) / max(len(rs), 1),  # <1 omits text, >1 invents or repeats
    }


# --- Commands -------------------------------------------------------------------------------


@app.command()
def export() -> None:
    """Write the page images the OCR sees and create empty ground-truth files."""
    settings = get_settings()
    total = 0
    for path in _submission_files():
        doc = load_document(path, settings.pdf_dpi, settings.ocr_max_side_px)
        pages_dir = _eval_dir() / "pages" / _doc_id(path)
        gt_dir = _eval_dir() / "ground_truth" / _doc_id(path)
        pages_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        for page in doc.pages:
            img = pages_dir / f"p{page.number}.{EXT[page.media_type]}"
            img.write_bytes(base64.b64decode(page.image_b64))
            gt = gt_dir / f"p{page.number}.txt"
            if not gt.exists():
                gt.touch()
            total += 1
        typer.echo(f"{_doc_id(path)}: {len(doc.pages)} page(s)")
    typer.echo(f"{total} pages -> {_eval_dir()}")


@app.command()
def run(
    only: Annotated[str | None, typer.Option(help="Substring filter on the document id.")] = None,
    force: Annotated[bool, typer.Option(help="Re-run pages that already have output.")] = False,
    pages_from: Annotated[str | None, typer.Option(
        help="Only OCR the pages already present in this run dir (same pages, other model).")]
    = None,
) -> None:
    """OCR every page with the production OCR model (HUC_OCR_MODEL). Resumable: finished
    pages are skipped. To try another model: HUC_OCR_MODEL=<model> python ... run"""
    settings = get_settings()
    model = get_ocr_model(settings)
    out_dir = _run_dir(_run_name())
    wanted = ({str(f.relative_to(_run_dir(pages_from))).removesuffix(".txt")
               for f in _run_dir(pages_from).glob("*/p*.txt")} if pages_from else None)
    timings_path = out_dir / "timings.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "ocr_model": settings.ocr_model, "max_side_px": settings.ocr_max_side_px,
        "pdf_dpi": settings.pdf_dpi, "system_prompt": prompts.OCR_SYSTEM,
    }, indent=1, ensure_ascii=False))
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    async def go() -> None:
        for path in _submission_files():
            doc_id = _doc_id(path)
            if only and only not in doc_id:
                continue
            doc = load_document(path, settings.pdf_dpi, settings.ocr_max_side_px)
            (out_dir / doc_id).mkdir(exist_ok=True)
            for page in doc.pages:
                out = out_dir / doc_id / f"p{page.number}.txt"
                if wanted is not None and f"{doc_id}/p{page.number}" not in wanted:
                    continue
                if out.exists() and not force:
                    continue
                start = time.monotonic()
                text = await transcribe_page(model, page)
                elapsed = time.monotonic() - start
                out.write_text(text + "\n")
                timings[f"{doc_id}/p{page.number}"] = round(elapsed, 1)
                timings_path.write_text(json.dumps(timings, indent=1, sort_keys=True))
                typer.echo(f"{doc_id}/p{page.number}: {elapsed:.0f}s, {len(text)} chars")

    asyncio.run(go())


@app.command()
def score(
    run: Annotated[str | None, typer.Option(help="Run dir name (default: current settings).")]
    = None,
) -> None:
    """Compare a run against the ground truth; write report.md and metrics.csv in the run dir."""
    run = run or _run_name()
    out_dir = _run_dir(run)
    gt_root = _eval_dir() / "ground_truth"
    timings_path = out_dir / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    rows = []
    for gt in sorted(gt_root.glob("*/p*.txt")):
        ref = gt.read_text().strip()
        hyp_path = out_dir / gt.parent.name / gt.name
        if not ref or not hyp_path.exists():
            continue
        key = f"{gt.parent.name}/{gt.stem}"
        rows.append({"page": key, **page_metrics(hyp_path.read_text(), ref),
                     "seconds": timings.get(key)})
    if not rows:
        typer.echo("Nothing to score: fill ground_truth/ and run the OCR first.")
        raise typer.Exit(1)

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ref_total = sum(r["ref_chars"] for r in rows)
    # Micro-average: weight each page by its reference length.
    micro = {m: sum(r[m] * r["ref_chars"] for r in rows) / ref_total
             for m in ("cer", "cer_nospace", "wer")}
    secs = [r["seconds"] for r in rows if r["seconds"] is not None]
    lines = [
        f"# OCR eval: {run}",
        "",
        f"Pages scored: {len(rows)} · reference chars: {ref_total}",
        (f"CER {micro['cer']:.1%} · CER sin espacios {micro['cer_nospace']:.1%} · "
         f"WER {micro['wer']:.1%} (promedio ponderado por largo)"),
        f"[illegible]: {sum(r['illegible'] for r in rows)} · "
        + (f"tiempo/página: media {sum(secs) / len(secs):.0f}s, máx {max(secs):.0f}s"
           if secs else "sin tiempos"),
        "",
        "| página | chars ref | CER | CER s/esp | WER | largo hyp/ref | illegible | s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: -r["cer_nospace"]):
        lines.append(
            f"| {r['page']} | {r['ref_chars']} | {r['cer']:.1%} | {r['cer_nospace']:.1%} | "
            f"{r['wer']:.1%} | {r['len_ratio']:.2f} | {r['illegible']} | {r['seconds'] or ''} |"
        )
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report)
    typer.echo(report)


if __name__ == "__main__":
    app()

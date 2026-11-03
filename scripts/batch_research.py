"""Prepare, draft and grade the local handwritten research batch in a resumable workspace."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pymupdf

from huc.config import Settings
from huc.services import HucService
from huc.storage.repository import FileRepository

DIRECTORY = re.compile(r"^Semana_(\d+)_Modulo_([LMW]\d)$")
FILE_TYPES = {".pdf", ".jpg", ".jpeg", ".png"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def prepare(data: Path, batch: Path) -> None:
    batch.mkdir(parents=True, exist_ok=True)
    manifest_path = batch / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Manifest exists; choose a new batch directory")
    (batch / "scans").mkdir(exist_ok=True)
    items = []
    for directory in sorted(data.glob("Semana_*")):
        match = DIRECTORY.match(directory.name)
        if not match:
            continue
        week, module = int(match[1]), match[2]
        exam_id = f"s{week}-{module.lower()}"
        groups = defaultdict(list)
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in FILE_TYPES and not path.name.upper().startswith("SOLUCION"):
                groups[path.name.split("__")[0]].append(path)
        solutions = sorted(directory.glob("SOLUCION*.pdf"))
        if len(solutions) != 1:
            raise ValueError(f"{directory.name}: expected one solutions PDF")
        for index, (_, sources) in enumerate(sorted(groups.items()), 1):
            participant = f"{exam_id}-{index:02d}"
            combined = batch / "scans" / f"{participant}.pdf"
            output = pymupdf.open()
            for source in sources:
                with pymupdf.open(source) as doc:
                    if source.suffix.lower() == ".pdf":
                        output.insert_pdf(doc)
                    else:
                        with pymupdf.open("pdf", doc.convert_to_pdf()) as converted:
                            output.insert_pdf(converted)
            count = len(output)
            output.save(combined, garbage=4, deflate=True)
            output.close()
            items.append({"id": participant, "exam_id": exam_id, "week": week,
                          "module": module, "scan": str(combined), "scan_sha256": sha(combined),
                          "pages": count, "source_hashes": [sha(p) for p in sources],
                          "solutions": str(solutions[0]), "solutions_sha256": sha(solutions[0]),
                          "status": "prepared"})
    write_json(manifest_path, {"items": items})
    print(f"Prepared {len(items)} submissions in {len({x['exam_id'] for x in items})} exams")


async def draft(batch: Path) -> None:
    path = batch / "manifest.json"
    data = json.loads(path.read_text())
    rubric_dir = batch / "rubrics"
    rubric_dir.mkdir(exist_ok=True)
    service = HucService(FileRepository(batch / "store"), settings=Settings(data_dir=batch / "store"))
    for item in data["items"]:
        dest = rubric_dir / f"{item['exam_id']}.yaml"
        if dest.exists():
            continue
        existing = {"s14-m1": Path("examples/taller_s14_m1.yaml"),
                    "s13-m2": Path("data/research/drafts/s13-m2.yaml")}.get(item["exam_id"])
        if existing and existing.exists():
            dest.write_text(existing.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"{item['exam_id']}: existing research rubric", flush=True)
            continue
        try:
            drafts = await service.draft_rubrics(Path(item["solutions"]), [item["module"]],
                                                 profile_id="mat0004", week=item["week"])
            dest.write_text(drafts[0][1], encoding="utf-8")
            print(f"{item['exam_id']}: drafted", flush=True)
        except Exception as exc:  # noqa: BLE001 - isolate failed batch items
            print(f"{item['exam_id']}: drafting failed: {type(exc).__name__}: {exc}", flush=True)


async def grade(batch: Path) -> None:
    path = batch / "manifest.json"
    data = json.loads(path.read_text())
    settings = Settings(data_dir=batch / "store")
    service = HucService(FileRepository(settings.data_dir), settings=settings)
    for item in data["items"]:
        if item["status"] == "graded":
            continue
        try:
            rubric = batch / "rubrics" / f"{item['exam_id']}.yaml"
            if not rubric.exists():
                raise FileNotFoundError(f"Missing rubric {rubric}")
            exam = service.load_exam_yaml(rubric)
            if item.get("submission_id"):
                submission = service.repo.get_submission(item["submission_id"])
            else:
                submission = await service.ingest_file(exam.id, Path(item["scan"]), item["id"])
                item["submission_id"] = submission.id
                item["status"] = "ingested"
                write_json(path, data)
            result = await service.grade(submission.id)
            item.update(status="graded", grade_id=result.id,
                        grade_needs_verification=result.needs_verification,
                        rubric_sha256=sha(rubric))
            item.pop("error", None)
            print(f"{item['id']}: {len(result.questions)} questions, "
                  f"{sum(q.feedback_needs_review for q in result.questions)} flagged", flush=True)
        except Exception as exc:  # noqa: BLE001 - preserve and continue failed batch items
            item["status"] = "failed"
            item["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{item['id']}: failed: {item['error']}", flush=True)
        write_json(path, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "draft", "grade"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.data, args.batch)
    else:
        asyncio.run((draft if args.stage == "draft" else grade)(args.batch))


if __name__ == "__main__":
    main()

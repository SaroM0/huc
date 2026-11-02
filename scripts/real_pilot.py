"""Run a small local pilot on one pseudonymized scan and a supplied course rubric."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from huc.config import Settings
from huc.services import HucService
from huc.storage.repository import FileRepository


async def run(exam_path: Path, source_dir: Path, output: Path, repeats: int) -> None:
    destination = output / "results.json"
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a new output directory")
    candidates = sorted(p for p in source_dir.iterdir()
                        if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
                        and not p.name.upper().startswith("SOLUCION"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one student document in the selected directory")
    output.mkdir(parents=True, exist_ok=True)
    source = candidates[0]
    scan = output / f"participant-01{source.suffix.lower()}"
    if scan.exists() and scan.read_bytes() != source.read_bytes():
        raise ValueError("Output directory already contains a different scan")
    if not scan.exists():
        shutil.copyfile(source, scan)
    settings = Settings(data_dir=output / "store")
    service = HucService(FileRepository(settings.data_dir), settings=settings)
    exam = service.load_exam_yaml(exam_path)
    submission = await service.ingest_file(exam.id, scan, student_id="participant-01")
    grades = [await service.grade(submission.id) for _ in range(repeats)]
    result = {"source_sha256": hashlib.sha256(scan.read_bytes()).hexdigest(),
              "exam": exam.model_dump(mode="json"),
              "submission": submission.model_dump(mode="json"),
              "grades": [grade.model_dump(mode="json") for grade in grades]}
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"exam_id": exam.id, "questions": len(exam.questions),
                      "extracted_answers": len(submission.answers),
                      "grades": [{"total": grade.total, "flagged": grade.needs_verification,
                                  "reasons": grade.verification_reasons} for grade in grades]},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    asyncio.run(run(args.exam, args.source_dir, args.out, args.repeats))


if __name__ == "__main__":
    main()

"""Create an auditable, pseudonymous index of all feedback and mechanical warnings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huc.storage.repository import FileRepository


def audit(batch: Path) -> None:
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    repo = FileRepository(batch / "store")
    cases = []
    coverage = []
    for item in manifest["items"]:
        if item["status"] != "graded":
            cases.append({"id": item["id"], "status": item["status"], "error": item.get("error")})
            continue
        submission = repo.get_submission(item["submission_id"])
        grade = repo.get_grade(submission.id)
        if grade is None:
            raise ValueError(f"Missing grade for {item['id']}")
        answers = {answer.question_id: answer for answer in submission.answers}
        assigned = {page for answer in submission.answers for page in answer.source_pages}
        coverage.append({"id": item["id"], "page_count": item["pages"],
                         "pages_not_assigned": sorted(set(range(1, item["pages"] + 1)) - assigned)})
        for q in grade.questions:
            answer = answers.get(q.question_id)
            cases.append({"id": item["id"], "question_id": q.question_id,
                          "status": "needs_review" if q.requires_feedback_review() else "unflagged",
                          "answer": answer.text if answer else None,
                          "ocr_confidence": answer.extraction_confidence if answer else None,
                          "pages": answer.source_pages if answer else [],
                          "scores": [score.model_dump() for score in q.scores],
                          "feedback": q.feedback,
                          "review_reasons": q.feedback_review_reasons,
                          "audit": q.feedback_audit.model_dump() if q.feedback_audit else None})
    out = batch / "audit_index.json"
    out.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    (batch / "page_coverage.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"submissions": len(manifest["items"]),
                      "completed": sum(item["status"] == "graded" for item in manifest["items"]),
                      "questions": sum("question_id" in case for case in cases),
                      "flagged": sum(case["status"] == "needs_review" for case in cases),
                      "with_unassigned_pages": sum(bool(item["pages_not_assigned"]) for item in coverage)},
                     ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    audit(parser.parse_args().batch)

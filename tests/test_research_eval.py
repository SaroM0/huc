import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("research_eval", "scripts/research_eval.py")
assert spec and spec.loader
research_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research_eval)


def test_checkpoint_rejects_other_configuration_and_duplicates(tmp_path):
    checkpoint = tmp_path / "results.jsonl"
    record = {"case": "sample", "repetition": 1}
    line = json.dumps({"fingerprint": "abc", "record": record})
    checkpoint.write_text(line + "\n", encoding="utf-8")
    assert research_eval.load_checkpoint(checkpoint, "abc") == {("sample", 1): record}
    with pytest.raises(ValueError, match="different experiment"):
        research_eval.load_checkpoint(checkpoint, "other")
    checkpoint.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        research_eval.load_checkpoint(checkpoint, "abc")


async def test_call_budget_rejects_before_any_model_calls():
    dataset = Path("research/synthetic/math_cases.yaml")
    assert research_eval.experiment_fingerprint(dataset, None)
    with pytest.raises(ValueError, match="exceed limit"):
        await research_eval.run(dataset, 1, max_model_calls=0)


async def test_completed_checkpoint_resumes_without_model_calls(tmp_path):
    dataset = Path("research/synthetic/math_cases.yaml")
    fingerprint = research_eval.experiment_fingerprint(dataset, None)
    record = {"case": "correct_substitution", "repetition": 1,
              "scores": {"reasoning": 2, "result": 1},
              "score_checks": {"reasoning": True, "result": True},
              "verification_reasons": [], "feedback_assessment": None}
    checkpoint = tmp_path / "completed.jsonl"
    checkpoint.write_text(json.dumps({"fingerprint": fingerprint, "record": record}) + "\n")
    result = await research_eval.run(dataset, 1, checkpoint=checkpoint, max_model_calls=0,
                                     case_ids=["correct_substitution"])
    assert result["estimated_calls_this_run"] == 0
    assert result["records"] == [record]
    assert result["summary"]["criterion_range_match"] == 1

import importlib.util
import json

import pymupdf

spec = importlib.util.spec_from_file_location("batch_research", "scripts/batch_research.py")
assert spec and spec.loader
batch_research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch_research)


def test_prepare_groups_parts_and_removes_identifiers_from_output_names(tmp_path):
    source = tmp_path / "input" / "Semana_03_Modulo_M2"
    source.mkdir(parents=True)
    for name in ("Person__01.pdf", "Person__02.pdf", "SOLUCION_reference.pdf"):
        doc = pymupdf.open()
        doc.new_page()
        doc.save(source / name)
        doc.close()
    batch = tmp_path / "batch"
    batch_research.prepare(tmp_path / "input", batch)
    manifest = json.loads((batch / "manifest.json").read_text())
    assert len(manifest["items"]) == 1
    item = manifest["items"][0]
    assert item["id"] == "s3-m2-01"
    assert item["pages"] == 2
    assert "Person" not in (batch / "manifest.json").read_text()
    with pymupdf.open(item["scan"]) as doc:
        assert len(doc) == 2

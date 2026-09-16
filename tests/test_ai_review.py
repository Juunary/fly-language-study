"""Synthetic Claude records exist only in pytest tmp directories; never review evidence."""
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from flystudy.ai_review import rows, text_key, validate_ai_evidence
from flystudy.data import generate
from flystudy.gates import certify_review
from flystudy.protocol import file_hash, write_json
from flystudy.workflow import review_provenance

ROOT = Path(__file__).resolve().parents[1]


def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


prepare_mod, build_mod = load("prepare_ai_review", "prepare_ai"), load("build_review_packages", "build_review_packages_ai")


def csv_write(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


@pytest.fixture
def evidence(tmp_path):
    data = tmp_path / "data"
    # A tiny dataset, its review catalog and blinded packages from the current builder, so that the checklist the
    # validator requires (language-specific case inventory) is the one the packages carry.
    generate(data, 40, 4)  # 208 examples per language, so the frozen audit sample has its full 200 items per language
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_review_catalog.py"), "--data", str(data)], check=True)
    build_mod.build(data, tmp_path / "review")
    path = prepare_mod.prepare(data, tmp_path / "review" / "dist", tmp_path / "ai")
    manifest = json.loads(path.read_text())
    manifest["data_origin"] = "actual_claude_responses"  # software fixture only
    originals = {text_key(r): r for r in rows(data / "audit-sample.csv")}
    merged, reviewers = [], {lang: [] for lang in ("en", "de", "ko")}
    for i, run in enumerate(manifest["runs"]):
        run.update(model_id="claude-test-fixture", model_version="fixture", interface="api",
                   session_id=f"synthetic-session-{i}", executed_at_utc="2026-09-16T00:00:00Z",
                   settings={"temperature": 0}, fresh_context=True, answer_key_exposed=False)
        reviewers[run["language"]].append(run["reviewer"])
        incoming = rows(path.parent / run["inputs"]["items"]["path"])
        for row in incoming:
            original = originals[text_key(row)]
            row.update(judged_label=original["label"], fluent="yes", comment="synthetic fixture")
            merged.append({**original, "reviewer": run["reviewer"], "judged_label": row["judged_label"],
                           "fluent": row["fluent"], "comment": row["comment"]})
        csv_write(path.parent / run["items_response"]["path"], incoming)
        checks = rows(path.parent / run["inputs"]["checklist"]["path"])
        for row in checks:
            row.update(checked="yes", issue="no", comment="synthetic fixture")
        csv_write(path.parent / run["checklist_response"]["path"], checks)
        (path.parent / run["transcript"]["path"]).write_text(f"SYNTHETIC TEST ONLY {i}")
    write_json(path, manifest)
    sealed = prepare_mod.seal(path)
    merged_path = tmp_path / "merged.csv"
    csv_write(merged_path, merged)
    attest = dict(review_mode="claude_only", reviewers={k: sorted(v) for k, v in reviewers.items()}, all_templates_and_forms_checked=True,
                  inventory_hash=file_hash(data / "template-inventory.json"), resolved_items={})
    return data, merged_path, attest, sealed


def test_certify_ai_has_distinct_gate_and_propagated_evidence(evidence, tmp_path, monkeypatch):
    data, merged, attest, manifest = evidence
    meta = json.loads((data / "manifest.json").read_text())
    monkeypatch.setattr("flystudy.data.audit", lambda _: {"passed": True, "dataset_hash": meta["dataset_hash"]})
    write_json(tmp_path / "attest.json", attest)
    report = certify_review(data, merged, tmp_path / "attest.json", tmp_path / "report.json", ai_evidence=manifest)
    assert report["gate"] == "ai_review" and report["human_reviewed"] is False
    assert len(report["ai_runs"]) == 6 and report["limitations"]
    assert review_provenance(report, "claude_only")
    with pytest.raises(ValueError, match="mode"):
        review_provenance(report, "human")
    artifact = Path(next(k for k in report["evidence_files"] if k.endswith("transcript.txt")))
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="Stale evidence"):
        review_provenance(report, "claude_only")


@pytest.mark.parametrize("change,match", [
    ("model", "metadata"), ("synthetic", "Synthetic"), ("context", "distinct fresh"),
    ("source", "source version"), ("answer", "exposure"), ("raw", "stale AI artifact"),
])
def test_missing_or_changed_provenance_is_rejected(evidence, change, match):
    data, merged, attest, path = evidence
    manifest = json.loads(path.read_text())
    if change == "model": manifest["runs"][0]["model_id"] = ""
    if change == "synthetic": manifest["data_origin"] = "synthetic"
    if change == "context": manifest["runs"][1]["session_id"] = manifest["runs"][0]["session_id"]
    if change == "source": manifest["dataset_hash"] = "other"
    if change == "answer": manifest["runs"][0]["answer_key_exposed"] = True
    if change == "raw": (path.parent / manifest["runs"][0]["items_response"]["path"]).write_text("changed")
    write_json(path, manifest)
    with pytest.raises(ValueError, match=match): validate_ai_evidence(data, merged, attest, path)


def test_missing_no_reason_and_unrecorded_resolution_are_rejected(evidence):
    data, merged, attest, path = evidence
    manifest = json.loads(path.read_text())
    ref = manifest["runs"][0]["items_response"]
    response = path.parent / ref["path"]
    records = rows(response)
    records[0].update(fluent="no", comment="")
    csv_write(response, records); ref["sha256"] = file_hash(response); write_json(path, manifest)
    with pytest.raises(ValueError, match="requires a reason"):
        validate_ai_evidence(data, merged, attest, path)
    records[0]["comment"] = "synthetic issue"
    csv_write(response, records); ref["sha256"] = file_hash(response); write_json(path, manifest)
    merged_rows = rows(merged)
    changed = next(r for r in merged_rows if r["reviewer"] == records[0]["reviewer"] and text_key(r) == text_key(records[0]))
    changed.update(fluent="no", comment="synthetic issue")
    csv_write(merged, merged_rows)
    attest["resolved_items"][changed["id"]] = dict(label=int(changed["label"]), fluent=True, rationale="unsupported")
    with pytest.raises(ValueError, match="recorded Claude adjudication"):
        validate_ai_evidence(data, merged, attest, path)


def test_checklist_issue_cannot_be_overridden_by_attestation(evidence):
    data, merged, attest, path = evidence
    manifest = json.loads(path.read_text())
    ref = manifest["runs"][0]["checklist_response"]
    target = path.parent / ref["path"]
    records = rows(target); records[0]["issue"] = "yes"
    csv_write(target, records); ref["sha256"] = file_hash(target); write_json(path, manifest)
    with pytest.raises(ValueError, match="template/form issue"):
        validate_ai_evidence(data, merged, attest, path)


def test_recorded_claude_adjudication_preserves_initial_no(evidence):
    data, merged, attest, path = evidence
    manifest = json.loads(path.read_text())
    run = manifest["runs"][0]
    ref = run["items_response"]
    target = path.parent / ref["path"]
    records = rows(target)
    records[0].update(fluent="no", comment="synthetic concern")
    csv_write(target, records); ref["sha256"] = file_hash(target)
    merged_rows = rows(merged)
    changed = next(r for r in merged_rows if r["reviewer"] == records[0]["reviewer"] and text_key(r) == text_key(records[0]))
    changed.update(fluent="no", comment="synthetic concern")
    csv_write(merged, merged_rows)
    attest["resolved_items"][changed["id"]] = dict(label=int(changed["label"]), fluent=True, rationale="synthetic resolution")
    resolution = path.parent / "resolution.json"
    write_json(resolution, dict(resolved_items=attest["resolved_items"]))
    manifest["adjudication"] = {**run, "session_id": "separate-adjudication-fixture",
                                "response": dict(path=resolution.name, sha256=file_hash(resolution))}
    write_json(path, manifest)
    before = merged.read_bytes()
    result = validate_ai_evidence(data, merged, attest, path)
    assert len(result["ai_runs"]) == 7 and merged.read_bytes() == before
    attest["resolved_items"][changed["id"]]["rationale"] = "edited"
    with pytest.raises(ValueError, match="differ from Claude adjudication"):
        validate_ai_evidence(data, merged, attest, path)


def test_sealing_does_not_overwrite_inputs_or_existing_snapshot(evidence):
    _, _, _, sealed = evidence
    with pytest.raises(FileExistsError): prepare_mod.seal(sealed.with_name("evidence.json"))
    with pytest.raises(FileExistsError): prepare_mod.prepare("unused", "unused", sealed.parent)


def test_human_gate_is_not_accepted_as_claude():
    with pytest.raises(ValueError, match="mode"):
        review_provenance(dict(gate="human_review", review_mode="human"), "claude_only")


def test_separate_claude_models_are_recorded_not_rejected(evidence):
    data, merged, attest, path = evidence
    manifest = json.loads(path.read_text())
    manifest["runs"][1]["model_id"] = "claude-other-test-fixture"
    write_json(path, manifest)
    report = validate_ai_evidence(data, merged, attest, path)
    assert report["ai_runs"][1]["model_id"] == "claude-other-test-fixture"
    attest["all_templates_and_forms_checked"] = "false"
    with pytest.raises(ValueError, match="attestation is incomplete"):
        validate_ai_evidence(data, merged, attest, path)

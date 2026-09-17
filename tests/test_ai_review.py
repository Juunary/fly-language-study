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


# ---- rule amendment v5.1-ai-review-2: checklist flags are adjudicated per issue, never auto-dismissed ----
from flystudy.ai_review import AMENDMENT, BLOCKING_CATEGORIES, NON_BLOCKING_CATEGORIES


def flag_row(path, run_index=0, row_index=0, comment="synthetic concern"):
    manifest = json.loads(path.read_text())
    run = manifest["runs"][run_index]
    ref = run["checklist_response"]; target = path.parent / ref["path"]
    records = rows(target); records[row_index].update(issue="yes", comment=comment)
    csv_write(target, records); ref["sha256"] = file_hash(target); write_json(path, manifest)
    r = records[row_index]
    return dict(reviewer=run["reviewer"], area=r["area"], language=r["language"], family=r["family"], split=r["split"], task=r["task"])


def add_adjudication(path, attest, issue_id, covers, blocking=False, category="auxiliary_only", scope=("auxiliary_evaluation",),
                     hashes_ok=True, with_amendment=True):
    manifest = json.loads(path.read_text())
    folder = path.parent / "checklist-adjudication" / issue_id; folder.mkdir(parents=True, exist_ok=True)
    (folder / "inputs.csv").write_text("synthetic input\n"); (folder / "prompt.md").write_text("synthetic prompt\n")
    (folder / "transcript.txt").write_text("synthetic transcript\n")
    verdict = dict(issue_id=issue_id, reviewed_input_hashes=[file_hash(folder / "inputs.csv") if hashes_ok else "0" * 64],
                   scope=list(scope), applies_to_languages=["en"], concrete_error_or_counterexample="synthetic: none found",
                   blocking=blocking, category=category, rationale="synthetic rationale", minimal_fix_or_interpretation_scope="synthetic scope")
    write_json(folder / "response.json", verdict)
    rel = lambda name: dict(path=f"checklist-adjudication/{issue_id}/{name}", sha256=file_hash(folder / name))
    manifest.setdefault("checklist_adjudications", []).append(dict(
        issue_id=issue_id, covers=covers, provider="anthropic", model_id="claude-test-fixture", model_version="fixture", interface="claude_code",
        session_id=f"synthetic-adjudication-{issue_id}", executed_at_utc="2026-09-17T00:00:00Z", settings={"availability": "not_exposed"},
        fresh_context=True, answer_key_exposed=False, inputs=[rel("inputs.csv")], prompt=rel("prompt.md"), transcript=rel("transcript.txt"),
        response=rel("response.json")))
    if with_amendment:
        manifest["rule_amendment"] = dict(id=AMENDMENT, adopted_at_utc="2026-09-17T00:00:00Z", adopted_after="synthetic round results",
                                          reason="synthetic fixture")
    write_json(path, manifest)
    attest.setdefault("checklist_adjudications", {})[issue_id] = dict(blocking=blocking, category=category, scope=sorted(scope))
    return manifest


def test_flag_without_adjudication_is_rejected_not_dismissed(evidence):
    data, merged, attest, path = evidence
    flag_row(path)
    with pytest.raises(ValueError, match="requires a recorded checklist adjudication"):
        validate_ai_evidence(data, merged, attest, path)


def test_non_blocking_adjudication_certifies_with_recorded_limitation(evidence):
    data, merged, attest, path = evidence
    covers = [flag_row(path)]
    add_adjudication(path, attest, "issue-A", covers)
    report = validate_ai_evidence(data, merged, attest, path)
    assert report["protocol_amendment"] == AMENDMENT and report["rule_amendment"]["id"] == AMENDMENT
    assert len(report["ai_runs"]) == 7 and report["checklist_flags"][0]["comment"] == "synthetic concern"
    assert report["checklist_adjudications"][0]["covers"] == covers and report["checklist_adjudications"][0]["blocking"] is False
    assert any("Auxiliary-evaluation limitation" in l and "issue-A" in l for l in report["limitations"])
    assert review_provenance(dict(report, gate="ai_review", status="passed"), "claude_only")


def test_blocking_adjudication_fails_certification(evidence):
    data, merged, attest, path = evidence
    covers = [flag_row(path)]
    add_adjudication(path, attest, "issue-B", covers, blocking=True, category=BLOCKING_CATEGORIES[0], scope=("primary_evaluation",))
    with pytest.raises(ValueError, match="Blocking template/form issue confirmed"):
        validate_ai_evidence(data, merged, attest, path)


@pytest.mark.parametrize("change,match", [
    ("no_amendment", "rule amendment"), ("bad_hashes", "exact input hashes"), ("uncovered", "without a recorded adjudication"),
    ("unflagged_cover", "was not flagged"), ("ack_mismatch", "acknowledgment differs"), ("category_mismatch", "non-blocking category"),
])
def test_adjudication_record_defects_are_rejected(evidence, change, match):
    data, merged, attest, path = evidence
    first = flag_row(path)
    second = flag_row(path, run_index=1, row_index=1)
    covers = [first, second]
    if change == "uncovered":
        covers = [first]
    if change == "unflagged_cover":
        covers = [first, dict(second, task="not-a-flagged-task")]
    add_adjudication(path, attest, "issue-C", covers, hashes_ok=change != "bad_hashes", with_amendment=change != "no_amendment")
    if change == "ack_mismatch":
        attest["checklist_adjudications"]["issue-C"]["category"] = "stylistic_preference"
    if change == "category_mismatch":
        manifest = json.loads(path.read_text())
        response = path.parent / manifest["checklist_adjudications"][0]["response"]["path"]
        verdict = json.loads(response.read_text()); verdict["category"] = BLOCKING_CATEGORIES[1]
        write_json(response, verdict); manifest["checklist_adjudications"][0]["response"]["sha256"] = file_hash(response); write_json(path, manifest)
    with pytest.raises(ValueError, match=match):
        validate_ai_evidence(data, merged, attest, path)


def test_flagged_rows_must_be_covered_exactly_once(evidence):
    data, merged, attest, path = evidence
    flag = flag_row(path)
    add_adjudication(path, attest, "issue-D", [flag])
    add_adjudication(path, attest, "issue-E", [flag])
    with pytest.raises(ValueError, match="already covered"):
        validate_ai_evidence(data, merged, attest, path)


def test_categories_are_disjoint_and_named():
    assert not set(BLOCKING_CATEGORIES) & set(NON_BLOCKING_CATEGORIES)
    assert "answer_changing_ambiguity" in BLOCKING_CATEGORIES and "auxiliary_only" in NON_BLOCKING_CATEGORIES


# ---- a blocking adjudication can be lifted only by a recorded revision plus a targeted re-review ----
def add_resolution(path, attest, issue_id, data, languages=("en",), tasks=("space",), disagree=False, include_doc=True, drop_language=False):
    manifest = json.loads(path.read_text())
    entry = next(e for e in manifest["checklist_adjudications"] if e["issue_id"] == issue_id)
    verdict_path = path.parent / entry["response"]["path"]
    verdict = json.loads(verdict_path.read_text()); verdict["applies_to_languages"] = list(languages); write_json(verdict_path, verdict)
    entry["response"]["sha256"] = file_hash(verdict_path)
    folder = path.parent / "checklist-adjudication" / issue_id / "resolution"; folder.mkdir(parents=True, exist_ok=True)
    (folder / "definition.md").write_text("synthetic revised definition\n")
    originals = rows(data / "audit-sample.csv")
    reviews = []
    for i, lang in enumerate(languages):
        if drop_language and i == 0:
            continue
        sub = folder / lang; sub.mkdir(exist_ok=True)
        items = [dict(review_id=f"rid{k}", language=r["language"], split=r["split"], task=r["task"], sentence_a=r["sentence_a"], sentence_b=r["sentence_b"])
                 for k, r in enumerate({r["id"]: r for r in originals}.values()) if r["language"] == lang and r["task"] in tasks]
        csv_write(sub / "items.csv", items)
        (sub / "prompt.md").write_text("synthetic prompt\n"); (sub / "transcript.txt").write_text("synthetic transcript\n")
        labels = {r["id"]: r["label"] for r in originals}
        by_text = {text_key(r): r for r in originals}
        judged = {it["review_id"]: dict(label=labels[by_text[text_key(it)]["id"]], fluent=True, comment="synthetic") for it in items}
        if disagree:
            first = next(iter(judged)); judged[first]["label"] = "1" if judged[first]["label"] == "0" else "0"
        inputs = [sub / "items.csv"] + ([folder / "definition.md"] if include_doc else [])
        response = dict(issue_id=f"{issue_id}-resolution", reviewed_input_hashes=[file_hash(p) for p in inputs], language=lang,
                        template_rows={}, items=judged, blocking=False, category="interpretation_outside_task_definition",
                        rationale="synthetic", minimal_fix_or_interpretation_scope="synthetic")
        write_json(sub / "response.json", response)
        rel = lambda p: dict(path=str(p.relative_to(path.parent)), sha256=file_hash(p))
        reviews.append(dict(language=lang, provider="anthropic", model_id="claude-test-fixture", model_version="fixture", interface="claude_code",
                            session_id=f"synthetic-resolution-{issue_id}-{lang}", executed_at_utc="2026-09-17T01:00:00Z", settings={"availability": "not_exposed"},
                            fresh_context=True, answer_key_exposed=False, inputs=[rel(p) for p in inputs], items_input=rel(sub / "items.csv"),
                            prompt=rel(sub / "prompt.md"), transcript=rel(sub / "transcript.txt"), response=rel(sub / "response.json")))
    entry["resolution"] = dict(kind="task_definition_revision", affected_tasks=list(tasks),
                               revision=dict(id="synthetic-revision-1", adopted_at_utc="2026-09-17T00:30:00Z", description="synthetic stipulation",
                                             documents=[dict(path=str((folder / "definition.md").relative_to(path.parent)), sha256=file_hash(folder / "definition.md"))]),
                               reviews=reviews)
    write_json(path, manifest)
    attest["checklist_adjudications"][issue_id]["resolved"] = True


def test_blocking_adjudication_is_lifted_by_recorded_revision_and_targeted_rereview(evidence):
    data, merged, attest, path = evidence
    covers = [flag_row(path)]
    add_adjudication(path, attest, "issue-R", covers, blocking=True, category=BLOCKING_CATEGORIES[3], scope=("primary_evaluation",))
    add_resolution(path, attest, "issue-R", data)
    report = validate_ai_evidence(data, merged, attest, path)
    adj = report["checklist_adjudications"][0]
    assert adj["blocking"] is True and adj["resolution"]["revision"]["id"] == "synthetic-revision-1"
    assert adj["resolution"]["reviews"][0]["language"] == "en" and adj["resolution"]["reviews"][0]["items"] > 0
    assert any("resolved by synthetic-revision-1" in l for l in report["limitations"])
    assert len(report["ai_runs"]) == 8  # six primary + adjudication + one targeted re-review


@pytest.mark.parametrize("defect,match", [("disagree", "disagrees with the reference"), ("no_doc", "did not receive the revised document"),
                                          ("missing_language", "exactly the languages")])
def test_incomplete_resolution_keeps_the_block(evidence, defect, match):
    data, merged, attest, path = evidence
    covers = [flag_row(path)]
    add_adjudication(path, attest, "issue-S", covers, blocking=True, category=BLOCKING_CATEGORIES[3], scope=("primary_evaluation",))
    add_resolution(path, attest, "issue-S", data, languages=("en", "de"), disagree=defect == "disagree", include_doc=defect != "no_doc",
                   drop_language=defect == "missing_language")
    with pytest.raises(ValueError, match=match):
        validate_ai_evidence(data, merged, attest, path)

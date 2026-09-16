"""Offline Claude review evidence; hashes establish file integrity, not provider authenticity."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

from .data import CASES
from .protocol import LANGUAGES, file_hash

MODE = "claude_only"
SOURCES = ("audit-sample.csv", "noun-forms.csv", "construction-examples.csv", "template-inventory.json")
LIMITATIONS = [
    "No human language review was performed.",
    "Separate Claude sessions are repeated AI assessments, not independent human raters.",
    "Agreement does not establish correctness; model errors may be correlated across languages and sessions.",
    "Archived records and hashes do not authenticate the provider or prove absence of prior answer exposure.",
    "Conclusions are conditional on an AI-reviewed generated corpus, not universal language difficulty.",
]


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def text_key(row):
    return tuple(row[k] for k in ("language", "split", "task", "sentence_a", "sentence_b"))


def truth(value):
    if str(value).strip().lower() in ("yes", "true", "1"):
        return True
    if str(value).strip().lower() in ("no", "false", "0"):
        return False
    raise ValueError("Incomplete AI boolean judgment")


def validate_ai_evidence(dataset, reviewed_csv, attestation, evidence_path):
    """Bind six raw Claude outputs and any adjudication to the certified merged judgments."""
    dataset, evidence_path = Path(dataset), Path(evidence_path)
    manifest = json.loads(evidence_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("review_mode") != MODE:
        raise ValueError("An explicit claude_only evidence manifest is required")
    if attestation.get("review_mode") != MODE:
        raise ValueError("Attestation must explicitly identify claude_only review")
    if attestation.get("all_templates_and_forms_checked") is not True:
        raise ValueError("Claude template/form attestation is incomplete")
    if manifest.get("data_origin") != "actual_claude_responses":
        raise ValueError("Synthetic/test/unknown responses cannot certify AI review")
    meta = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_hash") != meta["dataset_hash"] or manifest.get("source_hashes") != {
        name: file_hash(dataset / name) for name in SOURCES
    }:
        raise ValueError("AI review source version has changed")
    evidence = {str(evidence_path.resolve()): file_hash(evidence_path)}
    evidence.update({str((dataset / name).resolve()): file_hash(dataset / name) for name in (*SOURCES, "manifest.json")})

    def artifact(ref):
        if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
            raise ValueError("Missing AI artifact path/hash")
        path = (evidence_path.parent / ref["path"]).resolve()
        if not path.is_file() or path.stat().st_size == 0 or file_hash(path) != ref["sha256"]:
            raise ValueError(f"Missing/stale AI artifact: {path.name}")
        evidence[str(path)] = ref["sha256"]
        return path

    sessions, provenance = set(), []

    def run_metadata(run):
        for name in ("model_id", "model_version", "session_id", "executed_at_utc"):
            if not isinstance(run.get(name), str) or not run[name].strip():
                raise ValueError(f"Missing Claude metadata: {name}")
        if run.get("provider") != "anthropic" or "claude" not in run["model_id"].lower():
            raise ValueError("Claude provider/model must be recorded")
        if run.get("interface") not in ("api", "claude_web", "claude_code"):
            raise ValueError("Record the actual Claude interface")
        timestamp = datetime.fromisoformat(run["executed_at_utc"].replace("Z", "+00:00"))
        if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("Claude timestamp must include UTC timezone")
        if not isinstance(run.get("settings"), dict) or not run["settings"]:
            raise ValueError("Record generation settings or explicitly mark them not_exposed")
        if run.get("fresh_context") is not True or run["session_id"] in sessions:
            raise ValueError("Every Claude review/adjudication needs a distinct fresh session")
        sessions.add(run["session_id"])
        prompt, transcript = artifact(run.get("prompt")), artifact(run.get("transcript"))
        for prior in run.get("prior_artifacts", []):
            artifact(prior)
        provenance.append({k: run[k] for k in ("provider", "model_id", "model_version", "interface",
                                               "session_id", "executed_at_utc", "settings")})
        provenance[-1].update(prompt_hash=file_hash(prompt), transcript_hash=file_hash(transcript))

    originals = {r["id"]: r for r in rows(dataset / "audit-sample.csv")}
    by_text = {text_key(r): r for r in originals.values()}
    if len(by_text) != len(originals):
        raise ValueError("Ambiguous sentence identities; a new explicit mapping format is needed")
    merged = rows(reviewed_csv)
    merged_index = {(r["id"], r["reviewer"]): r for r in merged}
    if len(merged_index) != len(merged):
        raise ValueError("Duplicate merged AI judgments")
    runs = manifest.get("runs", [])
    if len(runs) != 6:
        raise ValueError("Exactly six primary Claude sessions required")
    reviewers = {lang: [] for lang in LANGUAGES}
    seen, output_hashes = set(), set()
    check_fields = ("area", "language", "family", "split", "task")
    unresolved = set()
    for run in runs:
        run_metadata(run)
        language, reviewer = run.get("language"), run.get("reviewer")
        if language not in reviewers or not reviewer or any(reviewer in v for v in reviewers.values()):
            raise ValueError("Missing/duplicate Claude reviewer code or language")
        reviewers[language].append(reviewer)
        if run.get("answer_key_exposed") is not False:
            raise ValueError("Primary Claude judgments must precede exposure to answers/other reviews")
        inputs = run.get("inputs", {})
        if set(inputs) != {"items", "checklist", "noun_forms", "construction_examples", "inventory"}:
            raise ValueError("Incomplete Claude review inputs")
        paths = {k: artifact(v) for k, v in inputs.items()}
        if file_hash(paths["inventory"]) != file_hash(dataset / "template-inventory.json"):
            raise ValueError("Claude saw a different template inventory")
        for kind, source in (("noun_forms", "noun-forms.csv"), ("construction_examples", "construction-examples.csv")):
            expected = [{k: v for k, v in r.items() if k != "label"} for r in rows(dataset / source) if r["language"] == language]
            if rows(paths[kind]) != expected:
                raise ValueError(f"Claude {kind} input differs from the reviewed data")
        inputs_rows = rows(paths["items"])
        item_index = {r["review_id"]: r for r in inputs_rows}
        if len(inputs_rows) != 200 or len(item_index) != 200 or any(
            r["language"] != language or "id" in r or "label" in r or
            any(r.get(k, "").strip() for k in ("judged_label", "fluent", "comment")) for r in inputs_rows
        ):
            raise ValueError("Claude needs the complete blinded 200-item input")
        expected_ids = {r["id"] for r in originals.values() if r["language"] == language}
        if {by_text.get(text_key(r), {}).get("id") for r in inputs_rows} != expected_ids:
            raise ValueError("Claude input is not the frozen audit sample")
        response = artifact(run.get("items_response"))
        if file_hash(response) in output_hashes:
            raise ValueError("Duplicate Claude output artifact")
        output_hashes.add(file_hash(response))
        responses = rows(response)
        if len(responses) != 200 or {r.get("review_id") for r in responses} != set(item_index):
            raise ValueError("Claude response contains missing/duplicate/unknown items")
        for row in responses:
            src = item_index[row["review_id"]]
            original = by_text[text_key(src)]
            key = (original["id"], reviewer)
            if text_key(row) != text_key(src) or row.get("reviewer") != reviewer or key in seen:
                raise ValueError("Changed sentence/reviewer or duplicate AI judgment")
            seen.add(key)
            if row.get("judged_label") not in ("0", "1"):
                raise ValueError("Missing Claude semantic judgment")
            fluent = truth(row.get("fluent"))
            if not fluent and not row.get("comment", "").strip():
                raise ValueError("Claude fluent=no requires a reason; request a recorded supplement")
            m = merged_index.get(key, {})
            if (m.get("judged_label") != row["judged_label"] or truth(m.get("fluent")) != fluent or
                    m.get("comment", "").strip() != row.get("comment", "").strip() or text_key(m) != text_key(src)):
                raise ValueError("Merged judgments differ from archived Claude output")
            if row["judged_label"] != original["label"] or not fluent:
                unresolved.add(original["id"])
        checklist_input = rows(paths["checklist"])
        checks = rows(artifact(run.get("checklist_response")))
        expected_checks = {tuple(r[k] for k in check_fields) for r in checklist_input}
        required_checks = {("construction", language, family, split, task)
                           for family, split in (("assertion", "train"), ("truth_question", "dev_a"),
                                                 ("reported_clause", "dev_b"), ("conditional", "test"))
                           for task in ("roles", "negation", "space", "quantity")}
        required_checks |= {("noun_forms", language, "", "", case) for case in (*CASES[language], "plural")}
        required_checks.add(("vocabulary", language, "", "", "template-inventory.json"))
        if (expected_checks != required_checks or len(checks) != len(required_checks) or
                {tuple(r[k] for k in check_fields) for r in checks} != required_checks):
            raise ValueError("Incomplete Claude template/form coverage")
        if any(not truth(r.get("checked")) or truth(r.get("issue")) for r in checks):
            raise ValueError("Unresolved Claude template/form issue; revise data and review the new version")
    if seen != set(merged_index) or any(len(v) != 2 for v in reviewers.values()):
        raise ValueError("Two Claude sessions per language must cover every merged judgment")
    if unresolved:
        adjudication = manifest.get("adjudication")
        if not isinstance(adjudication, dict):
            raise ValueError("Unresolved AI judgments require a recorded Claude adjudication")
        run_metadata(adjudication)
        resolution = json.loads(artifact(adjudication.get("response")).read_text(encoding="utf-8"))
        if resolution.get("resolved_items") != attestation.get("resolved_items") or set(resolution.get("resolved_items", {})) != unresolved:
            raise ValueError("Attestation resolutions differ from Claude adjudication")
    elif attestation.get("resolved_items"):
        raise ValueError("Unexpected resolutions without disputed AI judgments")
    return dict(review_mode=MODE, protocol_amendment="v4-ai-review-1", human_reviewed=False,
                ai_runs=provenance, limitations=LIMITATIONS, evidence_files=evidence)

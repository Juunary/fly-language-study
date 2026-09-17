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
# Rule amendment v5.1-ai-review-2 (adopted 2026-09-17 after the round-2 review results, before any main run):
# a checklist row marked issue=yes is no longer an automatic data-revision trigger. Every flagged row must be
# covered by exactly one recorded, independent issue-level adjudication session whose structured verdict names the
# scope, the concrete error or counterexample, whether it blocks certification and the minimal fix or interpretation
# scope. Blocking verdicts fail certification; non-blocking verdicts are carried into the report as limitations.
# Single flags are never dismissed for being single; a peer session's silence never passes a flag.
AMENDMENT = "v5.1-ai-review-2"
AMENDMENTS = ("v4-ai-review-1", AMENDMENT)
SCOPES = ("primary_evaluation", "auxiliary_evaluation", "documentation")
BLOCKING_CATEGORIES = ("primary_item_answer_error", "meaning_damaging_grammar_or_lexical_error",
                       "cross_language_meaning_mismatch", "answer_changing_ambiguity", "leakage_tampering_or_missing_review")
NON_BLOCKING_CATEGORIES = ("stylistic_preference", "interpretation_outside_task_definition", "auxiliary_only")
FLAG_FIELDS = ("reviewer", "area", "language", "family", "split", "task")
VERDICT_FIELDS = ("issue_id", "reviewed_input_hashes", "scope", "concrete_error_or_counterexample", "blocking", "category",
                  "rationale", "minimal_fix_or_interpretation_scope")
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
    unresolved, flags = set(), {}
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
        for r in checks:
            if not truth(r.get("checked")):
                raise ValueError("Unchecked Claude template/form row")
            if truth(r.get("issue")):
                flags[(reviewer, r["area"], language, r["family"], r["split"], r["task"])] = r.get("comment", "").strip()
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
    adjudicated, limitations = validate_checklist_adjudications(manifest, attestation, flags, run_metadata, artifact, by_text)
    return dict(review_mode=MODE, protocol_amendment=AMENDMENT, rule_amendment=manifest.get("rule_amendment"),
                human_reviewed=False, ai_runs=provenance, limitations=LIMITATIONS + limitations,
                checklist_flags=[dict(zip(FLAG_FIELDS, key), comment=comment) for key, comment in sorted(flags.items())],
                checklist_adjudications=adjudicated, evidence_files=evidence)


def validate_checklist_adjudications(manifest, attestation, flags, run_metadata, artifact, by_text):
    """Every flagged checklist row needs exactly one recorded issue-level adjudication; blocking verdicts fail unless a
    recorded resolution (revised definition/data of the affected part + targeted re-review of exactly that part) lifts them."""
    entries = manifest.get("checklist_adjudications")
    if not flags:
        if entries:
            raise ValueError("Unexpected checklist adjudications without flagged rows")
        return [], []
    if not isinstance(entries, list) or not entries:
        raise ValueError("Unresolved Claude template/form issue; every flagged row requires a recorded checklist adjudication")
    amendment = manifest.get("rule_amendment")
    if (not isinstance(amendment, dict) or amendment.get("id") != AMENDMENT or not amendment.get("adopted_at_utc")
            or not amendment.get("reason") or not amendment.get("adopted_after")):
        raise ValueError("Checklist issues can only be adjudicated under the recorded rule amendment " + AMENDMENT)
    acknowledged = attestation.get("checklist_adjudications")
    if not isinstance(acknowledged, dict):
        raise ValueError("Attestation must acknowledge every checklist adjudication verdict")
    covered, ids, summary, limitations, blocking = {}, set(), [], [], []
    for entry in entries:
        issue_id = entry.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id.strip() or issue_id in ids:
            raise ValueError("Checklist adjudication needs a unique issue_id")
        ids.add(issue_id)
        run_metadata(entry)
        inputs = entry.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"{issue_id}: adjudication must list the input artifacts the session saw")
        input_hashes = set()
        for ref in inputs:
            artifact(ref); input_hashes.add(ref["sha256"])
        verdict = json.loads(artifact(entry.get("response")).read_text(encoding="utf-8"))
        missing = [k for k in VERDICT_FIELDS if k not in verdict]
        if missing:
            raise ValueError(f"{issue_id}: adjudication verdict lacks {missing}")
        if verdict["issue_id"] != issue_id:
            raise ValueError(f"{issue_id}: verdict names a different issue")
        if not isinstance(verdict["reviewed_input_hashes"], list) or set(verdict["reviewed_input_hashes"]) != input_hashes:
            raise ValueError(f"{issue_id}: verdict does not acknowledge the exact input hashes")
        scope = verdict["scope"]
        if not isinstance(scope, list) or not scope or not set(scope) <= set(SCOPES):
            raise ValueError(f"{issue_id}: scope must be a non-empty subset of {SCOPES}")
        if not isinstance(verdict["blocking"], bool):
            raise ValueError(f"{issue_id}: blocking must be a JSON boolean")
        category = verdict["category"]
        if verdict["blocking"] and category not in BLOCKING_CATEGORIES:
            raise ValueError(f"{issue_id}: a blocking verdict must name a blocking category")
        if not verdict["blocking"] and category not in NON_BLOCKING_CATEGORIES:
            raise ValueError(f"{issue_id}: a non-blocking verdict must name a non-blocking category")
        for key in ("concrete_error_or_counterexample", "rationale", "minimal_fix_or_interpretation_scope"):
            if not isinstance(verdict[key], str) or not verdict[key].strip():
                raise ValueError(f"{issue_id}: {key} must be a non-empty string")
        covers = entry.get("covers")
        if not isinstance(covers, list) or not covers:
            raise ValueError(f"{issue_id}: adjudication must list the flagged rows it covers")
        for row in covers:
            key = tuple(str(row.get(f, "")) for f in FLAG_FIELDS)
            if key not in flags:
                raise ValueError(f"{issue_id}: covers a checklist row that was not flagged: {key}")
            if key in covered:
                raise ValueError(f"{issue_id}: flagged row already covered by {covered[key]}")
            covered[key] = issue_id
        resolution = validate_resolution(entry, verdict, run_metadata, artifact, by_text) if verdict["blocking"] else None
        expected_ack = dict(blocking=verdict["blocking"], category=category, scope=sorted(scope))
        if resolution is not None:
            expected_ack["resolved"] = True
        if acknowledged.get(issue_id) != expected_ack:
            raise ValueError(f"{issue_id}: attestation acknowledgment differs from the recorded verdict")
        summary.append(dict(issue_id=issue_id, covers=[dict(zip(FLAG_FIELDS, k)) for k, v in covered.items() if v == issue_id],
                            scope=sorted(scope), applies_to_languages=verdict.get("applies_to_languages"), blocking=verdict["blocking"],
                            category=category, concrete_error_or_counterexample=verdict["concrete_error_or_counterexample"],
                            rationale=verdict["rationale"], minimal_fix_or_interpretation_scope=verdict["minimal_fix_or_interpretation_scope"],
                            session_id=entry["session_id"], model_id=entry["model_id"], executed_at_utc=entry["executed_at_utc"],
                            resolution=resolution))
        if verdict["blocking"] and resolution is None:
            blocking.append(issue_id)
        elif verdict["blocking"]:
            limitations.append(f"Blocking checklist flag {issue_id} ({category}) resolved by {resolution['revision']['id']} "
                               f"({resolution['kind']}): {resolution['revision']['description']}")
        elif "auxiliary_evaluation" in scope or category == "auxiliary_only":
            limitations.append(f"Auxiliary-evaluation limitation from checklist adjudication {issue_id} ({category}): "
                               f"{verdict['minimal_fix_or_interpretation_scope']}")
        else:
            limitations.append(f"Checklist flag {issue_id} recorded as {category}: {verdict['minimal_fix_or_interpretation_scope']}")
    uncovered = sorted(set(flags) - set(covered))
    if uncovered:
        raise ValueError(f"Flagged template/form rows without a recorded adjudication: {uncovered}")
    if blocking:
        raise ValueError("Blocking template/form issue confirmed by adjudication (" + ", ".join(blocking) +
                         "); revise the affected data and review only the changed parts")
    return summary, limitations



RESOLUTION_KINDS = ("task_definition_revision", "data_revision")


def validate_resolution(entry, verdict, run_metadata, artifact, by_text):
    """A blocking adjudication is lifted only by a recorded revision of the affected part plus a targeted re-review of
    exactly that part: one fresh session per affected language that re-judges every audit item of the affected tasks
    under the revised definition/data and reproduces every reference label. Unchanged evidence is reused untouched."""
    resolution = entry.get("resolution")
    if not isinstance(resolution, dict):
        return None
    issue_id = entry["issue_id"]
    if resolution.get("kind") not in RESOLUTION_KINDS:
        raise ValueError(f"{issue_id}: resolution kind must be one of {RESOLUTION_KINDS}")
    revision = resolution.get("revision")
    if (not isinstance(revision, dict) or not revision.get("id") or not revision.get("adopted_at_utc") or not revision.get("description")
            or not isinstance(revision.get("documents"), list) or not revision["documents"]):
        raise ValueError(f"{issue_id}: resolution needs a revision id, adoption time, description and revised documents")
    document_hashes = {artifact(ref) and ref["sha256"] for ref in revision["documents"]}
    affected_tasks = resolution.get("affected_tasks")
    if not isinstance(affected_tasks, list) or not affected_tasks:
        raise ValueError(f"{issue_id}: resolution must name the affected tasks")
    languages = verdict.get("applies_to_languages") or []
    reviews = resolution.get("reviews")
    if not isinstance(reviews, list) or {r.get("language") for r in reviews} != set(languages) or len(reviews) != len(languages):
        raise ValueError(f"{issue_id}: targeted re-review must cover exactly the languages the verdict names, once each")
    for review in reviews:
        run_metadata(review)
        language = review["language"]
        inputs = review.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"{issue_id}/{language}: re-review must list its input artifacts")
        input_hashes = set()
        for ref in inputs:
            artifact(ref); input_hashes.add(ref["sha256"])
        if not document_hashes & input_hashes:
            raise ValueError(f"{issue_id}/{language}: re-review did not receive the revised document")
        items_input = rows(artifact(review.get("items_input")))
        expected = {r["id"]: r for r in by_text.values() if r["language"] == language and r["task"] in affected_tasks}
        seen_ids, by_review_id = set(), {}
        for r in items_input:
            original = by_text.get(text_key(r))
            if original is None or original["id"] not in expected or "label" in r or "id" in r:
                raise ValueError(f"{issue_id}/{language}: re-review items are not the blinded audit items of the affected tasks")
            seen_ids.add(original["id"]); by_review_id[r["review_id"]] = original
        if seen_ids != set(expected):
            raise ValueError(f"{issue_id}/{language}: re-review must cover every audit item of the affected tasks")
        response = json.loads(artifact(review.get("response")).read_text(encoding="utf-8"))
        if response.get("issue_id") != f"{issue_id}-resolution" or set(response.get("reviewed_input_hashes", [])) != input_hashes:
            raise ValueError(f"{issue_id}/{language}: re-review verdict does not acknowledge the issue and its exact inputs")
        if response.get("blocking") is not False or response.get("category") not in NON_BLOCKING_CATEGORIES:
            raise ValueError(f"{issue_id}/{language}: re-review still finds a blocking defect")
        for key in ("rationale", "minimal_fix_or_interpretation_scope"):
            if not isinstance(response.get(key), str) or not response[key].strip():
                raise ValueError(f"{issue_id}/{language}: re-review verdict lacks {key}")
        judged = response.get("items")
        if not isinstance(judged, dict) or set(judged) != set(by_review_id):
            raise ValueError(f"{issue_id}/{language}: re-review must judge every listed item exactly once")
        for review_id, j in judged.items():
            original = by_review_id[review_id]
            if not isinstance(j, dict) or j.get("label") not in ("0", "1") or not isinstance(j.get("fluent"), bool):
                raise ValueError(f"{issue_id}/{language}: malformed re-review judgment for {review_id}")
            if j["label"] != original["label"] or not j["fluent"]:
                raise ValueError(f"{issue_id}/{language}: re-review disagrees with the reference on {original['id']}; the defect is not resolved")
    return dict(kind=resolution["kind"], revision={k: revision[k] for k in ("id", "adopted_at_utc", "description")},
                revised_documents=revision["documents"], affected_tasks=affected_tasks,
                reviews=[dict(language=r["language"], session_id=r["session_id"], model_id=r["model_id"], executed_at_utc=r["executed_at_utc"],
                              items=len(rows(artifact(r["items_input"])))) for r in reviews])

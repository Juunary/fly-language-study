"""Join blinded, independent reviewer submissions back to original IDs for `certify-review`.

Inputs (operator side only):
  <review>/private/mapping.csv            review_id -> reviewer_code, original_id
  <review>/private/build-manifest.json    expected reviewer codes and per-language item counts
  <review>/submissions/raw/<code>/*.csv   exactly one returned items CSV per reviewer code (never edited)
  <data>/audit-sample.csv                 original items with labels (for the certify schema and disagreement report)
Outputs (<review>/submissions/merged/):
  reviewed-audit-sample.csv   certify-review input: original IDs, reviewer codes, both original judgments preserved
  disagreements.csv           operator-only consensus worklist (includes the original label)
  attestation-draft.json      reviewers per language and unresolved item keys; humans must complete every value
  receipt.json                hashes/counts/agreement of the raw submissions
No judgment is invented or altered: rows are copied verbatim after normalising 'yes'/'no' spelling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

AUDIT_FIELDS = ["id", "language", "split", "task", "sentence_a", "sentence_b", "label", "reviewer", "judged_label", "fluent", "comment"]
YES, NO = ("yes", "y", "true", "1", "예", "네", "ja"), ("no", "n", "false", "0", "아니오", "아니요", "nein")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", (text or "").strip())


def normalise_label(value: str):
    v = norm(value)
    if v in ("0", "1"):
        return v
    try:
        f = float(v)
        if f in (0.0, 1.0):
            return str(int(f))
    except ValueError:
        pass
    return None


def normalise_fluent(value: str):
    v = norm(value).lower()
    return "yes" if v in YES else "no" if v in NO else None


def find_submission(folder: Path):
    files = sorted(p for p in folder.glob("*.csv") if p.is_file()) if folder.exists() else []
    return files


def merge(review: Path, data: Path, output: Path | None = None):
    private = review / "private"
    manifest = json.loads((private / "build-manifest.json").read_text(encoding="utf-8"))
    mapping = {r["review_id"]: r for r in read_csv(private / "mapping.csv")}
    originals = {}
    for row in read_csv(data / "audit-sample.csv"):
        originals.setdefault(row["id"], row)
    codes = sorted(manifest["packages"])
    output = output or review / "submissions" / "merged"
    problems, judgments, receipt = defaultdict(list), {}, dict(merged_at_utc=datetime.now(timezone.utc).isoformat(), submissions={})
    for code in codes:
        expected = {rid: m for rid, m in mapping.items() if m["reviewer_code"] == code}
        files = find_submission(review / "submissions" / "raw" / code)
        if len(files) != 1:
            problems[code].append(f"expected exactly one CSV in submissions/raw/{code}, found {len(files)}")
            continue
        rows = read_csv(files[0])
        seen = set()
        for n, row in enumerate(rows, start=2):
            rid = norm(row.get("review_id"))
            if rid not in expected:
                problems[code].append(f"line {n}: unknown review_id {rid!r}"); continue
            if rid in seen:
                problems[code].append(f"line {n}: duplicate review_id {rid}"); continue
            seen.add(rid)
            original = originals[expected[rid]["original_id"]]
            if norm(row.get("sentence_a")) != norm(original["sentence_a"]) or norm(row.get("sentence_b")) != norm(original["sentence_b"]):
                problems[code].append(f"line {n}: sentence text was edited for review_id {rid}")
            reviewer = norm(row.get("reviewer"))
            if reviewer and reviewer != code:
                problems[code].append(f"line {n}: reviewer column {reviewer!r} differs from code {code}")
            label, fluent = normalise_label(row.get("judged_label")), normalise_fluent(row.get("fluent"))
            if label is None:
                problems[code].append(f"line {n}: judged_label must be 0 or 1 (got {row.get('judged_label')!r})")
            if fluent is None:
                problems[code].append(f"line {n}: fluent must be yes or no (got {row.get('fluent')!r})")
            if fluent == "no" and not norm(row.get("comment")):
                problems[code].append(f"line {n}: fluent=no requires a comment")
            judgments[(original["id"], code)] = dict(judged_label=label, fluent=fluent, comment=norm(row.get("comment")))
        missing = set(expected) - seen
        if missing:
            problems[code].append(f"{len(missing)} items missing (e.g. {sorted(missing)[:3]})")
        receipt["submissions"][code] = dict(file=str(files[0]), sha256=sha256(files[0]), rows=len(rows),
                                            language=manifest["packages"][code]["language"])
    if problems:
        output.mkdir(parents=True, exist_ok=True)
        (output / "merge-problems.json").write_text(json.dumps(problems, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise ValueError("Submissions are incomplete or invalid; see merge-problems.json:\n" +
                         "\n".join(f"  {code}: {len(v)} problem(s), first: {v[0]}" for code, v in problems.items()))

    by_language = defaultdict(list)
    for code in codes:
        by_language[manifest["packages"][code]["language"]].append(code)
    merged, disagreements, unresolved = [], [], []
    for item_id, original in originals.items():
        pair = sorted(by_language[original["language"]])
        rows = []
        for code in pair:
            j = judgments[(item_id, code)]
            rows.append({**{k: original[k] for k in AUDIT_FIELDS[:7]}, "reviewer": code, **j})
        merged.extend(rows)
        labels = [r["judged_label"] for r in rows]
        fluents = [r["fluent"] for r in rows]
        if len(set(labels)) > 1 or any(l != original["label"] for l in labels) or "no" in fluents:
            unresolved.append(item_id)
            disagreements.append(dict(id=item_id, language=original["language"], split=original["split"], task=original["task"],
                                      sentence_a=original["sentence_a"], sentence_b=original["sentence_b"], original_label=original["label"],
                                      **{f"{k}_{i+1}": rows[i][k] for i in range(2) for k in ("reviewer", "judged_label", "fluent", "comment")},
                                      reason=";".join(x for x, ok in (("reviewers_differ", len(set(labels)) > 1),
                                                                        ("differs_from_original", any(l != original["label"] for l in labels)),
                                                                        ("fluent_no", "no" in fluents)) if ok)))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reviewed-audit-sample.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS); w.writeheader(); w.writerows(merged)
    with (output / "disagreements.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = list(disagreements[0].keys()) if disagreements else ["id"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(disagreements)
    agreement = {}
    for lang, pair in by_language.items():
        ids = [i for i, o in originals.items() if o["language"] == lang]
        a = [judgments[(i, pair[0])]["judged_label"] for i in ids]
        b = [judgments[(i, pair[1])]["judged_label"] for i in ids]
        agree = sum(x == y for x, y in zip(a, b)) / len(ids)
        chance = sum((a.count(x) / len(ids)) * (b.count(x) / len(ids)) for x in ("0", "1"))
        agreement[lang] = dict(items=len(ids), raw_agreement=agree, kappa=(agree - chance) / (1 - chance) if chance < 1 else None,
                               fluent_no=sum(judgments[(i, c)]["fluent"] == "no" for i in ids for c in pair),
                               differs_from_original=sum(judgments[(i, c)]["judged_label"] != originals[i]["label"] for i in ids for c in pair))
    attestation = dict(reviewers={lang: sorted(pair) for lang, pair in by_language.items()},
                       all_templates_and_forms_checked=False,
                       inventory_hash=sha256(data / "template-inventory.json"),
                       resolved_items={i: dict(label=None, fluent=None, rationale="") for i in sorted(unresolved)},
                       note="DRAFT written by the merge tool. Humans must set all_templates_and_forms_checked after the template review "
                            "and fill label/fluent/rationale for every resolved item from the consensus meeting. Null values fail certify-review.")
    (output / "attestation-draft.json").write_text(json.dumps(attestation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt.update(items=len(originals), merged_rows=len(merged), unresolved_items=len(unresolved), agreement_by_language=agreement,
                   certify_thresholds=dict(items_per_language=200, min_agreement=.95, min_kappa=.8),
                   outputs={p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "receipt.json"})
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", default="review")
    parser.add_argument("--data", default="data/draft-v4.3")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    receipt = merge(Path(args.review), Path(args.data), Path(args.output) if args.output else None)
    print(json.dumps(dict(status="merged", items=receipt["items"], unresolved_items=receipt["unresolved_items"],
                          agreement_by_language=receipt["agreement_by_language"],
                          next="Hold the consensus meeting on disagreements.csv, complete attestation-draft.json, then run certify-review."),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

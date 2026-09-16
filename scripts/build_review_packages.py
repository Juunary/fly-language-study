"""Build blinded, independent human-review packages from the frozen audit sample.

Reads the dataset read-only. Writes under --output (default review/):
  private/    OPERATOR ONLY: review-id mapping, reviewer registry, build manifest. Never distribute.
  dist/       One package (folder + zip) per reviewer code. Items carry random review IDs, no labels,
              no original IDs, an independent row order per reviewer; language-filtered noun forms;
              construction examples with the label column removed; checklist; instructions.
  submissions/raw/<code>/   empty inboxes for returned files.
The build refuses to overwrite an existing output so distributed IDs stay stable.
Reviewer judgments are never generated here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

LANGUAGES = ("en", "de", "ko")
TASKS = ("roles", "negation", "space", "quantity")
FAMILIES = {"train": "assertion", "dev_a": "truth_question", "dev_b": "reported_clause", "test": "conditional"}
ITEM_FIELDS = ["review_id", "language", "split", "task", "sentence_a", "sentence_b", "reviewer", "judged_label", "fluent", "comment"]
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I
LANGUAGE_NAMES = {"en": "English", "de": "German (Deutsch)", "ko": "Korean (한국어)"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reviewer_code(lang: str, slot: int, rng: secrets.SystemRandom) -> str:
    return f"{lang.upper()}-R{slot}-" + "".join(rng.choice(CODE_ALPHABET) for _ in range(5))


def unique_items(audit_rows):
    """One record per item; the two delivered rows per item must be identical apart from the reviewer slot."""
    items, seen = [], {}
    for row in audit_rows:
        content = {k: row[k] for k in ("id", "language", "split", "task", "sentence_a", "sentence_b", "label")}
        if row["id"] in seen:
            if seen[row["id"]] != content:
                raise ValueError(f"Inconsistent duplicate audit rows for {row['id']}")
            continue
        seen[row["id"]] = content
        items.append(content)
    return items


def instructions(code: str, lang: str, n_items: int) -> str:
    name = LANGUAGE_NAMES[lang]
    return f"""# Human review instructions / 검수 안내

Reviewer code / 검수자 코드: **{code}**   Language / 언어: **{name}**   Items / 문항: **{n_items}**

Your code identifies you in this study instead of your name. Keep it.
이 코드는 실명 대신 사용하는 식별자입니다.

## 0. Independence / 독립성

The other reviewer of your language received the same {n_items} sentence pairs, with different review IDs and in a
different order. The random IDs only avoid exposing the answer key; they do not make the files incomparable.
Independence therefore depends on you: do not look at the other reviewer's file or at any answer key, do not discuss
any item before both of you have submitted, and submit your own judgments only.
같은 언어의 다른 검수자도 같은 {n_items}문항을 받았습니다(검수 ID와 순서만 다름). 무작위 ID는 정답 노출을 줄이는
장치일 뿐 두 파일을 비교할 수 없게 만드는 것은 아닙니다. 따라서 독립성은 절차로 확보합니다: 상대 파일이나 정답 자료를
보지 말고, 두 사람이 모두 제출하기 전에는 문항을 상의하지 말고, 본인 판정만 제출하십시오.

## 1. Item judgments / 문항 판정 (`items-{code}.csv`)

Each row shows two sentences in {name}. Both may be wrapped in the same frame (a question, reported speech, or a
conditional); judge the statements inside the frame.
각 행에는 두 문장이 있습니다. 두 문장이 같은 틀(의문문·인용·조건문)에 들어 있을 수 있으며, 틀 안의 진술을 판정합니다.

- `judged_label` = **1** if the two embedded statements express the same content, **0** if they differ.
  For the meaning to be the same, B must follow from A **and** A must follow from B. One-directional entailment is 0.
  두 문장의 내부 진술이 같은 내용을 표현하면 **1**, 다르면 **0**으로 판정합니다. 의미가 같으려면 A에서 B뿐 아니라
  B에서 A도 성립해야 합니다. 한 방향만 성립하면 0입니다.
- `fluent` = **yes** if both sentences are grammatical and natural for a native speaker; **no** otherwise.
  두 문장이 모두 문법적이고 자연스러우면 **yes**, 아니면 **no**. `no`이면 `comment`에 이유를 적어 주십시오.
- `comment`: optional otherwise. 그 외에는 선택.

Task column / 과제 열 (what "same content" means for each task / 과제별 '같은 내용'의 뜻):
- `roles`: the same participants play the same roles in both clauses. Active/passive or word-order changes with the
  same role assignment are the same content. 두 절에서 같은 참여자가 같은 역할. 태·어순만 다르면 같은 내용.
- `negation`: the same statement is affirmed and the same statement is denied. 같은 진술이 긍정되고 같은 진술이 부정됨.
- `space`: the same relative positions. Mirror phrasings (A left of B / B right of A) are the same content. 같은 상대 위치.
- `quantity`: the same number of each kind. Listing order does not matter. 각 종류의 개체 수가 같음.

Rules / 규칙: fill only `judged_label`, `fluent`, `comment`. Do not edit `review_id`, the sentences, or `reviewer`;
do not delete or reorder rows. Save as CSV (UTF-8). In Excel choose "CSV UTF-8".
`judged_label`, `fluent`, `comment`만 채우십시오. 다른 열·행 순서를 바꾸지 마십시오. UTF-8 CSV로 저장하십시오.

## 2. Templates and word forms / 문장 틀·활용표

- `noun-forms-{lang}.csv`: every noun phrase surface form (case, number). Mark errors by row.
  모든 명사구 표면형(격·수). 오류가 있는 행을 표시.
- `construction-examples-{lang}.csv`: one example per construction. Check grammaticality and naturalness.
  구문별 대표 문장. 문법성·자연스러움 확인.
- `template-inventory.json`: vocabulary tables. 어휘 표.
- `template-checklist-{code}.csv`: record `checked` (yes/no), `issue` (yes/no) and comments per row.

## 3. Return / 제출

Return `items-{code}.csv` and `template-checklist-{code}.csv` to the operator. Keep a copy.
두 파일을 운영자에게 보내고 사본을 보관하십시오.
"""


def checklist_rows(lang: str):
    rows = [dict(area="construction", language=lang, family=family, split=split, task=task, checked="", issue="", comment="")
            for split, family in FAMILIES.items() for task in TASKS]
    rows += [dict(area="noun_forms", language=lang, family="", split="", task=case, checked="", issue="", comment="")
             for case in ("nom", "acc", "dat", "plural")]
    rows.append(dict(area="vocabulary", language=lang, family="", split="", task="template-inventory.json", checked="", issue="", comment=""))
    return rows


def build(data: Path, output: Path, rng: secrets.SystemRandom | None = None):
    rng = rng or secrets.SystemRandom()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} exists; distributed review IDs must not be regenerated")
    audit_path = data / "audit-sample.csv"
    items = unique_items(read_csv(audit_path))
    per_language = {lang: [i for i in items if i["language"] == lang] for lang in LANGUAGES}
    if any(len(v) == 0 for v in per_language.values()):
        raise ValueError("Every language needs audit items")
    noun_forms = read_csv(data / "noun-forms.csv")
    constructions = read_csv(data / "construction-examples.csv")
    inventory_path = data / "template-inventory.json"
    manifest_data = json.loads((data / "manifest.json").read_text(encoding="utf-8"))

    private, dist, inbox = output / "private", output / "dist", output / "submissions" / "raw"
    for folder in (private, dist, inbox):
        folder.mkdir(parents=True, exist_ok=True)
    os.chmod(private, 0o700)

    mapping, registry, packages, used_ids = [], [], {}, set()
    for lang in LANGUAGES:
        for slot in (1, 2):
            code = reviewer_code(lang, slot, rng)
            while code in packages:
                code = reviewer_code(lang, slot, rng)
            order = rng.sample(per_language[lang], len(per_language[lang]))
            rows = []
            for position, item in enumerate(order, start=1):
                rid = secrets.token_hex(5)
                while rid in used_ids:
                    rid = secrets.token_hex(5)
                used_ids.add(rid)
                mapping.append(dict(review_id=rid, reviewer_code=code, original_id=item["id"], language=lang, position=position))
                rows.append(dict(review_id=rid, language=lang, split=item["split"], task=item["task"],
                                 sentence_a=item["sentence_a"], sentence_b=item["sentence_b"],
                                 reviewer=code, judged_label="", fluent="", comment=""))
            folder = dist / code
            folder.mkdir()
            write_csv(folder / f"items-{code}.csv", ITEM_FIELDS, rows)
            write_csv(folder / f"noun-forms-{lang}.csv", ["language", "noun", "color", "size", "case", "count_index", "surface"],
                      [r for r in noun_forms if r["language"] == lang])
            write_csv(folder / f"construction-examples-{lang}.csv", ["language", "split", "task", "verb", "neg", "sentence_a", "sentence_b"],
                      [r for r in constructions if r["language"] == lang])  # label column intentionally omitted
            shutil.copyfile(inventory_path, folder / "template-inventory.json")
            write_csv(folder / f"template-checklist-{code}.csv", ["area", "language", "family", "split", "task", "checked", "issue", "comment"],
                      checklist_rows(lang))
            (folder / "INSTRUCTIONS.md").write_text(instructions(code, lang, len(rows)), encoding="utf-8")
            files = sorted(p for p in folder.iterdir() if p.is_file())
            with zipfile.ZipFile(dist / f"{code}.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for p in files:
                    z.write(p, f"{code}/{p.name}")
            packages[code] = dict(language=lang, slot=slot, items=len(rows), files={p.name: sha256(p) for p in files},
                                  zip=sha256(dist / f"{code}.zip"))
            registry.append(dict(reviewer_code=code, language=lang, slot=slot, fluency_evidence="",
                                 date_sent="", date_received="", notes=""))
            (inbox / code).mkdir()

    write_csv(private / "mapping.csv", ["review_id", "reviewer_code", "original_id", "language", "position"], mapping)
    write_csv(private / "reviewer-registry.csv", list(registry[0].keys()), registry)
    leak_overlap = sum((r["language"], r["sentence_a"], r["sentence_b"]) in {(i["language"], i["sentence_a"], i["sentence_b"]) for i in items}
                       for r in constructions)
    manifest = dict(built_at_utc=datetime.now(timezone.utc).isoformat(), dataset=str(data.resolve()),
                    dataset_version=manifest_data.get("version"), dataset_hash=manifest_data.get("dataset_hash"),
                    source_hashes={name: sha256(data / name) for name in ("audit-sample.csv", "noun-forms.csv", "construction-examples.csv", "template-inventory.json")},
                    items_per_language={k: len(v) for k, v in per_language.items()}, packages=packages,
                    blinding=dict(labels_removed=True, original_ids_replaced=True, independent_row_order_per_reviewer=True,
                                  independent_review_ids_per_reviewer=True, construction_label_column_removed=True,
                                  construction_rows_overlapping_audit_items=leak_overlap),
                    human_judgments_generated=False)
    (private / "build-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (private / "README.md").write_text(
        "# OPERATOR ONLY — never distribute this folder\n\n"
        "- `mapping.csv`: review_id -> reviewer_code, original_id. Original IDs embed the label; this file plus\n"
        "  `data/.../audit-sample.csv` recovers every answer. Reviewers must never see it.\n"
        "- `reviewer-registry.csv`: one row per code; record the language-fluency evidence and send/receive dates.\n"
        "  Codes, not names, appear in submissions and in the attestation. If real names or contacts are needed,\n"
        "  the operator keeps them in a separate register outside this repository.\n"
        "- `build-manifest.json`: hashes of every distributed file for tamper checks at merge time.\n", encoding="utf-8")
    for p in private.iterdir():
        os.chmod(p, 0o600)
    return manifest


REGISTRY_FIELDS = ["reviewer_code", "language", "slot", "fluency_evidence", "date_sent", "date_received", "notes"]


def refresh_instructions(output: Path, reason: str):
    """Regenerate INSTRUCTIONS.md, the registry schema and the zips of an existing build.
    Items, review IDs and the mapping are never touched; the manifest records the refresh."""
    private, dist = output / "private", output / "dist"
    manifest_path = private / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = {code: sha256(dist / code / f"items-{code}.csv") for code in manifest["packages"]}
    mapping_before = sha256(private / "mapping.csv")
    for code, info in manifest["packages"].items():
        folder = dist / code
        (folder / "INSTRUCTIONS.md").write_text(instructions(code, info["language"], info["items"]), encoding="utf-8")
        files = sorted(p for p in folder.iterdir() if p.is_file())
        with zipfile.ZipFile(dist / f"{code}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, f"{code}/{p.name}")
        info["files"] = {p.name: sha256(p) for p in files}
        info["zip"] = sha256(dist / f"{code}.zip")
    registry_path = private / "reviewer-registry.csv"
    registry = read_csv(registry_path)
    filled = [r for r in registry if any(v.strip() for k, v in r.items() if k not in ("reviewer_code", "language", "slot"))]
    if filled:
        raise ValueError("Registry already contains entries; migrate it by hand instead of rewriting")
    write_csv(registry_path, REGISTRY_FIELDS, [{**{f: "" for f in REGISTRY_FIELDS}, **{k: r[k] for k in ("reviewer_code", "language", "slot")}} for r in registry])
    if any(sha256(dist / code / f"items-{code}.csv") != h for code, h in before.items()) or sha256(private / "mapping.csv") != mapping_before:
        raise RuntimeError("Refresh must not modify items or mapping")
    manifest.setdefault("refreshes", []).append(dict(at_utc=datetime.now(timezone.utc).isoformat(), reason=reason,
                                                    items_and_mapping_unchanged=True))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for p in private.iterdir():
        os.chmod(p, 0o600)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/draft-v4.3")
    parser.add_argument("--output", default="review")
    parser.add_argument("--refresh-instructions", metavar="REASON", help="Rewrite INSTRUCTIONS.md/registry schema/zips of an existing build; IDs unchanged")
    args = parser.parse_args(argv)
    if args.refresh_instructions:
        manifest = refresh_instructions(Path(args.output), args.refresh_instructions)
        print(json.dumps(dict(status="refreshed", output=args.output, refreshes=manifest["refreshes"],
                              zips={k: v["zip"][:12] for k, v in manifest["packages"].items()}), ensure_ascii=False, indent=2))
        return
    manifest = build(Path(args.data), Path(args.output))
    print(json.dumps(dict(status="built", output=args.output, items_per_language=manifest["items_per_language"],
                          packages={k: v["language"] for k, v in manifest["packages"].items()},
                          blinding=manifest["blinding"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

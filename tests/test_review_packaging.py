"""Blinded review packaging and merge. Synthetic judgments here are software fixtures in tmp_path only;
they are never written to the project review folder and are not human review evidence."""
import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import pytest
from flystudy.data import generate
from flystudy.gates import certify_review

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


build_mod, merge_mod = load("build_review_packages"), load("merge_review_submissions")


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def synthetic_submissions(review, data, judge):
    """Test fixture: fill each blinded package from the private mapping with `judge(original_row)`."""
    mapping = {r["review_id"]: r for r in read(review / "private" / "mapping.csv")}
    originals = {r["id"]: r for r in read(data / "audit-sample.csv")}
    for folder in (review / "dist").iterdir():
        if not folder.is_dir():
            continue
        code = folder.name
        rows = read(folder / f"items-{code}.csv")
        for row in rows:
            row["judged_label"], row["fluent"], row["comment"] = judge(originals[mapping[row["review_id"]]["original_id"]])
        target = review / "submissions" / "raw" / code / f"items-{code}.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("review")
    generate(root / "data", 18, 2)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_review_catalog.py"), "--data", str(root / "data")], check=True)
    build_mod.build(root / "data", root / "review")
    return root


def test_distribution_hides_labels_and_original_ids(tiny):
    manifest = json.loads((tiny / "review" / "private" / "build-manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["packages"]) == 6 and manifest["human_judgments_generated"] is False
    original_ids = {r["id"] for r in read(tiny / "data" / "audit-sample.csv")}
    all_review_ids = []
    for code, info in manifest["packages"].items():
        folder = tiny / "review" / "dist" / code
        rows = read(folder / f"items-{code}.csv")
        assert len(rows) == manifest["items_per_language"][info["language"]]
        assert "label" not in rows[0] and "id" not in rows[0]
        assert all(r["judged_label"] == "" and r["fluent"] == "" and r["reviewer"] == code for r in rows)
        assert all(re.fullmatch(r"[0-9a-f]{10}", r["review_id"]) for r in rows)
        all_review_ids += [r["review_id"] for r in rows]
        text = b"".join(p.read_bytes() for p in folder.iterdir())
        assert not any(i.encode() in text for i in original_ids)
        constructions = read(folder / f"construction-examples-{info['language']}.csv")
        assert "label" not in constructions[0] and {r["language"] for r in constructions} == {info["language"]}
        with zipfile.ZipFile(tiny / "review" / "dist" / f"{code}.zip") as z:
            assert all(n.startswith(code + "/") for n in z.namelist()) and not any("mapping" in n for n in z.namelist())
    assert len(all_review_ids) == len(set(all_review_ids))
    a, b = [c for c, i in manifest["packages"].items() if i["language"] == "en"]
    order_a = [r["review_id"] for r in read(tiny / "review" / "dist" / a / f"items-{a}.csv")]
    order_b = [r["review_id"] for r in read(tiny / "review" / "dist" / b / f"items-{b}.csv")]
    assert set(order_a).isdisjoint(order_b)
    registry = read(tiny / "review" / "private" / "reviewer-registry.csv")
    assert len(registry) == 6 and "real_name" not in registry[0] and all(r["fluency_evidence"] == "" for r in registry)
    text = (tiny / "review" / "dist" / a / "INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert "A must follow from B" in text and "B에서 A도 성립" in text and "do not make the files incomparable" in text
    assert "cannot" not in text.lower().replace("cannot be", "") or "incomparable" in text
    assert not (tiny / "review" / "dist" / a / f"items-{a}.csv").read_text(encoding="utf-8-sig").startswith("reviewer_")


def test_refresh_keeps_ids_and_mapping_and_updates_zips(tiny, tmp_path, monkeypatch):
    review = tmp_path / "review"; shutil.copytree(tiny / "review", review)
    items_before = {p.name: p.read_bytes() for p in review.glob("dist/*/items-*.csv")}
    mapping_before = (review / "private" / "mapping.csv").read_bytes()
    zips_before = {p.name: p.read_bytes() for p in review.glob("dist/*.zip")}
    original = build_mod.instructions
    monkeypatch.setattr(build_mod, "instructions", lambda code, lang, n: original(code, lang, n) + "\nREFRESH-MARKER\n")
    manifest = build_mod.refresh_instructions(review, "test refresh")
    assert {p.name: p.read_bytes() for p in review.glob("dist/*/items-*.csv")} == items_before
    assert (review / "private" / "mapping.csv").read_bytes() == mapping_before
    assert manifest["refreshes"][-1]["reason"] == "test refresh" and manifest["refreshes"][-1]["items_and_mapping_unchanged"]
    for code, info in manifest["packages"].items():
        zpath = review / "dist" / f"{code}.zip"
        assert build_mod.sha256(zpath) == info["zip"] and zpath.read_bytes() != zips_before[zpath.name]
        with zipfile.ZipFile(zpath) as z:
            text = z.read(f"{code}/INSTRUCTIONS.md").decode("utf-8")
            assert "REFRESH-MARKER" in text and "A must follow from B" in text
            assert z.read(f"{code}/items-{code}.csv") == items_before[f"items-{code}.csv"]
    registry = read(review / "private" / "reviewer-registry.csv")
    registry[0]["fluency_evidence"] = "native speaker"
    with (review / "private" / "reviewer-registry.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(registry[0].keys())); w.writeheader(); w.writerows(registry)
    with pytest.raises(ValueError, match="already contains entries"):
        build_mod.refresh_instructions(review, "second")


def test_rebuild_refuses_to_regenerate_ids(tiny):
    with pytest.raises(FileExistsError):
        build_mod.build(tiny / "data", tiny / "review")


def test_merge_rejects_missing_and_edited_submissions(tiny, tmp_path):
    review = tmp_path / "review"; shutil.copytree(tiny / "review", review)
    with pytest.raises(ValueError, match="incomplete or invalid"):
        merge_mod.merge(review, tiny / "data")
    problems = json.loads((review / "submissions" / "merged" / "merge-problems.json").read_text(encoding="utf-8"))
    assert len(problems) == 6 and all("expected exactly one CSV" in v[0] for v in problems.values())
    synthetic_submissions(review, tiny / "data", lambda o: (o["label"], "yes", ""))
    code = sorted(problems)[0]
    path = review / "submissions" / "raw" / code / f"items-{code}.csv"
    rows = read(path); rows[0]["sentence_a"] += " EDITED"; rows[1]["judged_label"] = "maybe"; del rows[2]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError):
        merge_mod.merge(review, tiny / "data")
    problems = json.loads((review / "submissions" / "merged" / "merge-problems.json").read_text(encoding="utf-8"))
    joined = " ".join(problems[code])
    assert "edited" in joined and "judged_label must be" in joined and "1 items missing" in joined


def test_merge_preserves_both_judgments_and_reports_disagreements(tiny, tmp_path):
    review = tmp_path / "review"; shutil.copytree(tiny / "review", review)
    counter = {"n": 0}
    def judge(o):
        counter["n"] += 1
        flip = counter["n"] % 7 == 0
        return (str(1 - int(o["label"])) if flip else o["label"], "no" if counter["n"] % 11 == 0 else "yes", "unnatural" if counter["n"] % 11 == 0 else "")
    synthetic_submissions(review, tiny / "data", judge)
    receipt = merge_mod.merge(review, tiny / "data")
    merged = read(review / "submissions" / "merged" / "reviewed-audit-sample.csv")
    originals = read(tiny / "data" / "audit-sample.csv")
    assert len(merged) == len(originals) and {r["id"] for r in merged} == {r["id"] for r in originals}
    for r in merged:
        assert not r["reviewer"].startswith("reviewer_") and r["judged_label"] in ("0", "1") and r["fluent"] in ("yes", "no")
    draft = json.loads((review / "submissions" / "merged" / "attestation-draft.json").read_text(encoding="utf-8"))
    assert draft["all_templates_and_forms_checked"] is False
    assert set(draft["resolved_items"]) == {r["id"] for r in read(review / "submissions" / "merged" / "disagreements.csv")}
    assert all(v["label"] is None and v["rationale"] == "" for v in draft["resolved_items"].values())
    assert receipt["unresolved_items"] > 0
    # The merged file reaches certify-review's item-count rule (tiny corpus has < 200 items), i.e. identities parse.
    attest = tmp_path / "attest.json"; attest.write_text(json.dumps(draft), encoding="utf-8")
    with pytest.raises(ValueError, match="200 audited items"):
        certify_review(tiny / "data", review / "submissions" / "merged" / "reviewed-audit-sample.csv", attest, tmp_path / "cert.json")


@pytest.mark.skipif(not (ROOT / "data" / "draft-v4.3" / "audit-sample.csv").exists(), reason="draft-v4.3 not present")
def test_real_sample_pipeline_reaches_certify(tmp_path):
    """Format check only: perfect synthetic agreement in tmp_path. Not human review; never touches review/."""
    data = tmp_path / "data"; shutil.copytree(ROOT / "data" / "draft-v4.3", data)
    review = tmp_path / "review"
    manifest = build_mod.build(data, review)
    assert manifest["items_per_language"] == {"en": 200, "de": 200, "ko": 200}
    assert manifest["blinding"]["construction_rows_overlapping_audit_items"] >= 1
    synthetic_submissions(review, data, lambda o: (o["label"], "yes", ""))
    merge_mod.merge(review, data)
    draft = json.loads((review / "submissions" / "merged" / "attestation-draft.json").read_text(encoding="utf-8"))
    assert draft["resolved_items"] == {}
    draft["all_templates_and_forms_checked"] = True  # fixture only
    attest = tmp_path / "attest.json"; attest.write_text(json.dumps(draft), encoding="utf-8")
    report = certify_review(data, review / "submissions" / "merged" / "reviewed-audit-sample.csv", attest, tmp_path / "cert.json")
    assert report["status"] == "passed" and report["sample_items"] == 600
    assert all(not name.startswith("reviewer_") for names in report["reviewers"].values() for name in names)

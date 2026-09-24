"""Hugging Face upload bundles: two dataset repositories built from the repository by an explicit allow-list.
Checkpoints (*.pt), graph files (*.npz) and review/ never enter a bundle; every bundled file is listed with its sha256."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_hf_bundles", ROOT / "scripts" / "build_hf_bundles.py")
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)


def touch(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")


def fake_repo(tmp_path):
    root = tmp_path / "repo"
    for name in ("train.jsonl", "dev_a.jsonl", "dev_b.jsonl", "test.jsonl", "manifest.json", "audit.json", "template-inventory.json"):
        touch(root / "data/draft-v4.6" / name, f"{name}\n")
    touch(root / "data/draft-v4.6/audit.json.partial", "junk")
    touch(root / "artifacts/tokenizer-wordbound-v4.6.json"); touch(root / "artifacts/tokenizer-wordbound-v4.6.meta.json")
    touch(root / "artifacts/graphs/real.npz", "graph"); touch(root / "artifacts/graphs/real.json", "{}")
    touch(root / "review/v5/secret.txt", "private")
    touch(root / "hf/items/README.md", "---\nlicense: cc-by-4.0\n---\nitems card\n")
    touch(root / "hf/main-runs/README.md", "---\nlicense: cc-by-4.0\n---\nruns card\n")
    touch(root / "docs/TASK_DEFINITION_V5.1.1.md", "task")
    for rel in mod.RUNS_EXTRA_FILES:
        touch(root / rel, "{}" if rel.endswith(".json") else "x")
    for name in ("extended-grid-v5.1.jsonl", "fwer-official-v5.1.jsonl", "fwer-official-v5.1.meta.json"):
        touch(root / "artifacts/power" / name, "row\n")
    runs = root / "runs/main-v5.1"
    for rid, seed, mode, order, stop, first in (("main-real-1-mixed", 1, "mixed", ["en", "de", "ko"], "mastered", 400000),
                                                 ("main-real-2-mono-de", 2, "mono", ["de"], "administrative_cap", None)):
        touch(runs / rid / "summary.json", dict(run_id=rid, seed=seed, mode=mode, order=order, stop_reason=stop, seen=first or 200000, cap=900000 if mode == "mixed" else 200000,
                                                first_global=first, gpu_hours=.1, elapsed_seconds=360.0, code_hash="c", protocol_hash="p", dataset_hash="d"))
        touch(runs / rid / "metadata.json", dict(device="cuda:0")); touch(runs / rid / "events.jsonl", '{"kind": "train"}\n')
        touch(runs / rid / "independent-test.json", dict(scores={})); touch(runs / rid / "terminal-auxiliary.json", dict(scores={}))
        touch(runs / rid / f"attempt-{rid}.json", dict(run_id=rid)); touch(runs / rid / "primary.pt", "CKPT"); touch(runs / rid / "latest.pt", "CKPT")
    touch(runs / "main-real-2-mono-de.main-real-2-mono-de-retry1.console.log", "log")
    failed = root / "runs/main-v5.1-failed-attempts/main-real-2-mono-de-attempt1"
    touch(failed / "failure.json", dict(error="x")); touch(failed / "kernel-xid.txt", "xid"); touch(failed / "run-folder/latest.pt", "CKPT"); touch(failed / "run-folder/events.jsonl", "e")
    analysis = root / "reports/main-v5.1-analysis-truncated"
    touch(analysis / "analysis.json", dict(selection=dict(included_seeds=[1]))); touch(analysis / "figures/rmst-by-order.png", "png")
    return root


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_of(bundle):
    return {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (bundle / "MANIFEST.sha256").read_text().splitlines() if line}


def test_items_bundle_copies_data_card_and_lists_every_file_with_sha256(tmp_path):
    root = fake_repo(tmp_path)
    bundle = mod.items_bundle(root, tmp_path / "out")
    assert (bundle / "README.md").read_text().startswith("---\nlicense: cc-by-4.0")
    assert (bundle / "data/train.jsonl").exists() and (bundle / "data/manifest.json").exists() and (bundle / "tokenizer/tokenizer-wordbound-v4.6.json").exists()
    assert not (bundle / "data/audit.json.partial").exists()
    listed = manifest_of(bundle)
    files = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"}
    assert set(listed) == files and all(listed[f] == sha(bundle / f) for f in files)


def test_runs_bundle_excludes_checkpoints_graph_and_review_but_keeps_records(tmp_path):
    root = fake_repo(tmp_path)
    bundle = mod.runs_bundle(root, tmp_path / "out")
    names = {p.name for p in bundle.rglob("*") if p.is_file()}
    assert not any(n.endswith((".pt", ".npz")) for n in names) and "secret.txt" not in names
    assert (bundle / "runs/main-real-1-mixed/events.jsonl").exists() and (bundle / "runs/main-real-1-mixed/attempt-main-real-1-mixed.json").exists()
    assert (bundle / "failed-attempts/main-real-2-mono-de-attempt1/run-folder/events.jsonl").exists() and (bundle / "failed-attempts/main-real-2-mono-de-attempt1/kernel-xid.txt").exists()
    assert (bundle / "console-logs/main-real-2-mono-de.main-real-2-mono-de-retry1.console.log").exists()
    assert (bundle / "analysis/analysis.json").exists() and (bundle / "analysis/figures/rmst-by-order.png").exists()
    assert (bundle / "design/fwer-official-v5.1.jsonl").exists() and (bundle / "README.md").read_text().endswith("runs card\n")
    listed = manifest_of(bundle)
    files = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"}
    assert set(listed) == files


def test_runs_table_has_one_row_per_run_with_prefix_flag(tmp_path):
    root = fake_repo(tmp_path)
    bundle = mod.runs_bundle(root, tmp_path / "out")
    rows = [json.loads(l) for l in (bundle / "runs.jsonl").read_text().splitlines()]
    assert [r["run_id"] for r in rows] == ["main-real-1-mixed", "main-real-2-mono-de"]
    assert rows[0]["in_analysis_prefix"] is True and rows[1]["in_analysis_prefix"] is False
    assert rows[0]["restricted_cost"] == 400000 and rows[1]["restricted_cost"] == 200000 and rows[1]["device"] == "cuda:0"
    assert rows[0]["condition"] == "mixed" and rows[1]["condition"] == "mono-de"


def test_bundles_refuse_to_overwrite(tmp_path):
    root = fake_repo(tmp_path)
    mod.items_bundle(root, tmp_path / "out")
    with pytest.raises(FileExistsError):
        mod.items_bundle(root, tmp_path / "out")

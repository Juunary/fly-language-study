#!/usr/bin/env python
"""Assemble the two Hugging Face dataset bundles from the repository by an explicit allow-list.

  items  -> <output>/fly-language-study-items      the sentence-pair benchmark (data/draft-v4.6), tokenizer, task
                                                   definition and certification records
  runs   -> <output>/fly-language-study-main-runs  the main-study run records (753 runs), failed attempts, ledger,
                                                   launch manifest, sample-size design files, analysis outputs, docs

Never bundled: checkpoints (*.pt), connectome graph files (*.npz, artifacts/graphs), anything under review/, partial or
pid files. Every bundled file is listed in MANIFEST.sha256 (sha256sum format). Bundles refuse to overwrite an existing
directory. Upload is done separately by the account owner (huggingface-cli upload <repo> <bundle> --repo-type dataset).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

EXCLUDED_SUFFIXES = (".pt", ".npz", ".partial", ".pid", ".pth", ".safetensors")
EXCLUDED_PARTS = ("review",)
RUN_FILES = ("summary.json", "metadata.json", "independent-test.json", "terminal-auxiliary.json", "events.jsonl")

ITEMS_EXTRA_FILES = {
    "artifacts/tokenizer-wordbound-v4.6.json": "tokenizer/tokenizer-wordbound-v4.6.json",
    "artifacts/tokenizer-wordbound-v4.6.meta.json": "tokenizer/tokenizer-wordbound-v4.6.meta.json",
    "docs/TASK_DEFINITION_V5.1.1.md": "docs/TASK_DEFINITION_V5.1.1.md",
}
ITEMS_EXTRA_GLOBS = {"docs/AI_REVIEW_GUIDE.md": "docs", "reports/ai-review-*.json": "certification"}

RUNS_EXTRA_FILES = {
    "reports/launch-manifest-v5.1-N80.json": "manifest/launch-manifest-v5.1-N80.json",
    "configs/protocol-v5.1-main-N80.json": "manifest/protocol-v5.1-main-N80.json",
    "configs/main-v5.1-retries.json": "manifest/main-v5.1-retries.json",
    "reports/design-extended-v5.1-N2.json": "design/design-extended-v5.1-N2.json",
    "runs/gpu-ledger.json": "ledger/gpu-ledger.json",
    "runs/main-v5.1-campaign.stopped-by-operator": "ledger/main-v5.1-campaign.stopped-by-operator",
    "reports/MAIN_STUDY_RESULTS_V5.1.md": "docs/MAIN_STUDY_RESULTS_V5.1.md",
    "reports/MAIN_STUDY_PLAN_V5.md": "docs/MAIN_STUDY_PLAN_V5.md",
    "docs/PROTOCOL_V5.md": "docs/PROTOCOL_V5.md",
}
RUNS_EXTRA_GLOBS = {
    "configs/main-v5.1-retry-reservations-*.json": "manifest",
    "reports/design-extended-v5.1*.json": "design",
    "reports/fwer-*-v5.1.json": "design",
    "artifacts/power/*": "design",
    "docs/EXTENDED_DESIGN_V5.1.md": "docs",
    "docs/FWER_EVALUATION_AMENDMENT_V5.1.md": "docs",
}


def excluded(path: Path) -> bool:
    return path.name.endswith(EXCLUDED_SUFFIXES) or any(part in EXCLUDED_PARTS for part in path.parts)


def copy_file(src: Path, dst: Path):
    if excluded(src):
        raise ValueError(f"Refusing to bundle excluded file: {src}")
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path):
    for p in sorted(src.rglob("*")):
        if p.is_file() and not excluded(p):
            copy_file(p, dst / p.relative_to(src))


def copy_extras(root: Path, bundle: Path, files: dict, globs: dict):
    missing = [rel for rel in files if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Required files missing: {missing}")
    for rel, dst in files.items():
        copy_file(root / rel, bundle / dst)
    for pattern, dst_dir in globs.items():
        for p in sorted(root.glob(pattern)):
            if p.is_file() and not excluded(p):
                copy_file(p, bundle / dst_dir / p.name)


def write_manifest(bundle: Path):
    lines = []
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(bundle).as_posix()}")
    (bundle / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _new_bundle(output: Path, name: str) -> Path:
    bundle = Path(output) / name
    if bundle.exists():
        raise FileExistsError(f"{bundle} exists; remove it to rebuild")
    bundle.mkdir(parents=True)
    return bundle


def items_bundle(root, output, data_version="draft-v4.6"):
    root = Path(root)
    bundle = _new_bundle(output, "fly-language-study-items")
    copy_file(root / "hf/items/README.md", bundle / "README.md")
    copy_tree(root / "data" / data_version, bundle / "data")
    copy_extras(root, bundle, ITEMS_EXTRA_FILES, ITEMS_EXTRA_GLOBS)
    write_manifest(bundle)
    return bundle


def condition_name(mode, order):
    return {"mono": f"mono-{order[0]}", "sequential": "seq-" + "-".join(order), "mixed": "mixed"}[mode]


def runs_table(runs_root: Path, included_seeds) -> list:
    rows = []
    for folder in sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "summary.json").is_file()):
        s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if (folder / "metadata.json").is_file() else {}
        first = s.get("first_global")
        rows.append(dict(run_id=s["run_id"], seed=s["seed"], mode=s["mode"], order="-".join(s["order"]), condition=condition_name(s["mode"], s["order"]),
                         stop_reason=s["stop_reason"], seen=s.get("seen"), cap=s["cap"], first_global=first, restricted_cost=s["cap"] if first is None else first,
                         gpu_hours=s.get("gpu_hours"), elapsed_seconds=s.get("elapsed_seconds"), device=meta.get("device", s.get("device")),
                         in_analysis_prefix=s["seed"] in set(included_seeds),
                         code_hash=s.get("code_hash"), protocol_hash=s.get("protocol_hash"), dataset_hash=s.get("dataset_hash")))
    return rows


def runs_bundle(root, output, runs="runs/main-v5.1", failed="runs/main-v5.1-failed-attempts", analysis="reports/main-v5.1-analysis-truncated"):
    root = Path(root)
    bundle = _new_bundle(output, "fly-language-study-main-runs")
    copy_file(root / "hf/main-runs/README.md", bundle / "README.md")
    runs_root = root / runs
    for folder in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for name in RUN_FILES:
            if (folder / name).is_file():
                copy_file(folder / name, bundle / "runs" / folder.name / name)
        for p in sorted(folder.glob("attempt-*.json")):
            copy_file(p, bundle / "runs" / folder.name / p.name)
    for p in sorted(runs_root.glob("*.console.log")):
        copy_file(p, bundle / "console-logs" / p.name)
    if (root / failed).is_dir():
        copy_tree(root / failed, bundle / "failed-attempts")
    copy_tree(root / analysis, bundle / "analysis")
    copy_extras(root, bundle, RUNS_EXTRA_FILES, RUNS_EXTRA_GLOBS)
    report = json.loads((root / analysis / "analysis.json").read_text(encoding="utf-8"))
    rows = runs_table(runs_root, report["selection"]["included_seeds"])
    (bundle / "runs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_manifest(bundle)
    return bundle


def summarize(bundle: Path):
    files = [p for p in bundle.rglob("*") if p.is_file()]
    return dict(bundle=str(bundle), files=len(files), megabytes=round(sum(p.stat().st_size for p in files) / 1048576, 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="hf-upload")
    parser.add_argument("--which", choices=("items", "runs", "both"), default="both")
    args = parser.parse_args(argv)
    built = []
    if args.which in ("items", "both"):
        built.append(items_bundle(args.root, args.output))
    if args.which in ("runs", "both"):
        built.append(runs_bundle(args.root, args.output))
    for b in built:
        print(json.dumps(summarize(b)))


if __name__ == "__main__":
    main()

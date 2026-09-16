"""Verify that draft-v4.4's primary/auxiliary evaluation path reproduces the exploratory diagnostics. Not study evidence.

CPU: v4.4 keeps v4.3's ids, meaning IDs, labels, scenes and training bytes; the primary rendering equals the
declarative diagnostic input (audit_token_exposure.assertion_view) and the auxiliary rendering equals the v4.3
sentences; token counts and unseen-token fractions match reports/token-exposure-v1.json.
GPU: runtime.evaluate on the v4.4 primary and auxiliary views of dev A/B reproduces, cell by cell, the recorded
declarative/original counts of the 5 exploratory-v1 and 12 exploratory-v2 checkpoints. Evaluation runs inside
immutable_evaluation, which raises if any parameter or buffer changes. No parameter update, no test split.
Actual wall time is charged to one new contingency ('reserve') ledger entry.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

from flystudy.budget import Ledger
from flystudy.data import encode_pair, load_rows
from flystudy.gates import code_hash, environment
from flystudy.graph import Graph
from flystudy.model import FlyClassifier
from flystudy.protocol import LANGUAGES, TASKS, Protocol, file_hash, write_json
from flystudy.runtime import Corpus, evaluate

SPLITS = ("dev_a", "dev_b")
_spec = importlib.util.spec_from_file_location("audit_token_exposure", Path(__file__).with_name("audit_token_exposure.py"))
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def cpu_checks(old, new, tokenizer, exposure_report):
    result = dict(training_bytes_identical=file_hash(Path(old)/"train.jsonl") == file_hash(Path(new)/"train.jsonl"), splits={})
    training = {lang: set() for lang in LANGUAGES}
    for row in load_rows(new, "train"):
        training[row["language"]].update(encode_pair(tokenizer, row))
    pooled = set().union(*training.values())
    exposure = {(c["split"], c["language"], c["view"]): c for c in exposure_report["cells"]}
    for split in SPLITS + ("test",):
        before = {r["id"]: r for r in load_rows(old, split)}
        after = load_rows(new, split)
        c = Counter(rows=len(after))
        tokens = defaultdict(lambda: Counter())
        for r in after:
            o = before.get(r["id"])
            c["id_known"] += o is not None
            if o is None:
                continue
            c["metadata_equal"] += all(o[k] == r[k] for k in ("meaning_id", "foil_group", "split", "task", "language", "label", "gender_pair", "scene"))
            c["auxiliary_equals_v43_sentences"] += (r["auxiliary"]["sentence_a"], r["auxiliary"]["sentence_b"]) == (o["sentence_a"], o["sentence_b"])
            c["auxiliary_family_equals_v43"] += r["auxiliary"]["template_family"] == o["template_family"]
            view = _audit.assertion_view(o)
            c["primary_equals_assertion_view"] += (r["sentence_a"], r["sentence_b"]) == (view["sentence_a"], view["sentence_b"])
            for name, item in (("primary", r), ("auxiliary", r["auxiliary"])):
                ids = encode_pair(tokenizer, item)
                t = tokens[(r["language"], name)]
                t["items"] += 1; t["tokens"] += len(ids); t["unseen_any"] += sum(i not in pooled for i in ids)
        c["counts_by_language_task_equal"] = Counter((r["language"], r["task"]) for r in after) == Counter((r["language"], r["task"]) for r in before.values())
        c["meaning_ids_equal"] = {r["meaning_id"] for r in after} == {r["meaning_id"] for r in before.values()}
        exposure_match = {}
        for (lang, name), t in tokens.items():
            recorded = exposure.get((split, lang, "assertion" if name == "primary" else "original"))
            entry = dict(items=t["items"], tokens=t["tokens"], mean_tokens=t["tokens"]/t["items"], unseen_fraction=t["unseen_any"]/t["tokens"])
            if recorded:
                entry["matches_token_exposure_v1"] = (recorded["tokens"] == t["tokens"] and recorded["items"] == t["items"]
                                                      and abs(recorded["unseen_in_any_training_fraction"]-entry["unseen_fraction"]) < 1e-12)
            exposure_match[f"{lang}/{name}"] = entry
        result["splits"][split] = dict(**c, tokens=exposure_match)
    return result


def recorded_cells(frame_report, v2_report):
    """(checkpoint path, hash, label) -> {(split, view, lang/task): [correct, items]} from the two exploratory reports."""
    targets = []
    for run in frame_report["runs"]:
        cells = {(c["split"], "primary" if c["view"] == "assertion" else "auxiliary", f"{c['language']}/{c['task']}"): [c["correct"], c["items"]]
                 for c in run["cells"]}
        targets.append(dict(source="frame-diagnostic-v1", run_id=run["run_id"], stage="final", path=Path("runs/exploratory-v1")/run["run_id"]/"primary.pt",
                            checkpoint_hash=run["checkpoint_hash"], languages=run["languages"], cells=cells))
    for e in v2_report["evaluations"]:
        if e["source"] != "v2":
            continue
        cells = {}
        for c in e["cells"]:
            if c["set"].startswith("dev_"):
                split, view = c["set"].rsplit("_", 1)
                cells[(split, "primary" if view == "assertion" else "auxiliary", f"{c['language']}/{c['task']}")] = [c["correct"], c["items"]]
        targets.append(dict(source="exploratory-v2", run_id=e["run_id"], stage=e["stage"], path=Path(e["checkpoint_path"]),
                            checkpoint_hash=e["checkpoint_hash"], languages=e["languages"], cells=cells))
    return targets


def gpu_checks(targets, data, tokenizer, graph, protocol, device, microbatch):
    corpus = Corpus(data, tokenizer)
    results = []
    for t in targets:
        if file_hash(t["path"]) != t["checkpoint_hash"]:
            raise ValueError(f"{t['path']}: checkpoint hash changed")
        begin = time.perf_counter()
        ck = torch.load(t["path"], map_location=device, weights_only=True)
        model = FlyClassifier(graph, tokenizer.get_vocab_size(), protocol.embed_dim, protocol.microsteps, ck["metadata"]["seed"], "cuda").to(device)
        model.load_state_dict(ck["model"])
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        scored, mismatches = {}, []
        for split in SPLITS:
            for view in ("primary", "auxiliary"):
                scores, _ = evaluate(model, corpus, split, tuple(t["languages"]), microbatch, view=view)  # immutable_evaluation inside
                for cell, count in scores.items():
                    scored[(split, view, cell)] = count
                    if t["cells"].get((split, view, cell)) != count:
                        mismatches.append(dict(split=split, view=view, cell=cell, recorded=t["cells"].get((split, view, cell)), now=count))
        unchanged = all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        results.append(dict(source=t["source"], run_id=t["run_id"], stage=t["stage"], checkpoint_hash=t["checkpoint_hash"], seen=ck["curriculum"]["seen"],
                            cells_compared=len(scored), cells_recorded=len(t["cells"]), mismatches=mismatches, reproduced=not mismatches and len(scored) == len(t["cells"]),
                            parameters_unchanged=unchanged, seconds=time.perf_counter()-begin))
        del model, ck
        torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", default="data/draft-v4.3"); p.add_argument("--new", default="data/draft-v4.4")
    p.add_argument("--tokenizer", default="artifacts/tokenizer-v4.4.json"); p.add_argument("--graph", default="artifacts/graphs/real.npz")
    p.add_argument("--protocol", default="configs/protocol-v4-ai.json"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=256); p.add_argument("--ledger", default="runs/gpu-ledger.json")
    p.add_argument("--reservation-id", default="eval-revision-check-1"); p.add_argument("--hours", type=float, default=.1)
    p.add_argument("--frame-report", default="reports/frame-diagnostic-v1.json"); p.add_argument("--v2-report", default="reports/exploratory-v2-report.json")
    p.add_argument("--output", default="reports/evaluation-revision-check.json")
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError("Keep previous checks; use a new output path")
    tokenizer = Tokenizer.from_file(a.tokenizer)
    frame_report = json.loads(Path(a.frame_report).read_text(encoding="utf-8"))
    v2_report = json.loads(Path(a.v2_report).read_text(encoding="utf-8"))
    exposure = json.loads(Path("reports/token-exposure-v1.json").read_text(encoding="utf-8"))
    report = dict(kind="evaluation_revision_check", study_evidence=False, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  code_hash=code_hash(), script_hash=file_hash(__file__), old=dict(path=a.old, manifest=json.loads((Path(a.old)/"manifest.json").read_text(encoding="utf-8"))["dataset_hash"]),
                  new=dict(path=a.new, manifest=json.loads((Path(a.new)/"manifest.json").read_text(encoding="utf-8"))["dataset_hash"]),
                  tokenizer_hash=file_hash(a.tokenizer), tokenizer_identical_to_v43=file_hash(a.tokenizer) == file_hash("artifacts/tokenizer-v4.3.json"),
                  parameter_updates=0, test_split_used=False)
    report["cpu"] = cpu_checks(a.old, a.new, tokenizer, exposure)
    ledger = Ledger(a.ledger)
    ledger.reserve_bundle([dict(run_id=a.reservation_id, category="reserve", hours=a.hours)])
    started, status = time.perf_counter(), "technical_failure"
    try:
        report["environment"] = environment()
        targets = recorded_cells(frame_report, v2_report)
        report["gpu"] = gpu_checks(targets, a.new, tokenizer, Graph.load(a.graph), Protocol.load(a.protocol), a.device, a.microbatch)
        status = "completed"
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        hours = (time.perf_counter()-started)/3600
        ledger.charge(a.reservation_id, hours, status)
        report["ledger"] = dict(path=a.ledger, reservation_id=a.reservation_id, reserved_hours=a.hours, charged_hours=hours, status=status)
        report["summary"] = dict(checkpoints=len(report.get("gpu", [])), reproduced=sum(r["reproduced"] for r in report.get("gpu", [])),
                                 parameters_unchanged=all(r["parameters_unchanged"] for r in report.get("gpu", [])) if report.get("gpu") else None,
                                 cpu_all_rows_matched=all(s["rows"] == s["id_known"] == s["metadata_equal"] == s["auxiliary_equals_v43_sentences"]
                                                          == s["primary_equals_assertion_view"] for s in report["cpu"]["splits"].values()),
                                 token_exposure_matched=all(e.get("matches_token_exposure_v1", True) for s in report["cpu"]["splits"].values() for e in s["tokens"].values()))
        write_json(a.output, report)
    print(json.dumps(report["summary"], indent=1)); print(json.dumps(report["ledger"], indent=1))


if __name__ == "__main__":
    main()

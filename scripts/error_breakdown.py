"""Per-item error breakdown of the exploratory-v2 900k checkpoints on the draft-v4.4 primary evaluation. Not study evidence.

Cells: EN space, DE roles, DE space (the cells that stay low in the declarative frame), plus EN roles and KO space as
contrasts. Accuracy is broken down by axis/direction (space), verb pair (roles) and label. Read-only; charged to a
new contingency ('reserve') ledger entry.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

from flystudy.budget import Ledger
from flystudy.gates import code_hash
from flystudy.graph import Graph
from flystudy.model import FlyClassifier, collate
from flystudy.protocol import Protocol, file_hash, write_json
from flystudy.runtime import Corpus, immutable_evaluation

CELLS = {"en": ("space", "roles"), "de": ("roles", "space"), "ko": ("space",)}
AXES = ("left/right", "above/below", "front/behind")


def breakdown(items, model, pad_id, microbatch, device):
    groups = defaultdict(lambda: [0, 0])
    ordered = sorted(items, key=lambda x: len(x[1]))
    for start in range(0, len(ordered), microbatch):
        part = ordered[start:start+microbatch]
        x, lengths, y = collate([(ids, r["label"]) for r, ids in part], pad_id, device)
        ok = (model(x, lengths).argmax(-1) == y).tolist()
        for hit, (r, _) in zip(ok, part):
            s = r["scene"]
            keys = [("all", ""), ("label", str(r["label"]))]
            if r["task"] == "space":
                keys += [("axis", AXES[s["axis"]]), ("direction", str(s["direction"])), ("axis+direction", f"{AXES[s['axis']]}:{s['direction']}")]
            if r["task"] == "roles":
                keys += [("verb_pair", f"{min(s['verb'], s['verb2'])}-{max(s['verb'], s['verb2'])}"), ("verb_order", f"{s['verb']}>{s['verb2']}")]
            for k in keys:
                groups[k][0] += hit; groups[k][1] += 1
    return {f"{k[0]}={k[1]}" if k[1] else k[0]: dict(correct=c, items=n, accuracy=c/n) for k, (c, n) in sorted(groups.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/draft-v4.4"); p.add_argument("--tokenizer", default="artifacts/tokenizer-v4.4.json")
    p.add_argument("--graph", default="artifacts/graphs/real.npz"); p.add_argument("--protocol", default="configs/protocol-v4-ai.json")
    p.add_argument("--root", default="runs/exploratory-v2"); p.add_argument("--stage", default="900k"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=256); p.add_argument("--ledger", default="runs/gpu-ledger.json")
    p.add_argument("--reservation-id", default="eval-revision-errors-1"); p.add_argument("--hours", type=float, default=.05)
    p.add_argument("--output", default="reports/remaining-issues-v4.4.json")
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError("Keep previous reports; use a new output path")
    ledger = Ledger(a.ledger)
    ledger.reserve_bundle([dict(run_id=a.reservation_id, category="reserve", hours=a.hours)])
    started, status = time.perf_counter(), "technical_failure"
    report = dict(kind="exploratory_error_breakdown", study_evidence=False, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  code_hash=code_hash(), data=a.data, stage=a.stage, parameter_updates=0, test_split_used=False, checkpoints=[])
    try:
        tokenizer = Tokenizer.from_file(a.tokenizer)
        corpus, graph, protocol = Corpus(a.data, tokenizer), Graph.load(a.graph), Protocol.load(a.protocol)
        pad_id = tokenizer.token_to_id("[PAD]")
        for lang, tasks in CELLS.items():
            run_dir = Path(a.root)/f"explore-v2-mono-{lang}"
            stage = next(s for s in json.loads((run_dir/"stages.json").read_text(encoding="utf-8")) if s["label"] == a.stage)
            if file_hash(stage["checkpoint"]) != stage["checkpoint_hash"]:
                raise ValueError("checkpoint changed")
            ck = torch.load(stage["checkpoint"], map_location=a.device, weights_only=True)
            model = FlyClassifier(graph, tokenizer.get_vocab_size(), protocol.embed_dim, protocol.microsteps, ck["metadata"]["seed"], "cuda").to(a.device)
            model.load_state_dict(ck["model"])
            entry = dict(run_id=ck["metadata"]["run_id"], seen=ck["curriculum"]["seen"], checkpoint_hash=stage["checkpoint_hash"], cells={})
            with immutable_evaluation(model):
                for task in tasks:
                    for split in ("dev_a", "dev_b"):
                        items = [(r, ids) for r, ids in corpus.split(split)[lang] if r["task"] == task]
                        entry["cells"][f"{lang}/{task}/{split}"] = breakdown(items, model, pad_id, a.microbatch, a.device)
            report["checkpoints"].append(entry)
            del model, ck
            torch.cuda.empty_cache()
        status = "completed"
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        hours = (time.perf_counter()-started)/3600
        ledger.charge(a.reservation_id, hours, status)
        report["ledger"] = dict(path=a.ledger, reservation_id=a.reservation_id, reserved_hours=a.hours, charged_hours=hours, status=status)
        write_json(a.output, report)
    for entry in report["checkpoints"]:
        for cell, groups in entry["cells"].items():
            print(cell, {k: round(v["accuracy"], 3) for k, v in groups.items()})
    print(json.dumps(report["ledger"], indent=1))


if __name__ == "__main__":
    main()

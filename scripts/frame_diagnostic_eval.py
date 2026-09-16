"""Frame diagnostic on the five exploratory-v1 primary checkpoints (reports/TOKEN_EXPOSURE_V1.md).

Each checkpoint scores dev A/B twice: the original rows and `assertion_view(row)`, which keeps the held-out
scene, IDs and label and renders it in the training-style declarative frame. Monolingual checkpoints score
their language; sequential and mixed checkpoints score all three. No parameter update, no test split, no
change to data, checkpoints, summaries or independent tests. The assertion view relaxes the outer-frame
holdout, so this is an exploratory diagnostic, not a confirmatory evaluation.

Actual wall time (failures included) is charged to one new contingency ('reserve') ledger entry.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

from flystudy.budget import Ledger
from flystudy.data import encode_pair, load_rows, render
from flystudy.gates import code_hash, environment
from flystudy.graph import Graph
from flystudy.model import FlyClassifier, collate
from flystudy.protocol import LANGUAGES, TASKS, Protocol, file_hash, write_json
from flystudy.runtime import immutable_evaluation, sync

SPLITS = ("dev_a", "dev_b")
VIEWS = ("original", "assertion")
RUN_FILES = ("primary.pt", "latest.pt", "summary.json", "independent-test.json", "events.jsonl", "metadata.json")
_spec = importlib.util.spec_from_file_location("audit_token_exposure", Path(__file__).with_name("audit_token_exposure.py"))
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
assertion_view = _audit.assertion_view


def wilson(correct, n, z=1.959964):
    p, d = correct/n, 1+z*z/n
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [(p+z*z/(2*n))/d-half, (p+z*z/(2*n))/d+half]


def preserved_hashes(args, manifest):
    paths = [Path(args.root)/"exploratory-manifest.json", Path(args.tokenizer), Path(args.tokenizer).with_suffix(".meta.json"),
             Path(args.graph), *(Path(args.data)/f for f in ("train.jsonl", "dev_a.jsonl", "dev_b.jsonl", "test.jsonl", "manifest.json"))]
    paths += [Path(args.root)/r["run_id"]/name for r in manifest["runs"] for name in RUN_FILES]
    return {str(p): file_hash(p) for p in paths}


def build_views(data, tokenizer):
    """CPU inputs shared by all checkpoints; verifies that only the outer frame changes."""
    train = load_rows(data, "train")
    training = {lang: set() for lang in LANGUAGES}
    for row in train:
        training[row["language"]].update(encode_pair(tokenizer, row))
    pooled = set().union(*training.values())
    train_pairs, train_meanings = {(r["sentence_a"], r["sentence_b"]) for r in train}, {r["meaning_id"] for r in train}
    views, checks, tokens = {}, {}, {}
    for split in SPLITS:
        rows, c = load_rows(data, split), Counter()
        for row in rows:
            view = assertion_view(row)
            c["rows"] += 1
            c["non_sentence_fields_preserved"] += view.keys() == row.keys() and all(
                view[k] == row[k] for k in row if k not in ("sentence_a", "sentence_b"))
            c["original_a_rerendered_exactly"] += render(row["scene"], row["language"], split) == row["sentence_a"]
            c["original_b_rerendered_exactly"] += render(row["scene"], row["language"], split, inverse=True,
                                                         foil=not row["label"]) == row["sentence_b"]
            c["assertion_pair_in_training"] += (view["sentence_a"], view["sentence_b"]) in train_pairs
            c["meaning_id_in_training"] += row["meaning_id"] in train_meanings
            for name, item in (("original", row), ("assertion", view)):
                ids = encode_pair(tokenizer, item)
                views.setdefault((split, name, row["language"]), []).append((row, ids))
                t = tokens.setdefault((split, name, row["language"], row["task"]), Counter())
                t["items"] += 1; t["tokens"] += len(ids)
                t["unseen_any"] += sum(i not in pooled for i in ids)
                t["unseen_language"] += sum(i not in training[row["language"]] for i in ids)
        checks[split] = dict(c)
        n = c["rows"]
        if (c["non_sentence_fields_preserved"] != n or c["original_a_rerendered_exactly"] != n or
                c["original_b_rerendered_exactly"] != n or c["assertion_pair_in_training"] or c["meaning_id_in_training"]):
            raise ValueError(f"Input check failed for {split}: {dict(c)}")
    for group in views.values():
        group.sort(key=lambda x: (x[0]["meaning_id"], x[0]["label"]))  # same order as runtime.Corpus.split
    return views, checks, tokens


def score(model, items, pad_id, microbatch, device):
    """Mirror of runtime.evaluate for one language, plus predicted-label counts."""
    result = {}
    for task in TASKS:
        selected = sorted([(r, ids) for r, ids in items if r["task"] == task], key=lambda x: len(x[1]))
        correct = count = predicted_positive = positives = 0
        for start in range(0, len(selected), microbatch):
            part = selected[start:start+microbatch]
            x, lengths, y = collate([(ids, r["label"]) for r, ids in part], pad_id, device)
            predicted = model(x, lengths).argmax(-1)
            correct += int((predicted == y).sum()); count += len(part)
            predicted_positive += int(predicted.sum()); positives += int(y.sum())
        result[task] = dict(correct=correct, items=count, predicted_positive=predicted_positive, label_positive=positives)
    return result


def gpu_processes(device):
    index = torch.device(device).index or 0
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def execute(args):
    root = Path(args.root)
    manifest = json.loads((root/"exploratory-manifest.json").read_text(encoding="utf-8"))
    if manifest["code_hash"] != code_hash():
        raise ValueError("Code changed since the exploratory checkpoints were trained")
    active = gpu_processes(args.device)
    if active:
        raise RuntimeError(f"Other compute processes are active on {args.device}: {active}")
    ledger_before = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    open_reservations = [k for k, r in ledger_before["runs"].items() if r["status"] == "reserved"]
    if open_reservations:
        raise RuntimeError(f"Open ledger reservations exist: {open_reservations}")
    Ledger(args.ledger).reserve_bundle([dict(run_id=args.reservation_id, category="reserve", hours=args.hours)])
    started, status = time.perf_counter(), "technical_failure"
    report = dict(kind="exploratory_frame_diagnostic", study_evidence=False, confirmatory=False,
                  source="reports/TOKEN_EXPOSURE_V1.md", created_at_utc=datetime.now(timezone.utc).isoformat(),
                  note=("Assertion view keeps held-out scenes, meaning IDs and labels but renders them in the training "
                        "declarative frame, relaxing the outer-frame holdout. Not a replacement for dev/test evaluation, "
                        "mastery timing or independent tests, which are unchanged."),
                  code_hash=code_hash(), script_hash=file_hash(__file__), audit_script_hash=file_hash(_spec.origin),
                  protocol_path=manifest["protocol_path"], dataset_hash=manifest["dataset_hash"],
                  tokenizer_hash=file_hash(args.tokenizer), graph_hash=manifest["graph_hash"],
                  device=args.device, microbatch=args.microbatch, parameter_updates=0,
                  other_compute_processes_at_start=active, open_ledger_reservations_at_start=open_reservations,
                  ledger=dict(path=args.ledger, reservation_id=args.reservation_id, category="reserve", reserved_hours=args.hours),
                  runs=[])
    try:
        before = preserved_hashes(args, manifest)
        protocol = Protocol.load(manifest["protocol_path"])
        graph = Graph.load(args.graph)
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        data_manifest = json.loads((Path(args.data)/"manifest.json").read_text(encoding="utf-8"))
        if (protocol.hash != manifest["protocol_hash"] or graph.hash != manifest["graph_hash"] or
                data_manifest["dataset_hash"] != manifest["dataset_hash"] or file_hash(args.tokenizer) != manifest["tokenizer_hash"]):
            raise ValueError("Artifacts differ from the exploratory manifest")
        views, report["input_checks"], tokens = build_views(args.data, tokenizer)
        report["environment"] = environment()
        pad_id = tokenizer.token_to_id("[PAD]")
        interrupted = False
        for planned in manifest["runs"]:
            if (time.perf_counter()-started)/3600 >= args.hours:
                interrupted = True
                break
            run_dir = root/planned["run_id"]
            summary = json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))
            checkpoint_hash = file_hash(run_dir/"primary.pt")
            if json.loads((run_dir/"independent-test.json").read_text(encoding="utf-8"))["checkpoint_hash"] != checkpoint_hash:
                raise ValueError(f"{planned['run_id']}: primary.pt differs from the independent-test checkpoint")
            sync(args.device); begin = time.perf_counter()
            ck = torch.load(run_dir/"primary.pt", map_location=args.device, weights_only=True)
            meta = ck["metadata"]
            for key in ("run_id", "protocol_hash", "graph_hash", "dataset_hash", "tokenizer_hash", "code_hash"):
                if meta[key] != (planned["run_id"] if key == "run_id" else manifest[key]):
                    raise ValueError(f"{planned['run_id']}: checkpoint {key} mismatch")
            model = FlyClassifier(graph, tokenizer.get_vocab_size(), protocol.embed_dim, protocol.microsteps,
                                  meta["seed"], "cuda").to(args.device)
            model.load_state_dict(ck["model"])
            languages = tuple(meta["order"]) if meta["mode"] == "mono" else LANGUAGES
            cells = []
            with immutable_evaluation(model):  # raises if any parameter or buffer changes
                for split in SPLITS:
                    for view in VIEWS:
                        for lang in languages:
                            for task, s in score(model, views[(split, view, lang)], pad_id, args.microbatch, args.device).items():
                                t = tokens[(split, view, lang, task)]
                                if t["items"] != s["items"]:
                                    raise ValueError("Token statistics and scored items disagree")
                                cells.append(dict(split=split, view=view, language=lang, task=task, **s,
                                                  accuracy=s["correct"]/s["items"], wilson95=wilson(s["correct"], s["items"]),
                                                  tokens=t["tokens"], mean_tokens=t["tokens"]/t["items"],
                                                  unseen_in_any_training_fraction=t["unseen_any"]/t["tokens"],
                                                  unseen_in_language_fraction=t["unseen_language"]/t["tokens"]))
            sync(args.device)
            # Primary.pt is saved right after the final scheduled dev_b panel at the cap: the original view must match it.
            events = [json.loads(line) for line in (run_dir/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            final = [e for e in events if e["kind"] == "scheduled"][-1]
            reproduced = None
            if final["seen"] == summary["seen"] and final["panel"] == "dev_b":
                ours = {f"{c['language']}/{c['task']}": [c["correct"], c["items"]] for c in cells
                        if c["split"] == "dev_b" and c["view"] == "original"}
                reproduced = ours == final["scores"]
            report["runs"].append(dict(run_id=planned["run_id"], mode=meta["mode"], order=meta["order"], seed=meta["seed"],
                                       seen=summary["seen"], stop_reason=summary["stop_reason"], checkpoint_hash=checkpoint_hash,
                                       languages=list(languages), final_scheduled_panel=dict(seen=final["seen"], panel=final["panel"]),
                                       original_dev_b_reproduces_final_scheduled_panel=reproduced,
                                       seconds=time.perf_counter()-begin, cells=cells))
            del model, ck
            torch.cuda.empty_cache()
        after = preserved_hashes(args, manifest)
        report["preserved_files"] = dict(sha256=before, unchanged=before == after)
        if before != after:
            raise RuntimeError("Preserved files changed during the diagnostic")
        status = "budget_interruption" if interrupted else "completed"
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        hours = (time.perf_counter()-started)/3600
        Ledger(args.ledger).charge(args.reservation_id, hours, status)
        report["ledger"].update(charged_hours=hours, status=status)
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["comparison"] = [
            dict(run_id=r["run_id"], split=o["split"], language=o["language"], task=o["task"],
                 original_accuracy=o["accuracy"], assertion_accuracy=a["accuracy"], delta=a["accuracy"]-o["accuracy"],
                 original_mean_tokens=o["mean_tokens"], assertion_mean_tokens=a["mean_tokens"],
                 original_unseen_fraction=o["unseen_in_any_training_fraction"],
                 assertion_unseen_fraction=a["unseen_in_any_training_fraction"])
            for r in report["runs"] for o in r["cells"] if o["view"] == "original"
            for a in r["cells"] if a["view"] == "assertion" and (a["split"], a["language"], a["task"]) == (o["split"], o["language"], o["task"])]
        write_json(args.output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/exploratory-v1"); p.add_argument("--ledger", default="runs/gpu-ledger.json")
    p.add_argument("--graph", default="artifacts/graphs/real.npz"); p.add_argument("--data", default="data/draft-v4.3")
    p.add_argument("--tokenizer", default="artifacts/tokenizer-v4.3.json"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=256); p.add_argument("--hours", type=float, default=.25)
    p.add_argument("--reservation-id", default="explore-v1-frame-diagnostic-1")
    p.add_argument("--output", default="reports/frame-diagnostic-v1.json")
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError("Keep previous diagnostics; use a new output path and reservation id")
    if not 0 < a.hours <= .25:
        raise ValueError("This diagnostic is limited to 0.25 GPU-hours")
    r = execute(a)
    for run in r["runs"]:
        print(run["run_id"], "reproduced_final_dev_b" if run["original_dev_b_reproduces_final_scheduled_panel"] else "", f"{run['seconds']:.1f}s")
    print(json.dumps(r["ledger"], indent=1))


if __name__ == "__main__":
    main()

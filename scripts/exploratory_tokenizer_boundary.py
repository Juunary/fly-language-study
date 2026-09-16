"""Exploratory v3: word-boundary byte BPE versus the study's whole-string BPE. NOT study evidence.

One comparison only. A shared byte-level BPE is trained on the same three-language training split with merges
confined to whitespace-delimited words (ByteLevel use_regex=True). EN, DE and KO monolingual runs then train from
scratch with seed 20001 on the exploratory-v2 configuration (same graph, learning rates, batch, microsteps,
protocol-v4-ai-mono900k, stage checkpoints at 199,936 / 300,032 / 600,064 / 899,840 exposures) and the preserved
checkpoints are scored on the draft-v4.4 primary/auxiliary evaluation and the fixed training sample exactly as in
exploratory v2. Scheduled panels during training use draft-v4.3, as in v2, so that the same non-mastering run
structure and cap apply; training bytes are identical in v4.3 and v4.4. The test split is not used.

Because the vocabulary and the embedding parameter count differ from the study tokenizer, the comparison is between
two input-representation policies, not a pure boundary effect.

  --tokenizer  train artifacts/tokenizer-wordbound-v1.json on data/draft-v4.3, record vocabulary, lengths, exposure
  --profile    time 41 updates + one scheduled panel per language on one GPU; project run cost (charged)
  --plan       reserve ledger entries from the projection; write <root>/exploratory-manifest.json
  --execute    run the three languages on two GPUs (one process per language); --retry for unfinished runs
  --single     (internal) one language's four stages in this process
  --evaluate   score the 12 preserved checkpoints; compare with reports/exploratory-v2-report.json (charged)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

from flystudy.budget import Ledger
from flystudy.data import train_tokenizer
from flystudy.gates import code_hash, environment
from flystudy.graph import Graph
from flystudy.model import FlyClassifier
from flystudy.protocol import LANGUAGES, TASKS, Protocol, file_hash, write_json
from flystudy.runtime import Corpus

SEED, BATCH = 20001, 256
STAGES = [("200k", 781), ("300k", 1172), ("600k", 2344), ("900k", 3515)]
PROFILE_ID, EVAL_ID = "explore-v3-wb-profile-1", "explore-v3-wb-eval-1"
LABEL = dict(cohort_label="exploratory", study_evidence=False, review_status="incomplete_ai_review_not_certified",
             question="input representation: word-boundary byte BPE vs whole-string byte BPE, everything else as exploratory v2",
             limitation="vocabulary size and embedding parameter count differ from the study tokenizer, so this compares two "
                        "input-representation policies rather than an isolated boundary effect",
             note="Runs are labelled cohort='smoke' by the training code because that is its only non-gated cohort; they use "
                  "the REAL graph and CUDA backend. Scheduled panels use draft-v4.3 (as in v2); reported comparisons use the "
                  "draft-v4.4 primary/auxiliary evaluation on preserved checkpoints. No mastery claim, no test split.")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v2 = load_module("exploratory_mono_exposure")
_audit = load_module("audit_token_exposure")


def run_id(lang):
    return f"explore-v3-wb-mono-{lang}"


def parameter_count(graph, vocab, protocol):
    model = FlyClassifier(graph, vocab, protocol.embed_dim, protocol.microsteps, SEED, "sparse")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tokenizer_step(args):
    output = Path(args.tokenizer)
    if output.exists():
        raise FileExistsError("Keep the existing tokenizer; use a new artifact name")
    report = train_tokenizer(args.train_data, output, args.vocab_size, boundary="word", strict_vocab=False)
    tokenizer = Tokenizer.from_file(str(output))
    tokens = [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())]
    report["cross_space_tokens"] = sum("Ġ" in t[1:] for t in tokens if not t.startswith("["))
    exposure = _audit.audit_exposure(args.train_data, output)  # on v4.3 rows: 'original' = outer frames, 'assertion' = primary frame
    exposure.update(tokenizer=str(output), boundary_policy=report["boundary_policy"])
    write_json(args.exposure_output, exposure)
    protocol = Protocol.load(args.protocol)
    graph = Graph.load(args.graph)
    reference = json.loads(Path(args.reference_meta).read_text(encoding="utf-8"))
    summary = dict(tokenizer=str(output), tokenizer_hash=report["tokenizer_hash"], boundary_policy=report["boundary_policy"],
                   vocab_requested=report["vocab_requested"], vocab_actual=report["vocab_actual"], vocab_shortfall=report["vocab_shortfall"],
                   cross_space_tokens=report["cross_space_tokens"], trainable_parameters=parameter_count(graph, report["vocab_actual"], protocol),
                   reference=dict(tokenizer_hash=reference["tokenizer_hash"], vocab_actual=reference["vocab_actual"],
                                  boundary_policy=reference["boundary_policy"], trainable_parameters=parameter_count(graph, reference["vocab_actual"], protocol)),
                   lengths={k: v["mean"] for k, v in report["lengths"].items()},
                   exposure={f"{c['split']}/{c['language']}/{c['view']}": dict(mean_tokens=c["mean_tokens"], unseen_fraction=c["unseen_in_any_training_fraction"])
                             for c in exposure["cells"]})
    write_json(Path(args.root)/"tokenizer-summary.json", summary)
    return summary


def profile(args):
    from flystudy.train import run
    root = Path(args.root)
    ledger = Ledger(args.ledger)
    ledger.reserve_bundle([dict(run_id=PROFILE_ID, category="reserve", hours=args.profile_hours)])
    started, status = time.perf_counter(), "technical_failure"
    result = dict(reservation_id=PROFILE_ID, device=args.device, updates=41, languages={})
    try:
        protocol = Protocol.load(args.protocol)
        for lang in LANGUAGES:
            out = root/"profile"/lang
            if out.exists():
                shutil.rmtree(out)
            summary = run(protocol, args.graph, args.train_data, args.tokenizer, out, "mono", (lang,), SEED, cohort="smoke",
                          device=args.device, backend="cuda", microbatch=args.microbatch, smoke=True, run_id=f"profile-wb-{lang}", max_updates=41)
            clocks = summary["clocks_seconds"]
            seconds_per_update, seconds_per_panel = clocks["train"]/41, clocks["eval"]
            projected = STAGES[-1][1]*seconds_per_update + 88*seconds_per_panel + 88*clocks["checkpoint"]/2 + 4*(summary["elapsed_seconds"]-sum(clocks.values()))
            result["languages"][lang] = dict(seconds_per_update=seconds_per_update, seconds_per_panel=seconds_per_panel,
                                             checkpoint_seconds=clocks["checkpoint"]/2, startup_seconds=summary["elapsed_seconds"]-sum(clocks.values()),
                                             projected_run_hours=projected/3600, mean_tokens_per_example=summary["counters"]["tokens"]/summary["seen"],
                                             trainable_parameters=summary["trainable_parameters"])
        status = "completed"
    finally:
        hours = (time.perf_counter()-started)/3600
        ledger.charge(PROFILE_ID, hours, status)
        result.update(charged_hours=hours, status=status)
        write_json(root/"profile.json", result)
    return result


def plan(args):
    root = Path(args.root)
    if (root/"exploratory-manifest.json").exists():
        raise FileExistsError("Plan exists; use a new exploratory version")
    profile_report = json.loads((root/"profile.json").read_text(encoding="utf-8"))
    tokenizer_summary = json.loads((root/"tokenizer-summary.json").read_text(encoding="utf-8"))
    if tokenizer_summary["tokenizer_hash"] != file_hash(args.tokenizer):
        raise ValueError("Tokenizer changed since its summary was written")
    protocol = Protocol.load(args.protocol)
    graph = Graph.load(args.graph)
    requests = []
    runs = []
    for lang in LANGUAGES:
        projected = profile_report["languages"][lang]["projected_run_hours"]
        hours = max(.1, math.ceil(projected*args.margin/.05)*.05)
        requests.append(dict(run_id=run_id(lang), category="reserve", hours=hours))
        runs.append(dict(run_id=run_id(lang), mode="mono", order=[lang], projected_hours=projected, reserved_hours=hours, status="reserved", attempts=[]))
    requests.append(dict(run_id=EVAL_ID, category="reserve", hours=args.eval_hours))
    Ledger(args.ledger).reserve_bundle(requests)
    manifest = dict(**LABEL, created_at_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, microbatch=args.microbatch,
                    protocol_path=args.protocol, protocol_hash=protocol.hash, training_hash=protocol.training_hash,
                    graph_hash=graph.hash, code_hash=code_hash(), tokenizer=args.tokenizer, tokenizer_hash=file_hash(args.tokenizer),
                    tokenizer_summary=tokenizer_summary, train_data=args.train_data,
                    train_dataset_hash=json.loads((Path(args.train_data)/"manifest.json").read_text(encoding="utf-8"))["dataset_hash"],
                    eval_data=args.eval_data, eval_dataset_hash=json.loads((Path(args.eval_data)/"manifest.json").read_text(encoding="utf-8"))["dataset_hash"],
                    training_bytes_identical=file_hash(Path(args.train_data)/"train.jsonl") == file_hash(Path(args.eval_data)/"train.jsonl"),
                    stages=[dict(label=l, updates=u, seen=u*BATCH) for l, u in STAGES], profile=profile_report,
                    ledger=args.ledger, budget_category="reserve", reserved_hours=sum(r["hours"] for r in requests),
                    evaluation=dict(reservation_id=EVAL_ID, reserved_hours=args.eval_hours, status="reserved"), runs=runs)
    write_json(root/"exploratory-manifest.json", manifest)
    return manifest


def single(args):
    from flystudy.train import run
    lang = args.single
    rid, out = run_id(lang), Path(args.root)/run_id(lang)
    protocol = Protocol.load(args.protocol)
    stages = json.loads((out/"stages.json").read_text(encoding="utf-8")) if (out/"stages.json").exists() else []
    preserved = {s["label"]: s for s in stages if "checkpoint_hash" in s}
    resume = None
    for label, updates in STAGES:
        if label in preserved:
            s = preserved[label]
            if s["updates"] != updates or file_hash(s["checkpoint"]) != s["checkpoint_hash"]:
                raise RuntimeError(f"{rid}: preserved stage {label} does not match the plan")
            resume = s["checkpoint"]
            continue
        resumed_from = resume
        if resume is not None and file_hash(out/"latest.pt") != file_hash(resume):
            # An interrupted stage leaves latest.pt ahead of the preserved checkpoint: crash recovery continues from it.
            latest = torch.load(out/"latest.pt", map_location="cpu", weights_only=True)
            preserved_updates = max(s["updates"] for s in stages if "checkpoint_hash" in s)
            if latest["metadata"]["run_id"] != rid or not preserved_updates < latest["counters"]["updates"] < updates:
                raise RuntimeError(f"{rid}: latest.pt is not between the preserved stage and the target")
            resumed_from = f"latest.pt@{latest['counters']['updates']}"
            resume = str(out/"latest.pt")
        begin = time.perf_counter()
        summary = run(protocol, args.graph, args.train_data, args.tokenizer, out, "mono", (lang,), SEED, cohort="smoke",
                      device=args.device, backend="cuda", microbatch=args.microbatch, smoke=True, run_id=rid, resume=resume, max_updates=updates)
        record = dict(label=label, target_updates=updates, attempt=args.attempt, resumed_from=resumed_from, stop_reason=summary["stop_reason"],
                      seen=summary["seen"], updates=summary["counters"]["updates"], stage_seconds=time.perf_counter()-begin, elapsed_seconds=summary["elapsed_seconds"])
        if summary["stop_reason"] != "debug_interruption":
            stages.append(record); write_json(out/"stages.json", stages)
            break
        checkpoint = out/f"checkpoint-{label}.pt"
        shutil.copy2(out/"latest.pt", checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (file_hash(checkpoint) != file_hash(out/"latest.pt") or state["counters"]["updates"] != updates
                or state["curriculum"]["seen"] != summary["seen"] or summary["seen"] != updates*BATCH):
            raise RuntimeError(f"{rid}: preserved checkpoint does not match stage {label}")
        record.update(checkpoint=str(checkpoint), checkpoint_hash=file_hash(checkpoint), language_exposures=state["sampler"]["exposures"])
        stages.append(record); write_json(out/"stages.json", stages)
        resume = str(checkpoint)
    return stages


def execute(args):
    root = Path(args.root)
    manifest_path = root/"exploratory-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["code_hash"] != code_hash():
        raise ValueError("Code changed after the exploratory plan was written")
    if manifest["tokenizer_hash"] != file_hash(args.tokenizer):
        raise ValueError("Tokenizer changed after the plan was written")
    active = _v2.gpu_processes(args.devices)
    if any(active.values()):
        raise RuntimeError(f"Other compute processes are active: {active}")
    ledger = Ledger(args.ledger)
    todo = [r for r in manifest["runs"] if r["status"] != "completed"]
    if args.retry:
        if not todo:
            raise ValueError("Every run is complete; nothing to retry")
        requests = []
        for r in todo:
            if r["status"] == "reserved":
                raise ValueError(f"{r['run_id']} has not been attempted; use --execute")
            r["pending_reservation"] = f"{r['run_id']}-r{len(r['attempts'])+1}"
            requests.append(dict(run_id=r["pending_reservation"], category="reserve", hours=args.retry_hours or r["reserved_hours"]))
        _v2.check_open_reservations(args, {EVAL_ID})
        ledger.reserve_bundle(requests)
        manifest["retry_reserved_hours"] = manifest.get("retry_reserved_hours", 0)+sum(q["hours"] for q in requests)
    else:
        if any(r["status"] != "reserved" for r in todo) or len(todo) != len(manifest["runs"]):
            raise ValueError("First attempts already made; use --retry for unfinished runs")
        for r in todo:
            r["pending_reservation"] = r["run_id"]
        _v2.check_open_reservations(args, {r["run_id"] for r in manifest["runs"]} | {EVAL_ID})
    manifest.setdefault("executions", []).append(dict(started_at_utc=datetime.now(timezone.utc).isoformat(), devices=args.devices, retry=bool(args.retry),
                                                     other_compute_processes_at_start=active, reservations=[r["pending_reservation"] for r in todo]))
    write_json(manifest_path, manifest)
    pending, active_jobs = list(todo), {}
    while pending or active_jobs:
        for device in args.devices:
            if device in active_jobs or not pending:
                continue
            r = pending.pop(0)
            reservation = ledger.reservation(r["pending_reservation"])
            log = (root/f"{r['pending_reservation']}.console.log").open("w", encoding="utf-8")
            cmd = [sys.executable, __file__, "--single", r["order"][0], "--device", device, "--root", str(root), "--ledger", args.ledger,
                   "--protocol", args.protocol, "--graph", args.graph, "--train-data", args.train_data, "--eval-data", args.eval_data,
                   "--tokenizer", args.tokenizer, "--microbatch", str(args.microbatch), "--attempt", r["pending_reservation"]]
            active_jobs[device] = dict(proc=subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log=log, run=r,
                                       start=time.perf_counter(), deadline=reservation["reserved"]*3600, reserved=reservation["reserved"])
        for device, job in list(active_jobs.items()):
            code = job["proc"].poll()
            elapsed = time.perf_counter()-job["start"]
            interrupted = False
            if code is None and elapsed > job["deadline"]:
                job["proc"].kill(); job["proc"].wait(); code, interrupted = -9, True
            if code is None:
                continue
            job["log"].close(); del active_jobs[device]
            hours = (time.perf_counter()-job["start"])/3600
            r, out = job["run"], root/job["run"]["run_id"]
            stages = json.loads((out/"stages.json").read_text(encoding="utf-8")) if (out/"stages.json").exists() else []
            summary = json.loads((out/"summary.json").read_text(encoding="utf-8")) if (out/"summary.json").exists() else {}
            complete = code == 0 and len(stages) == len(STAGES) and all("checkpoint_hash" in s for s in stages)
            status = "completed" if complete else ("budget_interruption" if interrupted else "technical_failure")
            reservation_id = r.pop("pending_reservation")
            ledger.charge(reservation_id, hours, status)
            r["attempts"].append(dict(reservation_id=reservation_id, reserved_hours=job["reserved"], device=device, wall_hours=hours, exit_code=code,
                                      status=status, interrupted="reservation_wall_clock_exceeded" if interrupted else None,
                                      stages_preserved_after=[s["label"] for s in stages if "checkpoint_hash" in s],
                                      console_log=str(root/f"{reservation_id}.console.log"), error=None if code == 0 else _v2.last_line(root/f"{reservation_id}.console.log")))
            r.update(status=status, device=device, wall_hours=sum(a["wall_hours"] for a in r["attempts"]), stages=stages,
                     stop_reason=summary.get("stop_reason"), seen=summary.get("seen"), updates=(summary.get("counters") or {}).get("updates"),
                     clocks_seconds=summary.get("clocks_seconds"), summary_gpu_hours=summary.get("gpu_hours"),
                     trainable_parameters=summary.get("trainable_parameters"), tokens=(summary.get("counters") or {}).get("tokens"),
                     independent_test_written=(out/"independent-test.json").exists())
            write_json(manifest_path, manifest)
        time.sleep(1)
    manifest["executions"][-1]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["actual_training_gpu_hours"] = sum(r.get("wall_hours", 0) for r in manifest["runs"])
    write_json(manifest_path, manifest)
    return manifest


def evaluate(args):
    root = Path(args.root)
    manifest_path = root/"exploratory-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["code_hash"] != code_hash():
        raise ValueError("Code changed after the exploratory plan was written")
    if manifest["evaluation"]["status"] != "reserved":
        raise ValueError("Evaluation reservation already used; reserve a new id for a re-run")
    if Path(args.output).exists():
        raise FileExistsError("Keep previous reports; use a new output path")
    active = _v2.gpu_processes([args.device])
    if any(active.values()):
        raise RuntimeError(f"Other compute processes are active: {active}")
    _v2.check_open_reservations(args, {EVAL_ID})
    v2 = json.loads(Path(args.v2_report).read_text(encoding="utf-8"))
    started, status = time.perf_counter(), "technical_failure"
    report = dict(kind="exploratory_v3_wordbound_report", **LABEL, created_at_utc=datetime.now(timezone.utc).isoformat(), manifest=str(manifest_path),
                  code_hash=code_hash(), script_hash=file_hash(__file__), tokenizer_summary=manifest["tokenizer_summary"], device=args.device,
                  microbatch=args.microbatch, parameter_updates_during_evaluation=0, test_split_used=False, eval_data=args.eval_data,
                  runs=[{k: v for k, v in r.items() if k != "clocks_seconds"} for r in manifest["runs"]], profile=manifest["profile"],
                  training_gpu_hours=manifest.get("actual_training_gpu_hours"), evaluations=[])
    try:
        protocol = Protocol.load(manifest["protocol_path"])
        graph = Graph.load(args.graph)
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        corpus = Corpus(args.eval_data, tokenizer)
        sample, report["training_sample"] = _v2.training_sample(args.eval_data, tokenizer)
        if report["training_sample"]["ids_sha256"] != v2["training_sample"]["ids_sha256"]:
            raise ValueError("Training sample differs from exploratory v2")
        sets = [("train_sample", sample)]
        for split in ("dev_a", "dev_b"):
            for view in ("primary", "auxiliary"):
                sets.append((f"{split}_{view}", corpus.split(split, view)))
        report["environment"] = environment()
        for r in manifest["runs"]:
            for s in r.get("stages", []):
                if "checkpoint_hash" not in s:
                    continue
                if file_hash(s["checkpoint"]) != s["checkpoint_hash"]:
                    raise ValueError(f"{s['checkpoint']}: checkpoint hash changed")
                begin = time.perf_counter()
                info = _v2.score_checkpoint(Path(s["checkpoint"]), graph, tokenizer, protocol.embed_dim, protocol.microsteps, sets, args.device, args.microbatch)
                report["evaluations"].append(dict(source="v3-wordbound", stage=s["label"], seconds=time.perf_counter()-begin, **info))
        rename = {"train_sample": "train_sample", "dev_a_assertion": "dev_a_primary", "dev_b_assertion": "dev_b_primary",
                  "dev_a_original": "dev_a_auxiliary", "dev_b_original": "dev_b_auxiliary"}
        v2_cells = {}
        for e in v2["evaluations"]:
            if e["source"] == "v2":
                for c in e["cells"]:
                    v2_cells[(e["languages"][0], e["stage"], rename[c["set"]], c["task"])] = c
        report["comparison"] = []
        for e in report["evaluations"]:
            lang = e["languages"][0]
            for c in e["cells"]:
                ref = v2_cells.get((lang, e["stage"], c["set"], c["task"]))
                report["comparison"].append(dict(language=lang, stage=e["stage"], seen=e["seen"], set=c["set"], task=c["task"],
                                                 wordbound_accuracy=c["accuracy"], whole_string_accuracy=ref["accuracy"] if ref else None,
                                                 delta=c["accuracy"]-ref["accuracy"] if ref else None, wordbound_correct=c["correct"], items=c["items"]))
        report["scheduled_panels"] = {}
        for r in manifest["runs"]:
            events = [json.loads(l) for l in (root/r["run_id"]/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            report["scheduled_panels"][r["run_id"]] = [dict(seen=e["seen"], panel=e["panel"], scores=e["scores"]) for e in events if e["kind"] == "scheduled"]
            train_events = [e for e in events if e["kind"] == "train"]
            for rr in report["runs"]:
                if rr["run_id"] == r["run_id"]:
                    rr["loss_first_last_min"] = [train_events[0]["loss"], train_events[-1]["loss"], min(e["loss"] for e in train_events)] if train_events else None
                    rr["mastered"] = any(e.get("first_task") for e in events if e["kind"] == "scheduled")
        status = "completed"
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        hours = (time.perf_counter()-started)/3600
        Ledger(args.ledger).charge(EVAL_ID, hours, status)
        manifest["evaluation"].update(status=status, charged_hours=hours, output=args.output)
        manifest["actual_total_gpu_hours"] = manifest.get("actual_training_gpu_hours", 0)+hours+manifest["profile"]["charged_hours"]
        write_json(manifest_path, manifest)
        report["ledger"] = dict(path=args.ledger, profile=dict(reservation_id=PROFILE_ID, charged_hours=manifest["profile"]["charged_hours"]),
                                evaluation=dict(reservation_id=EVAL_ID, reserved_hours=manifest["evaluation"]["reserved_hours"], charged_hours=hours, status=status),
                                training_attempts={r["run_id"]: r.get("attempts", []) for r in manifest["runs"]},
                                plan_reserved_hours=manifest["reserved_hours"], retry_reserved_hours=manifest.get("retry_reserved_hours", 0),
                                training_charged_hours=manifest.get("actual_training_gpu_hours"), total_charged_hours=manifest["actual_total_gpu_hours"])
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("tokenizer", "profile", "plan", "execute", "evaluate", "retry"):
        p.add_argument(f"--{flag}" if flag != "tokenizer" else "--train-tokenizer", action="store_true")
    p.add_argument("--single", choices=LANGUAGES); p.add_argument("--attempt", default=None)
    p.add_argument("--root", default="runs/exploratory-v3-wordbound"); p.add_argument("--ledger", default="runs/gpu-ledger.json")
    p.add_argument("--protocol", default="configs/protocol-v4-ai-mono900k.json"); p.add_argument("--graph", default="artifacts/graphs/real.npz")
    p.add_argument("--train-data", default="data/draft-v4.3"); p.add_argument("--eval-data", default="data/draft-v4.4")
    p.add_argument("--tokenizer", default="artifacts/tokenizer-wordbound-v1.json"); p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--reference-meta", default="artifacts/tokenizer-v4.3.meta.json")
    p.add_argument("--exposure-output", default="reports/token-exposure-wordbound-v1.json")
    p.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"]); p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=256); p.add_argument("--profile-hours", type=float, default=.05)
    p.add_argument("--margin", type=float, default=1.5); p.add_argument("--eval-hours", type=float, default=.1); p.add_argument("--retry-hours", type=float, default=None)
    p.add_argument("--v2-report", default="reports/exploratory-v2-report.json"); p.add_argument("--output", default="reports/exploratory-v3-wordbound-report.json")
    a = p.parse_args()
    Path(a.root).mkdir(parents=True, exist_ok=True)
    if a.train_tokenizer:
        print(json.dumps(tokenizer_step(a), indent=1, ensure_ascii=False))
    elif a.profile:
        print(json.dumps(profile(a), indent=1))
    elif a.plan:
        m = plan(a); print(json.dumps(dict(status="planned", runs={r["run_id"]: (r["projected_hours"], r["reserved_hours"]) for r in m["runs"]}, reserved_hours=m["reserved_hours"]), indent=1))
    elif a.execute:
        m = execute(a); print(json.dumps(dict(status="executed", actual_training_gpu_hours=m["actual_training_gpu_hours"],
                                              runs={r["run_id"]: (r.get("status"), r.get("stop_reason"), r.get("seen"), r.get("updates")) for r in m["runs"]}), indent=1))
    elif a.single:
        print(json.dumps(single(a), indent=1))
    elif a.evaluate:
        r = evaluate(a); print(json.dumps(r["ledger"], indent=1))
    else:
        p.error("choose --train-tokenizer, --profile, --plan, --execute, --single or --evaluate")


if __name__ == "__main__":
    main()

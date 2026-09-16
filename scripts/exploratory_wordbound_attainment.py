"""Exploratory v4: attainment time and reproducibility under the word-boundary BPE. NOT study evidence.

Six monolingual runs (seeds 20002 and 20003 x EN/DE/KO) train from scratch with the exploratory-v2/v3 configuration
(real graph, same learning rates, batch 256, microsteps 2) on draft-v4.4, whose scheduled panels are the primary
(training-frame) rendering of held-out meanings. Full A/B panels alternate every 5,120 exposures from the start; a run
stops when every task cell is >= 80% on two consecutive scheduled evaluations, or at the 200,000 cap. The input is
artifacts/tokenizer-wordbound-v4.4.json (byte-identical to tokenizer-wordbound-v1, 841 tokens, metadata linked to
draft-v4.4). No independent test is run; the auxiliary (outer-frame) evaluation is scored once on the terminal
checkpoint. Actual GPU time is charged to the contingency ('reserve') allocation.

  --plan       reserve six ledger entries and write <root>/exploratory-manifest.json
  --execute    run on two GPUs, one process per run (--retry resumes unfinished runs from latest.pt)
  --single     (internal) one run plus its terminal evaluation
  --report     CPU only: attainment costs, curves, counters, throughput; figures via reporting.plot_curves
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

from flystudy.budget import Ledger
from flystudy.gates import code_hash, environment
from flystudy.graph import Graph
from flystudy.model import FlyClassifier
from flystudy.protocol import LANGUAGES, TASKS, Protocol, file_hash, write_json
from flystudy.runtime import Corpus, evaluate

SEEDS = (20002, 20003)  # unused exploratory seeds (v1-v3 used 20001)
RUN_HOURS = .25  # v3 alone: 0.10-0.135 s/update -> ~0.06 h per capped run; v3 saw a 4x slowdown while sharing the host
LABEL = dict(cohort_label="exploratory", study_evidence=False, review_status="incomplete_ai_review_not_certified",
             question="attainment cost and seed reproducibility of the word-boundary BPE input on the primary evaluation",
             note="Runs are labelled cohort='smoke' by the training code because that is its only non-gated cohort; they use the "
                  "REAL graph and CUDA backend. Mastery here is the exploratory attainment rule on the draft-v4.4 primary rendering; "
                  "it is not a study result. No independent test; auxiliary evaluation only at the terminal checkpoint.")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v2 = load_module("exploratory_mono_exposure")


def run_id(seed, lang):
    return f"explore-v4-wb-{seed}-mono-{lang}"


def plan(args):
    root = Path(args.root)
    if (root/"exploratory-manifest.json").exists():
        raise FileExistsError("Plan exists; use a new exploratory version")
    protocol = Protocol.load(args.protocol)
    graph = Graph.load(args.graph)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    meta = json.loads(Path(args.tokenizer).with_suffix(".meta.json").read_text(encoding="utf-8"))
    data_meta = json.loads((Path(args.data)/"manifest.json").read_text(encoding="utf-8"))
    if meta["tokenizer_hash"] != file_hash(args.tokenizer) or meta["dataset_hash"] != data_meta["dataset_hash"]:
        raise ValueError("Tokenizer metadata is not linked to this dataset")
    if meta["training_file_hash"] != file_hash(Path(args.data)/"train.jsonl"):
        raise ValueError("Tokenizer was not trained on this training file")
    if tokenizer.get_vocab_size() != protocol.vocab_size:
        raise ValueError("Protocol vocab_size must state the actual vocabulary")
    root.mkdir(parents=True, exist_ok=True)
    runs = [dict(run_id=run_id(seed, lang), seed=seed, mode="mono", order=[lang], reserved_hours=RUN_HOURS, status="reserved", attempts=[])
            for lang in LANGUAGES for seed in SEEDS]
    Ledger(args.ledger).reserve_bundle([dict(run_id=r["run_id"], category="reserve", hours=RUN_HOURS) for r in runs])
    model = FlyClassifier(graph, tokenizer.get_vocab_size(), protocol.embed_dim, protocol.microsteps, SEEDS[0], "sparse")
    manifest = dict(**LABEL, created_at_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS), microbatch=args.microbatch,
                    protocol_path=args.protocol, protocol_hash=protocol.hash, training_hash=protocol.training_hash,
                    eval_interval=protocol.eval_interval, mono_cap=protocol.mono_cap, threshold=protocol.threshold,
                    graph_hash=graph.hash, code_hash=code_hash(), tokenizer=args.tokenizer, tokenizer_hash=file_hash(args.tokenizer),
                    tokenizer_identical_to_wordbound_v1=file_hash(args.tokenizer) == file_hash("artifacts/tokenizer-wordbound-v1.json"),
                    vocab_actual=tokenizer.get_vocab_size(), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    data=args.data, dataset_hash=data_meta["dataset_hash"], independent_test=False, auxiliary_panels="terminal_only",
                    ledger=args.ledger, budget_category="reserve", reserved_hours=RUN_HOURS*len(runs), runs=runs)
    write_json(root/"exploratory-manifest.json", manifest)
    return manifest


def terminal_evaluation(out, lang, args, protocol):
    checkpoint = out/"primary.pt"
    if not checkpoint.exists():
        return None
    tokenizer = Tokenizer.from_file(args.tokenizer)
    ck = torch.load(checkpoint, map_location=args.device, weights_only=True)
    model = FlyClassifier(Graph.load(args.graph), tokenizer.get_vocab_size(), protocol.embed_dim, protocol.microsteps, ck["metadata"]["seed"], "cuda").to(args.device)
    model.load_state_dict(ck["model"])
    corpus = Corpus(args.data, tokenizer)
    result = dict(checkpoint_hash=file_hash(checkpoint), seen=ck["curriculum"]["seen"], updates=ck["counters"]["updates"], scores={})
    begin = time.perf_counter()
    for split in ("dev_a", "dev_b"):
        for view in ("primary", "auxiliary"):
            scores, _ = evaluate(model, corpus, split, (lang,), args.microbatch, view=view)
            result["scores"][f"{split}/{view}"] = scores
    result["seconds"] = time.perf_counter()-begin
    write_json(out/"terminal-evaluation.json", result)
    return result


def single(args):
    from flystudy.train import run
    seed, lang = int(args.seed), args.single
    rid, out = run_id(seed, lang), Path(args.root)/run_id(seed, lang)
    protocol = Protocol.load(args.protocol)
    resume = str(out/"latest.pt") if (out/"latest.pt").exists() else None
    summary = run(protocol, args.graph, args.data, args.tokenizer, out, "mono", (lang,), seed, cohort="smoke", device=args.device,
                  backend="cuda", microbatch=args.microbatch, smoke=True, run_id=rid, resume=resume,
                  independent_test=False, auxiliary_panels=False)
    terminal = terminal_evaluation(out, lang, args, protocol)
    return dict(run_id=rid, stop_reason=summary["stop_reason"], seen=summary["seen"], first_global=summary["first_global"],
                first_task=summary["first_task"], resumed_from=resume, terminal=terminal is not None)


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
            requests.append(dict(run_id=r["pending_reservation"], category="reserve", hours=args.retry_hours or RUN_HOURS))
        _v2.check_open_reservations(args, set())
        ledger.reserve_bundle(requests)
        manifest["retry_reserved_hours"] = manifest.get("retry_reserved_hours", 0)+sum(q["hours"] for q in requests)
    else:
        if any(r["status"] != "reserved" for r in todo) or len(todo) != len(manifest["runs"]):
            raise ValueError("First attempts already made; use --retry for unfinished runs")
        for r in todo:
            r["pending_reservation"] = r["run_id"]
        _v2.check_open_reservations(args, {r["run_id"] for r in manifest["runs"]})
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
            cmd = [sys.executable, __file__, "--single", r["order"][0], "--seed", str(r["seed"]), "--device", device, "--root", str(root),
                   "--ledger", args.ledger, "--protocol", args.protocol, "--graph", args.graph, "--data", args.data,
                   "--tokenizer", args.tokenizer, "--microbatch", str(args.microbatch)]
            active_jobs[device] = dict(proc=subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log=log, run=r, start=time.perf_counter(),
                                       started_at=datetime.now(timezone.utc).isoformat(), deadline=reservation["reserved"]*3600, reserved=reservation["reserved"])
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
            summary = json.loads((out/"summary.json").read_text(encoding="utf-8")) if (out/"summary.json").exists() else {}
            complete = code == 0 and summary.get("stop_reason") in ("mastered", "administrative_cap") and (out/"terminal-evaluation.json").exists()
            status = "completed" if complete else ("budget_interruption" if interrupted else "technical_failure")
            reservation_id = r.pop("pending_reservation")
            ledger.charge(reservation_id, hours, status)
            r["attempts"].append(dict(reservation_id=reservation_id, reserved_hours=job["reserved"], device=device, wall_hours=hours, exit_code=code,
                                      status=status, started_at_utc=job["started_at"], ended_at_utc=datetime.now(timezone.utc).isoformat(),
                                      interrupted="reservation_wall_clock_exceeded" if interrupted else None,
                                      concurrent_with=[j["run"]["run_id"] for j in active_jobs.values()],
                                      console_log=str(root/f"{reservation_id}.console.log"), error=None if code == 0 else _v2.last_line(root/f"{reservation_id}.console.log")))
            r.update(status=status, device=device, wall_hours=sum(a["wall_hours"] for a in r["attempts"]), stop_reason=summary.get("stop_reason"),
                     seen=summary.get("seen"), updates=(summary.get("counters") or {}).get("updates"), first_global=summary.get("first_global"),
                     first_task=summary.get("first_task"), clocks_seconds=summary.get("clocks_seconds"), counters=summary.get("counters"),
                     summary_gpu_hours=summary.get("gpu_hours"), trainable_parameters=summary.get("trainable_parameters"),
                     independent_test=summary.get("independent_test"), independent_test_written=(out/"independent-test.json").exists())
            write_json(manifest_path, manifest)
        time.sleep(1)
    manifest["executions"][-1]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["actual_training_gpu_hours"] = sum(r.get("wall_hours", 0) for r in manifest["runs"])
    write_json(manifest_path, manifest)
    return manifest


def report(args):
    from flystudy.reporting import plot_curves
    root = Path(args.root)
    manifest = json.loads((root/"exploratory-manifest.json").read_text(encoding="utf-8"))
    if Path(args.output).exists():
        raise FileExistsError("Keep previous reports; use a new output path")
    profile = json.loads(Path(args.reference_profile).read_text(encoding="utf-8"))["languages"]
    out = dict(kind="exploratory_v4_wordbound_attainment_report", **LABEL, created_at_utc=datetime.now(timezone.utc).isoformat(),
               manifest=str(root/"exploratory-manifest.json"), code_hash=code_hash(), script_hash=file_hash(__file__),
               protocol=dict(path=manifest["protocol_path"], hash=manifest["protocol_hash"], eval_interval=manifest["eval_interval"],
                             mono_cap=manifest["mono_cap"], threshold=manifest["threshold"], vocab_size=manifest["vocab_actual"]),
               tokenizer=dict(path=manifest["tokenizer"], hash=manifest["tokenizer_hash"], identical_to_wordbound_v1=manifest["tokenizer_identical_to_wordbound_v1"],
                              vocab=manifest["vocab_actual"]), trainable_parameters=manifest["trainable_parameters"],
               data=dict(path=manifest["data"], hash=manifest["dataset_hash"]), gpu_used_for_report=False, test_split_used=False, runs=[])
    for r in manifest["runs"]:
        lang, folder = r["order"][0], root/r["run_id"]
        events = [json.loads(l) for l in (folder/"events.jsonl").read_text(encoding="utf-8").splitlines()] if (folder/"events.jsonl").exists() else []
        scheduled = [e for e in events if e["kind"] == "scheduled"]
        train_events = [e for e in events if e["kind"] == "train"]
        curve = [dict(seen=e["seen"], panel=e["panel"], accuracy={k: c/n for k, (c, n) in e["scores"].items()}) for e in scheduled]
        terminal = json.loads((folder/"terminal-evaluation.json").read_text(encoding="utf-8")) if (folder/"terminal-evaluation.json").exists() else None
        clocks, counters = r.get("clocks_seconds") or {}, r.get("counters") or {}
        seconds_per_update = clocks.get("train", 0)/counters["updates"] if counters.get("updates") else None
        out["runs"].append(dict(run_id=r["run_id"], seed=r["seed"], language=lang, status=r["status"], stop_reason=r.get("stop_reason"), seen=r.get("seen"),
                                updates=counters.get("updates"), attainment=dict(all_tasks=r.get("first_global"),
                                                                                  per_task={task: (r.get("first_task") or {}).get(f"{lang}/{task}") for task in TASKS}),
                                counters=counters, clocks_seconds=clocks, gpu_hours=r.get("wall_hours"), summary_gpu_hours=r.get("summary_gpu_hours"),
                                attempts=r.get("attempts", []), seconds_per_update=seconds_per_update,
                                seconds_per_update_alone_v3_profile=profile[lang]["seconds_per_update"],
                                slowdown_vs_alone=seconds_per_update/profile[lang]["seconds_per_update"] if seconds_per_update else None,
                                loss_first_last_min=[train_events[0]["loss"], train_events[-1]["loss"], min(e["loss"] for e in train_events)] if train_events else None,
                                scheduled_panels=len(scheduled), curve=curve, terminal_evaluation=terminal,
                                independent_test=r.get("independent_test"), independent_test_written=r.get("independent_test_written")))
    figures = Path(args.figures)
    plot_curves(root, figures)
    out["figures"] = sorted(str(p) for p in figures.glob("*.png"))
    out["ledger"] = dict(path=manifest["ledger"], plan_reserved_hours=manifest["reserved_hours"], retry_reserved_hours=manifest.get("retry_reserved_hours", 0),
                         training_charged_hours=manifest.get("actual_training_gpu_hours"),
                         attempts={r["run_id"]: [dict(reservation_id=a["reservation_id"], status=a["status"], wall_hours=a["wall_hours"]) for a in r.get("attempts", [])] for r in manifest["runs"]})
    write_json(args.output, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("plan", "execute", "retry", "report"):
        p.add_argument(f"--{flag}", action="store_true")
    p.add_argument("--single", choices=LANGUAGES); p.add_argument("--seed", type=int)
    p.add_argument("--root", default="runs/exploratory-v4-wordbound-attainment"); p.add_argument("--ledger", default="runs/gpu-ledger.json")
    p.add_argument("--protocol", default="configs/protocol-v4-ai-wordbound-5120.json"); p.add_argument("--graph", default="artifacts/graphs/real.npz")
    p.add_argument("--data", default="data/draft-v4.4"); p.add_argument("--tokenizer", default="artifacts/tokenizer-wordbound-v4.4.json")
    p.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"]); p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=256); p.add_argument("--retry-hours", type=float, default=None)
    p.add_argument("--reference-profile", default="runs/exploratory-v3-wordbound/profile.json")
    p.add_argument("--output", default="reports/exploratory-v4-wordbound-attainment.json")
    p.add_argument("--figures", default="reports/figures/exploratory-v4-wordbound-attainment")
    a = p.parse_args()
    if a.plan:
        m = plan(a); print(json.dumps(dict(status="planned", runs=[r["run_id"] for r in m["runs"]], reserved_hours=m["reserved_hours"], vocab=m["vocab_actual"], parameters=m["trainable_parameters"]), indent=1))
    elif a.execute:
        m = execute(a); print(json.dumps(dict(status="executed", actual_training_gpu_hours=m["actual_training_gpu_hours"],
                                              runs={r["run_id"]: (r.get("status"), r.get("stop_reason"), r.get("seen"), r.get("first_global")) for r in m["runs"]}), indent=1))
    elif a.single:
        print(json.dumps(single(a), indent=1))
    elif a.report:
        r = report(a); print(json.dumps({x["run_id"]: dict(stop=x["stop_reason"], seen=x["seen"], all=x["attainment"]["all_tasks"], per_task=x["attainment"]["per_task"], slowdown=x["slowdown_vs_alone"]) for x in r["runs"]}, indent=1))
    else:
        p.error("choose --plan, --execute, --single or --report")


if __name__ == "__main__":
    main()

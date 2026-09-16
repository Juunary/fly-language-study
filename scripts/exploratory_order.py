"""Exploratory v5: language order under the word-boundary BPE input. NOT study evidence.

Fourteen runs: for each of the seeds 20002 and 20003, the six sequential language orders and the 1:1:1 mixed
baseline, trained from scratch with the exploratory-v4 configuration (draft-v4.4 primary panels, 841-token
word-boundary tokenizer, real graph, same learning rates, batch 256, microsteps 2, protocol-v4-ai-wordbound-5120).
The training loop's own curriculum applies: all twelve cells are evaluated on alternating A/B panels every 5,120
exposures from the start; a sequential run moves to the next language when the current language's four cells pass
>= 80% on two consecutive scheduled evaluations or after 200,000 exposures in that stage (model and optimizer state
are kept; non-mastered transitions are recorded as reason='stage_cap'); after three stages training is 1:1:1
review; the run ends when all twelve cells pass on the same two consecutive evaluations or at 900,000 exposures.
No independent test; the auxiliary (outer-frame) evaluation is scored once on the terminal checkpoint. The v4
monolingual runs of the same seeds are the exploratory single-language baselines. Actual GPU time is charged to
the contingency ('reserve') allocation.

  --plan       reserve fourteen ledger entries and write <root>/exploratory-manifest.json
  --execute    run on two GPUs, one process per run (--retry resumes unfinished runs from latest.pt)
  --single     (internal) one run plus its terminal evaluation
  --report     CPU only: attainment, stage tables, forgetting, quantity, baselines, counters, figures
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
from flystudy.gates import code_hash
from flystudy.graph import Graph
from flystudy.model import FlyClassifier
from flystudy.protocol import LANGUAGES, ORDERS, TASKS, Protocol, file_hash, write_json
from flystudy.runtime import Corpus, evaluate

SEEDS = (20002, 20003)
# Projection per capped run (v4 measurements): 3,516 updates x 0.13 s + 176 twelve-cell panels x 2.0 s
# + 176 checkpoints x 0.22 s + startup ~ 0.24 h; reserved with x2.5 for the throughput variation seen in v3/v4.
RUN_HOURS = .6
LABEL = dict(cohort_label="exploratory", study_evidence=False, review_status="incomplete_ai_review_not_certified",
             question="language order (six sequential orders vs 1:1:1 mixed) under the word-boundary BPE input, two seeds",
             note="Runs are labelled cohort='smoke' by the training code because that is its only non-gated cohort; they use the "
                  "REAL graph and CUDA backend. Order rankings from two seeds are exploratory only. No independent test; auxiliary "
                  "evaluation only at the terminal checkpoint. Review certification status is unchanged.")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v2 = load_module("exploratory_mono_exposure")
_v4 = load_module("exploratory_wordbound_attainment")


def run_id(seed, mode, order):
    return f"explore-v5-wb-{seed}-" + ("mixed" if mode == "mixed" else "seq-" + "-".join(order))


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
    if tokenizer.get_vocab_size() != protocol.vocab_size:
        raise ValueError("Protocol vocab_size must state the actual vocabulary")
    baseline = json.loads((Path(args.baseline_root)/"exploratory-manifest.json").read_text(encoding="utf-8"))
    for key in ("protocol_hash", "graph_hash", "tokenizer_hash", "dataset_hash"):
        if baseline[key] != {"protocol_hash": protocol.hash, "graph_hash": graph.hash, "tokenizer_hash": file_hash(args.tokenizer), "dataset_hash": data_meta["dataset_hash"]}[key]:
            raise ValueError(f"Monolingual baselines differ in {key}")
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for order in ORDERS:
        for seed in SEEDS:
            runs.append(dict(run_id=run_id(seed, "sequential", order), seed=seed, mode="sequential", order=list(order), reserved_hours=RUN_HOURS, status="reserved", attempts=[]))
    for seed in SEEDS:
        runs.append(dict(run_id=run_id(seed, "mixed", LANGUAGES), seed=seed, mode="mixed", order=list(LANGUAGES), reserved_hours=RUN_HOURS, status="reserved", attempts=[]))
    Ledger(args.ledger).reserve_bundle([dict(run_id=r["run_id"], category="reserve", hours=RUN_HOURS) for r in runs])
    model = FlyClassifier(graph, tokenizer.get_vocab_size(), protocol.embed_dim, protocol.microsteps, SEEDS[0], "sparse")
    manifest = dict(**LABEL, created_at_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS), microbatch=args.microbatch,
                    protocol_path=args.protocol, protocol_hash=protocol.hash, training_hash=protocol.training_hash,
                    eval_interval=protocol.eval_interval, mono_cap=protocol.mono_cap, total_cap=protocol.total_cap, threshold=protocol.threshold,
                    graph_hash=graph.hash, code_hash=code_hash(), tokenizer=args.tokenizer, tokenizer_hash=file_hash(args.tokenizer),
                    vocab_actual=tokenizer.get_vocab_size(), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    data=args.data, dataset_hash=data_meta["dataset_hash"], independent_test=False, auxiliary_panels="terminal_only",
                    baseline_root=args.baseline_root, baseline_code_hash=baseline["code_hash"], code_hash_equals_baseline=baseline["code_hash"] == code_hash(),
                    projection=dict(hours_per_capped_run=.24, basis="v4: 0.13 s/update, ~2.0 s per twelve-cell panel, 0.22 s per checkpoint", margin=2.5),
                    ledger=args.ledger, budget_category="reserve", reserved_hours=RUN_HOURS*len(runs), runs=runs)
    write_json(root/"exploratory-manifest.json", manifest)
    return manifest


def terminal_evaluation(out, args, protocol):
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
            scores, _ = evaluate(model, corpus, split, LANGUAGES, args.microbatch, view=view)
            result["scores"][f"{split}/{view}"] = scores
    result["seconds"] = time.perf_counter()-begin
    write_json(out/"terminal-evaluation.json", result)
    return result


def single(args):
    from flystudy.train import run
    manifest = json.loads((Path(args.root)/"exploratory-manifest.json").read_text(encoding="utf-8"))
    r = next(x for x in manifest["runs"] if x["run_id"] == args.single)
    out = Path(args.root)/r["run_id"]
    protocol = Protocol.load(args.protocol)
    resume = str(out/"latest.pt") if (out/"latest.pt").exists() else None
    summary = run(protocol, args.graph, args.data, args.tokenizer, out, r["mode"], tuple(r["order"]), r["seed"], cohort="smoke",
                  device=args.device, backend="cuda", microbatch=args.microbatch, smoke=True, run_id=r["run_id"], resume=resume,
                  independent_test=False, auxiliary_panels=False)
    terminal = terminal_evaluation(out, args, protocol)
    return dict(run_id=r["run_id"], stop_reason=summary["stop_reason"], seen=summary["seen"], first_global=summary["first_global"],
                stages=summary["stages"], resumed_from=resume, terminal=terminal is not None)


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
            cmd = [sys.executable, __file__, "--single", r["run_id"], "--device", device, "--root", str(root), "--ledger", args.ledger,
                   "--protocol", args.protocol, "--graph", args.graph, "--data", args.data, "--tokenizer", args.tokenizer, "--microbatch", str(args.microbatch)]
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
                     first_task=summary.get("first_task"), first_language=summary.get("first_language"), stages=summary.get("stages"),
                     review_start=summary.get("review_start"), language_exposures=summary.get("language_exposures"),
                     clocks_seconds=summary.get("clocks_seconds"), counters=summary.get("counters"), summary_gpu_hours=summary.get("gpu_hours"),
                     trainable_parameters=summary.get("trainable_parameters"), independent_test=summary.get("independent_test"),
                     independent_test_written=(out/"independent-test.json").exists())
            write_json(manifest_path, manifest)
        time.sleep(1)
    manifest["executions"][-1]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["actual_training_gpu_hours"] = sum(r.get("wall_hours", 0) for r in manifest["runs"])
    write_json(manifest_path, manifest)
    (root/"execution-done").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return manifest


def stage_tables(scheduled, stages, review_start, seen):
    """Accuracy of every cell at the last scheduled panel at or before each stage end, plus the review phase."""
    out = []
    phases = [dict(language=s["language"], start=s["start"], end=s["end"], exposures=s["exposures"], reason=s["reason"]) for s in stages]
    if review_start is not None and seen > review_start:
        phases.append(dict(language="review_1:1:1", start=review_start, end=seen, exposures=seen-review_start, reason="run_end"))
    for ph in phases:
        panel = max((e for e in scheduled if e["seen"] <= ph["end"]), key=lambda e: e["seen"], default=None)
        ph["panel_seen"] = panel["seen"] if panel else None
        ph["accuracy"] = {k: c/n for k, (c, n) in panel["scores"].items()} if panel else None
        out.append(ph)
    return out


def forgetting(scheduled, stages):
    """For each language stage that is followed by other training, the drop from its stage-end accuracy."""
    result = []
    for i, s in enumerate(stages):
        lang = s["language"]
        end_panel = max((e for e in scheduled if e["seen"] <= s["end"]), key=lambda e: e["seen"], default=None)
        later = [e for e in scheduled if e["seen"] > s["end"]]
        if end_panel is None or not later:
            continue
        entry = dict(language=lang, stage_index=i, stage_end=s["end"], cells={})
        for task in TASKS:
            key = f"{lang}/{task}"
            at_end = end_panel["scores"][key][0]/end_panel["scores"][key][1]
            after = [e["scores"][key][0]/e["scores"][key][1] for e in later]
            entry["cells"][key] = dict(at_stage_end=at_end, min_after=min(after), min_after_seen=later[after.index(min(after))]["seen"],
                                       drop_to_min=at_end-min(after), at_run_end=after[-1], drop_at_run_end=at_end-after[-1])
        result.append(entry)
    return result


def report(args):
    from flystudy.reporting import plot_curves
    root = Path(args.root)
    manifest = json.loads((root/"exploratory-manifest.json").read_text(encoding="utf-8"))
    if Path(args.output).exists():
        raise FileExistsError("Keep previous reports; use a new output path")
    baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    profile = json.loads(Path(args.reference_profile).read_text(encoding="utf-8"))["languages"]
    out = dict(kind="exploratory_v5_order_report", **LABEL, created_at_utc=datetime.now(timezone.utc).isoformat(), manifest=str(root/"exploratory-manifest.json"),
               code_hash=code_hash(), script_hash=file_hash(__file__), protocol=dict(path=manifest["protocol_path"], hash=manifest["protocol_hash"],
               eval_interval=manifest["eval_interval"], mono_cap=manifest["mono_cap"], total_cap=manifest["total_cap"], threshold=manifest["threshold"], vocab_size=manifest["vocab_actual"]),
               tokenizer=dict(path=manifest["tokenizer"], hash=manifest["tokenizer_hash"]), trainable_parameters=manifest["trainable_parameters"],
               data=dict(path=manifest["data"], hash=manifest["dataset_hash"]), gpu_used_for_report=False, test_split_used=False,
               baselines=dict(monolingual_v4=[dict(run_id=b["run_id"], seed=b["seed"], language=b["language"], stop_reason=b["stop_reason"], seen=b["seen"],
                                                   attainment=b["attainment"], gpu_hours=b["gpu_hours"], counters=b["counters"]) for b in baseline["runs"]]), runs=[])
    for r in manifest["runs"]:
        folder = root/r["run_id"]
        events = [json.loads(l) for l in (folder/"events.jsonl").read_text(encoding="utf-8").splitlines()] if (folder/"events.jsonl").exists() else []
        scheduled = [e for e in events if e["kind"] == "scheduled"]
        train_events = [e for e in events if e["kind"] == "train"]
        stages = r.get("stages") or []
        terminal = json.loads((folder/"terminal-evaluation.json").read_text(encoding="utf-8")) if (folder/"terminal-evaluation.json").exists() else None
        clocks, counters = r.get("clocks_seconds") or {}, r.get("counters") or {}
        spu = clocks.get("train", 0)/counters["updates"] if counters.get("updates") else None
        alone = sum(profile[l]["seconds_per_update"] for l in LANGUAGES)/3
        out["runs"].append(dict(run_id=r["run_id"], seed=r["seed"], mode=r["mode"], order=r["order"], status=r["status"], stop_reason=r.get("stop_reason"),
                                seen=r.get("seen"), updates=counters.get("updates"), attempts=r.get("attempts", []),
                                attainment=dict(all_twelve_cells=r.get("first_global"), per_language=r.get("first_language") or {}, per_cell=r.get("first_task") or {},
                                                quantity={lang: (r.get("first_task") or {}).get(f"{lang}/quantity") for lang in LANGUAGES}),
                                stages=stage_tables(scheduled, stages, r.get("review_start"), r.get("seen") or 0), non_mastered_transitions=[s for s in stages if s["reason"] != "mastered"],
                                forgetting=forgetting(scheduled, stages) if r["mode"] == "sequential" else [],
                                language_exposures=r.get("language_exposures"), counters=counters, clocks_seconds=clocks, gpu_hours=r.get("wall_hours"),
                                seconds_per_update=spu, seconds_per_update_alone_v3_profile_mean=alone, slowdown_vs_alone=spu/alone if spu else None,
                                loss_first_last_min=[train_events[0]["loss"], train_events[-1]["loss"], min(e["loss"] for e in train_events)] if train_events else None,
                                scheduled_panels=len(scheduled), curve=[dict(seen=e["seen"], panel=e["panel"], stage=e["stage"], accuracy={k: c/n for k, (c, n) in e["scores"].items()}) for e in scheduled],
                                terminal_evaluation=terminal, independent_test=r.get("independent_test"), independent_test_written=r.get("independent_test_written")))
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
    p.add_argument("--single")
    p.add_argument("--root", default="runs/exploratory-v5-wordbound-order"); p.add_argument("--ledger", default="runs/gpu-ledger.json")
    p.add_argument("--protocol", default="configs/protocol-v4-ai-wordbound-5120.json"); p.add_argument("--graph", default="artifacts/graphs/real.npz")
    p.add_argument("--data", default="data/draft-v4.4"); p.add_argument("--tokenizer", default="artifacts/tokenizer-wordbound-v4.4.json")
    p.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"]); p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=256); p.add_argument("--retry-hours", type=float, default=None)
    p.add_argument("--baseline-root", default="runs/exploratory-v4-wordbound-attainment")
    p.add_argument("--baseline-report", default="reports/exploratory-v4-wordbound-attainment.json")
    p.add_argument("--reference-profile", default="runs/exploratory-v3-wordbound/profile.json")
    p.add_argument("--output", default="reports/exploratory-v5-wordbound-order.json")
    p.add_argument("--figures", default="reports/figures/exploratory-v5-wordbound-order")
    a = p.parse_args()
    if a.plan:
        m = plan(a); print(json.dumps(dict(status="planned", runs=[r["run_id"] for r in m["runs"]], reserved_hours=m["reserved_hours"], code_hash_equals_baseline=m["code_hash_equals_baseline"]), indent=1))
    elif a.execute:
        m = execute(a); print(json.dumps(dict(status="executed", actual_training_gpu_hours=m["actual_training_gpu_hours"],
                                              runs={r["run_id"]: (r.get("status"), r.get("stop_reason"), r.get("seen"), r.get("first_global")) for r in m["runs"]}), indent=1))
    elif a.single:
        print(json.dumps(single(a), indent=1))
    elif a.report:
        r = report(a); print(json.dumps({x["run_id"]: dict(stop=x["stop_reason"], seen=x["seen"], all12=x["attainment"]["all_twelve_cells"],
                                                            stages=[(s["language"], s["exposures"], s["reason"]) for s in x["stages"]]) for x in r["runs"]}, indent=1))
    else:
        p.error("choose --plan, --execute, --single or --report")


if __name__ == "__main__":
    main()

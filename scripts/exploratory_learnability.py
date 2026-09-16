"""Exploratory learnability runs on the real cb5k graph under an INCOMPLETE data-review record.

NOT study evidence. No certified review and no pilot-ready manifest exist, so these runs must not be
labelled pilot or main. They use the official training loop (`flystudy.train.run`) through the code's
non-gated `smoke` cohort with the real graph and the validated CUDA backend, so training, evaluation
schedule, mastery rule and caps are identical to study runs. Actual GPU time is charged to the ledger's
contingency ('reserve') allocation. Failures, interruptions and non-attainment are preserved.

  --plan     write <root>/exploratory-manifest.json and reserve the ledger entries
  --execute  run the plan on two GPUs (one process per run), charge actual hours, collect results
  --single   (internal) execute one run in this process
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from flystudy.budget import Ledger
from flystudy.gates import code_hash
from flystudy.graph import Graph
from flystudy.protocol import Protocol, file_hash, write_json

SEED = 20001  # outside main (1-20), pilot (10001/10002) and auxiliary (7001/7002) seeds
PLAN = [  # run_id, mode, order, reserved hours (cap-bounded projections: mono ~0.02 h, multi ~0.10 h)
    ("explore-v1-mono-en", "mono", ["en"], .25),
    ("explore-v1-mono-de", "mono", ["de"], .25),
    ("explore-v1-mono-ko", "mono", ["ko"], .25),
    ("explore-v1-mixed", "mixed", ["en", "de", "ko"], .75),
    ("explore-v1-seq-en-de-ko", "sequential", ["en", "de", "ko"], .75),
]
LABEL = dict(cohort_label="exploratory", study_evidence=False, review_status="incomplete_ai_review_not_certified",
             gates_passed=["G0 (reports/g0-ai-server.json)", "G1 (reports/g1-ai-server.json)"],
             gates_not_passed=["certify-ai-review", "pilot-ready", "G2", "G3", "order pilots", "power/budget", "freeze"],
             note="Runs are labelled cohort='smoke' by the training code because that is its only non-gated cohort; "
                  "they use the REAL graph and CUDA backend. Interpret as learnability/cost exploration only. "
                  "No language-ordering conclusion is drawn from these runs.")


def plan(args):
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(f"{root} exists; use a new exploratory version")
    root.mkdir(parents=True)
    protocol = Protocol.load(args.protocol)
    graph = Graph.load(args.graph)
    requests = [dict(run_id=rid, category="reserve", hours=hours) for rid, _, _, hours in PLAN]
    Ledger(args.ledger).reserve_bundle(requests)
    manifest = dict(**LABEL, created_at_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, microbatch=args.microbatch,
                    protocol_path=args.protocol, protocol_hash=protocol.hash, training_hash=protocol.training_hash,
                    graph_hash=graph.hash, code_hash=code_hash(), tokenizer_hash=file_hash(args.tokenizer),
                    dataset_hash=json.loads((Path(args.data) / "manifest.json").read_text())["dataset_hash"],
                    ledger=args.ledger, budget_category="reserve", reserved_hours=sum(r["hours"] for r in requests),
                    runs=[dict(run_id=rid, mode=mode, order=order, reserved_hours=hours, status="reserved") for rid, mode, order, hours in PLAN])
    write_json(root / "exploratory-manifest.json", manifest)
    return manifest


def single(args):
    from flystudy.train import run
    rid, mode, order, _ = next(p for p in PLAN if p[0] == args.single)
    protocol = Protocol.load(args.protocol)
    return run(protocol, args.graph, args.data, args.tokenizer, Path(args.root) / rid, mode, tuple(order), SEED,
               cohort="smoke", device=args.device, backend="cuda", microbatch=args.microbatch, smoke=True, run_id=rid)


def execute(args):
    root = Path(args.root)
    manifest_path = root / "exploratory-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["code_hash"] != code_hash():
        raise ValueError("Code changed after the exploratory plan was written")
    ledger = Ledger(args.ledger)
    pending = [r for r in manifest["runs"] if r["status"] == "reserved"]
    active = {}
    while pending or active:
        for device in args.devices:
            if device in active or not pending:
                continue
            r = pending.pop(0)
            log = (root / f"{r['run_id']}.console.log").open("w", encoding="utf-8")
            cmd = [sys.executable, __file__, "--single", r["run_id"], "--device", device, "--root", str(root), "--ledger", args.ledger,
                   "--protocol", args.protocol, "--graph", args.graph, "--data", args.data, "--tokenizer", args.tokenizer,
                   "--microbatch", str(args.microbatch)]
            active[device] = dict(proc=subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log=log, run=r,
                                  start=time.perf_counter(), deadline=r["reserved_hours"] * 3600)
        for device, job in list(active.items()):
            code = job["proc"].poll()
            elapsed = time.perf_counter() - job["start"]
            if code is None and elapsed > job["deadline"]:
                job["proc"].kill(); job["proc"].wait(); code = -9
                job["run"]["interrupted"] = "reservation_wall_clock_exceeded"
            if code is None:
                continue
            job["log"].close(); del active[device]
            hours = (time.perf_counter() - job["start"]) / 3600
            summary_path = root / job["run"]["run_id"] / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            stop = summary.get("stop_reason")
            status = "completed" if stop in ("mastered", "administrative_cap") else ("budget_interruption" if code == -9 else "technical_failure")
            ledger.charge(job["run"]["run_id"], hours, status)
            job["run"].update(status=status, exit_code=code, device=device, wall_hours=hours, stop_reason=stop,
                              seen=summary.get("seen"), first_task=summary.get("first_task"), first_language=summary.get("first_language"),
                              first_global=summary.get("first_global"), stages=summary.get("stages"),
                              clocks_seconds=summary.get("clocks_seconds"), trainable_parameters=summary.get("trainable_parameters"))
            write_json(manifest_path, manifest)
        time.sleep(1)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["actual_gpu_hours"] = sum(r.get("wall_hours", 0) for r in manifest["runs"])
    write_json(manifest_path, manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true"); p.add_argument("--execute", action="store_true"); p.add_argument("--single")
    p.add_argument("--root", default="runs/exploratory-v1"); p.add_argument("--ledger", default="runs/gpu-ledger.json")
    p.add_argument("--protocol", default="configs/protocol-v4-ai.json"); p.add_argument("--graph", default="artifacts/graphs/real.npz")
    p.add_argument("--data", default="data/draft-v4.3"); p.add_argument("--tokenizer", default="artifacts/tokenizer-v4.3.json")
    p.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"]); p.add_argument("--device", default="cuda:0")
    p.add_argument("--microbatch", type=int, default=256)
    a = p.parse_args()
    if a.plan:
        m = plan(a); print(json.dumps(dict(status="planned", runs=[r["run_id"] for r in m["runs"]], reserved_hours=m["reserved_hours"]), indent=1))
    elif a.execute:
        m = execute(a); print(json.dumps(dict(status="executed", actual_gpu_hours=m["actual_gpu_hours"],
                                              runs={r["run_id"]: (r.get("status"), r.get("stop_reason"), r.get("seen")) for r in m["runs"]}), indent=1))
    elif a.single:
        s = single(a); print(json.dumps(dict(run_id=s["run_id"], stop_reason=s["stop_reason"], seen=s["seen"], gpu_hours=s["gpu_hours"])))
    else:
        p.error("choose --plan, --execute or --single")


if __name__ == "__main__":
    main()

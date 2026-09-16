from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys

from .protocol import Protocol, write_json


def parser():
    p = argparse.ArgumentParser(description="Fly language study v4; primary training is gated, smoke tests are labeled")
    commands = p.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init"); init.add_argument("--output", required=True)
    prep = commands.add_parser("prepare-data"); prep.add_argument("--output", required=True); prep.add_argument("--tiny", action="store_true")
    for name in ("audit-data", "baselines", "cue-baselines"):
        sub = commands.add_parser(name); sub.add_argument("--data", required=True)
    tok = commands.add_parser("tokenizer"); tok.add_argument("--data", required=True); tok.add_argument("--output", required=True); tok.add_argument("--vocab-size", type=int, default=4096)
    graph = commands.add_parser("fetch-graph"); graph.add_argument("--output", required=True); graph.add_argument("--cache", required=True)
    scramble = commands.add_parser("scramble"); scramble.add_argument("--graph", required=True); scramble.add_argument("--output", required=True); scramble.add_argument("--seed", type=int, required=True)
    sim = commands.add_parser("simulate"); sim.add_argument("--output", required=True); sim.add_argument("--quick", action="store_true"); sim.add_argument("--repetitions", type=int, default=5000)
    ps = commands.add_parser("power-report"); ps.add_argument("--input", required=True); ps.add_argument("--output", required=True)
    g0 = commands.add_parser("g0"); g0.add_argument("--graph", required=True); g0.add_argument("--output", required=True)
    g0.add_argument("--ledger"); g0.add_argument("--reservation-id")
    smoke = commands.add_parser("smoke"); smoke.add_argument("--output", required=True)
    review = commands.add_parser("certify-review"); review.add_argument("--data", required=True); review.add_argument("--csv", required=True); review.add_argument("--templates", required=True); review.add_argument("--output", required=True)
    ready = commands.add_parser("pilot-ready")
    for arg in ("g0", "review", "data", "graph", "tokenizer", "output"):
        ready.add_argument("--"+arg, required=True)
    plan = commands.add_parser("reserve-pilots"); plan.add_argument("--protocol", required=True); plan.add_argument("--output", required=True); plan.add_argument("--ledger", required=True)
    for mode in ("mono", "mixed", "sequential"):
        plan.add_argument("--"+mode+"-hours", type=float, required=True)
    pilot_report = commands.add_parser("pilot-report"); pilot_report.add_argument("--protocol", required=True); pilot_report.add_argument("--runs", required=True); pilot_report.add_argument("--output", required=True)
    assess = commands.add_parser("assess-design")
    for arg in ("simulation", "pilots", "costs", "resolution", "output"):
        assess.add_argument("--"+arg, required=True)
    assess.add_argument("--main-hours", type=float, default=448)
    freeze = commands.add_parser("freeze")
    for arg in ("protocol", "readiness", "g1", "pilots", "design", "output", "ledger"):
        freeze.add_argument("--"+arg, required=True)
    train = commands.add_parser("run")
    for arg in ("protocol", "graph", "data", "tokenizer", "output", "manifest", "matrix", "run-id", "ledger"):
        train.add_argument("--"+arg, required=True)
    train.add_argument("--device", default="cuda:0"); train.add_argument("--microbatch", type=int, default=32); train.add_argument("--resume")
    train.add_argument("--reservation-id", help="A new reservation ID for a retry; original accounting is preserved")
    g1 = commands.add_parser("g1")
    for arg in ("protocol", "graph", "data", "tokenizer", "output", "g0", "ledger", "reservation-id"):
        g1.add_argument("--"+arg, required=True)
    g1.add_argument("--device", default="cuda:0")
    reserve = commands.add_parser("reserve"); reserve.add_argument("--ledger", required=True); reserve.add_argument("--requests", required=True)
    transfer=commands.add_parser('transfer-reserve'); transfer.add_argument('--ledger',required=True)
    transfer.add_argument('--destination',required=True); transfer.add_argument('--hours',type=float,required=True)
    release = commands.add_parser("release-auxiliary"); release.add_argument("--ledger", required=True); release.add_argument("--category", choices=("review","shuffle","tokenizer"), required=True)
    analyze = commands.add_parser("analyze"); analyze.add_argument("--runs", required=True); analyze.add_argument("--output", required=True)
    calibration = commands.add_parser("calibrate")
    for arg in ("protocol","graph","data","tokenizer","output","readiness","ledger","reservation-id"):
        calibration.add_argument("--"+arg,required=True)
    calibration.add_argument("--language",choices=("en","de","ko"),required=True)
    calibration.add_argument("--seed",type=int,required=True); calibration.add_argument("--device",default="cuda:0")
    calibration.add_argument("--microbatch",type=int,default=32)
    res = commands.add_parser("resolution-report")
    for arg in ("calibrations","pilots","output"): res.add_argument("--"+arg,required=True)
    cost = commands.add_parser("cost-report")
    for arg in ("runs","output"): cost.add_argument("--"+arg,required=True)
    cost.add_argument("--remaining-hours",type=float,required=True); cost.add_argument("--serial-overhead-hours",type=float,required=True)
    prof = commands.add_parser("profile")
    for arg in ("protocol","graph","data","tokenizer","g0","output","ledger","reservation-id"):
        prof.add_argument("--"+arg,required=True)
    prof.add_argument("--device",default="cuda:0")
    aux = commands.add_parser("fixed-review")
    for arg in ("runs","output","ledger"): aux.add_argument("--"+arg,required=True)
    aux.add_argument("--target",type=int,choices=(100000,300000),required=True)
    aux.add_argument("--hours-per-run",type=float,required=True); aux.add_argument("--device",default="cuda:0")
    aux.add_argument("--microbatch",type=int,default=32)
    ap = commands.add_parser('reserve-auxiliary')
    for arg in ('main-manifest','graph','other','tokenizer','data','output','ledger'):
        ap.add_argument('--'+arg,required=True)
    ap.add_argument('--category',choices=('shuffle','tokenizer'),required=True)
    ap.add_argument('--hours-per-run',type=float,required=True)
    camp=commands.add_parser("campaign")
    for arg in ("protocol","graph","data","tokenizer","manifest","matrix","ledger","output"):
        camp.add_argument("--"+arg,required=True)
    camp.add_argument("--devices",nargs=2,default=["cuda:0","cuda:1"])
    camp.add_argument("--microbatch",type=int,default=32)
    return p


def smoke(output):
    import torch
    from .data import generate, train_tokenizer
    from .graph import synthetic
    from .train import run
    from .reporting import plot_curves
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    generate(output/"data", train_per_task=18, panel_per_task=2)
    train_tokenizer(output/"data", output/"tokenizer.json", vocab_size=320)
    synthetic().save(output/"graph.npz")
    protocol = Protocol(vocab_size=320, embed_dim=8, effective_batch=4, eval_interval=8,
                        mono_cap=12, total_cap=32, panel_per_task=2)
    write_json(output/"protocol.json", asdict(protocol))
    result = run(protocol, output/"graph.npz", output/"data", output/"tokenizer.json", output/"runs"/"smoke-seq",
                 "sequential", ("en","de","ko"), 7, cohort="smoke", device="cpu", backend="dense", microbatch=3, smoke=True)
    plot_curves(output/"runs", output/"figures")
    return dict(status="smoke_completed", actual_study_evidence=False, exposures=result["seen"], result=str(output/"runs"/"smoke-seq"/"summary.json"))


def dispatch(a):
    if a.command == "init":
        from .gates import environment
        write_json(Path(a.output)/"protocol.json", asdict(Protocol()))
        env = environment(); write_json(Path(a.output)/"environment.json", env)
        return env
    if a.command == "prepare-data":
        from .data import generate
        return generate(a.output, 18, 2) if a.tiny else generate(a.output)
    if a.command in ("audit-data", "baselines", "tokenizer"):
        from .data import audit, baselines, train_tokenizer
        return {"audit-data": lambda: audit(a.data), "baselines": lambda: baselines(a.data),
                "tokenizer": lambda: train_tokenizer(a.data, a.output, a.vocab_size)}[a.command]()
    if a.command == "cue-baselines":
        from .cues import run_baselines
        return run_baselines(a.data)
    if a.command == "fetch-graph":
        from .graph import fetch_real
        g = fetch_real(a.output, a.cache)
        return dict(neurons=g.n, edges=len(g.src), graph_hash=g.hash)
    if a.command == "scramble":
        from .graph import Graph, scramble
        g = scramble(Graph.load(a.graph), a.seed); g.save(a.output)
        return g.provenance
    if a.command == "simulate":
        from .statistics import simulate_grid
        return simulate_grid(a.output, a.repetitions, a.quick)
    if a.command == "power-report":
        from .reporting import power_summary
        return power_summary(a.input, a.output)
    if a.command == "g0":
        import time
        import torch
        from .budget import Ledger
        from .gates import g0
        from .graph import Graph
        if torch.cuda.is_available():
            if not a.ledger or not a.reservation_id:
                raise ValueError("GPU G0 requires a reserved g0_g1 budget entry")
            ledger=Ledger(a.ledger); entry=ledger.reservation(a.reservation_id)
            if entry["category"] != "g0_g1": raise ValueError("Wrong G0 budget category")
            start=time.perf_counter()
            result=g0(Graph.load(a.graph),a.output)
            hours=2*(time.perf_counter()-start)/3600
            result["gpu_hours"]=hours
            write_json(a.output,result)
            ledger.charge(a.reservation_id,hours,"completed" if result["status"] == "passed" else "technical_failure")
            return result
        return g0(Graph.load(a.graph), a.output)
    if a.command == "smoke":
        return smoke(a.output)
    if a.command == "certify-review":
        from .gates import certify_review
        return certify_review(a.data, a.csv, a.templates, a.output)
    if a.command == "pilot-ready":
        from .workflow import pilot_ready
        return pilot_ready(a.g0,a.review,a.data,a.graph,a.tokenizer,a.output)
    if a.command == "reserve-pilots":
        from .workflow import reserve_pilots
        return reserve_pilots(Protocol.load(a.protocol), a.output, a.ledger,
                              dict(mono=a.mono_hours, mixed=a.mixed_hours, sequential=a.sequential_hours))
    if a.command == "pilot-report":
        from .gates import summarize_pilot
        return summarize_pilot(sorted(Path(a.runs).glob("*/summary.json")), Protocol.load(a.protocol), a.output)
    if a.command == "assess-design":
        from .workflow import design_decision
        return design_decision(a.simulation,a.pilots,a.costs,a.resolution,a.output,a.main_hours)
    if a.command == "freeze":
        from .workflow import freeze
        return freeze(Protocol.load(a.protocol),a.readiness,a.g1,a.pilots,a.design,a.output,a.ledger)
    if a.command == "run":
        from .train import run
        matrix = json.loads(Path(a.matrix).read_text(encoding="utf-8"))
        found = [r for r in matrix["runs"] if r["run_id"] == a.run_id]
        if len(found) != 1:
            raise ValueError("Run not found uniquely in matrix")
        r = found[0]
        return run(Protocol.load(a.protocol),a.graph,a.data,a.tokenizer,a.output,r["mode"],r["order"],r["seed"],
                   cohort=r["cohort"],device=a.device,microbatch=a.microbatch,launch=a.manifest,ledger_path=a.ledger,
                   run_id=a.run_id,resume=a.resume, reservation_id=a.reservation_id,
                   review_samples=r["cohort"] == "main" and r["seed"] in (1,2) and r["mode"] == "sequential")
    if a.command == "g1":
        import time
        from .budget import Ledger
        from .gates import code_hash
        from .train import memorize
        g0 = json.loads(Path(a.g0).read_text())
        if g0["status"] != "passed" or g0["environment"]["code_hash"] != code_hash():
            raise ValueError("G0 has not passed on the current code")
        ledger = Ledger(a.ledger); entry = ledger.reservation(a.reservation_id)
        if entry['category'] != 'g0_g1': raise ValueError('Wrong G1 budget category')
        start, status = time.perf_counter(), "technical_failure"
        try:
            result = memorize(Protocol.load(a.protocol),a.graph,a.data,a.tokenizer,a.output,a.device,max_seconds=entry['reserved']*3600)
            status = "completed"
            return result
        finally:
            ledger.charge(a.reservation_id,(time.perf_counter()-start)/3600,status)
    if a.command in ("reserve", "release-auxiliary"):
        from .budget import Ledger
        ledger = Ledger(a.ledger)
        if a.command == "reserve":
            ledger.reserve_bundle(json.loads(Path(a.requests).read_text()))
        else:
            ledger.release_auxiliary(a.category)
        return dict(status="recorded")
    if a.command == 'transfer-reserve':
        from .budget import Ledger
        Ledger(a.ledger).transfer_reserve(a.destination,a.hours)
        return dict(status='recorded')
    if a.command == "analyze":
        from .reporting import analyze
        return analyze(a.runs,a.output)
    if a.command == "calibrate":
        from .measurement import calibrate
        return calibrate(Protocol.load(a.protocol),a.graph,a.data,a.tokenizer,a.output,a.language,a.seed,a.readiness,
                         a.ledger,a.reservation_id,a.device,a.microbatch)
    if a.command == "resolution-report":
        from .measurement import resolution_report
        return resolution_report(a.calibrations,a.pilots,a.output)
    if a.command == "cost-report":
        from .measurement import cost_report
        return cost_report(a.runs,a.output,a.remaining_hours,a.serial_overhead_hours)
    if a.command == "profile":
        import time
        from .measurement import profile
        from .budget import Ledger
        ledger=Ledger(a.ledger); ledger.reservation(a.reservation_id)
        start,status=time.perf_counter(),"technical_failure"
        try:
            result=profile(Protocol.load(a.protocol),a.graph,a.data,a.tokenizer,a.g0,a.output,a.device)
            status="completed"
            return result
        finally:
            ledger.charge(a.reservation_id,(time.perf_counter()-start)/3600,status)
    if a.command == "fixed-review":
        from .auxiliary import fixed_review_bundle
        return fixed_review_bundle(a.runs,a.output,a.target,a.ledger,a.hours_per_run,a.device,a.microbatch)
    if a.command == 'reserve-auxiliary':
        from .auxplan import reserve_auxiliary
        return reserve_auxiliary(a.main_manifest,a.category,a.graph,a.other,a.tokenizer,a.data,a.output,a.ledger,a.hours_per_run)
    if a.command == "campaign":
        from .campaign import campaign
        return campaign(a.protocol,a.graph,a.data,a.tokenizer,a.manifest,a.matrix,a.ledger,a.output,a.devices,a.microbatch)
    raise ValueError(a.command)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if isinstance(result, dict) and (result.get("status") in ("blocked", "failed", "incomplete") or
                result.get('stop_reason') in ('technical_failure','budget_interruption','debug_interruption')):
            return 2
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

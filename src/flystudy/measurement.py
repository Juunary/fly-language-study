"""Independent calibration runs; no counterfactual replay of sequential curricula."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
from tokenizers import Tokenizer

from .budget import Ledger
from .gates import code_hash
from .graph import Graph
from .model import FlyClassifier, train_batch
from .protocol import LANGUAGES, TASKS, INTERVALS, file_hash, write_json
from .runtime import Corpus, AlignedSampler, evaluate, append_event, sync


def calibrate(protocol, graph_path, dataset, tokenizer_path, output, language, seed, readiness_path,
              ledger_path, reservation_id, device="cuda:0", microbatch=32):
    if seed in protocol.main_seeds or language not in LANGUAGES or not device.startswith("cuda"):
        raise ValueError("Calibration needs an independent seed and CUDA")
    ready = json.loads(Path(readiness_path).read_text())
    graph = Graph.load(graph_path)
    if ready.get("status") != "pilot_ready" or ready["graph_hash"] != graph.hash or ready["code_hash"] != code_hash():
        raise ValueError("Missing current pilot readiness")
    metadata = json.loads((Path(dataset)/"manifest.json").read_text())
    if ready["dataset_hash"] != metadata["dataset_hash"] or ready["tokenizer_hash"] != file_hash(tokenizer_path):
        raise ValueError("Calibration input mismatch")
    ledger = Ledger(ledger_path)
    budget = ledger.reservation(reservation_id)
    if budget["category"] != "g2":
        raise ValueError("Calibration must be charged to G2")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        corpus, model = Corpus(dataset,tokenizer), FlyClassifier(graph,tokenizer.get_vocab_size(),protocol.embed_dim,
                                     protocol.microsteps,seed,"cuda").to(device)
        sampler = AlignedSampler(corpus,seed)
        optimizer = torch.optim.AdamW(model.parameter_groups(protocol),weight_decay=protocol.weight_decay)
    except Exception:
        ledger.charge(reservation_id,(time.perf_counter()-start)/3600,'technical_failure')
        raise
    seen, next_eval, status = 0, 5120, "technical_failure"
    try:
        while seen < protocol.mono_cap:
            if (time.perf_counter()-start)/3600 >= budget["reserved"]:
                status = "budget_interruption"
                break
            count = min(protocol.effective_batch,next_eval-seen,protocol.mono_cap-seen)
            train_batch(model,optimizer,sampler.sample(count,language),corpus.pad_id,microbatch,protocol)
            seen += count
            if seen == next_eval or seen == protocol.mono_cap:
                a,_ = evaluate(model,corpus,"dev_a",(language,),microbatch,optimizer,sampler)
                b,_ = evaluate(model,corpus,"dev_b",(language,),microbatch,optimizer,sampler)
                append_event(output/"calibration.jsonl",dict(seen=seen,scores_a=a,scores_b=b))
                next_eval = min(next_eval+5120,protocol.mono_cap)
        if seen == protocol.mono_cap:
            status = "completed"
    finally:
        sync(device)
        report = dict(status=status,language=language,seed=seed,seen=seen,cap=protocol.mono_cap,
                      dataset_hash=metadata["dataset_hash"],graph_hash=graph.hash,tokenizer_hash=file_hash(tokenizer_path),
                      code_hash=code_hash(),gpu_hours=(time.perf_counter()-start)/3600,
                      continued_after_mastery=True,charged_category="g2")
        write_json(output/"calibration-meta.json",report)
        ledger.charge(reservation_id,report["gpu_hours"],status)
    return report


def resolution_report(root, pilot_report_path, output):
    pilot = json.loads(Path(pilot_report_path).read_text())
    paired = defaultdict(dict)
    evidence = {str(Path(pilot_report_path).resolve()): file_hash(pilot_report_path)}
    for meta_path in Path(root).glob("*/calibration-meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta["status"] != "completed":
            raise ValueError("Incomplete calibration trajectory")
        if any(meta[k] != pilot[k] for k in ("dataset_hash","graph_hash","tokenizer_hash","code_hash")):
            raise ValueError("Calibration/pilot provenance mismatch")
        log = meta_path.parent/"calibration.jsonl"
        records = {r["seen"]: r for r in (json.loads(line) for line in log.read_text().splitlines())}
        times = {}
        for interval in INTERVALS:
            grid = list(range(interval,meta["cap"],interval))+[meta["cap"]]
            previous = False
            event = None
            for i, exposure in enumerate(grid):
                row = records.get(exposure)
                if row is None:
                    raise ValueError("Dense calibration must include both full panels at every 5,120 point and cap")
                scores = row["scores_a" if i % 2 == 0 else "scores_b"]
                passed = len(scores) == 4 and all(n == 1000 and c/n >= .8 for c,n in scores.values())
                if passed and previous:
                    event = exposure
                    break
                previous = passed
            times[interval] = event or meta["cap"]
        if meta["language"] in paired[meta["seed"]]:
            raise ValueError("Duplicate calibration seed/language")
        paired[meta["seed"]][meta["language"]] = times
        evidence[str(meta_path.resolve())] = file_hash(meta_path)
        evidence[str(log.resolve())] = file_hash(log)
    if len(paired) < 2 or any(set(group) != set(LANGUAGES) for group in paired.values()):
        raise ValueError("Need at least two complete three-language calibration seed blocks")
    result = dict(intervals={}, evidence_files=evidence, threshold=.05,
                  interpretation="Monolingual measurement resolution only; sequential policies require actual pilots.")
    for interval in INTERVALS:
        distortions = []
        for group in paired.values():
            reference = np.mean([group[lang][5120] for lang in LANGUAGES])
            for i,a in enumerate(LANGUAGES):
                for b in LANGUAGES[i+1:]:
                    fine_delta = group[a][5120]-group[b][5120]
                    coarse_delta = group[a][interval]-group[b][interval]
                    distortions.append(abs(coarse_delta-fine_delta)/reference)
        actual = pilot.get("eval_interval") == interval and pilot.get("status") == "passed"
        result["intervals"][str(interval)] = dict(status="passed" if max(distortions) <= .05 else "failed",
                  maximum_paired_distortion=max(distortions),actual_sequence_pilot=actual,n_calibration_seeds=len(paired))
    write_json(output,result)
    return result


def cost_report(run_root, output, remaining_calendar_hours, serial_overhead_hours):
    if remaining_calendar_hours <= 0 or serial_overhead_hours < 0:
        raise ValueError("Invalid remaining time")
    records = [json.loads(p.read_text()) for p in Path(run_root).glob("*/summary.json")]
    intervals = defaultdict(lambda: defaultdict(list))
    evidence = {}
    for path in Path(run_root).glob("*/summary.json"):
        r = json.loads(path.read_text())
        if r["cohort"] != "pilot" or r["synthetic"] or r["stop_reason"] not in ("mastered","administrative_cap"):
            raise ValueError("Cost projection requires completed actual GPU pilots")
        key = "mixed" if r["mode"] == "mixed" else ("mono-"+r["order"][0] if r["mode"] == "mono" else "seq-"+"-".join(r["order"]))
        intervals[str(r["protocol"]["eval_interval"])][key].append(r["gpu_hours"]*r["cap"]/r["seen"])
        evidence[str(path.resolve())] = file_hash(path)
    result = dict(intervals={k:dict(per_run_upper_hours={c:max(v) for c,v in groups.items()}) for k,groups in intervals.items()},
                  remaining_calendar_hours=remaining_calendar_hours,serial_overhead_hours=serial_overhead_hours,
                  evidence_files=evidence,method="Maximum observed full-run cost per exposure extrapolated to cap; design adds 25% contingency.")
    write_json(output,result)
    return result


def profile(protocol, graph_path, dataset, tokenizer_path, g0_path, output, device="cuda:0"):
    gate = json.loads(Path(g0_path).read_text())
    graph = Graph.load(graph_path)
    if gate.get("status") != "passed" or gate["graph_hash"] != graph.hash or gate["environment"]["code_hash"] != code_hash():
        raise ValueError("G0 must pass before measuring the CUDA path")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    corpus = Corpus(dataset,tokenizer)
    longest = sorted([(ids,r["label"]) for lang in LANGUAGES for r,ids in corpus.split("train")[lang]],key=lambda x:len(x[0]),reverse=True)[:256]
    rows = []
    for batch in (32,64,128,256):
        torch.cuda.empty_cache()
        try:
            model = FlyClassifier(graph,tokenizer.get_vocab_size(),protocol.embed_dim,protocol.microsteps,40001,"cuda").to(device)
            optimizer = torch.optim.AdamW(model.parameter_groups(protocol),weight_decay=protocol.weight_decay)
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(2): train_batch(model,optimizer,longest,corpus.pad_id,batch,protocol)
            sync(device); start=time.perf_counter()
            for _ in range(5): train_batch(model,optimizer,longest,corpus.pad_id,batch,protocol)
            sync(device)
            seconds=(time.perf_counter()-start)/5
            peak=torch.cuda.max_memory_allocated(device)
            total=torch.cuda.get_device_properties(device).total_memory
            rows.append(dict(microbatch=batch,seconds_per_256=seconds,peak_bytes=peak,within_80_percent_vram=peak <= .8*total))
            del model,optimizer
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(microbatch=batch,status="out_of_memory",within_80_percent_vram=False))
            del model
            if "optimizer" in locals(): del optimizer
    feasible=[r for r in rows if r["within_80_percent_vram"]]
    result=dict(status="passed" if feasible else "failed",device=device,rows=rows,
                selected_microbatch=min(feasible,key=lambda r:r["seconds_per_256"])["microbatch"] if feasible else None,
                longest_examples_tokens=[len(ids) for ids,_ in longest],includes_evaluation=False,
                note="Final run pricing comes from complete pilots, including full evaluation and test, not this training-only benchmark.")
    write_json(output,result)
    return result

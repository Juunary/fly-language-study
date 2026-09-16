"""Evaluation, independent-test and checkpoint throughput on the validated CUDA path.

The delivered `profile` command times training only. This script times the full-panel
evaluation exactly as `run` performs it (immutable_evaluation with optimizer/sampler
guards), plus one checkpoint write, and projects cap-bounded per-run hours for the
pilot reservation. Charged to the g0_g1 (G0/G1/throughput) allocation. It is a planning
measurement, not a gate and not study evidence; actual run pricing comes from pilots.
"""
import argparse, json, time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from flystudy.budget import Ledger
from flystudy.gates import code_hash, environment
from flystudy.graph import Graph
from flystudy.model import FlyClassifier, train_batch
from flystudy.protocol import Protocol, LANGUAGES, INTERVALS, write_json
from flystudy.runtime import Corpus, AlignedSampler, evaluate, save_checkpoint, sync, rng_state

p = argparse.ArgumentParser()
for name in ("protocol", "graph", "data", "tokenizer", "g0", "profile", "output", "ledger", "reservation-id"):
    p.add_argument("--" + name, required=True)
p.add_argument("--device", default="cuda:0")
p.add_argument("--microbatches", type=int, nargs="+", default=[32, 256])
a = p.parse_args()

gate = json.loads(Path(a.g0).read_text())
graph = Graph.load(a.graph)
if gate.get("status") != "passed" or gate["graph_hash"] != graph.hash or gate["environment"]["code_hash"] != code_hash():
    raise SystemExit("G0 must pass on the current code before measuring the CUDA path")
train_profile = json.loads(Path(a.profile).read_text())
if train_profile.get("status") != "passed":
    raise SystemExit("Training profile must pass first")
ledger = Ledger(a.ledger)
entry = ledger.reservation(a.reservation_id)
if entry["category"] != "g0_g1":
    raise SystemExit("Charge evaluation throughput to g0_g1")

start, status = time.perf_counter(), "technical_failure"
report = dict(status=status, device=a.device, graph_hash=graph.hash, code_hash=code_hash(), environment=environment(),
              is_gate=False, is_study_evidence=False, rows=[], per_language_dev_a=[], checkpoint=None, projection=None)
try:
    protocol = Protocol.load(a.protocol)
    tokenizer = Tokenizer.from_file(a.tokenizer)
    corpus = Corpus(a.data, tokenizer)
    seed = 40002  # throughput-only seed; outside main/pilot/auxiliary seeds
    model = FlyClassifier(graph, tokenizer.get_vocab_size(), protocol.embed_dim, protocol.microsteps, seed, "cuda").to(a.device)
    optimizer = torch.optim.AdamW(model.parameter_groups(protocol), weight_decay=protocol.weight_decay)
    sampler = AlignedSampler(corpus, seed)
    train_batch(model, optimizer, sampler.sample(protocol.effective_batch), corpus.pad_id, 256, protocol)  # populate optimizer state
    for mb in a.microbatches:
        for split in ("dev_a", "dev_b", "test"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(a.device)
            sync(a.device); t0 = time.perf_counter()
            scores, _ = evaluate(model, corpus, split, LANGUAGES, mb, optimizer, sampler)
            sync(a.device); sec = time.perf_counter() - t0
            items = sum(v[1] for v in scores.values())
            report["rows"].append(dict(microbatch=mb, split=split, cells=len(scores), items=items, seconds=sec,
                                       items_per_second=items / sec, peak_bytes=torch.cuda.max_memory_allocated(a.device)))
    mb = train_profile["selected_microbatch"]
    for lang in LANGUAGES:
        sync(a.device); t0 = time.perf_counter()
        scores, _ = evaluate(model, corpus, "dev_a", (lang,), mb, optimizer, sampler)
        sync(a.device)
        report["per_language_dev_a"].append(dict(language=lang, microbatch=mb, items=sum(v[1] for v in scores.values()),
                                                 seconds=time.perf_counter() - t0))
    tmp = Path(a.output).with_suffix(".tmp.pt")
    sync(a.device); t0 = time.perf_counter()
    save_checkpoint(tmp, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state_dict(), rng=rng_state()))
    report["checkpoint"] = dict(seconds=time.perf_counter() - t0, bytes=tmp.stat().st_size, path_filesystem=str(tmp.parent.resolve()))
    tmp.unlink()

    # Cap-bounded projection with the selected microbatch: worst case = every run reaches its cap.
    train_sec_per_update = next(r["seconds_per_256"] for r in train_profile["rows"] if r["microbatch"] == mb)
    eval12 = max(r["seconds"] for r in report["rows"] if r["microbatch"] == mb and r["split"] in ("dev_a", "dev_b"))
    test12 = next(r["seconds"] for r in report["rows"] if r["microbatch"] == mb and r["split"] == "test")
    eval4 = max(r["seconds"] for r in report["per_language_dev_a"])
    ck = report["checkpoint"]["seconds"]
    projection = dict(assumptions="Every run reaches its administrative cap; training time uses the 256 longest training items "
                                  "(upper bound); evaluation time uses the slowest full panel measured here; one checkpoint per "
                                  "evaluation; one independent test; no contingency (reserve-pilots adds 25%).",
                      selected_microbatch=mb, train_seconds_per_update=train_sec_per_update,
                      eval_seconds_12_cells=eval12, eval_seconds_4_cells=eval4, test_seconds_12_cells=test12, checkpoint_seconds=ck,
                      by_interval={})
    for interval in INTERVALS:
        def hours(cap, cells):
            updates = -(-cap // protocol.effective_batch)
            evals = -(-cap // interval)
            per_eval = eval12 if cells == 12 else eval4
            per_test = test12 if cells == 12 else test12 / 3
            return (updates * train_sec_per_update + evals * (per_eval + ck) + per_test) / 3600
        mono = hours(protocol.mono_cap, 4)
        multi = hours(protocol.total_cap, 12)
        projection["by_interval"][str(interval)] = dict(mono_hours=mono, mixed_hours=multi, sequential_hours=multi,
            calibration_hours=(-(-protocol.mono_cap // protocol.effective_batch) * train_sec_per_update
                               + -(-protocol.mono_cap // 5120) * 2 * eval4) / 3600 if interval == 5120 else None,
            pilot_block_two_seeds_hours=2 * (3 * mono + 7 * multi))
    report["projection"] = projection
    status = "completed"
finally:
    sync(a.device)
    report.update(status=status, gpu_hours=(time.perf_counter() - start) / 3600, charged_category="g0_g1",
                  reservation_id=a.reservation_id)
    write_json(a.output, report)
    ledger.charge(a.reservation_id, report["gpu_hours"], status)
print(json.dumps(dict(status=status, gpu_hours=report["gpu_hours"], checkpoint=report["checkpoint"],
                      rows=[dict(mb=r["microbatch"], split=r["split"], s=round(r["seconds"], 2), ips=round(r["items_per_second"]),
                                 peak_GiB=round(r["peak_bytes"] / 2**30, 2)) for r in report["rows"]],
                      per_language=[dict(l=r["language"], s=round(r["seconds"], 2)) for r in report["per_language_dev_a"]],
                      projection=report["projection"] and report["projection"]["by_interval"]), indent=1))

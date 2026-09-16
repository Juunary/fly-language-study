"""Exploratory v2: monolingual exposure to 900k on the exploratory-v1 configuration. NOT study evidence.

Only the exposure budget differs from exploratory v1: EN, DE and KO monolingual runs with seed 20001 continue to a
mono cap of 900,000 exposures instead of 200,000, with the same data, tokenizer, graph, model, learning rates,
threshold, evaluation schedule and code. Each run goes through the official training loop (`flystudy.train.run`,
smoke cohort, real graph, CUDA backend) in four stages, using the loop's debugging update cap and its crash-recovery
resume, so that a checkpoint is preserved near 200k, 300k, 600k and 900k exposures without changing src/. Every
batch holds 256 examples, so the stage boundaries are exact update counts. The 900k stage stops one update (160
exposures) before the administrative cap: the terminal path, which would run the independent test, is never taken,
and the test split is not used anywhere here. Actual GPU time is charged to the contingency ('reserve') allocation.

  --plan      write <root>/exploratory-manifest.json and reserve the ledger bundle (three runs + one evaluation)
  --execute   run the three languages on two GPUs (one process per language), charge actual hours
  --execute --retry  new reservations for runs that did not complete; resume from their preserved stage checkpoints
  --single    (internal) execute one language's four stages in this process
  --evaluate  score the preserved checkpoints, plus the v1 200k monolingual and 900k mixed checkpoints, on a fixed
              training sample and on original / declarative (assertion) dev A/B; charge the evaluation reservation
"""
from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import time
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
from flystudy.runtime import immutable_evaluation, sync

SEED = 20001  # same exploratory seed as v1
BATCH = 256
STAGES = [("200k", 781), ("300k", 1172), ("600k", 2344), ("900k", 3515)]  # cumulative updates; seen = updates * 256
RUN_HOURS, EVAL_HOURS = .3, .1  # cap-bounded projection per run ~0.1 h (v1 mono 200k: 0.017-0.020 h); bundle total 1.0 h
EVAL_ID = "explore-v2-checkpoint-eval-1"
V1_ROOT = Path("runs/exploratory-v1")
V1_FILES = ("primary.pt", "summary.json", "independent-test.json", "events.jsonl", "metadata.json")
SAMPLE_PER_TASK = 1000
LABEL = dict(cohort_label="exploratory", study_evidence=False, review_status="incomplete_ai_review_not_certified",
             question="exposure budget only: monolingual 200k -> 900k with everything else fixed",
             note="Runs are labelled cohort='smoke' by the training code because that is its only non-gated cohort; "
                  "they use the REAL graph and CUDA backend. Threshold and per-language settings are unchanged. "
                  "No language-ordering conclusion is drawn. The test split is not used.")
_spec = importlib.util.spec_from_file_location("frame_diagnostic_eval", Path(__file__).with_name("frame_diagnostic_eval.py"))
_frame = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_frame)


def run_id(lang):
    return f"explore-v2-mono-{lang}"


@contextmanager
def rng_state_on_cpu():
    """train.run loads a resume checkpoint with map_location=<cuda device>, which moves the saved CPU RNG-state
    tensors to the GPU; torch.set_rng_state then rejects them ('RNG state must be a torch.ByteTensor'). This
    puts only those tensors back on the CPU after loading. Model, optimizer, sampler and curriculum state are
    untouched, and src/ is not modified."""
    original = torch.load

    def load(*args, **kwargs):
        state = original(*args, **kwargs)
        if isinstance(state, dict) and isinstance(state.get("rng"), dict):
            state["rng"]["torch"] = state["rng"]["torch"].cpu()
            state["rng"]["cuda"] = [t.cpu() for t in state["rng"]["cuda"]]
        return state
    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def preserved_paths(args, v1_manifest):
    paths = [V1_ROOT/"exploratory-manifest.json", Path(args.tokenizer), Path(args.tokenizer).with_suffix(".meta.json"),
             Path(args.graph), Path(args.base_protocol), Path(args.protocol),
             *(Path(args.data)/f for f in ("train.jsonl", "dev_a.jsonl", "dev_b.jsonl", "test.jsonl", "manifest.json"))]
    paths += [V1_ROOT/r["run_id"]/name for r in v1_manifest["runs"] for name in V1_FILES]
    return {str(p): file_hash(p) for p in paths}


def protocol_difference(args):
    base, new = Protocol.load(args.base_protocol), Protocol.load(args.protocol)
    diff = {k: [getattr(base, k), getattr(new, k)] for k in base.__dataclass_fields__ if getattr(base, k) != getattr(new, k)}
    if set(diff) != {"version", "mono_cap"} or diff["mono_cap"] != [200000, 900000]:
        raise ValueError(f"Exploratory protocol must differ from v1 only in version and mono_cap: {diff}")
    return base, new, diff


def plan(args):
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(f"{root} exists; use a new exploratory version")
    v1 = json.loads((V1_ROOT/"exploratory-manifest.json").read_text(encoding="utf-8"))
    if v1["code_hash"] != code_hash():
        raise ValueError("Code changed since exploratory v1; checkpoints would not be comparable")
    base, protocol, diff = protocol_difference(args)
    if base.hash != v1["protocol_hash"]:
        raise ValueError("Base protocol differs from the exploratory-v1 protocol")
    graph = Graph.load(args.graph)
    dataset_hash = json.loads((Path(args.data)/"manifest.json").read_text(encoding="utf-8"))["dataset_hash"]
    if (graph.hash, dataset_hash, file_hash(args.tokenizer)) != (v1["graph_hash"], v1["dataset_hash"], v1["tokenizer_hash"]):
        raise ValueError("Artifacts differ from exploratory v1")
    root.mkdir(parents=True)
    requests = [dict(run_id=run_id(lang), category="reserve", hours=RUN_HOURS) for lang in LANGUAGES]
    requests.append(dict(run_id=EVAL_ID, category="reserve", hours=EVAL_HOURS))
    Ledger(args.ledger).reserve_bundle(requests)
    manifest = dict(**LABEL, created_at_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, microbatch=args.microbatch,
                    protocol_path=args.protocol, protocol_hash=protocol.hash, training_hash=protocol.training_hash,
                    base_protocol_path=args.base_protocol, base_protocol_hash=base.hash, base_training_hash=base.training_hash,
                    protocol_difference_from_v1=diff, graph_hash=graph.hash, code_hash=code_hash(),
                    tokenizer_hash=file_hash(args.tokenizer), dataset_hash=dataset_hash, v1_manifest=str(V1_ROOT/"exploratory-manifest.json"),
                    stages=[dict(label=l, updates=u, seen=u*BATCH) for l, u in STAGES],
                    ledger=args.ledger, budget_category="reserve", reserved_hours=sum(r["hours"] for r in requests),
                    evaluation=dict(reservation_id=EVAL_ID, reserved_hours=EVAL_HOURS, status="reserved"),
                    preserved_v1_sha256=preserved_paths(args, v1),
                    runs=[dict(run_id=run_id(lang), mode="mono", order=[lang], reserved_hours=RUN_HOURS, status="reserved") for lang in LANGUAGES])
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
    for label, updates in (STAGES if not args.dry_run else [("a", 2), ("b", 4)]):
        if label in preserved:  # preserved by an earlier attempt: verify it and continue from it
            s = preserved[label]
            if s["updates"] != updates or file_hash(s["checkpoint"]) != s["checkpoint_hash"]:
                raise RuntimeError(f"{rid}: preserved stage {label} does not match the plan")
            resume = s["checkpoint"]
            continue
        if resume is not None and file_hash(out/"latest.pt") != file_hash(resume):
            raise RuntimeError(f"{rid}: run directory is not at the preserved stage")
        begin = time.perf_counter()
        with rng_state_on_cpu():
            summary = run(protocol, args.graph, args.data, args.tokenizer, out, "mono", (lang,), SEED, cohort="smoke",
                          device=args.device, backend=args.backend, microbatch=args.microbatch, smoke=True, run_id=rid,
                          resume=resume, max_updates=updates)
        record = dict(label=label, target_updates=updates, attempt=args.attempt, stop_reason=summary["stop_reason"],
                      seen=summary["seen"], updates=summary["counters"]["updates"], stage_seconds=time.perf_counter()-begin,
                      elapsed_seconds=summary["elapsed_seconds"])
        if summary["stop_reason"] != "debug_interruption":  # mastered or capped: the loop already took its terminal path
            stages.append(record)
            write_json(out/"stages.json", stages)
            break
        checkpoint = out/f"checkpoint-{label}.pt"
        shutil.copy2(out/"latest.pt", checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (file_hash(checkpoint) != file_hash(out/"latest.pt") or state["counters"]["updates"] != updates
                or state["curriculum"]["seen"] != summary["seen"] or (not args.dry_run and summary["seen"] != updates*BATCH)):
            raise RuntimeError(f"{rid}: preserved checkpoint does not match stage {label}")
        record.update(checkpoint=str(checkpoint), checkpoint_hash=file_hash(checkpoint),
                      language_exposures=state["sampler"]["exposures"], unique_training_items=len(state["sampler"]["unique"]))
        stages.append(record)
        write_json(out/"stages.json", stages)
        resume = str(checkpoint)
    return stages


def gpu_processes(devices):
    found = {}
    for device in devices:
        index = torch.device(device).index or 0
        out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
        found[device] = [line.strip() for line in out.splitlines() if line.strip()]
    return found


def last_line(path):
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines() if Path(path).exists() else []
    return lines[-1] if lines else None


def check_open_reservations(args, allowed):
    data = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    other = [k for k, r in data["runs"].items() if r["status"] == "reserved" and k not in allowed]
    if other:
        raise RuntimeError(f"Unrelated open ledger reservations: {other}")


def execute(args):
    root = Path(args.root)
    manifest_path = root/"exploratory-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["code_hash"] != code_hash():
        raise ValueError("Code changed after the exploratory plan was written")
    v1 = json.loads((V1_ROOT/"exploratory-manifest.json").read_text(encoding="utf-8"))
    if preserved_paths(args, v1) != manifest["preserved_v1_sha256"]:
        raise RuntimeError("Preserved v1 files changed since planning")
    active = gpu_processes(args.devices)
    if any(active.values()):
        raise RuntimeError(f"Other compute processes are active: {active}")
    ledger = Ledger(args.ledger)
    for r in manifest["runs"]:  # first attempts made before attempt records existed are migrated from run-level fields
        if "attempts" not in r:
            r["attempts"] = [] if r["status"] == "reserved" else [dict(
                reservation_id=r["run_id"], reserved_hours=r["reserved_hours"], device=r.get("device"), wall_hours=r.get("wall_hours"),
                exit_code=r.get("exit_code"), status=r["status"], interrupted=r.get("interrupted"),
                stages_preserved_after=[s["label"] for s in r.get("stages", []) if "checkpoint_hash" in s],
                console_log=str(root/f"{r['run_id']}.console.log"), error=last_line(root/f"{r['run_id']}.console.log"))]
    todo = [r for r in manifest["runs"] if r["status"] != "completed"]
    if args.retry:
        if not todo:
            raise ValueError("Every run is complete; nothing to retry")
        requests = []
        for r in todo:
            if r["status"] == "reserved":
                raise ValueError(f"{r['run_id']} has not been attempted; use --execute")
            r["pending_reservation"] = f"{r['run_id']}-r{len(r['attempts'])+1}"
            requests.append(dict(run_id=r["pending_reservation"], category="reserve", hours=args.retry_hours))
        check_open_reservations(args, {EVAL_ID})
        ledger.reserve_bundle(requests)
        manifest["retry_reserved_hours"] = manifest.get("retry_reserved_hours", 0)+sum(q["hours"] for q in requests)
    else:
        if any(r["status"] != "reserved" for r in todo) or len(todo) != len(manifest["runs"]):
            raise ValueError("First attempts already made; use --retry for unfinished runs")
        for r in todo:
            r["pending_reservation"] = r["run_id"]
        check_open_reservations(args, {r["run_id"] for r in manifest["runs"]} | {EVAL_ID})
    manifest.setdefault("executions", []).append(dict(started_at_utc=datetime.now(timezone.utc).isoformat(), devices=args.devices,
                                                     retry=bool(args.retry), other_compute_processes_at_start=active,
                                                     reservations=[r["pending_reservation"] for r in todo]))
    write_json(manifest_path, manifest)
    pending, active_jobs = list(todo), {}
    while pending or active_jobs:
        for device in args.devices:
            if device in active_jobs or not pending:
                continue
            r = pending.pop(0)
            reservation = ledger.reservation(r["pending_reservation"])
            log = (root/f"{r['pending_reservation']}.console.log").open("w", encoding="utf-8")
            cmd = [sys.executable, __file__, "--single", r["order"][0], "--device", device, "--root", str(root),
                   "--ledger", args.ledger, "--protocol", args.protocol, "--graph", args.graph, "--data", args.data,
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
            ledger.charge(r["pending_reservation"], hours, status)
            reservation_id = r.pop("pending_reservation")
            r["attempts"].append(dict(reservation_id=reservation_id, reserved_hours=job["reserved"], device=device,
                                      wall_hours=hours, exit_code=code, status=status,
                                      interrupted="reservation_wall_clock_exceeded" if interrupted else None,
                                      stages_preserved_after=[s["label"] for s in stages if "checkpoint_hash" in s],
                                      console_log=str(root/f"{reservation_id}.console.log"),
                                      error=None if code == 0 else last_line(root/f"{reservation_id}.console.log")))
            r.update(status=status, device=device, wall_hours=sum(a["wall_hours"] for a in r["attempts"]), stages=stages,
                     stop_reason=summary.get("stop_reason"), seen=summary.get("seen"),
                     updates=(summary.get("counters") or {}).get("updates"), clocks_seconds=summary.get("clocks_seconds"),
                     summary_gpu_hours=summary.get("gpu_hours"), trainable_parameters=summary.get("trainable_parameters"),
                     independent_test_written=(out/"independent-test.json").exists())
            write_json(manifest_path, manifest)
        time.sleep(1)
    manifest["executions"][-1]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["actual_training_gpu_hours"] = sum(r.get("wall_hours", 0) for r in manifest["runs"])
    manifest["preserved_v1_unchanged_after_execution"] = preserved_paths(args, v1) == manifest["preserved_v1_sha256"]
    write_json(manifest_path, manifest)
    return manifest


def training_sample(data, tokenizer):
    """Fixed, language-aligned sample: the same (meaning_id, label) keys per task in every language."""
    rows = load_rows(data, "train")
    keys = {task: sorted({(r["meaning_id"], r["label"]) for r in rows if r["language"] == "en" and r["task"] == task}) for task in TASKS}
    chosen = {task: set(random.Random(SEED*10+i).sample(keys[task], SAMPLE_PER_TASK)) for i, task in enumerate(TASKS)}
    sample = {lang: [] for lang in LANGUAGES}
    for row in rows:
        if (row["meaning_id"], row["label"]) in chosen[row["task"]]:
            sample[row["language"]].append((row, encode_pair(tokenizer, row)))
    for lang in LANGUAGES:
        counts = {task: sum(r["task"] == task for r, _ in sample[lang]) for task in TASKS}
        if any(c != SAMPLE_PER_TASK for c in counts.values()):
            raise ValueError(f"Training sample is not {SAMPLE_PER_TASK} per task for {lang}: {counts}")
        sample[lang].sort(key=lambda x: (x[0]["meaning_id"], x[0]["label"]))
    aligned = len({tuple((r["meaning_id"], r["label"]) for r, _ in sample[lang]) for lang in LANGUAGES}) == 1
    ids = sorted(r["id"] for lang in LANGUAGES for r, _ in sample[lang])
    meta = dict(per_task=SAMPLE_PER_TASK, sample_seed_base=SEED*10, aligned_across_languages=aligned, items=len(ids),
                ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                label_positive={lang: sum(r["label"] for r, _ in sample[lang]) for lang in LANGUAGES})
    return sample, meta


def score_checkpoint(path, graph, tokenizer, protocol_embed, protocol_microsteps, sets, device, microbatch):
    ck = torch.load(path, map_location=device, weights_only=True)
    meta = ck["metadata"]
    model = FlyClassifier(graph, tokenizer.get_vocab_size(), protocol_embed, protocol_microsteps, meta["seed"], "cuda").to(device)
    model.load_state_dict(ck["model"])
    languages = tuple(meta["order"]) if meta["mode"] == "mono" else LANGUAGES
    pad_id = tokenizer.token_to_id("[PAD]")
    cells = []
    with immutable_evaluation(model):
        for set_name, items_by_lang in sets:
            for lang in languages:
                for task, s in _frame.score(model, items_by_lang[lang], pad_id, microbatch, device).items():
                    cells.append(dict(set=set_name, language=lang, task=task, **s, accuracy=s["correct"]/s["items"],
                                      wilson95=_frame.wilson(s["correct"], s["items"])))
    sync(device)
    info = dict(run_id=meta["run_id"], mode=meta["mode"], order=meta["order"], seed=meta["seed"], seen=ck["curriculum"]["seen"],
                updates=ck["counters"]["updates"], language_exposures=ck["sampler"]["exposures"], checkpoint_hash=file_hash(path),
                checkpoint_path=str(path), languages=list(languages), cells=cells)
    del model, ck
    torch.cuda.empty_cache()
    return info


def loss_trace_check(v2_dir, v1_dir, updates):
    """The first 781 updates of a 900k-cap run repeat the 200k-cap run exactly if the kernels are deterministic."""
    def losses(path):
        return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    a = [e for e in losses(v2_dir/"events.jsonl") if e["kind"] == "train"][:updates]
    b = [e for e in losses(v1_dir/"events.jsonl") if e["kind"] == "train"][:updates]
    if len(a) != updates or len(b) != updates:
        return dict(compared_updates=min(len(a), len(b)), complete=False)
    diffs = [abs(x["loss"]-y["loss"]) for x, y in zip(a, b)]
    return dict(compared_updates=updates, complete=True, identical_losses=all(d == 0 for d in diffs), max_abs_loss_difference=max(diffs),
                identical_examples=all(x["examples"] == y["examples"] == BATCH for x, y in zip(a, b)),
                max_abs_grad_norm_difference=max(abs(x["grad_norm"]-y["grad_norm"]) for x, y in zip(a, b)))


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
    v1 = json.loads((V1_ROOT/"exploratory-manifest.json").read_text(encoding="utf-8"))
    active = gpu_processes([args.device])
    if any(active.values()):
        raise RuntimeError(f"Other compute processes are active: {active}")
    check_open_reservations(args, {EVAL_ID})
    started, status = time.perf_counter(), "technical_failure"
    report = dict(kind="exploratory_v2_mono_exposure_report", **LABEL, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  manifest=str(manifest_path), code_hash=code_hash(), script_hash=file_hash(__file__),
                  frame_script_hash=file_hash(_spec.origin), protocol=dict(path=manifest["protocol_path"], hash=manifest["protocol_hash"],
                  training_hash=manifest["training_hash"], difference_from_v1=manifest["protocol_difference_from_v1"]),
                  device=args.device, microbatch=args.microbatch, parameter_updates_during_evaluation=0, test_split_used=False,
                  runs=[{k: v for k, v in r.items() if k != "clocks_seconds"} for r in manifest["runs"]],
                  training_gpu_hours=manifest.get("actual_training_gpu_hours"), evaluations=[])
    try:
        protocol = Protocol.load(manifest["protocol_path"])
        graph = Graph.load(args.graph)
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        sample, report["training_sample"] = training_sample(args.data, tokenizer)
        views, report["dev_input_checks"], _ = _frame.build_views(args.data, tokenizer)
        sets = [("train_sample", sample)]
        for split in ("dev_a", "dev_b"):
            for view in ("assertion", "original"):
                sets.append((f"{split}_{view}", {lang: views[(split, view, lang)] for lang in LANGUAGES}))
        report["environment"] = environment()
        targets = []
        for r in manifest["runs"]:
            for s in r.get("stages", []):
                if "checkpoint_hash" in s:
                    targets.append(dict(source="v2", stage=s["label"], path=Path(s["checkpoint"]), expected_hash=s["checkpoint_hash"]))
        for r in v1["runs"]:
            if r["mode"] in ("mono", "mixed"):
                path = V1_ROOT/r["run_id"]/"primary.pt"
                targets.append(dict(source="v1", stage="final", path=path,
                                    expected_hash=json.loads((path.parent/"independent-test.json").read_text(encoding="utf-8"))["checkpoint_hash"]))
        for t in targets:
            if file_hash(t["path"]) != t["expected_hash"]:
                raise ValueError(f"{t['path']}: checkpoint hash changed")
            begin = time.perf_counter()
            info = score_checkpoint(t["path"], graph, tokenizer, protocol.embed_dim, protocol.microsteps, sets, args.device, args.microbatch)
            report["evaluations"].append(dict(source=t["source"], stage=t["stage"], seconds=time.perf_counter()-begin, **info))
        report["loss_trace_check"] = {}
        for r in manifest["runs"]:
            lang = r["order"][0]
            report["loss_trace_check"][lang] = loss_trace_check(root/r["run_id"], V1_ROOT/f"explore-v1-mono-{lang}", STAGES[0][1])
        report["scheduled_panels"] = {}
        for r in report["runs"]:
            events = [json.loads(l) for l in (root/r["run_id"]/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            report["scheduled_panels"][r["run_id"]] = [dict(seen=e["seen"], panel=e["panel"], scores=e["scores"]) for e in events if e["kind"] == "scheduled"]
            train_events = [e for e in events if e["kind"] == "train"]
            r["loss_first_last_min"] = [train_events[0]["loss"], train_events[-1]["loss"], min(e["loss"] for e in train_events)] if train_events else None
            r["mastered"] = any(e.get("first_task") for e in events if e["kind"] == "scheduled")
        report["preserved_v1_unchanged"] = preserved_paths(args, v1) == manifest["preserved_v1_sha256"]
        if not report["preserved_v1_unchanged"]:
            raise RuntimeError("Preserved v1 files changed")
        status = "completed"
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        hours = (time.perf_counter()-started)/3600
        Ledger(args.ledger).charge(EVAL_ID, hours, status)
        manifest["evaluation"].update(status=status, charged_hours=hours, output=args.output)
        manifest["actual_total_gpu_hours"] = manifest.get("actual_training_gpu_hours", 0)+hours
        write_json(manifest_path, manifest)
        report["ledger"] = dict(path=args.ledger, evaluation_reservation_id=EVAL_ID, evaluation_reserved_hours=EVAL_HOURS,
                                evaluation_charged_hours=hours, evaluation_status=status,
                                training_attempts={r["run_id"]: r.get("attempts", []) for r in manifest["runs"]},
                                plan_reserved_hours=manifest["reserved_hours"], retry_reserved_hours=manifest.get("retry_reserved_hours", 0),
                                training_charged_hours=manifest.get("actual_training_gpu_hours"), total_charged_hours=manifest["actual_total_gpu_hours"])
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true"); p.add_argument("--execute", action="store_true")
    p.add_argument("--single", choices=LANGUAGES); p.add_argument("--evaluate", action="store_true")
    p.add_argument("--root", default="runs/exploratory-v2"); p.add_argument("--ledger", default="runs/gpu-ledger.json")
    p.add_argument("--protocol", default="configs/protocol-v4-ai-mono900k.json"); p.add_argument("--base-protocol", default="configs/protocol-v4-ai.json")
    p.add_argument("--graph", default="artifacts/graphs/real.npz"); p.add_argument("--data", default="data/draft-v4.3")
    p.add_argument("--tokenizer", default="artifacts/tokenizer-v4.3.json")
    p.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"]); p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", default="cuda"); p.add_argument("--microbatch", type=int, default=256)
    p.add_argument("--output", default="reports/exploratory-v2-report.json")
    p.add_argument("--dry-run", action="store_true", help="internal: two tiny stages for --single on any device")
    p.add_argument("--retry", action="store_true", help="with --execute: new reservations for runs that did not complete")
    p.add_argument("--retry-hours", type=float, default=.25); p.add_argument("--attempt", default=None)
    a = p.parse_args()
    if a.plan:
        m = plan(a); print(json.dumps(dict(status="planned", runs=[r["run_id"] for r in m["runs"]], reserved_hours=m["reserved_hours"]), indent=1))
    elif a.execute:
        m = execute(a); print(json.dumps(dict(status="executed", actual_training_gpu_hours=m["actual_training_gpu_hours"],
                                              runs={r["run_id"]: (r.get("status"), r.get("stop_reason"), r.get("seen"), r.get("updates")) for r in m["runs"]}), indent=1))
    elif a.single:
        print(json.dumps(single(a), indent=1))
    elif a.evaluate:
        r = evaluate(a); print(json.dumps(r["ledger"], indent=1))
    else:
        p.error("choose --plan, --execute, --single or --evaluate")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time
import inspect
from functools import wraps
import torch
from tokenizers import Tokenizer

from .budget import Ledger
from .data import require_confirmatory_test
from .gates import code_hash, environment, verify_main_launch
from .graph import Graph
from .model import FlyClassifier, train_batch
from .protocol import LANGUAGES, TASKS, Protocol, file_hash, write_json
from .runtime import Corpus, AlignedSampler, evaluate, append_event, rng_state, restore_rng, save_checkpoint, sync
from .schedule import Curriculum

REVIEW_POINTS = (0, 100000, 200000, 300000)


def account_attempt(function):
    """Charge each reserved attempt, including setup failures; retries use new IDs."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        rid = a['run_id'] or f"{a['cohort']}-real-{a['seed']}-{'seq' if a['mode'] == 'sequential' else a['mode']}" + (f"-{'-'.join(a['order'])}" if a['mode'] != 'mixed' else '')
        reservation_id = a['reservation_id'] or rid
        summary_path = Path(a['output'])/'summary.json'
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if a['resume'] and previous.get('stop_reason') in ('mastered','administrative_cap'):
            raise ValueError('Run is already terminal; its independent test and checkpoint are immutable')
        ledger = Ledger(a['ledger_path']) if a['ledger_path'] and not a['smoke'] else None
        if ledger:
            entry = ledger.reservation(reservation_id)
            if a['resume'] and reservation_id == rid:
                raise ValueError('Reserve a separate retry ID; preserve the original attempt accounting')
        start, status = time.perf_counter(), 'technical_failure'
        try:
            result = function(*args, **kwargs)
            status = ('completed' if result['stop_reason'] in ('mastered','administrative_cap')
                      else 'budget_interruption' if result['stop_reason'] == 'budget_interruption' else 'technical_failure')
            return result
        finally:
            hours = (time.perf_counter()-start)/3600 if str(a['device']).startswith('cuda') else 0.
            if ledger:
                ledger.charge(reservation_id, hours, status)
                write_json(Path(a['output'])/f'attempt-{reservation_id}.json',
                           dict(reservation_id=reservation_id,run_id=rid,status=status,gpu_hours=hours))
                if summary_path.exists():
                    saved = json.loads(summary_path.read_text())
                    saved.update(gpu_hours=previous.get('gpu_hours',0.)+hours,
                                 last_attempt_gpu_hours=hours, last_reservation_id=reservation_id)
                    write_json(summary_path, saved)
                    if 'result' in locals(): result.update(saved)
    return wrapped


@account_attempt
def run(protocol, graph_path, dataset, tokenizer_path, output, mode, order, seed,
        cohort="pilot", device="cuda:0", backend="cuda", microbatch=32,
        launch=None, ledger_path=None, run_id=None, resume=None, smoke=False,
        max_updates=None, review_samples=False, reservation_id=None,
        independent_test=True, auxiliary_panels=True):
    """One bounded run. Technical/budget interruption never becomes a censored success.
    Smoke (exploratory) runs may skip the terminal independent test and the per-panel auxiliary
    evaluation; study cohorts always perform both."""
    protocol.validate(production=not smoke)
    if cohort in ('main','pilot'): protocol.validate_primary_model()
    if microbatch < 1 or microbatch > protocol.effective_batch:
        raise ValueError("Invalid physical microbatch")
    graph = Graph.load(graph_path)
    output = Path(output)
    if cohort not in ("main", "pilot", "smoke", "auxiliary"):
        raise ValueError("Invalid cohort")
    if smoke != (cohort == "smoke"):
        raise ValueError("Smoke experiments require an explicit smoke cohort")
    if not smoke and not (independent_test and auxiliary_panels):
        raise ValueError("Study cohorts always run the independent test and the auxiliary panels")
    data_meta = json.loads((Path(dataset)/"manifest.json").read_text(encoding="utf-8"))
    if not smoke:
        require_confirmatory_test(data_meta)
    if not smoke and (graph.n != 5000 or device == "cpu" or backend != "cuda"):
        raise ValueError("Production/pilot runs require cb5k and the validated CUDA backend")
    if cohort == "pilot" and seed in protocol.main_seeds:
        raise ValueError("Pilot seed overlaps main seeds")
    if max_updates is not None and not smoke:
        raise ValueError("A debugging update cap is not allowed in study runs")
    run_id = run_id or f"{cohort}-real-{seed}-{'seq' if mode == 'sequential' else mode}" + (f"-{'-'.join(order)}" if mode != "mixed" else "")
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Use an empty run directory or explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    token_meta_path = Path(tokenizer_path).with_suffix(".meta.json")
    token_meta = json.loads(token_meta_path.read_text(encoding="utf-8"))
    if token_meta["dataset_hash"] != data_meta["dataset_hash"] or token_meta["tokenizer_hash"] != file_hash(tokenizer_path):
        raise ValueError("Tokenizer provenance mismatch")
    reservation = None
    if cohort == "main":
        if not launch or not ledger_path:
            raise ValueError("Main runs require frozen gates and a budget reservation")
        frozen, specified = verify_main_launch(launch, protocol, graph, dataset, tokenizer_path, run_id)
        if specified["seed"] != seed or specified["mode"] != mode or specified["order"] != list(order):
            raise ValueError("Run arguments differ from the frozen design")
        reservation = Ledger(ledger_path).reservation(reservation_id or run_id)
    elif cohort == 'auxiliary':
        from .workflow import verified_evidence, read
        if not launch or not ledger_path:
            raise ValueError('Auxiliary runs require a complete reserved bundle')
        manifest = read(launch)
        if manifest.get('status') != 'auxiliary_reserved' or manifest['code_hash'] != code_hash():
            raise ValueError('Missing current auxiliary bundle')
        verified_evidence(manifest)
        selected = [r for r in manifest['runs'] if r['run_id'] == run_id]
        if len(selected) != 1:
            raise ValueError('Auxiliary run outside the prespecified bundle')
        selected = selected[0]
        expected = dict(protocol_hash=protocol.hash, graph_hash=graph.hash,
                        tokenizer_hash=file_hash(tokenizer_path), dataset_hash=data_meta['dataset_hash'],
                        seed=seed, mode=mode, order=list(order))
        if any(selected.get(k) != v for k,v in expected.items()):
            raise ValueError('Auxiliary arguments differ from the reserved paired design')
        reservation = Ledger(ledger_path).reservation(reservation_id or run_id)
    elif not smoke:
        if not launch or not ledger_path:
            raise ValueError("Pilots need a readiness manifest (G0 + certified data review) and a reservation")
        readiness = json.loads(Path(launch).read_text(encoding="utf-8"))
        if readiness.get("status") != "pilot_ready" or readiness.get("dataset_hash") != data_meta["dataset_hash"] or readiness.get("graph_hash") != graph.hash or readiness.get("code_hash") != code_hash():
            raise ValueError("Pilot readiness has not been established for these artifacts")
        if readiness.get("tokenizer_hash") != file_hash(tokenizer_path):
            raise ValueError("Pilot tokenizer differs from the reviewed artifacts")
        for path, sha in readiness["evidence_files"].items():
            if file_hash(path) != sha:
                raise ValueError("Readiness evidence changed")
        reservation = Ledger(ledger_path).reservation(reservation_id or run_id)
    metadata = dict(run_id=run_id, cohort=cohort, mode=mode, order=list(order), seed=seed,
                    protocol_hash=protocol.hash, training_hash=protocol.training_hash, protocol=asdict(protocol), graph_hash=graph.hash,
                    dataset_hash=data_meta["dataset_hash"], tokenizer_hash=file_hash(tokenizer_path),
                    code_hash=code_hash(), synthetic=graph.provenance.get("condition") == "synthetic",
                    backend=backend, device=device, graph_condition=graph.provenance.get("condition"),
                    physical_microbatch=microbatch, review_samples=review_samples,
                    independent_test=independent_test, auxiliary_panels=auxiliary_panels,
                    environment=environment(), paths=dict(graph=str(Path(graph_path).resolve()),
                        dataset=str(Path(dataset).resolve()), tokenizer=str(Path(tokenizer_path).resolve())))
    review_record = json.loads(Path(launch).read_text(encoding="utf-8")) if launch and not smoke else {}
    metadata.update({k: review_record.get(k) for k in ("review_mode", "human_reviewed", "review_limitations", "protocol_amendment")})
    started = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if not smoke and tokenizer.get_vocab_size() != protocol.vocab_size:
        raise ValueError("Actual vocabulary differs from the protocol")
    corpus = Corpus(dataset, tokenizer)
    sampler = AlignedSampler(corpus, seed)
    curriculum = Curriculum(protocol, mode, tuple(order))
    model = FlyClassifier(graph, tokenizer.get_vocab_size(), protocol.embed_dim,
                          protocol.microsteps, seed, backend).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(protocol), weight_decay=protocol.weight_decay)
    clocks = dict(train=0., eval=0., auxiliary=0., checkpoint=0., test=0., review_eval=0.)
    counters = dict(updates=0, tokens=0, padded_tokens=0, recurrent_updates=0, evaluation_examples=0)
    completed_reviews = []
    elapsed_before = 0.
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=True)
        for key in ("run_id", "protocol_hash", "graph_hash", "dataset_hash", "tokenizer_hash", "code_hash"):
            if ck["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        sampler.load_state_dict(ck["sampler"]); restore_rng(ck["rng"])
        curriculum = Curriculum.restore(protocol, ck["curriculum"])
        clocks, counters = ck["clocks"], ck["counters"]
        clocks.setdefault("auxiliary", 0.)
        completed_reviews = ck["completed_reviews"]
        elapsed_before = ck["elapsed_seconds"]
        # Crash recovery truncates events after the latest complete checkpoint.
        log = output / "events.jsonl"
        if log.exists():
            with log.open("r+b") as f:
                f.truncate(ck["log_offset"])
    write_json(output / "metadata.json", metadata)
    log = output / "events.jsonl"
    def checkpoint(name="latest.pt"):
        sync(device)
        begin = time.perf_counter()
        state = dict(metadata=metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     sampler=sampler.state_dict(), rng=rng_state(), curriculum=curriculum.checkpoint(),
                     clocks=clocks, counters=counters, completed_reviews=completed_reviews,
                     elapsed_seconds=elapsed_before+time.perf_counter()-started,
                     log_offset=log.stat().st_size if log.exists() else 0)
        save_checkpoint(output / name, state)
        clocks["checkpoint"] += time.perf_counter()-begin
    def diagnostic_review():
        if mode != "sequential" or curriculum.review_start is None:
            return
        r = curriculum.seen-curriculum.review_start
        if r in REVIEW_POINTS and r not in completed_reviews:
            completed_reviews.append(r)
            if review_samples:
                # Store states only. Full fixed-review evaluations are performed
                # later under the separately reserved auxiliary budget.
                append_event(log, dict(kind="review_checkpoint", seen=curriculum.seen, review_exposures=r))
                checkpoint(f"review-{r}.pt")
    status = None
    try:
        while not curriculum.stop_reason:
            if reservation and (time.perf_counter()-started)/3600 >= reservation["reserved"]:
                status = "budget_interruption"
                break
            if curriculum.seen == curriculum.next_eval:
                panel = curriculum.next_panel
                sync(device); begin = time.perf_counter()
                scores, strata = evaluate(model, corpus, panel, curriculum.languages, microbatch, optimizer, sampler)
                sync(device); clocks["eval"] += time.perf_counter()-begin
                counters["evaluation_examples"] += sum(v[1] for v in scores.values())
                if auxiliary_panels and corpus.has_auxiliary(panel):
                    # Outer-frame transfer on the same items; recorded only, never a mastery observation.
                    sync(device); begin = time.perf_counter()
                    aux_scores, aux_strata = evaluate(model, corpus, panel, curriculum.languages, microbatch, optimizer, sampler, view="auxiliary")
                    sync(device); clocks["auxiliary"] += time.perf_counter()-begin
                    counters["evaluation_examples"] += sum(v[1] for v in aux_scores.values())
                    append_event(log, dict(kind="auxiliary", seen=curriculum.seen, panel=panel, scores=aux_scores, strata=aux_strata))
                curriculum.evaluate(scores, panel)
                append_event(log, dict(kind="scheduled", seen=curriculum.seen, panel=panel, scores=scores,
                                      strata=strata, stage=curriculum.stage, confirmed=curriculum.confirmed,
                                      first_task=curriculum.first_task, first_language=curriculum.first_language))
                diagnostic_review()
                checkpoint()
                continue
            curriculum.advance()
            diagnostic_review()
            if max_updates is not None and counters["updates"] >= max_updates:
                status = "debug_interruption"
                break
            count = curriculum.batch_size()
            # These batch boundaries apply to ALL sequential runs, regardless of
            # whether auxiliary evaluation is funded. The optimizer path is paired.
            if curriculum.review_start is not None and mode == "sequential":
                r = curriculum.seen-curriculum.review_start
                pending = [point-r for point in REVIEW_POINTS if point > r]
                if pending:
                    count = min(count, min(pending))
            if count <= 0:
                raise RuntimeError("Curriculum made no progress")
            examples = sampler.sample(count, curriculum.current_language)
            sync(device); begin = time.perf_counter()
            result = train_batch(model, optimizer, examples, corpus.pad_id, microbatch, protocol)
            sync(device); clocks["train"] += time.perf_counter()-begin
            curriculum.train_exposures(count)
            counters["updates"] += 1
            for key in ("tokens", "padded_tokens", "recurrent_updates"):
                counters[key] += result[key]
            append_event(log, dict(kind="train", seen=curriculum.seen, language=curriculum.current_language,
                                  stage=curriculum.stage, examples=count, **result))
        status = status or curriculum.stop_reason
        checkpoint("primary.pt" if status in ("mastered", "administrative_cap") else "latest.pt")
        tests, test_strata = None, None
        test_path = output / "independent-test.json"
        if status in ("mastered", "administrative_cap") and independent_test:
            if test_path.exists():
                saved = json.loads(test_path.read_text())
                if saved["checkpoint_hash"] != file_hash(output / "primary.pt"):
                    raise ValueError("Independent test already ran on a different terminal checkpoint")
                tests, test_strata = saved["scores"], saved["strata"]
            else:
                sync(device); begin = time.perf_counter()
                tests, test_strata = evaluate(model, corpus, "test", curriculum.languages, microbatch, optimizer, sampler)
                record = dict(scores=tests, strata=test_strata, checkpoint_hash=file_hash(output / "primary.pt"), used_for_training=False)
                if corpus.has_auxiliary("test"):
                    aux_tests, aux_strata = evaluate(model, corpus, "test", curriculum.languages, microbatch, optimizer, sampler, view="auxiliary")
                    record.update(auxiliary_scores=aux_tests, auxiliary_strata=aux_strata)
                sync(device); clocks["test"] += time.perf_counter()-begin
                write_json(test_path, record)
    except Exception as exc:
        status = "technical_failure"
        try:
            checkpoint()
        except Exception:
            pass
        write_json(output / "failure.json", dict(error=repr(exc), seen=curriculum.seen, status=status))
        raise
    finally:
        elapsed = elapsed_before+time.perf_counter()-started
        summary = {**metadata, "seen": curriculum.seen, "stop_reason": status,
                   "independent_test": "recorded" if tests is not None else ("skipped_exploratory" if not independent_test else None),
                   "first_global": curriculum.first_global, "first_task": curriculum.first_task,
                   "first_language": curriculum.first_language, "stages": curriculum.stages,
                   "review_start": curriculum.review_start, "clocks_seconds": clocks,
                   "elapsed_seconds": elapsed, "gpu_hours": elapsed/3600 if str(device).startswith("cuda") else 0.,
                   "cpu_hours": elapsed/3600 if device == "cpu" else 0.,
                   "unique_items": len(sampler.unique), "language_exposures": sampler.exposures,
                   "counters": counters, "cap": curriculum.cap,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        write_json(output / "summary.json", summary)
    return summary


def memorize(protocol, graph_path, dataset, tokenizer_path, output, device="cuda:0", backend="cuda", seed=9001, max_seconds=None):
    """G1, intentionally evaluated on its 128 training items; not generalization."""
    if device == "cpu" or backend != "cuda" or seed in protocol.main_seeds:
        raise ValueError("Official G1 needs the validated CUDA path and an independent seed")
    graph = Graph.load(graph_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    corpus = Corpus(dataset, tokenizer)
    examples, task_indices = [], {}
    for task in TASKS:
        selected = [(ids, row["label"]) for row, ids in corpus.split("train")["en"] if row["task"] == task][:32]
        if len(selected) != 32:
            raise ValueError("G1 requires 32 examples per task")
        task_indices[task] = (len(examples), len(examples)+32)
        examples.extend(selected)
    model = FlyClassifier(graph, tokenizer.get_vocab_size(), protocol.embed_dim, protocol.microsteps, seed, backend).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(protocol), weight_decay=protocol.weight_decay)
    from .model import collate
    from .runtime import immutable_evaluation
    history, passed = [], False
    start = time.perf_counter()
    for update in range(1, 2001):
        if max_seconds is not None and time.perf_counter()-start >= max_seconds:
            raise RuntimeError("G1 reservation exhausted; learnability was not established")
        train_batch(model, optimizer, examples, corpus.pad_id, 32, protocol)
        if update % 10:
            continue
        predictions = []
        with immutable_evaluation(model, optimizer):
            for offset in range(0, 128, 32):
                x, lengths, y = collate(examples[offset:offset+32], corpus.pad_id, device)
                predictions.extend((model(x, lengths).argmax(-1) == y).tolist())
        scores = {task: sum(predictions[a:b])/32 for task, (a,b) in task_indices.items()}
        history.append(dict(update=update, scores=scores, overall=sum(predictions)/128))
        if min(scores.values()) >= .95 and sum(predictions)/128 >= .98:
            passed = True
            break
    report = dict(gate="G1", status="passed" if passed else "failed", history=history,
                  seed=seed, graph_hash=graph.hash, tokenizer_hash=file_hash(tokenizer_path),
                  protocol_hash=protocol.hash, training_hash=protocol.training_hash, code_hash=code_hash(), gpu_hours=(time.perf_counter()-start)/3600)
    write_json(output, report)
    return report

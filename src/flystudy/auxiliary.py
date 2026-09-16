from __future__ import annotations

from pathlib import Path
import json
import time
import torch
from tokenizers import Tokenizer

from .budget import Ledger
from .gates import code_hash
from .graph import Graph
from .model import FlyClassifier, train_batch
from .protocol import LANGUAGES, ORDERS, Protocol, file_hash, write_json
from .runtime import Corpus, AlignedSampler, evaluate, restore_rng, append_event, sync


def fixed_review_bundle(run_root, output, target, ledger_path, hours_per_run, device="cuda:0", microbatch=32):
    if target not in (100000,300000) or not device.startswith("cuda"):
        raise ValueError("Fixed review uses a prespecified common target and CUDA")
    selected = {}
    for path in Path(run_root).glob("*/summary.json"):
        row = json.loads(path.read_text())
        if row["cohort"] == "main" and row["mode"] == "sequential" and row["seed"] in (1,2):
            key = (row["seed"],tuple(row["order"]))
            if key in selected: raise ValueError("Duplicate main run")
            if row["stop_reason"] not in ("mastered","administrative_cap") or row["code_hash"] != code_hash():
                raise ValueError("Primary run is incomplete or code changed")
            selected[key] = (path.parent,row)
    expected = {(s,o) for s in (1,2) for o in ORDERS}
    if set(selected) != expected:
        raise ValueError("Require both preselected seeds and all six orders; no outcome-based subsets")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    ledger = Ledger(ledger_path)
    requests = [dict(run_id=r["run_id"]+f"-review-{target}",category="review",hours=hours_per_run*1.25) for _,r in selected.values()]
    ledger.reserve_bundle(requests)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"bundle.json",dict(target=target,primary_seeds=[1,2],runs=[r["run_id"] for _,r in selected.values()],
                                         requests=requests,primary_endpoints_immutable=True))
    completed = []
    for folder,row in selected.values():
        rid = row["run_id"]+f"-review-{target}"
        reservation = ledger.reservation(rid)
        start,status=time.perf_counter(),"technical_failure"
        report = dict(run_id=rid,primary_run_id=row["run_id"],primary_endpoint=row["first_global"],points={})
        try:
            ck = torch.load(folder/"primary.pt",map_location=device,weights_only=True)
            protocol=Protocol(**{**row["protocol"],"main_seeds":tuple(row["protocol"]["main_seeds"]),"pilot_seeds":tuple(row["protocol"]["pilot_seeds"])})
            paths=row["paths"]
            graph=Graph.load(paths["graph"])
            tokenizer=Tokenizer.from_file(paths["tokenizer"])
            corpus=Corpus(paths["dataset"],tokenizer)
            model=FlyClassifier(graph,tokenizer.get_vocab_size(),protocol.embed_dim,protocol.microsteps,row["seed"],"cuda").to(device)
            optimizer=torch.optim.AdamW(model.parameter_groups(protocol),weight_decay=protocol.weight_decay)
            sampler=AlignedSampler(corpus,row["seed"])
            review_start = row['review_start'] if row['review_start'] is not None else row['seen']
            primary_review=row["seen"]-review_start
            for point in (p for p in (0,100000,200000,300000) if p <= target and p <= primary_review):
                saved=(ck if point == 0 and row['review_start'] is None else
                       torch.load(folder/f"review-{point}.pt",map_location=device,weights_only=True))
                model.load_state_dict(saved["model"])
                scores,_=evaluate(model,corpus,"dev_a",LANGUAGES,microbatch)
                report["points"][point]=scores
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            sampler.load_state_dict(ck["sampler"]); restore_rng(ck["rng"])
            exposure=primary_review
            pending=[p for p in (100000,200000,300000) if primary_review < p <= target]
            for point in pending:
                while exposure < point:
                    if (time.perf_counter()-start)/3600 >= reservation["reserved"]:
                        status="budget_interruption"
                        raise RuntimeError("Auxiliary reservation exhausted; primary results are unchanged")
                    total = review_start+exposure
                    # Preserve regular-grid optimizer batch boundaries after the
                    # primary endpoint, even when diagnostic evaluations are omitted.
                    to_grid = protocol.eval_interval-total % protocol.eval_interval
                    to_cap = protocol.total_cap-total if total < protocol.total_cap else to_grid
                    count=min(protocol.effective_batch,point-exposure,to_grid,to_cap)
                    train_batch(model,optimizer,sampler.sample(count),corpus.pad_id,microbatch,protocol)
                    exposure+=count
                scores,_=evaluate(model,corpus,"dev_a",LANGUAGES,microbatch,optimizer,sampler)
                report["points"][point]=scores
            status="completed"
        finally:
            sync(device)
            report.update(status=status,gpu_hours=(time.perf_counter()-start)/3600,charged_category="review")
            write_json(output/f"{rid}.json",report)
            ledger.charge(rid,report["gpu_hours"],status)
        completed.append(rid)
    return dict(status="completed",runs=completed,target=target)

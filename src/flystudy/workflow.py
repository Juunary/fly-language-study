"""Evidence-linked readiness, design selection, and launch manifests."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math

from .budget import Ledger, choose_design
from .data import audit
from .ai_review import AMENDMENTS
from .gates import code_hash
from .protocol import LANGUAGES, ORDERS, Protocol, file_hash, run_matrix, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_evidence(*reports):
    evidence = {}
    for report in reports:
        for path, sha in report.get('evidence_files', {}).items():
            if file_hash(path) != sha:
                raise ValueError(f'Stale evidence: {path}')
            evidence[path] = sha
    return evidence


def review_provenance(review, review_mode):
    expected_gate = {"claude_only": "ai_review", "human": "human_review"}.get(review_mode)
    if not expected_gate or review.get("gate") != expected_gate or review.get("review_mode") != review_mode:
        raise ValueError("Review mode does not match the requested certification")
    if review_mode == "claude_only" and (review.get("human_reviewed") is not False or
            review.get("protocol_amendment") not in AMENDMENTS or not review.get("ai_runs") or
            not review.get("limitations") or not review.get("evidence_files")):
        raise ValueError("Claude review provenance is incomplete")
    return verified_evidence(review)


def pilot_ready(g0_path, review_path, dataset, graph, tokenizer, output, review_mode="claude_only"):
    from .graph import Graph
    g0, review = read(g0_path), read(review_path)
    review_evidence = review_provenance(review, review_mode)
    checked = audit(dataset)
    baseline_path = Path(dataset)/"nuisance-baselines.json"
    baseline = read(baseline_path)
    baseline_meta_path = Path(dataset)/'nuisance-baselines.meta.json'
    baseline_meta = read(baseline_meta_path)
    if baseline_meta['report_hash'] != file_hash(baseline_path) or any(
            file_hash(Path(dataset)/f'{split}.jsonl') != sha for split,sha in baseline_meta['data_files'].items()):
        raise ValueError('Nuisance baseline belongs to another dataset/report')
    cue_path=Path(dataset)/"cue-baselines.json"
    cues=read(cue_path)
    cue_meta=read(Path(dataset)/"cue-baselines.meta.json")
    if any(file_hash(Path(dataset)/f"{split}.jsonl") != sha for split,sha in cue_meta["data_files"].items()):
        raise ValueError("Cue baseline belongs to another data version")
    gh = Graph.load(graph).hash
    if g0.get("status") != "passed" or g0.get("graph_hash") != gh or g0["environment"]["code_hash"] != code_hash():
        raise ValueError("Current G0 evidence is missing")
    if not checked["passed"] or review.get("status") != "passed" or review.get("dataset_hash") != checked["dataset_hash"]:
        raise ValueError("Current certified data review is missing")
    if any(r["audit_required"] for r in baseline.values()):
        raise ValueError("Unresolved nuisance-feature shortcut audit")
    if any(r["audit_required"] for r in cues.values()):
        raise ValueError("Unresolved hypothesis-only/cue coverage audit")
    meta = read(Path(tokenizer).with_suffix(".meta.json"))
    if meta["dataset_hash"] != checked["dataset_hash"] or meta["tokenizer_hash"] != file_hash(tokenizer):
        raise ValueError("Tokenizer is not derived from this frozen training split")
    evidence = [g0_path, review_path, baseline_path, baseline_meta_path, cue_path, Path(dataset)/"cue-baselines.meta.json", Path(dataset)/"audit.json"]
    result = dict(status="pilot_ready", dataset_hash=checked["dataset_hash"], graph_hash=gh,
                  tokenizer_hash=file_hash(tokenizer), code_hash=code_hash(),
                  evidence_files={str(Path(p).resolve()): file_hash(p) for p in evidence})
    result["evidence_files"].update(review_evidence)
    result.update(review_mode=review_mode, human_reviewed=review.get("human_reviewed"),
                  review_limitations=review.get("limitations", []),
                  protocol_amendment=review.get("protocol_amendment"))
    write_json(output, result)
    return result


def design_decision(simulation_path, pilot_path, cost_path, resolution_path, output, available_main_hours=448):
    """A conservative stress-envelope decision; never select on observed effects.

    Cost input contains per-run upper hours and remaining calendar capacity.
    All modeled CVs/families are retained; zero latent pairing correlation is used
    for planning. Small pilot censor bounds outside the simulated grid block entry.
    """
    pilot, costs, resolution = read(pilot_path), read(cost_path), read(resolution_path)
    nested_evidence = verified_evidence(pilot, costs, resolution)
    meta = read(Path(simulation_path).with_suffix(".meta.json"))
    if not meta.get("complete") or meta.get("quick") or meta.get("repetitions", 0) < 5000:
        raise ValueError("A completed full 5,000-repetition grid is required")
    if pilot.get("status") != "passed":
        raise ValueError("Learnability pilot evidence is incomplete")
    groups = pilot["groups"]
    seq = [g for k, g in groups.items() if k.startswith("sequential/")]
    upper = max(g["censor_upper_95"] for g in seq)
    grid = [json.loads(line) for line in Path(simulation_path).read_text().splitlines()]
    censor_grid = sorted({r["censoring"] for r in grid})
    compatible = [c for c in censor_grid if c >= upper]
    cv_bound=pilot.get("dispersion_uncertainty",{}).get("restricted_cv_upper")
    result = dict(status="blocked", reason=None, candidates=[], pilot_censor_upper=upper,
                  selection_uses_observed_effect=False, stress_envelope="all CVs and families; latent rho=0",
                  evidence_files={str(Path(p).resolve()): file_hash(p) for p in
                                  (simulation_path, pilot_path, cost_path, resolution_path)})
    result['evidence_files'].update(nested_evidence)
    result['evidence_files'][str(Path(simulation_path).with_suffix('.meta.json').resolve())] = file_hash(Path(simulation_path).with_suffix('.meta.json'))
    if not compatible:
        result["reason"] = "Pilot censoring uncertainty exceeds the simulated envelope; add independent pilot blocks or extend the grid."
    else:
        ceiling = compatible[0]
        if cv_bound is not None:
            result["stress_envelope"]="Pilot-informed inflated bootstrap dispersion bound, rounded outward to the DGM grid; all families; latent rho=0. Conditional power only."
        for n in (10, 15, 20):
            for interval in (5120, 10240, 20480):
                rows = [r for r in grid if r["n"] == n and r["interval"] == interval and
                        r["censoring"] <= ceiling and r["latent_rho"] == 0]
                if cv_bound is not None:
                    kept=[]
                    for family in ("lognormal","weibull"):
                        for censoring in sorted({r["censoring"] for r in rows}):
                            group=[r for r in rows if r["family"] == family and r["censoring"] == censoring]
                            above=[r["reference_restricted_cv"] for r in group if r["reference_restricted_cv"] >= cv_bound]
                            edge=min(above) if above else float("inf")
                            kept.extend(r for r in group if r["reference_restricted_cv"] <= edge+1e-9)
                    rows=kept
                power = [r["conservative_power_ci"][0] for r in rows if r["effect"] == .2]
                errors = [r["mc_ci"][1] for r in rows if r["effect"] == 0]
                measured = costs.get("intervals", {}).get(str(interval))
                res = resolution.get("intervals", {}).get(str(interval), {})
                if not measured or not power or not errors:
                    continue
                if set(measured["per_run_upper_hours"]) != {"mono-en", "mono-de", "mono-ko", "mixed"} | {"seq-"+"-".join(o) for o in ORDERS}:
                    raise ValueError("Cost evidence must cover every primary condition")
                hours = n*sum(measured["per_run_upper_hours"].values())
                result["candidates"].append(dict(n=n, eval_interval=interval,
                    measurement_passed=res.get("status") == "passed" and res.get("actual_sequence_pilot") is True,
                    power_lower=min(power), null_fwer_upper=max(errors), gpu_hours=hours,
                    available_main_hours=available_main_hours, calendar_hours=hours/2+costs["serial_overhead_hours"],
                    available_calendar_hours=costs["remaining_calendar_hours"],
                    per_run_upper_hours=measured["per_run_upper_hours"]))
        chosen = choose_design(result["candidates"])
        if chosen:
            result.update(status="passed", chosen=chosen)
        else:
            result["reason"] = "No candidate meets resolution, conservative power, false-positive, GPU and calendar limits."
    write_json(output, result)
    return result


def freeze(protocol, readiness_path, g1_path, pilot_path, design_path, output, ledger_path):
    protocol.validate().validate_primary_model()
    ready, g1, pilots, decision = map(read, (readiness_path, g1_path, pilot_path, design_path))
    verified_evidence(ready, pilots, decision)
    if ready.get("status") != "pilot_ready" or any(x.get("status") != "passed" for x in (g1, pilots, decision)):
        raise ValueError("Readiness, G1, pilots, and design feasibility must all pass")
    if ready["code_hash"] != code_hash() or pilots["code_hash"] != code_hash() or g1["code_hash"] != code_hash():
        raise ValueError("Evidence does not match the current code")
    for key in ("graph_hash", "tokenizer_hash"):
        if ready[key] != pilots[key] or ready[key] != g1[key]:
            raise ValueError(f"Gate provenance mismatch: {key}")
    if pilots["dataset_hash"] != ready["dataset_hash"] or pilots["training_hash"] != protocol.training_hash or g1["training_hash"] != protocol.training_hash:
        raise ValueError("Pilot protocol/data mismatch")
    chosen = decision["chosen"]
    if len(protocol.main_seeds) != chosen["n"] or protocol.eval_interval != chosen["eval_interval"]:
        raise ValueError("Protocol must match the selected N and evaluation interval")
    # Main seeds may be frozen only before any primary outcome is consulted.
    if Path(output).exists():
        raise FileExistsError("A launch manifest is immutable; create a new protocol version")
    runs = list(run_matrix(protocol.main_seeds))
    requests = []
    for run in runs:
        condition = "mixed" if run["mode"] == "mixed" else ("mono-"+run["order"][0] if run["mode"] == "mono" else "seq-"+"-".join(run["order"]))
        requests.append(dict(run_id=run["run_id"], category="main", hours=1.25*chosen["per_run_upper_hours"][condition]))
    ledger = Ledger(ledger_path)
    ledger.reserve_bundle(requests)
    with ledger.transaction() as state:
        current_allocations = dict(state['allocations'])
    evidence = dict(ready["evidence_files"])
    evidence.update(decision["evidence_files"])
    evidence.update({str(Path(p).resolve()): file_hash(p) for p in (readiness_path, g1_path, pilot_path, design_path)})
    result = dict(status="frozen", protocol=asdict(protocol), protocol_hash=protocol.hash,
                  code_hash=code_hash(), graph_hash=ready["graph_hash"], dataset_hash=ready["dataset_hash"],
                  tokenizer_hash=ready["tokenizer_hash"], evidence_files=evidence, runs=runs,
                  fixed_review_seeds=[1,2], auxiliary_training_mandatory=False,
                  budgets=current_allocations, decision=chosen)
    result.update({k: ready.get(k) for k in ("review_mode", "human_reviewed", "review_limitations", "protocol_amendment")})
    write_json(output, result)
    return result


def reserve_pilots(protocol, output, ledger, hours_by_mode):
    runs = list(run_matrix(protocol.pilot_seeds, cohort="pilot"))
    categories = dict(mono="g2", mixed="g3", sequential="order_pilot")
    Ledger(ledger).reserve_bundle([dict(run_id=r["run_id"], category=categories[r["mode"]],
                                      hours=hours_by_mode[r["mode"]]*1.25) for r in runs])
    write_json(output, dict(status="pilot_matrix", runs=runs, protocol_hash=protocol.hash))
    return runs

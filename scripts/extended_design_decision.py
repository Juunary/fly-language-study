"""Design decision over the extended-N grid (design version v5.1-extended-N-1); rules fixed in docs/EXTENDED_DESIGN_V5.1.md.

Primary rule = the existing conservative rule of flystudy.workflow.design_decision, applied to more candidate N:
evaluation interval 5,120; latent pairing correlation 0; both families; grid censoring up to the smallest grid value at
or above the pilot's sequential censoring upper bound; per family and censoring level, every grid CV up to the smallest
grid reference restricted CV at or above the pilot dispersion bound (a bound above the grid is outside the simulation
range and blocks); power = lowest Wilson lower bound of the raw p < .025 rate at a 20% effect over both targets;
false-positive rate = highest Wilson upper bound at no effect. Selection = flystudy.budget.choose_design (resolution
passed with an actual sequence pilot, power >= .8, FWER <= .06, 1.25 x GPU and calendar hours within budget; largest N).
Auxiliary allocations are released to the main study (review -> shuffle -> tokenizer) only if no candidate fits the main
allocation alone. The correlation-informed envelope and the analytic contrast cross-check are reported, never decisive.
Observed contrast means are not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flystudy.budget import choose_design
from flystudy.protocol import ORDERS, file_hash, write_json
from flystudy.workflow import verified_evidence

INTERVAL = 5120
CONDITIONS = {"mono-en", "mono-de", "mono-ko", "mixed"} | {"seq-" + "-".join(o) for o in ORDERS}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def envelope(grid, n, ceiling, cv_bound, rho):
    """Rows of one N inside the stress envelope, or (None, reason) when the bound lies outside the grid."""
    rows = [r for r in grid if r["n"] == n and r["interval"] == INTERVAL and r["censoring"] <= ceiling + 1e-9 and abs(r["latent_rho"] - rho) < 1e-9]
    if cv_bound is None:
        return rows, None
    kept = []
    for family in ("lognormal", "weibull"):
        for censoring in sorted({r["censoring"] for r in rows}):
            group = [r for r in rows if r["family"] == family and abs(r["censoring"] - censoring) < 1e-9]
            above = [r["reference_restricted_cv"] for r in group if r["reference_restricted_cv"] >= cv_bound]
            if not above:
                return None, f"dispersion bound {cv_bound:.3f} exceeds the grid for {family}, censoring {censoring}"
            edge = min(above)
            kept.extend(r for r in group if r["reference_restricted_cv"] <= edge + 1e-9)
    return kept, None


def null_key(r):
    return (r["n"], r["family"], round(r["censoring"], 6), round(r["cv"], 6), r["target"])


def load_official_fwer(path):
    """Official 50,000-repetition null rows (design version v5.1-extended-N-2), indexed like the grid's null rows."""
    meta = read(Path(path).with_suffix(".meta.json"))
    if meta.get("design_version") != "v5.1-extended-N-2" or not meta.get("complete"):
        raise ValueError("The official FWER verification must be complete and of design version v5.1-extended-N-2")
    if meta.get("repetitions") != 50000:
        raise ValueError("The official FWER verification uses exactly 50,000 repetitions per scenario")
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    if any(r["repetitions"] != 50000 or r["effect"] != 0 or r["latent_rho"] != 0 for r in rows):
        raise ValueError("Official FWER rows must be rho-0 null scenarios with 50,000 repetitions")
    index = {null_key(r): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate official FWER rows")
    return index, meta


def power_table(grid, ns, ceiling, cv_bound, rho, official=None):
    """Power from the grid; the null bound from the grid, or from the official report when one is given (every envelope
    null row must then have its official counterpart - no deduplication, no exclusion)."""
    table = {}
    for n in ns:
        rows, reason = envelope(grid, n, ceiling, cv_bound, rho)
        if rows is None:
            table[n] = dict(outside_grid=reason); continue
        power = [r["conservative_power_ci"][0] for r in rows if r["effect"] == .2]
        null_rows = [r for r in rows if r["effect"] == 0]
        errors = [r["mc_ci"][1] for r in null_rows]
        table[n] = dict(power_lower=min(power) if power else None, null_fwer_upper=max(errors) if errors else None, rows=len(rows))
        if official is not None:
            missing = [null_key(r) for r in null_rows if null_key(r) not in official]
            if missing:
                raise ValueError(f"missing official FWER rows for N={n}: {missing[:3]}")
            matched = [official[null_key(r)] for r in null_rows]
            worst = max(matched, key=lambda r: r["mc_ci"][1])
            table[n].update(null_fwer_upper=max(r["mc_ci"][1] for r in matched), grid_null_fwer_upper=max(errors) if errors else None,
                            null_rows=len(null_rows), official_rows=len(matched), official_max_rate=max(r["rate"] for r in matched),
                            official_worst=dict(family=worst["family"], censoring=worst["censoring"], cv=worst["cv"], target=worst["target"],
                                                rate=worst["rate"], wilson_upper=worst["mc_ci"][1]))
    return table


def decide(grid_path, pilot_path, cost_path, resolution_path, dispersion_path, prereg_path, ledger_path, output,
           assess_path=None, fwer_report=None, amendment_path=None):
    meta = read(Path(grid_path).with_suffix(".meta.json"))
    if not meta.get("complete") or meta.get("quick") or meta.get("repetitions", 0) < 5000 or meta.get("interval") != INTERVAL:
        raise ValueError("A complete 5,000-repetition extended grid at interval 5,120 is required")
    pilot, costs, resolution, dispersion, ledger = map(read, (pilot_path, cost_path, resolution_path, dispersion_path, ledger_path))
    nested = verified_evidence(pilot, costs, resolution, dispersion)
    grid = [json.loads(l) for l in Path(grid_path).read_text(encoding="utf-8").splitlines()]
    ns = sorted({r["n"] for r in grid})
    seq = [g for k, g in pilot["groups"].items() if k.startswith("sequential/")]
    censor_upper = max(g["censor_upper_95"] for g in seq)
    grid_censoring = sorted({r["censoring"] for r in grid})
    ceilings = [c for c in grid_censoring if c >= censor_upper]
    cv_bound = pilot.get("dispersion_uncertainty", {}).get("restricted_cv_upper")
    evidence = {str(Path(p).resolve()): file_hash(p) for p in (grid_path, Path(grid_path).with_suffix(".meta.json"), pilot_path,
                                                             cost_path, resolution_path, dispersion_path, prereg_path)}
    if assess_path and Path(assess_path).exists():
        evidence[str(Path(assess_path).resolve())] = file_hash(assess_path)
    evidence.update(nested)
    existing = read(assess_path) if assess_path and Path(assess_path).exists() else None
    official, official_meta = load_official_fwer(fwer_report) if fwer_report else (None, None)
    if fwer_report:
        for extra in (fwer_report, Path(fwer_report).with_suffix(".meta.json"), amendment_path):
            if extra and Path(extra).exists():
                evidence[str(Path(extra).resolve())] = file_hash(extra)
    result = dict(design_version="v5.1-extended-N-2" if fwer_report else "v5.1-extended-N-1", status="blocked", reason=None, selection_uses_observed_effect=False,
                  preregistration=str(prereg_path), candidate_ns=ns, eval_interval=INTERVAL,
                  existing_assess_design=None if existing is None else dict(path=str(assess_path), status=existing.get("status"), reason=existing.get("reason")),
                  pilot_status=pilot.get("status"), pilot_censor_upper=censor_upper, dispersion_bound=cv_bound,
                  stress_envelope="rho=0; both families; censoring up to the grid ceiling; CVs up to the grid edge at the pilot bound",
                  candidates=[], evidence_files=evidence)
    measurement = resolution.get("intervals", {}).get(str(INTERVAL), {})
    measured = costs.get("intervals", {}).get(str(INTERVAL))
    if not measured or set(measured["per_run_upper_hours"]) != CONDITIONS:
        raise ValueError("Cost evidence must cover every primary condition at interval 5,120")
    per_run = measured["per_run_upper_hours"]
    block_hours = sum(per_run.values())
    allocations = ledger["allocations"]
    releasable = sum(allocations[c] for c in ("review", "shuffle", "tokenizer")
                     if not any(r["category"] == c for r in ledger["runs"].values()))
    result["cost"] = dict(per_run_upper_hours=per_run, seed_block_upper_hours=block_hours, main_allocation=allocations["main"],
                          releasable_auxiliary_hours=releasable, serial_overhead_hours=costs["serial_overhead_hours"],
                          remaining_calendar_hours=costs["remaining_calendar_hours"],
                          method=costs.get("method"))
    if not ceilings:
        result["reason"] = "Pilot censoring upper bound exceeds the simulated grid"
        write_json(output, result); return result
    ceiling = ceilings[0]
    primary = power_table(grid, ns, ceiling, cv_bound, 0., official)
    result["primary_power_table"] = {str(n): v for n, v in primary.items()}
    if official is not None:
        result["fwer_evaluation"] = dict(
            rule="FWER approval bound 0.06 unchanged; null scenarios evaluated with 50,000 repetitions each (amendment recorded after the "
                 "5,000-repetition block and the unofficial 50,000-repetition diagnostic were known, before the main study)",
            amendment=None if amendment_path is None else str(amendment_path), report=str(fwer_report), repetitions=official_meta["repetitions"],
            simulation_seed_base=official_meta.get("simulation_seed_base"), power_rows="reused from the 5,000-repetition extended grid",
            grid_null_bound_not_used={str(n): v.get("grid_null_fwer_upper") for n, v in primary.items() if "outside_grid" not in v},
            envelope_null_rows_per_n={str(n): v.get("null_rows") for n, v in primary.items() if "outside_grid" not in v},
            official_rows_matched_per_n={str(n): v.get("official_rows") for n, v in primary.items() if "outside_grid" not in v})
    rho_lower = dispersion["correlation"]["sequential_orders"]["mean_offdiagonal_bootstrap_95"][0]
    grid_rhos = sorted({r["latent_rho"] for r in grid})
    rho_sens = max([x for x in grid_rhos if x <= max(0., rho_lower or 0.) + 1e-9], default=0.)
    result["sensitivity_correlation_informed"] = dict(latent_rho=rho_sens, basis="largest grid rho <= lower 2.5% bootstrap bound of the mean inter-order correlation",
                                                     power_table={str(n): v for n, v in power_table(grid, ns, ceiling, cv_bound, rho_sens, official if rho_sens == 0 else None).items()},
                                                     decisive=False)
    result["analytic_contrast_cross_check"] = {k: v["analytic_required_n_20pct"] for k, v in dispersion["contrasts"].items()}
    passing = [n for n in ns if "outside_grid" not in primary[n] and primary[n]["power_lower"] is not None
               and primary[n]["power_lower"] >= .8 and primary[n]["null_fwer_upper"] <= .06]
    result["minimum_passing_n"] = min(passing) if passing else None

    def candidates(available):
        out = []
        for n in ns:
            row = primary[n]
            if "outside_grid" in row:
                continue
            hours = n * block_hours
            out.append(dict(n=n, eval_interval=INTERVAL,
                            measurement_passed=measurement.get("status") == "passed" and measurement.get("actual_sequence_pilot") is True,
                            power_lower=row["power_lower"], null_fwer_upper=row["null_fwer_upper"], gpu_hours=hours,
                            available_main_hours=available, calendar_hours=hours / 2 + costs["serial_overhead_hours"],
                            available_calendar_hours=costs["remaining_calendar_hours"], per_run_upper_hours=per_run))
        return out
    result["candidates"] = candidates(allocations["main"])
    if pilot.get("status") != "passed":
        result["reason"] = "Learnability pilot evidence is incomplete (pilot-report status is not passed)"
        write_json(output, result); return result
    chosen = choose_design(result["candidates"])
    release = 0.
    if chosen is None and releasable:
        widened = candidates(allocations["main"] + releasable)
        chosen = choose_design(widened)
        if chosen is not None:
            result["candidates"] = widened
            release = max(0., 1.25 * chosen["gpu_hours"] - allocations["main"])
    if chosen is None:
        blocking = []
        if not passing:
            blocking.append("no candidate N meets power >= .8 and FWER <= .06 in the conservative envelope")
        if not (measurement.get("status") == "passed" and measurement.get("actual_sequence_pilot") is True):
            blocking.append("evaluation-resolution evidence at 5,120 is not passed with an actual sequence pilot")
        if passing and min(passing) * block_hours * 1.25 > allocations["main"] + releasable:
            blocking.append("the smallest passing N exceeds the main and releasable auxiliary GPU budget")
        result["reason"] = "; ".join(blocking) or "no candidate meets the selection criteria"
    else:
        result.update(status="passed", chosen=chosen, auxiliary_release_required_hours=release,
                      main_seeds=list(range(30001, 30001 + chosen["n"])))
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in ("grid", "pilots", "costs", "resolution", "dispersion", "preregistration", "ledger", "output"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--assess-design")
    p.add_argument("--fwer-report", help="official 50,000-repetition null verification (design version v5.1-extended-N-2)")
    p.add_argument("--amendment", help="amendment document recorded as evidence with --fwer-report")
    a = p.parse_args()
    r = decide(a.grid, a.pilots, a.costs, a.resolution, a.dispersion, a.preregistration, a.ledger, a.output, a.assess_design,
               a.fwer_report, a.amendment)
    print(json.dumps({k: r.get(k) for k in ("status", "reason", "minimum_passing_n", "chosen", "auxiliary_release_required_hours")}, indent=1, default=str))


if __name__ == "__main__":
    main()

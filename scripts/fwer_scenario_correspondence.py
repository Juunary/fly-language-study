"""Correspondence of the null scenarios behind the FWER approval check (design version v5.1-extended-N-2).

Lists, for the decision envelope of the preserved blocked report, every null row of the 5,000-repetition grid (34 per N),
its counterpart in the unofficial 50,000-repetition diagnostic (17 rows, target 0 only) and in the official verification
(both targets), with scenario indices and seeds. Also verifies on a throwaway seed that the target label does not change
the generated null data and states where the familywise rate is computed. Reads no official result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from flystudy.protocol import file_hash, write_json
from flystudy.statistics import simulate

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in ("grid", "blocked-decision", "diagnostic", "output"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    grid_mod, official, design = load("extended_power_grid"), load("fwer_official_verification"), load("extended_design_decision")
    grid = [json.loads(l) for l in Path(a.grid).read_text(encoding="utf-8").splitlines()]
    blocked = json.loads(Path(a.blocked_decision).read_text(encoding="utf-8"))
    diagnostic = json.loads(Path(a.diagnostic).read_text(encoding="utf-8"))
    ceiling = min(c for c in sorted({r["censoring"] for r in grid}) if c >= blocked["pilot_censor_upper"])
    key = lambda s: (s["family"], round(s["censoring"], 6), round(s["cv"], 6), s["target"])
    grid_index = {key(s): i for i, s in enumerate(grid_mod.scenarios()) if s["effect"] == 0 and s["latent_rho"] == 0}
    official_index = {key(s): i for i, s in official.scenarios()}
    diag_seed = {(r["family"], round(r["censoring"], 6), round(r["cv"], 6)): r["simulation_seed"] for r in diagnostic["records"]}
    per_n = {}
    for n in sorted({r["n"] for r in grid}):
        rows, reason = design.envelope(grid, n, ceiling, blocked["dispersion_bound"], 0.)
        per_n[n] = sorted(key(r) for r in rows if r["effect"] == 0)
    reference = per_n[min(per_n)]
    table = []
    for k in reference:
        family, censoring, cv, target = k
        row = next(r for r in grid if r["n"] == min(per_n) and r["effect"] == 0 and r["latent_rho"] == 0 and key(r) == k)
        table.append(dict(family=family, censoring=censoring, cv=cv, target=target, reference_restricted_cv=row["reference_restricted_cv"],
                          grid_scenario_index=grid_index[k], grid_seed=grid_mod.SEED_BASE + grid_index[k],
                          diagnostic_seed_target0_only=diag_seed.get(k[:3]), diagnostic_has_own_row=target == 0 and k[:3] in diag_seed,
                          official_scenario_index=official_index[k], official_seed=official.SEED_BASE + official_index[k]))
    # Same seed, only the target label differs: every simulated quantity must be identical under the null.
    sample = dict(cv=.5, latent_rho=0., censoring=.3, family="weibull", interval=5120, effect=0.)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "target"} for r in rows]
    same = strip(simulate({**sample, "target": 0}, repetitions=2000, seed=424242, ns=(20, 60, 100))) == \
        strip(simulate({**sample, "target": 1}, repetitions=2000, seed=424242, ns=(20, 60, 100)))
    envelope_keys = set(reference)
    report = dict(
        design_version=official.DESIGN_VERSION, envelope=dict(latent_rho=0., censoring_ceiling=ceiling, dispersion_bound=blocked["dispersion_bound"],
                                                              identical_for_every_n=all(v == reference for v in per_n.values())),
        counts=dict(grid_null_rows_per_n=len(reference), distinct_generating_distributions=len({k[:3] for k in reference}),
                    diagnostic_rows_per_n=sum(k[:3] in diag_seed for k in {k[:3] for k in reference}),
                    official_scenarios=len(official_index), official_in_envelope=len(envelope_keys & set(official_index)),
                    official_reported_only=len(set(official_index) - envelope_keys)),
        null_target_equivalence=dict(
            code="flystudy.statistics.simulate: with effect 0 the scale factor is 1.0, so 'raw[:, :, treatment] *= factor' leaves the data unchanged "
                 "for either target; the target label then enters no computation of the null rate",
            same_seed_outputs_identical=same, check="throwaway seed 424242, 2,000 repetitions, N 20/60/100; tests/test_fwer_official.py repeats it"),
        rate_definition="effect 0: successes = rejects.any(1).sum() with rejects = Holm-adjusted p < .05 for the two planned contrasts, i.e. the "
                        "probability that at least one of the two contrasts is falsely rejected; mc_ci is its Wilson 95% interval",
        deduplication=dict(used_in_official_decision=False,
                           note="The two targets of a null scenario are independent Monte Carlo replicates of one distribution. The unofficial "
                                "diagnostic simulated target 0 only (17 envelope rows). The official verification simulates both targets so that "
                                "each of the 34 grid rows per N has its own counterpart; the decision takes the maximum Wilson upper bound over "
                                "all of them and excludes none."),
        envelope_rows=table,
        official_scenarios=[dict(index=i, seed=official.SEED_BASE + i, in_decision_envelope=key(s) in envelope_keys, **s) for i, s in official.scenarios()],
        sources={name: file_hash(path) for name, path in (("grid", a.grid), ("blocked_decision", a.blocked_decision), ("diagnostic", a.diagnostic),
                                                         ("official_script", ROOT / "scripts" / "fwer_official_verification.py"))})
    write_json(a.output, report)
    print(json.dumps({**report["counts"], "null_target_equivalence": same, "envelope_identical_for_every_n": report["envelope"]["identical_for_every_n"]}, indent=1))


if __name__ == "__main__":
    main()

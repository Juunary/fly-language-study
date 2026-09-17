"""Variability of the two planned contrasts and correlation between conditions in complete pilot seed blocks.

Design nuisance parameters only. Contrast means (observed effects) are deliberately not reported and must not be used
to choose N. Inputs are the pilot run summaries of the listed matrices that share the protocol's training hash; only
seed blocks with all ten conditions terminal (mastered or administrative cap) are used, and excluded blocks are listed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from flystudy.protocol import LANGUAGES, ORDERS, Protocol, file_hash, write_json
from flystudy.statistics import CONTRASTS, CONTRAST_NAMES

CONDITIONS = [("mono", (l,)) for l in LANGUAGES] + [("sequential", o) for o in ORDERS] + [("mixed", LANGUAGES)]


def label(mode, order):
    return "mixed" if mode == "mixed" else ("mono-" + order[0] if mode == "mono" else "seq-" + "-".join(order))


def load_blocks(runs_root, matrices, protocol):
    runs_root = Path(runs_root)
    ids = [r for m in matrices for r in json.loads(Path(m).read_text(encoding="utf-8"))["runs"]]
    blocks, excluded, evidence = {}, {}, {}
    for r in ids:
        path = runs_root / r["run_id"] / "summary.json"
        seed = r["seed"]
        if not path.exists():
            excluded.setdefault(seed, []).append(f"{r['run_id']}: no summary"); continue
        s = json.loads(path.read_text(encoding="utf-8"))
        if s["training_hash"] != protocol.training_hash or s["cohort"] != "pilot" or s.get("graph_condition") != "real":
            excluded.setdefault(seed, []).append(f"{r['run_id']}: incompatible provenance"); continue
        if s["stop_reason"] not in ("mastered", "administrative_cap"):
            excluded.setdefault(seed, []).append(f"{r['run_id']}: {s['stop_reason']}"); continue
        blocks.setdefault(seed, {})[label(s["mode"], tuple(s["order"]))] = s["first_global"] or s["cap"]
        evidence[str(path.resolve())] = file_hash(path)
    complete = {seed: b for seed, b in blocks.items() if len(b) == 10 and seed not in excluded}
    for seed, b in blocks.items():
        if seed not in complete:
            excluded.setdefault(seed, []).append(f"{len(b)}/10 terminal conditions")
    return complete, excluded, evidence


def bootstrap(values, fn, draws=10000, seed=39101):
    """Seed-block bootstrap; draws where the statistic is undefined (e.g. a constant resampled column) are NaN."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), (draws, len(values)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.array([fn(values[i]) for i in index], dtype=float)


def q(values, level):
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, level)) if len(finite) else None


def undefined(values):
    return int((~np.isfinite(values)).sum())


def mean_offdiag(matrix):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(matrix, rowvar=False)
    values = c[~np.eye(c.shape[0], dtype=bool)]
    return float(np.mean(values)) if np.isfinite(values).all() else float("nan")


def required_n(relative_sd, effect=.2, alpha=.025, power=.8, n_max=1000):
    """Smallest n with two-sided paired t power >= power for a contrast difference of effect x reference mean."""
    if relative_sd <= 0:
        return None
    d = effect / relative_sd
    for n in range(3, n_max + 1):
        crit = stats.t.ppf(1 - alpha / 2, n - 1)
        nc = d * np.sqrt(n)
        if stats.nct.sf(crit, n - 1, nc) + stats.nct.cdf(-crit, n - 1, nc) >= power:
            return n
    return None


def report(runs_root, matrices, protocol_path, output):
    protocol = Protocol.load(protocol_path)
    blocks, excluded, evidence = load_blocks(runs_root, matrices, protocol)
    seeds = sorted(blocks)
    if len(seeds) < 3:
        raise ValueError("Need at least three complete seed blocks")
    seq = np.array([[blocks[s]["seq-" + "-".join(o)] for o in ORDERS] for s in seeds], dtype=float)
    allc = np.array([[blocks[s][label(m, o)] for m, o in CONDITIONS] for s in seeds], dtype=float)
    reference = float(seq.mean())
    order_var = seq.var(0, ddof=1)
    out = dict(kind="pilot_dispersion", design_nuisance_only=True, contrast_means_reported=False,
               note="Observed contrast means are omitted on purpose; N is never chosen from observed effects.",
               protocol=protocol_path, training_hash=protocol.training_hash, n_seed_blocks=len(seeds), seeds=seeds,
               excluded_seeds={str(k): v for k, v in sorted(excluded.items())},
               reference_mean_restricted_cost=reference,
               orders={"-".join(o): dict(mean=float(seq[:, i].mean()), sd=float(np.sqrt(order_var[i])),
                                         cv=float(np.sqrt(order_var[i]) / seq[:, i].mean()),
                                         censored=int((seq[:, i] >= protocol.total_cap).sum())) for i, o in enumerate(ORDERS)},
               contrasts={}, correlation={}, evidence_files=evidence)
    for k, name in enumerate(CONTRAST_NAMES):
        coef = CONTRASTS[k]
        delta = seq @ coef
        rel_sd = float(delta.std(ddof=1) / reference)
        boot = bootstrap(seq, lambda v: (v @ coef).std(ddof=1) / v.mean(), seed=39101 + k)
        sum_sq = float((coef ** 2).sum())
        rho_eff = float(1 - delta.var(ddof=1) / (sum_sq * order_var.mean()))
        boot_rho = bootstrap(seq, lambda v: 1 - (v @ coef).var(ddof=1) / (sum_sq * v.var(0, ddof=1).mean()), seed=39201 + k)
        out["contrasts"][name] = dict(
            coefficients={"-".join(o): float(c) for o, c in zip(ORDERS, coef)}, sum_squared_coefficients=sum_sq,
            sd=float(delta.std(ddof=1)), relative_sd=rel_sd,
            relative_sd_bootstrap_95=[q(boot, .025), q(boot, .975)], relative_sd_upper_975=q(boot, .975),
            implied_exchangeable_rho=rho_eff,
            implied_rho_bootstrap_95=[q(boot_rho, .025), q(boot_rho, .975)], bootstrap_undefined_draws=undefined(boot_rho),
            analytic_required_n_20pct=dict(point=required_n(rel_sd), upper_975_sd=required_n(q(boot, .975)),
                                           note="paired t, two-sided alpha .025, power .8, difference 0.2 x reference mean; cross-check only, not the decision"))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr6 = np.corrcoef(seq, rowvar=False)
    boot_r = bootstrap(seq, mean_offdiag, seed=39301)
    out["correlation"]["sequential_orders"] = dict(
        labels=["-".join(o) for o in ORDERS], matrix=[[None if not np.isfinite(x) else round(float(x), 4) for x in row] for row in corr6],
        mean_offdiagonal=mean_offdiag(seq),
        mean_offdiagonal_bootstrap_95=[q(boot_r, .025), q(boot_r, .975)], bootstrap_undefined_draws=undefined(boot_r))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr10 = np.corrcoef(allc, rowvar=False)
    out["correlation"]["all_conditions"] = dict(
        labels=[label(m, o) for m, o in CONDITIONS], matrix=[[None if not np.isfinite(x) else round(float(x), 4) for x in row] for row in corr10],
        note="Monolingual EN/DE are mostly censored at 200,000; their correlations are unstable or undefined (constant columns).")
    write_json(output, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True); p.add_argument("--protocol", required=True)
    p.add_argument("--matrices", nargs="+", required=True); p.add_argument("--output", required=True)
    a = p.parse_args()
    r = report(a.runs, a.matrices, a.protocol, a.output)
    print(json.dumps(dict(blocks=r["n_seed_blocks"], excluded=r["excluded_seeds"],
                          contrasts={k: dict(relative_sd=round(v["relative_sd"], 3), upper=round(v["relative_sd_upper_975"], 3),
                                             implied_rho=round(v["implied_exchangeable_rho"], 3)) for k, v in r["contrasts"].items()},
                          mean_order_correlation=round(r["correlation"]["sequential_orders"]["mean_offdiagonal"], 3)), indent=1))


if __name__ == "__main__":
    main()

from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import math
import numpy as np
from scipy import optimize, special, stats

from .protocol import ORDERS, INTERVALS, digest, write_json

LAST = np.array([o[-1] == "ko" for o in ORDERS])
FIRST = np.array([o[0] == "ko" for o in ORDERS])
MIDDLE = ~(LAST | FIRST)
CONTRASTS = np.array([LAST.astype(float)/2-FIRST.astype(float)/2,
                      (~MIDDLE).astype(float)/4-MIDDLE.astype(float)/2])
CONTRAST_NAMES = ("ko_last_minus_first", "en_de_adjacent_minus_separated")


def holm(p):
    p = np.asarray(p, dtype=float)
    indices = np.argsort(p, axis=-1)
    ordered = np.take_along_axis(p, indices, axis=-1)
    adjusted = np.minimum(1, np.maximum.accumulate(ordered*np.arange(p.shape[-1], 0, -1), axis=-1))
    return np.take_along_axis(adjusted, np.argsort(indices, axis=-1), axis=-1)


def paired_tests(values):
    """Input (..., seed, order). Orders within a seed are not replicates."""
    delta = np.asarray(values) @ CONTRASTS.T
    n = delta.shape[-2]
    mean, se = delta.mean(-2), delta.std(-2, ddof=1)/np.sqrt(n)
    t = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
    # Zero empirical variance is uninformative; do not invent certainty.
    p = np.where(se > 0, 2*stats.t.sf(abs(t), n-1), 1.)
    return mean, se, p, holm(p)


def summarize(values, cap=900000, bootstrap=10000, seed=921):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Require complete seed-by-six-order observations")
    if np.any(values <= 0) or np.any(values > cap):
        raise ValueError("Invalid restricted costs")
    mean, se, p, adjusted = paired_tests(values)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(values), (bootstrap, len(values)))
    boot = (values[draw] @ CONTRASTS.T).mean(1)
    critical = stats.t.ppf(.9875, len(values)-1)  # 97.5% CIs; two-comparison Bonferroni coverage
    report = dict(n_seeds=len(values), cap=cap, contrasts={},
                  rmst_by_order={"-".join(o): float(values[:, i].mean()) for i, o in enumerate(ORDERS)},
                  inference="Conditional on the frozen benchmark, graph, tokenizer and training policy.")
    for i, name in enumerate(CONTRAST_NAMES):
        report["contrasts"][name] = dict(difference=float(mean[i]), p=float(p[i]), holm_p=float(adjusted[i]),
            simultaneous_ci=None if se[i] == 0 else [float(mean[i]-critical*se[i]), float(mean[i]+critical*se[i])],
            seed_bootstrap_ci=None if se[i] == 0 else np.quantile(boot[:, i], [.025, .975]).tolist(),
            informative=bool(se[i] > 0))
    return report


def wilson(successes, total, confidence=.95):
    if total == 0:
        return [0., 1.]
    z = stats.norm.ppf((1+confidence)/2)
    p, denom = successes/total, 1+z*z/total
    center = (p+z*z/(2*total))/denom
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [max(0., center-radius), min(1., center+radius)]


def exact_censor_upper(censored, n):
    if n == 0 or censored == n:
        return 1.
    return float(stats.beta.ppf(.95, censored+1, n-censored))


def _shape_quantile(u, cv, family):
    if family == "lognormal":
        return np.exp(math.sqrt(math.log1p(cv*cv))*special.ndtri(u))
    if family == "weibull":
        shape = optimize.brentq(lambda k: math.expm1(special.gammaln(1+2/k)-2*special.gammaln(1+1/k))-cv*cv, .2, 20)
        return (-np.log1p(-u))**(1/shape)
    raise ValueError(f"Unknown family: {family}")


def restricted_times(raw, interval, cap):
    # Synthetic monotone crossing followed by the second scheduled confirmation.
    # This is an explicit statistical DGM, not a simulation of learned neural dynamics.
    confirmed = (np.ceil(raw/interval)+1)*interval
    # Include the fixed cap evaluation after the last regular grid point.
    prior_grid = math.floor((cap-1)/interval)*interval
    confirmed = np.where((confirmed > cap) & (raw <= prior_grid), cap, confirmed)
    event = confirmed <= cap
    return np.minimum(confirmed, cap), event


def scenarios(quick=False):
    cvs = (.5,) if quick else (.25, .5, 1.)
    rhos = (.5,) if quick else (0., .5, .8)
    censoring = (.3,) if quick else (.1, .3, .5)
    families = ("lognormal",) if quick else ("lognormal", "weibull")
    for cv, rho, censor, family, interval, target, effect in product(
            cvs, rhos, censoring, families, INTERVALS, (0, 1), (0., .1, .2, .3)):
        yield dict(cv=cv, latent_rho=rho, censoring=censor, family=family,
                   interval=interval, target=target, effect=effect)


def simulate(scenario, repetitions=5000, seed=8401, ns=(10, 15, 20), cap=900000):
    rng = np.random.default_rng(seed)
    nmax = max(ns)
    common = rng.normal(size=(repetitions, nmax, 1))
    independent = rng.normal(size=(repetitions, nmax, 6))
    rho = scenario["latent_rho"]
    z = np.sqrt(rho)*common + np.sqrt(1-rho)*independent
    unit = _shape_quantile(special.ndtr(z).clip(1e-12, 1-1e-12), scenario["cv"], scenario["family"])
    interval = scenario["interval"]
    last_confirmable_crossing = math.floor((cap-1)/interval)*interval
    quantile = float(_shape_quantile(np.array(1-scenario["censoring"]), scenario["cv"], scenario["family"]))
    scale = last_confirmable_crossing/quantile
    raw = unit*scale
    # Calibrate actual discrete RMST reduction, not an assumed raw-time scale factor.
    integration = _shape_quantile((np.arange(100000)+.5)/100000, scenario["cv"], scenario["family"])*scale
    base_mean = restricted_times(integration, interval, cap)[0].mean()
    reference_cv = float(restricted_times(integration, interval, cap)[0].std()/base_mean)
    target_mean = base_mean*(1-scenario["effect"])
    factor = 1.
    if scenario["effect"]:
        factor = optimize.brentq(lambda f: restricted_times(integration*f, interval, cap)[0].mean()-target_mean, .0001, 1.)
    treatment = LAST if scenario["target"] == 0 else ~MIDDLE
    raw[:, :, treatment] *= factor
    observed, events = restricted_times(raw, interval, cap)
    results = []
    for n in ns:
        values = observed[:, :n]
        _, _, raw_p, adjusted = paired_tests(values)
        rejects = adjusted < .05
        all_null = scenario["effect"] == 0
        successes = int(rejects.any(1).sum()) if all_null else int(rejects[:, scenario["target"]].sum())
        rate = successes/repetitions
        conservative_successes = int((raw_p[:, scenario["target"]] < .025).sum())
        censor = (~events[:, :n]).mean(axis=(0, 1)).tolist()
        results.append({**scenario, "n": n, "repetitions": repetitions, "simulation_seed": seed,
                        "rate_kind": "familywise_type_I_error" if all_null else "power",
                        "rate": rate, "mc_se": math.sqrt(rate*(1-rate)/repetitions),
                        "conservative_power_ci": wilson(conservative_successes, repetitions) if not all_null else None,
                        "mc_ci": wilson(successes, repetitions), "censoring_by_order": censor,
                        "reference_rmst": float(base_mean), "calibrated_scale_factor": float(factor),
                        "reference_restricted_cv": reference_cv,
                        "calibrated_relative_rmst_effect": float(1-restricted_times(integration*factor, interval, cap)[0].mean()/base_mean),
                        "dgm": "correlated event times, monotone crossing, two-check confirmation; not neural dynamics"})
    return results


def simulate_grid(output, repetitions=5000, quick=False):
    from threadpoolctl import threadpool_limits
    # Small matrix products otherwise oversubscribe CPU threads per scenario.
    limits = threadpool_limits(limits=1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as f:
        for i, scenario in enumerate(scenarios(quick)):
            for result in simulate(scenario, repetitions, seed=8401+i):
                f.write(json.dumps(result) + "\n")
                total += 1
            f.flush()
            if i % 20 == 0:
                print(f"simulation scenarios completed: {i+1}", flush=True)
    write_json(output.with_suffix(".meta.json"), dict(quick=quick, records=total, repetitions=repetitions,
               complete=True, design_effect=.2, minimum_power=.8, mc_interval="Wilson 95%"))
    return dict(records=total, output=str(output))

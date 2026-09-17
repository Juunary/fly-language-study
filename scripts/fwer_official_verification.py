"""Official familywise-error verification for the extended design decision (design version v5.1-extended-N-2), CPU only.

Amendment recorded in docs/FWER_EVALUATION_AMENDMENT_V5.1.md: the FWER approval bound stays at 0.06, but the null
scenarios are evaluated with 50,000 repetitions each instead of 5,000. This script simulates every null (effect 0)
scenario of the extended grid at latent correlation 0 - 9 raw CVs x 3 censoring levels x 2 families x 2 targets = 108
scenarios, for N 20..100 - with flystudy.statistics.simulate unchanged. Under the null the target label does not enter
the generated data (scale factor 1) and the recorded rate is the probability that either Holm-adjusted contrast is
rejected; both targets are nevertheless simulated so that every null row of the 5,000-repetition grid has its own
official counterpart and nothing is deduplicated or excluded. Code, scenarios, seeds and repetitions are fixed before the
single run; the script refuses to run twice.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from multiprocessing import Pool
from pathlib import Path

DESIGN_VERSION = "v5.1-extended-N-2"
NS = (20, 30, 40, 50, 60, 70, 80, 90, 100)
CVS = (.25, .375, .5, .625, .75, 1., 1.25, 1.5, 2.)
CENSORING = (.1, .3, .5)
FAMILIES = ("lognormal", "weibull")
TARGETS = (0, 1)
INTERVAL = 5120
REPETITIONS = 50000
SEED_BASE = 29401  # new; the 5,000-repetition grid used 9401+, the unofficial diagnostic 19401+, the original grid 8401+


def scenarios():
    return list(enumerate(dict(cv=cv, latent_rho=0., censoring=censor, family=family, interval=INTERVAL, target=target, effect=0.)
                          for cv, censor, family, target in product(CVS, CENSORING, FAMILIES, TARGETS)))


def work(args):
    index, scenario = args
    from threadpoolctl import threadpool_limits
    from flystudy.statistics import simulate
    with threadpool_limits(limits=1):
        return simulate(scenario, repetitions=REPETITIONS, seed=SEED_BASE + index, ns=NS)


def run(output, workers=8):
    from flystudy.gates import code_hash
    from flystudy.protocol import file_hash, write_json
    out = Path(output); marker = Path(str(out) + ".started")
    if out.exists() or marker.exists() or out.with_suffix(".meta.json").exists():
        raise FileExistsError(f"{out}: the official verification runs once; seeds and repetitions are not retried")
    fixed = dict(design_version=DESIGN_VERSION, repetitions=REPETITIONS, simulation_seed_base=SEED_BASE, scenarios=len(scenarios()),
                 ns=list(NS), cvs=list(CVS), censoring=list(CENSORING), families=list(FAMILIES), targets=list(TARGETS), latent_rho=0.,
                 interval=INTERVAL, effect=0., code_hash=code_hash(), script_hash=file_hash(__file__))
    marker.write_text(json.dumps(dict(started_at_utc=datetime.now(timezone.utc).isoformat(), **fixed), indent=1) + "\n", encoding="utf-8")
    items, total = scenarios(), 0
    with Pool(workers) as pool, out.open("w", encoding="utf-8") as f:
        for k, results in enumerate(pool.imap(work, items, chunksize=1)):
            for r in results:
                f.write(json.dumps(r) + "\n"); total += 1
            f.flush()
            print(f"scenarios completed: {k + 1}/{len(items)}", flush=True)
    write_json(out.with_suffix(".meta.json"), dict(complete=True, records=total, finished_at_utc=datetime.now(timezone.utc).isoformat(),
               mc_interval="Wilson 95%", rate_kind="familywise_type_I_error (either Holm-adjusted contrast rejected at .05)",
               dgm="flystudy.statistics.simulate (correlated event times, monotone crossing, two-check confirmation); not neural dynamics", **fixed))
    return dict(records=total, output=str(out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True); p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.workers)))


if __name__ == "__main__":
    main()

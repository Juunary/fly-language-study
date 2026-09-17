"""Extended-N power grid for protocol v5.1 (design version v5.1-extended-N-1), CPU only.

Same data-generating model and test as flystudy.statistics.simulate (correlated event times, two-check confirmation,
paired t with Holm, conservative raw p < .025), evaluation interval fixed at 5,120, cap 900,000, 5,000 repetitions per
scenario. Axes were fixed in docs/EXTENDED_DESIGN_V5.1.md before the 15-block pilot results were read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from multiprocessing import Pool
from pathlib import Path

NS = (20, 30, 40, 50, 60, 70, 80, 90, 100)
CVS = (.25, .375, .5, .625, .75, 1., 1.25, 1.5, 2.)
RHOS = (0., .25, .5, .65, .8)
CENSORING = (.1, .3, .5)
FAMILIES = ("lognormal", "weibull")
INTERVAL = 5120
TARGETS = (0, 1)
EFFECTS = (0., .1, .2, .3)
REPETITIONS = 5000
SEED_BASE = 9401


def scenarios():
    for cv, rho, censor, family, target, effect in product(CVS, RHOS, CENSORING, FAMILIES, TARGETS, EFFECTS):
        yield dict(cv=cv, latent_rho=rho, censoring=censor, family=family, interval=INTERVAL, target=target, effect=effect)


def work(args):
    index, scenario = args
    from threadpoolctl import threadpool_limits
    from flystudy.statistics import simulate
    with threadpool_limits(limits=1):
        return simulate(scenario, repetitions=REPETITIONS, seed=SEED_BASE + index, ns=NS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True); p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    items = list(enumerate(scenarios()))
    total = 0
    with Pool(a.workers) as pool, out.open("w", encoding="utf-8") as f:
        for k, results in enumerate(pool.imap(work, items, chunksize=4)):
            for r in results:
                f.write(json.dumps(r) + "\n"); total += 1
            if k % 200 == 0:
                print(f"scenarios completed: {k + 1}/{len(items)}", flush=True)
    from flystudy.gates import code_hash
    from flystudy.protocol import file_hash, write_json
    write_json(out.with_suffix(".meta.json"), dict(
        design_version="v5.1-extended-N-1", complete=True, quick=False, records=total, scenarios=len(items), repetitions=REPETITIONS,
        ns=list(NS), cvs=list(CVS), latent_rhos=list(RHOS), censoring=list(CENSORING), families=list(FAMILIES), interval=INTERVAL,
        targets=list(TARGETS), effects=list(EFFECTS), simulation_seed_base=SEED_BASE, design_effect=.2, minimum_power=.8,
        mc_interval="Wilson 95%", code_hash=code_hash(), script_hash=file_hash(__file__),
        dgm="flystudy.statistics.simulate (correlated event times, monotone crossing, two-check confirmation); not neural dynamics"))
    print(json.dumps(dict(records=total, output=str(out))))


if __name__ == "__main__":
    main()

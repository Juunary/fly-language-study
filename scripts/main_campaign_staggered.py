"""Campaign launcher with serialized run start-up; scheduling only, the frozen flystudy package is not changed.

Same rules as flystudy.campaign.campaign: complete 10-condition seed blocks in manifest order, two GPU workers, completed
runs (summary.json matching the frozen hashes) are skipped, and after a failed run nothing new is dispatched and the
campaign stops - incomplete blocks are not censored observations. The one difference: a run is launched only after every
run launched before it has passed its start-up phase (its metadata.json exists) or has exited. Two runs that start in
the same instant both rewrite <data>/audit.json through the same audit.json.partial, and the second rename fails
(technical failure of main-real-30002-mono-de on 2026-09-17, before any training). Each run is started with exactly the
command flystudy.campaign uses; an explicitly listed retry adds --reservation-id so the failed attempt's accounting stays.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def campaign(runs, expected, output, devices, launch, poll=.2, retries=None, startup_timeout=600.):
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("Specify two distinct CUDA devices")
    output, retries = Path(output), dict(retries or {})
    output.mkdir(parents=True, exist_ok=True)
    launched = skipped = 0
    for seed in sorted({r["seed"] for r in runs}):
        seed_block = [r for r in runs if r["seed"] == seed]
        if len(seed_block) != 10:
            raise ValueError("Campaign scheduling unit is a complete 10-condition seed block")
        pending, active, failed = list(seed_block), {}, False
        while pending or active:
            for device in devices:
                if device in active or not pending or failed:
                    continue
                # Start-up gate: every active run must have written metadata.json (past the dataset audit) first.
                if any(not (folder / "metadata.json").exists() and time.time() - t0 < startup_timeout for _, _, folder, t0 in active.values()):
                    continue
                run = pending.pop(0); folder = output / run["run_id"]
                if (folder / "summary.json").exists():
                    row = json.loads((folder / "summary.json").read_text())
                    if row.get("run_id") != run["run_id"] or any(row.get(k) != v for k, v in expected.items()):
                        raise ValueError(f"Existing run {run['run_id']} does not match this campaign")
                    if row["stop_reason"] in ("mastered", "administrative_cap"):
                        skipped += 1
                        continue
                    raise RuntimeError(f"Explicit failure recovery required for {run['run_id']}")
                extra = ["--reservation-id", retries[run["run_id"]]] if run["run_id"] in retries else []
                active[device] = (launch(run, device, folder, extra), run["run_id"], folder, time.time()); launched += 1
            for device, (process, rid, folder, t0) in list(active.items()):
                code = process.poll()
                if code is not None:
                    del active[device]
                    if code:
                        failed = True
            if failed and not active:
                raise RuntimeError("Campaign stopped after a failed run; incomplete blocks are not censored observations")
            time.sleep(poll)
    return dict(status="completed", runs=len(runs), launched=launched, skipped_complete=skipped)


def main():
    from flystudy.gates import code_hash
    from flystudy.graph import Graph
    from flystudy.protocol import Protocol, file_hash
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in ("protocol", "graph", "data", "tokenizer", "manifest", "matrix", "ledger", "output"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"]); p.add_argument("--microbatch", type=int, default=32)
    p.add_argument("--retries", help="JSON file {run_id: new reservation id} for explicitly recovered technical failures")
    a = p.parse_args()
    runs = json.loads(Path(a.matrix).read_text())["runs"]
    expected = dict(protocol_hash=Protocol.load(a.protocol).hash, graph_hash=Graph.load(a.graph).hash, tokenizer_hash=file_hash(a.tokenizer),
                    code_hash=code_hash(), dataset_hash=json.loads((Path(a.data) / "manifest.json").read_text())["dataset_hash"])
    output = Path(a.output)
    logs = {}

    def launch(run, device, folder, extra):
        command = [sys.executable, "-m", "flystudy", "run", "--protocol", str(a.protocol), "--graph", str(a.graph), "--data", str(a.data),
                   "--tokenizer", str(a.tokenizer), "--manifest", str(a.manifest), "--matrix", str(a.matrix), "--ledger", str(a.ledger),
                   "--output", str(folder), "--run-id", run["run_id"], "--device", device, "--microbatch", str(a.microbatch), *extra]
        name = run["run_id"] + (".console.log" if not extra else f".{extra[1]}.console.log")
        log = (output / name).open("w", encoding="utf-8"); logs[run["run_id"]] = log
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

    retries = json.loads(Path(a.retries).read_text()) if a.retries else {}
    try:
        print(json.dumps(campaign(runs, expected, output, a.devices, launch, retries=retries)), flush=True)
    finally:
        for log in logs.values():
            log.close()


if __name__ == "__main__":
    main()

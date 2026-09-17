"""Staggered campaign launcher (scripts only): identical scheduling to flystudy.campaign, except that a new run starts
only after every run launched before it has passed its start-up phase (metadata.json written) or exited. Two runs that
start in the same instant race on the dataset's audit.json.partial; the frozen package code is not changed."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("main_campaign_staggered", ROOT / "scripts" / "main_campaign_staggered.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

CONDITIONS = ["mono-en", "mono-de", "mono-ko", "seq-a", "seq-b", "seq-c", "seq-d", "seq-e", "seq-f", "mixed"]
EXPECTED = dict(protocol_hash="p", graph_hash="g", tokenizer_hash="t", code_hash="c", dataset_hash="d")


def block(seed):
    return [dict(run_id=f"main-real-{seed}-{c}", seed=seed) for c in CONDITIONS]


class FakeProcess:
    def __init__(self, folder, run_id, log, startup=.15, total=.3, code=0, summary=True):
        self.folder, self.run_id, self.log, self.code, self.summary = folder, run_id, log, code, summary
        self.t0, self.startup, self.total = time.time(), startup, total
        log.append(("launch", run_id, time.time()))

    def poll(self):
        age = time.time() - self.t0
        if age >= self.startup and self.code == 0 and not (self.folder / "metadata.json").exists():
            self.folder.mkdir(parents=True, exist_ok=True); (self.folder / "metadata.json").write_text("{}")
            self.log.append(("started", self.run_id, time.time()))
        if age < self.total:
            return None
        if self.code == 0 and self.summary and not (self.folder / "summary.json").exists():
            (self.folder / "summary.json").write_text(json.dumps(dict(run_id=self.run_id, stop_reason="mastered", **EXPECTED)))
        return self.code


def test_a_run_starts_only_after_earlier_runs_passed_startup(tmp_path):
    log = []
    launch = lambda run, device, folder, extra: FakeProcess(folder, run["run_id"], log)
    result = mod.campaign(block(1), EXPECTED, tmp_path, ["cuda:0", "cuda:1"], launch, poll=.01)
    assert result == dict(status="completed", runs=10, launched=10, skipped_complete=0)
    events = sorted(log, key=lambda e: e[2])
    launches = [e for e in events if e[0] == "launch"]
    for earlier, later in zip(launches, launches[1:]):
        started = next(e[2] for e in events if e[0] == "started" and e[1] == earlier[1])
        assert later[2] >= started, (earlier, later)  # never two runs inside the start-up phase together
    assert max(sum(1 for l in launches if l[2] <= t and t < l[2] + .3) for _, _, t in launches) == 2  # still two GPUs busy


def test_completed_runs_are_skipped_and_mismatched_ones_refused(tmp_path):
    runs = block(1)
    for r in runs[:4]:
        (tmp_path / r["run_id"]).mkdir(); (tmp_path / r["run_id"] / "summary.json").write_text(json.dumps(dict(run_id=r["run_id"], stop_reason="administrative_cap", **EXPECTED)))
    log = []
    result = mod.campaign(runs, EXPECTED, tmp_path, ["cuda:0", "cuda:1"], lambda run, device, folder, extra: FakeProcess(folder, run["run_id"], log, .02, .05), poll=.01)
    assert result["launched"] == 6 and result["skipped_complete"] == 4
    (tmp_path / runs[0]["run_id"] / "summary.json").write_text(json.dumps(dict(run_id=runs[0]["run_id"], stop_reason="mastered", **{**EXPECTED, "code_hash": "other"})))
    with pytest.raises(ValueError, match="does not match"):
        mod.campaign(runs, EXPECTED, tmp_path, ["cuda:0", "cuda:1"], lambda *a: None, poll=.01)


def test_failure_stops_the_campaign_after_active_runs_finish_and_retries_pass_a_reservation(tmp_path):
    log, extras = [], {}
    def launch(run, device, folder, extra):
        extras[run["run_id"]] = extra
        return FakeProcess(folder, run["run_id"], log, .02, .05, code=1 if run["run_id"].endswith("mono-ko") else 0)
    with pytest.raises(RuntimeError, match="failed run"):
        mod.campaign(block(1), EXPECTED, tmp_path, ["cuda:0", "cuda:1"], launch, poll=.01, retries={"main-real-1-mono-de": "main-real-1-mono-de-retry1"})
    assert extras["main-real-1-mono-de"] == ["--reservation-id", "main-real-1-mono-de-retry1"] and extras["main-real-1-mono-en"] == []
    launched = [e[1] for e in log if e[0] == "launch"]
    assert "main-real-1-mixed" not in launched and len(launched) <= 4  # nothing new starts after the failure


def test_incomplete_seed_blocks_are_refused(tmp_path):
    with pytest.raises(ValueError, match="10-condition"):
        mod.campaign(block(1)[:9], EXPECTED, tmp_path, ["cuda:0", "cuda:1"], lambda *a: None, poll=.01)

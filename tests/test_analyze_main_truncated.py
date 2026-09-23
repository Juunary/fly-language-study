"""Truncated main-study analysis (operator stop at 753/800 runs): the analysis set is the longest prefix of the
pre-registered seed order whose 10-condition blocks are complete; the pre-registered summary statistics run unchanged on
that prefix, and every other quantity is computed with cap-censoring (no exclusion of non-attainment). Synthetic runs only."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from flystudy.protocol import ORDERS, write_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_main_truncated", ROOT / "scripts" / "analyze_main_truncated.py")
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)

LANGS = ("en", "de", "ko"); TASKS = ("roles", "negation", "space", "quantity")
CELLS = [f"{l}/{t}" for l in LANGS for t in TASKS]
HASHES = dict(protocol_hash="p", code_hash="c", dataset_hash="d", graph_hash="g", tokenizer_hash="t", training_hash="tr", review_mode="claude_only", protocol_amendment="a")


def scheduled(seen, acc):
    return dict(kind="scheduled", seen=seen, scores={c: [int(round(acc.get(c, .5) * 1000)), 1000] for c in CELLS})


def write_run(root, seed, mode, order, first_global, stop="mastered", mono_first=None, events=None, test_acc=.95):
    rid = f"main-real-{seed}-" + ("mixed" if mode == "mixed" else ("mono-" if mode == "mono" else "seq-") + "-".join(order))
    d = root / rid; d.mkdir(parents=True)
    cap = 200000 if mode == "mono" else 900000
    stages = []
    if mode == "sequential":
        bounds = [0, 100000, 200000, 300000]
        stages = [dict(language=l, start=bounds[i], end=bounds[i + 1], exposures=100000, reason="mastered") for i, l in enumerate(order)]
    summary = dict(run_id=rid, cohort="main", synthetic=False, seed=seed, mode=mode, order=list(order), stop_reason=stop, seen=first_global or cap, cap=cap,
                   first_global=first_global, first_language={}, first_task=mono_first or {}, stages=stages, review_start=300000 if mode == "sequential" else None,
                   gpu_hours=0.1, device="cuda:0", protocol=dict(main_seeds=[1, 2, 3]), **HASHES)
    write_json(d / "summary.json", summary); write_json(d / "metadata.json", dict(device="cuda:0"))
    write_json(d / "independent-test.json", dict(scores={c: [int(test_acc * 1000), 1000] for c in CELLS}, auxiliary_scores={c: [400, 1000] for c in CELLS}, checkpoint_hash="x"))
    write_json(d / "terminal-auxiliary.json", dict(scores={"dev_a": {c: [300, 1000] for c in CELLS}, "dev_b": {c: [350, 1000] for c in CELLS}}, used_for_mastery=False))
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in (events or [])) + "\n")
    return rid


def build(tmp_path, seeds_complete=(1, 2), partial=3):
    root = tmp_path / "runs"; root.mkdir()
    for seed in seeds_complete:
        for i, order in enumerate(ORDERS):
            ev = ([scheduled(100000, {c: (.95 if c.startswith(order[0]) else .5) for c in CELLS}),   # end of stage 1
                   scheduled(150000, {c: (.60 if c.startswith(order[0]) else .5) for c in CELLS}),   # forgetting during stage 2
                   scheduled(300000, {c: (.70 if c.startswith(order[0]) else .5) for c in CELLS}),   # review start
                   scheduled(320000, {c: .9 for c in CELLS})])                                          # recovered
            write_run(root, seed, "sequential", order, 300000 + 20000 * i + seed * 1000, events=ev)
        for lang in LANGS:
            write_run(root, seed, "mono", (lang,), 150000 if lang != "de" else None, stop="mastered" if lang != "de" else "administrative_cap",
                      mono_first={f"{lang}/{t}": 120000 for t in TASKS} if lang != "de" else {"de/roles": 190000})
        write_run(root, seed, "mixed", LANGS, 400000)
    for lang in LANGS:  # partial block: only the monolingual runs exist
        write_run(root, partial, "mono", (lang,), 150000, mono_first={f"{lang}/{t}": 120000 for t in TASKS})
    manifest = dict(runs=[dict(run_id=f"main-real-{s}-{'mixed' if m == 'mixed' else ('mono-' if m == 'mono' else 'seq-') + '-'.join(o)}", seed=s, mode=m, order=list(o))
                          for s in (1, 2, 3) for m, o in [("mono", (l,)) for l in LANGS] + [("sequential", o) for o in ORDERS] + [("mixed", LANGS)]],
                    protocol=dict(main_seeds=[1, 2, 3]))
    write_json(tmp_path / "manifest.json", manifest)
    return root, tmp_path / "manifest.json"


def test_analysis_set_is_the_complete_prefix_and_exclusions_are_reported(tmp_path):
    root, manifest = build(tmp_path)
    records = mod.load_records(root, manifest)
    sel = mod.select_prefix(records, [1, 2, 3])
    assert sel["included_seeds"] == [1, 2] and sel["excluded"] == {"3": dict(reason="incomplete block", present=3, missing=7)}
    assert sel["rule"].startswith("longest prefix")


def test_prefix_stops_at_the_first_incomplete_block_even_if_later_blocks_are_complete(tmp_path):
    root, manifest = build(tmp_path, seeds_complete=(1, 3), partial=2)
    records = mod.load_records(root, manifest)
    sel = mod.select_prefix(records, [1, 2, 3])
    assert sel["included_seeds"] == [1] and set(sel["excluded"]) == {"2", "3"} and sel["excluded"]["3"]["reason"] == "after the first incomplete block"


def test_order_values_mono_costs_and_mixed_use_cap_censoring(tmp_path):
    root, manifest = build(tmp_path)
    records = mod.load_records(root, manifest); seeds = [1, 2]
    values = mod.order_values(records, seeds)
    assert values.shape == (2, 6) and values[0, 0] == 301000 and values[1, 5] == 402000
    mono = mod.mono_costs(records, seeds)
    assert mono["de/roles"]["restricted_costs"] == [190000, 190000] and mono["de/negation"]["restricted_costs"] == [200000, 200000] and mono["de/negation"]["censored"] == 2
    assert mono["en/all"]["restricted_costs"] == [200000, 200000]  # first_language missing -> cap, never excluded
    mixed = mod.mixed_summary(records, seeds); assert mixed["rmst"] == 400000 and mixed["events"] == 2
    per_order = mod.per_order(values, records, seeds)
    assert per_order["ko-de-en"]["rmst"] == pytest.approx(np.mean([values[0, 5], values[1, 5]])) and per_order["ko-de-en"]["events"] == 2


def test_forgetting_and_recovery_are_aligned_to_stage_end_and_review(tmp_path):
    root, manifest = build(tmp_path)
    records = mod.load_records(root, manifest)
    fr = mod.forgetting_recovery(records, [1, 2])
    first = fr["by_order"]["en-de-ko"]["first_language"]
    assert first["language"] == "en" and first["at_stage_end"]["mean"] == pytest.approx(.95) and first["min_after"]["mean"] == pytest.approx(.60)
    assert first["at_review_start"]["mean"] == pytest.approx(.70) and first["recovered_within_review"]["fraction"] == 1.0
    assert first["recovery_exposures"]["values"] == [20000, 20000]
    curve = fr["by_order"]["en-de-ko"]["first_language"]["curve"]
    assert curve[0] == dict(since_stage_end=0, mean=pytest.approx(.95), n=2) and any(p["since_stage_end"] == 50000 and abs(p["mean"] - .6) < 1e-9 for p in curve)


def test_independent_test_aggregates_primary_and_auxiliary_separately(tmp_path):
    root, manifest = build(tmp_path)
    records = mod.load_records(root, manifest)
    it = mod.independent_test(records, [1, 2])
    assert it["primary"]["mixed"]["cells"]["en/roles"]["mean"] == pytest.approx(.95) and it["primary"]["mixed"]["cells"]["en/roles"]["n"] == 2
    assert it["auxiliary"]["mixed"]["test"]["en/roles"]["mean"] == pytest.approx(.4) and it["auxiliary"]["mixed"]["dev_a"]["en/roles"]["mean"] == pytest.approx(.3)
    assert it["auxiliary_note"]


def test_budget_counts_ledger_status_and_failed_attempts(tmp_path):
    root, manifest = build(tmp_path)
    records = mod.load_records(root, manifest)
    ledger = dict(allocations=dict(main=448), runs={"main-real-1-mixed": dict(reserved=1.0, used=.4, status="completed"),
                                                    "main-real-9-mixed": dict(reserved=1.0, used=0, status="reserved"),
                                                    "main-real-2-mono-de": dict(reserved=.1, used=.01, status="technical_failure"),
                                                    "g0-x": dict(reserved=4, used=.007, status="completed")}, transfers=[])
    write_json(tmp_path / "ledger.json", ledger)
    failed = tmp_path / "failed"; (failed / "main-real-2-mono-de-attempt1").mkdir(parents=True)
    write_json(failed / "main-real-2-mono-de-attempt1" / "failure.json", dict(error="RuntimeError('x')", seen=100, status="technical_failure"))
    write_json(failed / "main-real-2-mono-de-attempt1" / "attempt-main-real-2-mono-de.json", dict(reservation_id="main-real-2-mono-de", gpu_hours=.01))
    b = mod.budget_and_failures(records, tmp_path / "ledger.json", failed)
    assert b["completed_runs"] == 23 and b["gpu_hours_completed_runs"] == pytest.approx(2.3)
    assert b["ledger"]["by_status"]["reserved"] == dict(count=1, used_hours=0, reserved_hours=1.0) and b["ledger"]["main_entries"] == 3
    assert b["failed_attempts"][0]["failure"]["seen"] == 100 and b["failed_attempts"][0]["attempt"]["gpu_hours"] == .01
    assert b["missing_runs"] == [f"main-real-3-{n}" for n in ("seq-en-de-ko", "seq-en-ko-de", "seq-de-en-ko", "seq-de-ko-en", "seq-ko-en-de", "seq-ko-de-en", "mixed")]


def test_end_to_end_analysis_refuses_to_overwrite_and_writes_report(tmp_path):
    root, manifest = build(tmp_path)
    out = tmp_path / "out"
    report = mod.analyze(root, manifest, out, figures=False)
    assert report["selection"]["included_seeds"] == [1, 2] and report["prespecified_summary"]["n_seeds"] == 2
    assert report["deviation"]["analyzed_n"] == 2 and report["deviation"]["pre_registered_n"] == 3 and (out / "analysis.json").exists()
    with pytest.raises(FileExistsError):
        mod.analyze(root, manifest, out, figures=False)


def test_records_with_mixed_hashes_are_refused(tmp_path):
    root, manifest = build(tmp_path)
    s = root / "main-real-1-mixed" / "summary.json"; d = json.loads(s.read_text()); d["code_hash"] = "other"; s.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Incompatible"):
        mod.load_records(root, manifest)

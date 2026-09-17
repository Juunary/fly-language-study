"""FWER evaluation amendment (design version v5.1-extended-N-2): the null false-positive bound of the extended design
decision comes from one official 50,000-repetition simulation of every rho-0 null scenario (both targets), while power
rows are reused from the 5,000-repetition extended grid. Tests use tiny synthetic inputs; nothing here runs the official
simulation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flystudy import statistics
from flystudy.protocol import write_json

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


official, design = load("fwer_official_verification"), load("extended_design_decision")
from test_extended_design import setup  # noqa: E402  (synthetic grid/pilot/cost fixtures)

NULL = dict(cv=.5, latent_rho=0., censoring=.3, family="weibull", interval=5120, effect=0.)


def test_null_scenarios_generate_the_same_distribution_for_both_targets():
    a = statistics.simulate({**NULL, "target": 0}, repetitions=400, seed=77, ns=(20, 40))
    b = statistics.simulate({**NULL, "target": 1}, repetitions=400, seed=77, ns=(20, 40))
    for x, y in zip(a, b):  # same seed, only the target label differs: every simulated quantity is identical
        assert {k: v for k, v in x.items() if k != "target"} == {k: v for k, v in y.items() if k != "target"}
        assert x["calibrated_scale_factor"] == 1.0 and x["rate_kind"] == "familywise_type_I_error"


def test_null_rate_counts_a_false_rejection_of_either_contrast(monkeypatch):
    def fake_tests(values):
        reps = values.shape[0]
        adjusted = np.ones((reps, 2)); adjusted[: reps // 10, 0] = .01; adjusted[reps // 10: reps // 10 + reps // 20, 1] = .01
        return None, None, np.ones((reps, 2)), adjusted
    monkeypatch.setattr(statistics, "paired_tests", fake_tests)
    null = statistics.simulate({**NULL, "target": 0}, repetitions=200, seed=5, ns=(20,))[0]
    assert null["rate"] == pytest.approx(.15)  # 10% only contrast 1 + 5% only contrast 2, disjoint
    powered = statistics.simulate({**NULL, "effect": .2, "target": 1}, repetitions=200, seed=5, ns=(20,))[0]
    assert powered["rate"] == pytest.approx(.05)  # with an effect only the targeted contrast counts


def test_official_scenarios_are_every_rho0_null_row_of_the_extended_grid_with_fixed_seeds():
    grid = load("extended_power_grid")
    items = official.scenarios()
    assert len(items) == len(grid.CVS) * len(grid.CENSORING) * len(grid.FAMILIES) * 2 == 108
    assert all(s["effect"] == 0 and s["latent_rho"] == 0 and s["interval"] == 5120 for _, s in items)
    assert {(s["cv"], s["censoring"], s["family"], s["target"]) for _, s in items} == {
        (cv, c, f, t) for cv in grid.CVS for c in grid.CENSORING for f in grid.FAMILIES for t in (0, 1)}
    assert [i for i, _ in items] == list(range(108)) and official.REPETITIONS == 50000 and official.NS == grid.NS
    assert official.SEED_BASE not in (grid.SEED_BASE, 19401, 8401) and official.DESIGN_VERSION == "v5.1-extended-N-2"


def test_official_run_is_single_shot(tmp_path):
    out = tmp_path / "fwer.jsonl"; out.write_text("")
    with pytest.raises(FileExistsError):
        official.run(out, workers=1)
    (tmp_path / "other.jsonl.started").write_text("x")
    with pytest.raises(FileExistsError):  # a started marker also blocks: no second attempt after an interrupted or failed run
        official.run(tmp_path / "other.jsonl", workers=1)


def fwer_report(tmp_path, grid_path, rate=.051, repetitions=50000, drop=None, complete=True):
    rows = []
    for r in (json.loads(l) for l in Path(grid_path).read_text().splitlines()):
        if r["effect"] == 0 and r["latent_rho"] == 0 and (drop is None or not drop(r)):
            rows.append({**r, "repetitions": repetitions, "rate": rate, "mc_ci": [rate - .002, rate + .002], "cv": r["reference_restricted_cv"]})
    path = tmp_path / "fwer.jsonl"; path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    write_json(path.with_suffix(".meta.json"), dict(design_version="v5.1-extended-N-2", complete=complete, repetitions=repetitions, records=len(rows)))
    return path


def test_decision_takes_the_false_positive_bound_from_the_official_report(tmp_path):
    args = setup(tmp_path, {20: .5, 40: .9, 60: .95})
    grid = args[0]  # the synthetic grid's own 5,000-repetition null bound is 0.07 here: blocked without the official report
    rows = [json.loads(l) for l in grid.read_text().splitlines()]
    for r in rows:
        if r["effect"] == 0: r["mc_ci"] = [.06, .07]
        r["cv"] = r["reference_restricted_cv"]
    grid.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    blocked = design.decide(*args, tmp_path / "old.json")
    assert blocked["status"] == "blocked" and blocked["design_version"] == "v5.1-extended-N-1"
    report = fwer_report(tmp_path, grid)
    r = design.decide(*args, tmp_path / "new.json", fwer_report=report)
    assert r["status"] == "passed" and r["design_version"] == "v5.1-extended-N-2" and r["minimum_passing_n"] == 40
    assert r["chosen"]["null_fwer_upper"] == pytest.approx(.053) and r["primary_power_table"]["40"]["power_lower"] == pytest.approx(.85)
    assert r["fwer_evaluation"]["repetitions"] == 50000 and r["fwer_evaluation"]["grid_null_bound_not_used"]["40"] == pytest.approx(.07)
    assert r["fwer_evaluation"]["envelope_null_rows_per_n"]["40"] == r["fwer_evaluation"]["official_rows_matched_per_n"]["40"]


def test_official_report_must_cover_every_envelope_row_at_50000_repetitions(tmp_path):
    args = setup(tmp_path, {40: .9})
    grid = args[0]
    rows = [json.loads(l) for l in grid.read_text().splitlines()]
    for r in rows: r["cv"] = r["reference_restricted_cv"]
    grid.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="missing"):
        design.decide(*args, tmp_path / "a.json", fwer_report=fwer_report(tmp_path, grid, drop=lambda r: r["family"] == "weibull" and r["target"] == 1))
    with pytest.raises(ValueError, match="50,000"):
        design.decide(*args, tmp_path / "b.json", fwer_report=fwer_report(tmp_path, grid, repetitions=5000))
    with pytest.raises(ValueError, match="complete"):
        design.decide(*args, tmp_path / "c.json", fwer_report=fwer_report(tmp_path, grid, complete=False))

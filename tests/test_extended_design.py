"""Extended-N design decision and pilot dispersion report (scripts only; synthetic inputs in tmp_path)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flystudy.protocol import ORDERS, file_hash, write_json
from flystudy.statistics import CONTRASTS

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


design, dispersion = load("extended_design_decision"), load("pilot_dispersion_report")
CONDITIONS = sorted(design.CONDITIONS)


def grid_rows(ns, power_by_n, ref_cvs=(0.3, 0.5, 0.7), fwer=0.05):
    rows = []
    for n in ns:
        for family in ("lognormal", "weibull"):
            for censoring in (0.1, 0.3, 0.5):
                for rho in (0.0, 0.5):
                    for i, cv in enumerate(ref_cvs):
                        for target in (0, 1):
                            for effect in (0.0, 0.2):
                                p = power_by_n[n] - 0.05 * i + (0.1 if rho else 0)
                                rows.append(dict(n=n, interval=5120, family=family, censoring=censoring, latent_rho=rho, target=target,
                                                 effect=effect, reference_restricted_cv=cv,
                                                 conservative_power_ci=None if effect == 0 else [p, p + .01], mc_ci=[fwer - .01, fwer]))
    return rows


def setup(tmp_path, power_by_n, cv_bound=0.45, pilot_status="passed", block_hours=4.0, censor_upper=0.18, ref_cvs=(0.3, 0.5, 0.7)):
    grid = tmp_path / "grid.jsonl"
    grid.write_text("\n".join(json.dumps(r) for r in grid_rows(sorted(power_by_n), power_by_n, ref_cvs)) + "\n")
    write_json(grid.with_suffix(".meta.json"), dict(complete=True, quick=False, repetitions=5000, interval=5120))
    groups = {"sequential/" + "-".join(o): dict(censor_upper_95=censor_upper) for o in ORDERS}
    write_json(tmp_path / "pilot.json", dict(status=pilot_status, groups=groups, dispersion_uncertainty=dict(restricted_cv_upper=cv_bound)))
    per_run = {c: block_hours / 10 for c in CONDITIONS}
    write_json(tmp_path / "cost.json", dict(intervals={"5120": dict(per_run_upper_hours=per_run)}, serial_overhead_hours=2, remaining_calendar_hours=336))
    write_json(tmp_path / "res.json", dict(intervals={"5120": dict(status="passed", actual_sequence_pilot=True)}))
    write_json(tmp_path / "disp.json", dict(contrasts={"c": dict(analytic_required_n_20pct=dict(point=30))},
                                            correlation=dict(sequential_orders=dict(mean_offdiagonal_bootstrap_95=[0.52, 0.8]))))
    write_json(tmp_path / "ledger.json", dict(allocations=dict(main=448, review=16, shuffle=24, tokenizer=24), runs={}))
    (tmp_path / "prereg.md").write_text("rules")
    return [tmp_path / n for n in ("grid.jsonl", "pilot.json", "cost.json", "res.json", "disp.json", "prereg.md", "ledger.json")]


def test_largest_affordable_passing_n_is_chosen_with_freeze_fields(tmp_path):
    args = setup(tmp_path, {20: .5, 40: .9, 60: .95, 100: .99})
    r = design.decide(*args, tmp_path / "out.json")
    assert r["status"] == "passed" and r["minimum_passing_n"] == 40
    assert r["chosen"]["n"] == 60  # 100 x 4.0 h x 1.25 = 500 > 448; 60 x 4 x 1.25 = 300 fits
    for key in ("n", "eval_interval", "per_run_upper_hours", "power_lower", "null_fwer_upper", "gpu_hours"):
        assert key in r["chosen"]
    assert r["main_seeds"] == list(range(30001, 30061)) and r["auxiliary_release_required_hours"] == 0
    assert r["selection_uses_observed_effect"] is False and r["sensitivity_correlation_informed"]["latent_rho"] == 0.5
    assert r["primary_power_table"]["40"]["power_lower"] == pytest.approx(.85)  # rows up to edge CV 0.5 (index 1): 0.9 - 0.05


def test_auxiliary_released_only_when_main_alone_cannot_fit(tmp_path):
    args = setup(tmp_path, {100: .95}, block_hours=3.8)  # 100 x 3.8 x 1.25 = 475 > 448, <= 512
    r = design.decide(*args, tmp_path / "out.json")
    assert r["status"] == "passed" and r["chosen"]["n"] == 100 and r["auxiliary_release_required_hours"] == pytest.approx(27)


def test_dispersion_bound_above_grid_blocks(tmp_path):
    args = setup(tmp_path, {20: .9, 40: .99}, cv_bound=0.9)
    r = design.decide(*args, tmp_path / "out.json")
    assert r["status"] == "blocked" and "outside_grid" in r["primary_power_table"]["20"]


def test_incomplete_pilot_reports_table_but_blocks(tmp_path):
    args = setup(tmp_path, {20: .5, 40: .9}, pilot_status="incomplete")
    r = design.decide(*args, tmp_path / "out.json")
    assert r["status"] == "blocked" and "pilot" in r["reason"] and r["minimum_passing_n"] == 40


def test_underpowered_everywhere_blocks_with_reason(tmp_path):
    args = setup(tmp_path, {20: .3, 40: .5, 100: .7})
    r = design.decide(*args, tmp_path / "out.json")
    assert r["status"] == "blocked" and "power" in r["reason"] and r["minimum_passing_n"] is None


def test_dispersion_report_recovers_contrast_sd_and_correlation(tmp_path):
    rng = np.random.default_rng(3)
    n, rho, sd, mean = 400, 0.6, 50000., 400000.
    z = np.sqrt(rho) * rng.normal(size=(n, 1)) + np.sqrt(1 - rho) * rng.normal(size=(n, 6))
    seq = mean + sd * z
    for k, coef in enumerate(CONTRASTS):
        expected = sd * np.sqrt((1 - rho) * (coef ** 2).sum()) / mean
        assert abs((seq @ coef).std(ddof=1) / seq.mean() - expected) < 0.01
    assert abs(dispersion.mean_offdiag(seq) - rho) < 0.05
    assert dispersion.required_n(0.2) < dispersion.required_n(0.4)

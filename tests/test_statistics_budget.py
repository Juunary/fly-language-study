import json
import numpy as np
import pytest
from flystudy.statistics import holm, summarize, restricted_times, simulate, exact_censor_upper, CONTRASTS
from flystudy.budget import Ledger, choose_design
from flystudy.protocol import BUDGETS


def test_holm_and_contrast_coefficients():
    np.testing.assert_allclose(holm([.04,.01]),[.04,.02])
    np.testing.assert_allclose(CONTRASTS.sum(1),0)
    np.testing.assert_allclose(CONTRASTS[0] @ CONTRASTS[1],0)


def test_all_censored_does_not_produce_equivalence_ci():
    result = summarize(np.full((10,6),900000.),bootstrap=100)
    assert all(v == 900000 for v in result["rmst_by_order"].values())
    for c in result["contrasts"].values():
        assert c["simultaneous_ci"] is None and c["holm_p"] == 1 and not c["informative"]


def test_confirmation_at_nonmultiple_cap():
    values,events = restricted_times(np.array([80,99,101]),20,105)
    np.testing.assert_array_equal(values,[100,105,105])
    np.testing.assert_array_equal(events,[True,True,False])


def test_simulation_calibrates_rmst_not_raw_scaling():
    s = dict(cv=.5,latent_rho=.5,censoring=.3,family="lognormal",interval=10240,target=0,effect=.2)
    results = simulate(s,repetitions=100,ns=(10,),seed=99)
    assert abs(results[0]["calibrated_relative_rmst_effect"]-.2) < .0001
    assert results[0]["calibrated_scale_factor"] != pytest.approx(.8,abs=.001)
    assert results[0]["mc_ci"][0] <= results[0]["rate"] <= results[0]["mc_ci"][1]


def test_two_successes_leave_large_censor_uncertainty():
    assert exact_censor_upper(0,2) == pytest.approx(.7763932)


def test_ledger_atomic_bundle_and_no_double_charge(tmp_path):
    ledger = Ledger(tmp_path/"ledger.json")
    with pytest.raises(ValueError):
        ledger.reserve_bundle([dict(run_id="a",category="review",hours=10),dict(run_id="b",category="review",hours=10)])
    ledger.reserve_bundle([dict(run_id="a",category="main",hours=1)])
    ledger.charge("a",.7,"completed")
    with pytest.raises(ValueError): ledger.charge("a",.7,"completed")
    assert json.loads((tmp_path/"ledger.json").read_text())["runs"]["a"]["used"] == .7


def test_auxiliary_is_cut_before_seed_count(tmp_path):
    assert sum(BUDGETS.values()) == 672
    ledger = Ledger(tmp_path/"ledger.json")
    with pytest.raises(ValueError): ledger.release_auxiliary("shuffle")
    for category in ("review","shuffle","tokenizer"): ledger.release_auxiliary(category)
    data = json.loads((tmp_path/"ledger.json").read_text())
    assert data["allocations"]["main"] == 512 and sum(data["allocations"].values()) == 672


def test_design_lexicographic_selection():
    base = dict(measurement_passed=True,power_lower=.85,null_fwer_upper=.055,gpu_hours=300,available_main_hours=448,
                calendar_hours=150,available_calendar_hours=300)
    candidates = [{**base,"n":15,"eval_interval":5120},{**base,"n":20,"eval_interval":10240},
                  {**base,"n":20,"eval_interval":5120,"power_lower":.7}]
    assert choose_design(candidates)["eval_interval"] == 10240

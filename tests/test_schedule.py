from dataclasses import replace
import pytest
from flystudy.protocol import Protocol, LANGUAGES, TASKS, run_matrix
from flystudy.schedule import Curriculum


def protocol(**changes):
    return replace(Protocol(), effective_batch=4, eval_interval=8, mono_cap=20, total_cap=64, panel_per_task=10, **changes)


def scores(c, bad=()):
    return {f"{lang}/{task}": [7 if f"{lang}/{task}" in bad else 8, 10] for lang in c.languages for task in TASKS}


def reach(c, time):
    while c.seen < time:
        c.train_exposures(min(c.batch_size(), time-c.seen))


def test_two_distinct_full_panels_required():
    c = Curriculum(protocol(), "mono", ("en",))
    reach(c, 8); c.evaluate(scores(c), "dev_a")
    assert not c.stop_reason
    reach(c, 16); c.evaluate(scores(c), "dev_b")
    assert c.first_global == 16 and c.stop_reason == "mastered"


def test_average_cannot_hide_failed_task():
    c = Curriculum(protocol(), "mono", ("en",))
    for time in (8,16,20):
        reach(c,time); c.evaluate(scores(c, ("en/quantity",)), c.next_panel)
    assert c.stop_reason == "administrative_cap" and c.first_global is None
    assert c.first_task["en/roles"] == 16


def test_diagnostics_do_not_confirm_or_reset():
    c = Curriculum(protocol(), "mono", ("en",))
    before = c.checkpoint()
    c.evaluate({}, "any", kind="diagnostic")
    assert c.checkpoint() == before


def test_partial_wrong_panel_and_early_eval_rejected():
    c = Curriculum(protocol(), "mixed", LANGUAGES)
    with pytest.raises(ValueError): c.evaluate(scores(c), "dev_a")
    reach(c,8)
    with pytest.raises(ValueError): c.evaluate(scores(c), "dev_b")
    missing = scores(c); missing.pop("de/space")
    with pytest.raises(ValueError): c.evaluate(missing,"dev_a")
    partial = scores(c); partial["de/space"] = [8,9]
    with pytest.raises(ValueError): c.evaluate(partial,"dev_a")


def test_same_two_times_for_all_twelve_cells():
    c = Curriculum(protocol(), "mixed", LANGUAGES)
    for time, bad in ((8,("ko/quantity",)), (16,()), (24,("en/roles",)), (32,()), (40,())):
        reach(c,time); c.evaluate(scores(c,bad),c.next_panel)
        if time < 40: assert c.first_global is None
    assert c.first_global == 40


def test_phase_cap_is_not_terminal_censoring():
    c = Curriculum(protocol(),"sequential",LANGUAGES)
    for time in (8,16):
        reach(c,time); c.evaluate(scores(c,tuple(scores(c))),c.next_panel)
    reach(c,20); c.advance()
    assert c.stage == 1 and c.stage_start == 20 and c.stop_reason is None
    assert c.stages[0]["reason"] == "stage_cap"


def test_zero_exposure_skip_uses_current_confirmation():
    c = Curriculum(protocol(),"sequential",LANGUAGES)
    for time in (8,16):
        reach(c,time); c.evaluate(scores(c,("ko/space",)),c.next_panel)
    assert c.stage == 2
    assert c.stages[1]["language"] == "de" and c.stages[1]["exposures"] == 0


def test_restore_and_batch_does_not_cross_boundary():
    c = Curriculum(protocol(),"sequential",LANGUAGES)
    with pytest.raises(ValueError): c.train_exposures(9)
    reach(c,8); c.evaluate(scores(c,("en/roles",)),c.next_panel)
    restored = Curriculum.restore(c.protocol,c.checkpoint())
    assert restored.checkpoint() == c.checkpoint()
    assert restored.next_panel == "dev_b"


def test_manifest_keeps_all_orders_and_seeds():
    runs = list(run_matrix(range(1,21)))
    assert len(runs) == 200 and len({r["run_id"] for r in runs}) == 200
    assert sum(r["mode"] == "sequential" for r in runs) == 120

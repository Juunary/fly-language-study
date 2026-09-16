"""Primary (same-frame held-out meanings) and auxiliary (outer-frame transfer) renderings of the same items."""
import json
import shutil

import pytest
from tokenizers import Tokenizer

from flystudy.data import (FAMILIES, PRIMARY_FAMILY, SPLITS, audit, generate, load_rows, render,
                           require_confirmatory_test, revise_evaluation, train_tokenizer)
from flystudy.graph import synthetic
from flystudy.protocol import LANGUAGES, Protocol, digest, file_hash, write_json
from flystudy.runtime import Corpus
from flystudy.train import run

EVAL_SPLITS = ("dev_a", "dev_b", "test")


@pytest.fixture(scope="module")
def corpus_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("views")
    generate(root/"data", 18, 2)
    train_tokenizer(root/"data", root/"tokenizer.json", 320)
    synthetic().save(root/"graph.npz")
    return root


def test_evaluation_rows_carry_primary_and_auxiliary_renderings(corpus_files):
    for split in EVAL_SPLITS:
        rows = load_rows(corpus_files/"data", split)
        assert rows
        for r in rows:
            assert r["template_family"] == PRIMARY_FAMILY == "assertion"
            assert r["sentence_a"] == render(r["scene"], r["language"], "train")
            assert r["sentence_b"] == render(r["scene"], r["language"], "train", inverse=True, foil=not r["label"])
            assert r["auxiliary"]["template_family"] == FAMILIES[split] != PRIMARY_FAMILY
            assert r["auxiliary"]["sentence_a"] == render(r["scene"], r["language"], split)
            assert r["auxiliary"]["sentence_b"] == render(r["scene"], r["language"], split, inverse=True, foil=not r["label"])
            assert r["auxiliary"]["sentence_a"] != r["sentence_a"]
    assert all("auxiliary" not in r for r in load_rows(corpus_files/"data", "train"))


def test_manifest_declares_two_renderings_of_the_same_items(corpus_files):
    manifest = json.loads((corpus_files/"data"/"manifest.json").read_text(encoding="utf-8"))
    evaluation = manifest["evaluation"]
    assert evaluation["primary"]["template_family"] == "assertion"
    assert evaluation["primary"]["shared_with_training"] is True
    assert evaluation["auxiliary"]["same_items_as_primary"] is True
    assert evaluation["auxiliary"]["template_families"] == {s: FAMILIES[s] for s in EVAL_SPLITS}
    assert manifest["independent_test"]["confirmatory"] is True


def test_audit_passes_and_reports_view_checks(corpus_files):
    report = audit(corpus_files/"data")
    assert report["passed"], report["errors"]
    checks = report["view_checks"]
    assert checks["auxiliary_rows"] == 2*3*4*3 and checks["primary_rerendered"] == checks["auxiliary_rows"]
    assert checks["auxiliary_rerendered"] == checks["auxiliary_rows"]
    assert checks["primary_pairs_in_train"] == 0


def test_audit_detects_auxiliary_primary_mismatch(corpus_files, tmp_path):
    target = tmp_path/"copy"
    shutil.copytree(corpus_files/"data", target)
    rows = load_rows(target, "dev_a")
    other = next(r for r in rows if r["language"] == rows[0]["language"] and r["meaning_id"] != rows[0]["meaning_id"])
    rows[0]["auxiliary"]["sentence_a"] = other["auxiliary"]["sentence_a"]
    with (target/"dev_a.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    report = audit(target)
    assert not report["passed"] and "auxiliary/primary mismatch" in report["errors"]


def test_audit_detects_primary_sentence_leak_into_train(corpus_files, tmp_path):
    target = tmp_path/"copy"
    shutil.copytree(corpus_files/"data", target)
    held = load_rows(target, "dev_b")[0]
    leaked = {k: v for k, v in held.items() if k != "auxiliary"}
    leaked.update(id="fresh:0:"+held["language"], meaning_id="fresh", foil_group="fresh", split="train")
    with (target/"train.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(leaked, ensure_ascii=False)+"\n")
    report = audit(target)
    assert not report["passed"] and "primary sentence leakage" in report["errors"]


def old_format_copy(source, target):
    """Rewrite a new-format dataset the way draft-v4.3 stored it: held-out frames in sentence_a/b, no auxiliary."""
    target.mkdir()
    files = {}
    for split in SPLITS:
        with (target/f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in load_rows(source, split):
                if "auxiliary" in r:
                    aux = r.pop("auxiliary")
                    r.update(template_family=aux["template_family"], sentence_a=aux["sentence_a"], sentence_b=aux["sentence_b"])
                f.write(json.dumps(r, ensure_ascii=False)+"\n")
        files[f"{split}.jsonl"] = file_hash(target/f"{split}.jsonl")
    new = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    write_json(target/"manifest.json", dict(version="old", seed=new["seed"], files=files, train_per_task=new["train_per_task"],
                                            panel_per_task=new["panel_per_task"], human_review="pending", families=FAMILIES,
                                            note="old", dataset_hash=digest(files)))


def test_revise_evaluation_reproduces_generated_dataset_and_keeps_training_bytes(corpus_files, tmp_path):
    old = tmp_path/"old"
    old_format_copy(corpus_files/"data", old)
    report = revise_evaluation(old, tmp_path/"revised", test_used_in_exploration=True)
    assert report["passed"], report["errors"]
    for split in SPLITS:
        assert file_hash(tmp_path/"revised"/f"{split}.jsonl") == file_hash(corpus_files/"data"/f"{split}.jsonl")
    assert file_hash(tmp_path/"revised"/"train.jsonl") == file_hash(old/"train.jsonl")
    manifest = json.loads((tmp_path/"revised"/"manifest.json").read_text(encoding="utf-8"))
    generated = json.loads((corpus_files/"data"/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_hash"] == generated["dataset_hash"]
    assert manifest["derived_from"]["dataset_hash"] == digest({f"{s}.jsonl": file_hash(old/f"{s}.jsonl") for s in SPLITS})
    assert manifest["independent_test"]["confirmatory"] is False


def test_revise_evaluation_refuses_to_overwrite(corpus_files, tmp_path):
    with pytest.raises(FileExistsError):
        revise_evaluation(corpus_files/"data", corpus_files/"data")


def test_corpus_views_encode_the_requested_rendering(corpus_files):
    tokenizer = Tokenizer.from_file(str(corpus_files/"tokenizer.json"))
    corpus = Corpus(corpus_files/"data", tokenizer)
    primary = corpus.split("dev_a")["en"]
    auxiliary = corpus.split("dev_a", view="auxiliary")["en"]
    assert [r["id"] for r, _ in primary] == [r["id"] for r, _ in auxiliary]
    assert all(ids == tokenizer.encode(r["sentence_a"]).ids + [tokenizer.token_to_id("[SEP]")]
               + tokenizer.encode(r["sentence_b"]).ids + [tokenizer.token_to_id("[EOS]")] for r, ids in primary)
    assert all(ids == tokenizer.encode(r["auxiliary"]["sentence_a"]).ids + [tokenizer.token_to_id("[SEP]")]
               + tokenizer.encode(r["auxiliary"]["sentence_b"]).ids + [tokenizer.token_to_id("[EOS]")] for r, ids in auxiliary)
    assert corpus.has_auxiliary("dev_a") and not corpus.has_auxiliary("train")
    with pytest.raises(ValueError, match="auxiliary"):
        corpus.split("train", view="auxiliary")


def test_training_scores_auxiliary_panels_without_touching_mastery(corpus_files, tmp_path):
    p = Protocol(vocab_size=320, embed_dim=8, effective_batch=4, eval_interval=8, mono_cap=12, total_cap=32, panel_per_task=2)
    summary = run(p, corpus_files/"graph.npz", corpus_files/"data", corpus_files/"tokenizer.json", tmp_path/"run",
                  "mono", ("en",), 7, cohort="smoke", device="cpu", backend="dense", microbatch=3, smoke=True, auxiliary_panels=True)
    events = [json.loads(l) for l in (tmp_path/"run"/"events.jsonl").read_text(encoding="utf-8").splitlines()]
    scheduled = [e for e in events if e["kind"] == "scheduled"]
    auxiliary = [e for e in events if e["kind"] == "auxiliary"]
    assert scheduled and [(e["seen"], e["panel"]) for e in auxiliary] == [(e["seen"], e["panel"]) for e in scheduled]
    assert all(set(e["scores"]) == set(s["scores"]) and "first_task" not in e for e, s in zip(auxiliary, scheduled))
    assert summary["clocks_seconds"]["auxiliary"] >= 0
    test = json.loads((tmp_path/"run"/"independent-test.json").read_text(encoding="utf-8"))
    assert set(test["auxiliary_scores"]) == set(test["scores"])


def test_non_confirmatory_test_split_blocks_study_cohorts(corpus_files, tmp_path):
    assert require_confirmatory_test(dict(independent_test=dict(confirmatory=True))) is None
    assert require_confirmatory_test(dict()) is None
    with pytest.raises(ValueError, match="confirmatory"):
        require_confirmatory_test(dict(independent_test=dict(confirmatory=False, reason="used in exploration")))
    target = tmp_path/"data"
    shutil.copytree(corpus_files/"data", target)
    manifest = json.loads((target/"manifest.json").read_text(encoding="utf-8"))
    manifest["independent_test"] = dict(confirmatory=False, reason="used in exploration")
    write_json(target/"manifest.json", manifest)
    with pytest.raises(ValueError, match="confirmatory"):
        run(Protocol(), corpus_files/"graph.npz", target, corpus_files/"tokenizer.json", tmp_path/"pilot",
            "mono", ("en",), 10001, cohort="pilot", device="cpu", backend="dense")
    summary = run(Protocol(vocab_size=320, embed_dim=8, effective_batch=4, eval_interval=8, mono_cap=12, total_cap=32, panel_per_task=2),
                  corpus_files/"graph.npz", target, corpus_files/"tokenizer.json", tmp_path/"smoke", "mono", ("en",), 7,
                  cohort="smoke", device="cpu", backend="dense", microbatch=3, smoke=True)
    assert summary["stop_reason"] == "administrative_cap"


def test_cli_revise_evaluation_writes_new_version(corpus_files, tmp_path, capsys):
    from flystudy.cli import main
    old = tmp_path/"old"
    old_format_copy(corpus_files/"data", old)
    assert main(["revise-evaluation", "--source", str(old), "--output", str(tmp_path/"v2")]) == 0
    manifest = json.loads((tmp_path/"v2"/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["derived_from"]["version"] == "old" and manifest["independent_test"]["confirmatory"] is False
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_smoke_run_can_skip_independent_test_and_auxiliary_panels(corpus_files, tmp_path):
    p = Protocol(vocab_size=320, embed_dim=8, effective_batch=4, eval_interval=8, mono_cap=12, total_cap=32, panel_per_task=2)
    summary = run(p, corpus_files/"graph.npz", corpus_files/"data", corpus_files/"tokenizer.json", tmp_path/"run",
                  "mono", ("en",), 7, cohort="smoke", device="cpu", backend="dense", microbatch=3, smoke=True,
                  independent_test=False, auxiliary_panels=False)
    assert summary["stop_reason"] == "administrative_cap" and summary["independent_test"] == "skipped_exploratory"
    assert (tmp_path/"run"/"primary.pt").exists() and not (tmp_path/"run"/"independent-test.json").exists()
    events = [json.loads(l) for l in (tmp_path/"run"/"events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "scheduled" for e in events) and not any(e["kind"] == "auxiliary" for e in events)
    assert summary["clocks_seconds"]["test"] == 0 and (tmp_path/"run"/"terminal-auxiliary.json").exists()

"""Protocol v5 revision: word-boundary vocabulary 841 is the fixed primary-model input; a new confirmatory test split
comes from unused semantic orbits; auxiliary (outer-frame) evaluation happens only at the terminal checkpoint."""
import json

import pytest
from tokenizers import Tokenizer

from flystudy.data import EVALUATION_SPLITS, FAMILIES, SPLITS, audit, generate, load_rows, new_test_split, train_tokenizer
from flystudy.graph import synthetic
from flystudy.protocol import Protocol, file_hash
from flystudy.train import run


@pytest.fixture(scope="module")
def corpus_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("v5")
    generate(root/"data", 18, 2)
    train_tokenizer(root/"data", root/"tokenizer.json", 320)
    synthetic().save(root/"graph.npz")
    return root


def test_primary_model_fixes_the_word_boundary_vocabulary():
    # Protocol v5.1: the word-boundary tokenizer of data/draft-v4.6 has 832 actual tokens (v5.0 on draft-v4.5 had 841).
    assert Protocol().version == "v5.1-wordbound" and Protocol().vocab_size == 832 and Protocol().validate_primary_model()
    for stale in (4096, 841):
        with pytest.raises(ValueError, match="protocol v5"):
            Protocol(vocab_size=stale).validate_primary_model()
    assert Protocol.load("configs/protocol-v5.1.json").vocab_size == 832
    assert Protocol.load("configs/protocol-v5.json").vocab_size == 841  # historical configs keep their recorded hash
    assert Protocol.load("configs/protocol-v4-ai.json").vocab_size == 4096


def test_new_test_split_uses_unused_orbits_and_keeps_other_splits(corpus_files, tmp_path):
    source, target = corpus_files/"data", tmp_path/"v-next"
    report = new_test_split(source, target, seed=99)
    assert report["passed"], report["errors"]
    for split in ("train", "dev_a", "dev_b"):
        assert file_hash(target/f"{split}.jsonl") == file_hash(source/f"{split}.jsonl")
    old_orbits = {r["meaning_id"] for s in SPLITS for r in load_rows(source, s)}
    old_pairs = {(r["sentence_a"], r["sentence_b"]) for s in SPLITS for r in load_rows(source, s)}
    old_pairs |= {(r["auxiliary"]["sentence_a"], r["auxiliary"]["sentence_b"]) for s in EVALUATION_SPLITS for r in load_rows(source, s)}
    new = load_rows(target, "test")
    assert new and all(r["meaning_id"] not in old_orbits for r in new)
    assert all((r["sentence_a"], r["sentence_b"]) not in old_pairs and (r["auxiliary"]["sentence_a"], r["auxiliary"]["sentence_b"]) not in old_pairs for r in new)
    assert all(r["split"] == "test" and r["auxiliary"]["template_family"] == FAMILIES["test"] for r in new)
    assert len(new) == 2*4*3 and all(r["meaning_id"] != old for r in new for old in ())
    manifest = json.loads((target/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["independent_test"]["confirmatory"] is True
    assert manifest["derived_from"]["dataset_hash"] == json.loads((source/"manifest.json").read_text(encoding="utf-8"))["dataset_hash"]
    assert manifest["overlap_check"]["orbits_shared_with_source"] == 0 and manifest["overlap_check"]["sentence_pairs_shared_with_source"] == 0
    with pytest.raises(FileExistsError):
        new_test_split(source, target, seed=99)


def test_new_test_split_can_exclude_extra_datasets(corpus_files, tmp_path):
    first = new_test_split(corpus_files/"data", tmp_path/"a", seed=5)
    second = new_test_split(corpus_files/"data", tmp_path/"b", seed=5, exclude=[tmp_path/"a"])
    a = {r["meaning_id"] for r in load_rows(tmp_path/"a", "test")}
    assert first["passed"] and second["passed"] and all(r["meaning_id"] not in a for r in load_rows(tmp_path/"b", "test"))


def test_auxiliary_is_scored_only_at_the_terminal_checkpoint(corpus_files, tmp_path):
    p = Protocol(vocab_size=320, embed_dim=8, effective_batch=4, eval_interval=8, mono_cap=12, total_cap=32, panel_per_task=2)
    summary = run(p, corpus_files/"graph.npz", corpus_files/"data", corpus_files/"tokenizer.json", tmp_path/"run",
                  "mono", ("en",), 7, cohort="smoke", device="cpu", backend="dense", microbatch=3, smoke=True)
    events = [json.loads(l) for l in (tmp_path/"run"/"events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not any(e["kind"] == "auxiliary" for e in events)
    terminal = json.loads((tmp_path/"run"/"terminal-auxiliary.json").read_text(encoding="utf-8"))
    assert set(terminal["scores"]) == {"dev_a", "dev_b"} and set(terminal["scores"]["dev_a"]) == {f"en/{t}" for t in ("roles", "negation", "space", "quantity")}
    assert terminal["checkpoint_hash"] == file_hash(tmp_path/"run"/"primary.pt") and terminal["used_for_mastery"] is False
    test = json.loads((tmp_path/"run"/"independent-test.json").read_text(encoding="utf-8"))
    assert set(test["auxiliary_scores"]) == set(test["scores"]) and summary["clocks_seconds"]["auxiliary"] > 0


def test_study_cohorts_cannot_skip_the_test_or_enable_auxiliary_panels(corpus_files, tmp_path):
    for kwargs in (dict(independent_test=False), dict(auxiliary_panels=True)):
        with pytest.raises(ValueError, match="Study cohorts"):
            run(Protocol(), corpus_files/"graph.npz", corpus_files/"data", corpus_files/"tokenizer.json", tmp_path/"pilot",
                "mono", ("en",), 10001, cohort="pilot", device="cpu", backend="dense", **kwargs)

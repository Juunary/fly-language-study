from dataclasses import replace
import json
import pytest
import torch
from tokenizers import Tokenizer
from flystudy.data import generate, load_rows, audit, train_tokenizer, nuisance_features
from flystudy.graph import synthetic
from flystudy.protocol import Protocol, LANGUAGES, TASKS
from flystudy.runtime import Corpus, AlignedSampler
from flystudy.train import run


@pytest.fixture(scope="module")
def corpus_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    generate(root/"data",18,2)
    train_tokenizer(root/"data",root/"tokenizer.json",320)
    synthetic().save(root/"graph.npz")
    return root


def test_no_split_foil_or_template_leaks(corpus_files):
    report = audit(corpus_files/"data")
    assert report["passed"] and report["human_review"] == "pending"
    assert report["examples"] == (18+2*3)*4*3


def test_korean_copular_endings_are_inflected(corpus_files):
    for split,ending in (('dev_a','사실이라는'),('dev_b','사실이라고'),('test','사실이라면')):
        rows=[r for r in load_rows(corpus_files/'data',split) if r['language']=='ko' and r['task']=='roles']
        assert rows and all(ending in r['sentence_a'] for r in rows)
        assert all('이다는' not in r['sentence_a'] and '이다고' not in r['sentence_a'] for r in rows)


def test_dataset_tampering_detected(corpus_files,tmp_path):
    import shutil
    target = tmp_path/"copy"
    shutil.copytree(corpus_files/"data",target)
    with (target/"train.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(load_rows(target,"dev_a")[0],ensure_ascii=False)+"\n")
    assert not audit(target)["passed"]


def test_pair_balancing_removes_nuisance_features(corpus_files):
    groups = {}
    for row in load_rows(corpus_files/"data","train"):
        groups.setdefault((row["meaning_id"],row["language"]),[]).append(row)
    for pair in groups.values():
        assert len(pair) == 2
        assert nuisance_features(pair[0]) == nuisance_features(pair[1])


def test_sampler_is_aligned_and_exactly_balanced(corpus_files):
    tokenizer = Tokenizer.from_file(str(corpus_files/"tokenizer.json"))
    corpus = Corpus(corpus_files/"data",tokenizer)
    sampler = AlignedSampler(corpus,7)
    assert sampler.indices["en"] == sampler.indices["de"] == sampler.indices["ko"]
    saved = sampler.state_dict()
    a = sampler.sample(10)
    assert max(sampler.exposures.values())-min(sampler.exposures.values()) == 1
    sampler.load_state_dict(saved)
    assert sampler.sample(10) == a


def test_checkpoint_resume_matches_uninterrupted(corpus_files,tmp_path):
    p = Protocol(vocab_size=320,embed_dim=8,effective_batch=4,eval_interval=8,mono_cap=12,total_cap=32,panel_per_task=2)
    args = (p,corpus_files/"graph.npz",corpus_files/"data",corpus_files/"tokenizer.json")
    kwargs = dict(mode="sequential",order=LANGUAGES,seed=7,cohort="smoke",device="cpu",backend="dense",microbatch=3,smoke=True)
    full = run(*args,tmp_path/"full",**kwargs)
    interrupted = run(*args,tmp_path/"resumed",max_updates=3,**kwargs)
    assert interrupted["stop_reason"] == "debug_interruption"
    resumed = run(*args,tmp_path/"resumed",resume=tmp_path/"resumed"/"latest.pt",**kwargs)
    assert (full["seen"],full["stop_reason"],full["first_task"]) == (resumed["seen"],resumed["stop_reason"],resumed["first_task"])
    a = torch.load(tmp_path/"full"/"primary.pt",weights_only=True)
    b = torch.load(tmp_path/"resumed"/"primary.pt",weights_only=True)
    for key in a["model"]: torch.testing.assert_close(a["model"][key],b["model"][key],atol=0,rtol=0)
    assert (tmp_path/"resumed"/"independent-test.json").exists()


def test_cpu_or_synthetic_cannot_launch_main(corpus_files,tmp_path):
    with pytest.raises(ValueError,match="cb5k"):
        run(Protocol(),corpus_files/"graph.npz",corpus_files/"data",corpus_files/"tokenizer.json",tmp_path/"main",
            "mono",("en",),1,cohort="main",device="cpu",backend="dense")

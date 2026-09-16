from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json
import random
import time
import numpy as np
import torch

from .data import encode_pair, load_rows
from .model import collate
from .protocol import LANGUAGES, TASKS


def rng_state():
    ns = np.random.get_state()
    return dict(python=random.getstate(), numpy=[ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    # torch.load(map_location="cuda:...") also moves serialized RNG byte tensors.
    # Generator state APIs consume CPU byte tensors even for CUDA generators.
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


@contextmanager
def immutable_evaluation(model, optimizer=None, sampler=None):
    state = rng_state()
    modules = [(m, m.training) for m in model.modules()]
    params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = deepcopy(optimizer.state_dict()) if optimizer is not None else None
    sample = deepcopy(sampler.state_dict()) if sampler is not None else None
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        restore_rng(state)
        for module, mode in modules:
            module.training = mode
        if not equal_state(params, model.state_dict()):
            raise RuntimeError("Evaluation mutated model parameters/persistent buffers")
        if optimizer is not None and not equal_state(opt, optimizer.state_dict()):
            raise RuntimeError("Evaluation mutated optimizer")
        if sampler is not None and not equal_state(sample, sampler.state_dict()):
            raise RuntimeError("Evaluation mutated sampler")


class Corpus:
    def __init__(self, root, tokenizer):
        self.root, self.tokenizer = Path(root), tokenizer
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cache = {}

    def split(self, split):
        if split not in self.cache:
            groups = {}
            for lang in LANGUAGES:
                groups[lang] = []
            for row in load_rows(self.root, split):
                groups[row["language"]].append((row, encode_pair(self.tokenizer, row)))
            for lang in LANGUAGES:
                groups[lang].sort(key=lambda x: (x[0]["meaning_id"], x[0]["label"]))
            self.cache[split] = groups
        return self.cache[split]


class AlignedSampler:
    def __init__(self, corpus, seed):
        self.rows = corpus.split("train")
        self.generators = {lang: random.Random(seed+23456) for lang in LANGUAGES}
        self.indices, self.cursors = {}, {}
        for lang in LANGUAGES:
            self.indices[lang] = list(range(len(self.rows[lang])))
            self.generators[lang].shuffle(self.indices[lang])
            self.cursors[lang] = 0
        alignment = [[(r["meaning_id"], r["label"]) for r, _ in self.rows[lang]] for lang in LANGUAGES]
        if not alignment[0] == alignment[1] == alignment[2]:
            raise ValueError("Training translations are not aligned")
        self.mix_cursor = 0
        self.unique = set()
        self.exposures = {lang: 0 for lang in LANGUAGES}

    def sample(self, count, language=None):
        result = []
        for _ in range(count):
            lang = language or LANGUAGES[self.mix_cursor % 3]
            if language is None:
                self.mix_cursor += 1
            if self.cursors[lang] == len(self.indices[lang]):
                self.generators[lang].shuffle(self.indices[lang])
                self.cursors[lang] = 0
            idx = self.indices[lang][self.cursors[lang]]
            self.cursors[lang] += 1
            row, tokens = self.rows[lang][idx]
            self.unique.add(row["id"])
            self.exposures[lang] += 1
            result.append((tokens, row["label"]))
        return result

    def state_dict(self):
        return dict(indices=deepcopy(self.indices), cursors=dict(self.cursors), mix_cursor=self.mix_cursor,
                    generators={k: g.getstate() for k, g in self.generators.items()},
                    unique=sorted(self.unique), exposures=dict(self.exposures))

    def load_state_dict(self, state):
        self.indices, self.cursors = deepcopy(state["indices"]), dict(state["cursors"])
        self.mix_cursor = state["mix_cursor"]
        for k, v in state["generators"].items():
            self.generators[k].setstate(v)
        self.unique, self.exposures = set(state["unique"]), dict(state["exposures"])


def evaluate(model, corpus, split, languages, microbatch, optimizer=None, sampler=None):
    scores, strata = {}, {}
    with immutable_evaluation(model, optimizer, sampler):
        for lang in languages:
            items = corpus.split(split)[lang]
            for task in TASKS:
                selected = [(r, tokens) for r, tokens in items if r["task"] == task]
                correct, count, male_correct, male_count = 0, 0, 0, 0
                selected.sort(key=lambda x: len(x[1]))
                for start in range(0, len(selected), microbatch):
                    part = selected[start:start+microbatch]
                    x, lengths, y = collate([(tokens, r["label"]) for r, tokens in part], corpus.pad_id, model.edge_values.device)
                    ok = (model(x, lengths).argmax(-1) == y).tolist()
                    correct += sum(ok); count += len(ok)
                    for hit, (row, _) in zip(ok, part):
                        if row["gender_pair"] == "mm":
                            male_correct += hit; male_count += 1
                scores[f"{lang}/{task}"] = [correct, count]
                if lang == "de":
                    strata[f"{lang}/{task}/male_only"] = [male_correct, male_count]
    return scores, strata


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def append_event(path, event):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def save_checkpoint(path, state):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(state, tmp)
    tmp.replace(path)

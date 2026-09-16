"""Pure state machine: only prescheduled full-panel observations change mastery."""
from __future__ import annotations

from dataclasses import dataclass, field
from .protocol import LANGUAGES, TASKS, Protocol


@dataclass
class Curriculum:
    protocol: Protocol
    mode: str
    order: tuple[str, ...]
    seen: int = 0
    stage: int = 0
    stage_start: int = 0
    eval_index: int = 0
    previous_pass: dict = field(default_factory=dict)
    confirmed: dict = field(default_factory=dict)
    first_task: dict = field(default_factory=dict)
    first_language: dict = field(default_factory=dict)
    first_global: int | None = None
    last_global_pass: bool = False
    stages: list = field(default_factory=list)
    review_start: int | None = None
    stop_reason: str | None = None

    def __post_init__(self):
        if self.mode not in ("mono", "mixed", "sequential"):
            raise ValueError("Unknown curriculum")
        if self.mode == "mono" and (len(self.order) != 1 or self.order[0] not in LANGUAGES):
            raise ValueError("Mono requires one language")
        if self.mode != "mono" and (len(self.order) != 3 or set(self.order) != set(LANGUAGES)):
            raise ValueError("Three-language curricula require a permutation")
        if self.mode == "mixed":
            self.review_start = 0

    @property
    def languages(self):
        return self.order if self.mode == "mono" else LANGUAGES

    @property
    def cap(self):
        return self.protocol.mono_cap if self.mode == "mono" else self.protocol.total_cap

    @property
    def next_eval(self):
        return min((self.eval_index + 1) * self.protocol.eval_interval, self.cap)

    @property
    def next_panel(self):
        return "dev_a" if self.eval_index % 2 == 0 else "dev_b"

    @property
    def current_language(self):
        if self.mode == "mono":
            return self.order[0]
        if self.mode == "sequential" and self.stage < 3:
            return self.order[self.stage]
        return None

    def batch_size(self):
        if self.stop_reason:
            return 0
        limits = [self.protocol.effective_batch, self.next_eval - self.seen, self.cap - self.seen]
        if self.mode == "sequential" and self.stage < 3:
            limits.append(self.stage_start + self.protocol.mono_cap - self.seen)
        return max(0, min(limits))

    def train_exposures(self, count):
        if count <= 0 or count > self.batch_size():
            raise ValueError("Batch crosses an evaluation, phase boundary, or cap")
        self.seen += count

    def evaluate(self, scores, panel, kind="scheduled"):
        if kind != "scheduled":
            return  # diagnostics never mutate confirmation history
        if self.stop_reason or self.seen != self.next_eval or panel != self.next_panel:
            raise ValueError("Unexpected evaluation time or panel")
        expected = {f"{lang}/{task}" for lang in self.languages for task in TASKS}
        if set(scores) != expected:
            raise ValueError("Full evaluation must cover every required language/task cell")
        passed = {}
        for key, count in scores.items():
            correct, total = count
            if total != self.protocol.panel_per_task or not 0 <= correct <= total:
                raise ValueError("Partial or invalid panel cannot confirm mastery")
            passed[key] = correct >= self.protocol.threshold * total
            if passed[key] and self.previous_pass.get(key, False):
                self.first_task.setdefault(key, self.seen)
        for lang in self.languages:
            keys = [f"{lang}/{task}" for task in TASKS]
            self.confirmed[lang] = all(passed[k] and self.previous_pass.get(k, False) for k in keys)
            if self.confirmed[lang]:
                self.first_language.setdefault(lang, self.seen)
        global_pass = all(passed.values())
        if global_pass and self.last_global_pass:
            self.first_global = self.seen
            self.stop_reason = "mastered"
        self.previous_pass = passed
        self.last_global_pass = global_pass
        self.eval_index += 1
        self.advance()
        if self.seen == self.cap and not self.stop_reason:
            self.stop_reason = "administrative_cap"

    def advance(self):
        if self.mode != "sequential":
            return
        while self.stage < 3:
            lang = self.order[self.stage]
            mastered = self.confirmed.get(lang, False)
            capped = self.seen - self.stage_start >= self.protocol.mono_cap
            if not mastered and not capped:
                break
            self.stages.append(dict(language=lang, start=self.stage_start, end=self.seen,
                                    exposures=self.seen - self.stage_start,
                                    reason="mastered" if mastered else "stage_cap"))
            self.stage += 1
            self.stage_start = self.seen
        if self.stage == 3 and self.review_start is None:
            self.review_start = self.seen

    def checkpoint(self):
        return {k: v for k, v in self.__dict__.items() if k != "protocol"}

    @classmethod
    def restore(cls, protocol, state):
        state = dict(state)
        state["order"] = tuple(state["order"])
        return cls(protocol=protocol, **state)


from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from itertools import permutations
from pathlib import Path
import json

LANGUAGES = ("en", "de", "ko")
TASKS = ("roles", "negation", "space", "quantity")
ORDERS = tuple(permutations(LANGUAGES))
INTERVALS = (5120, 10240, 20480)
UPSTREAM = {
    "flygpt": "d8ce5e286b638167a0dcb51bd0e1defdbd510d37",
    "kernels": "c6be4cfea8ca2210098ef52344b88e85c683fd5c",
    "hf": "68282fad45efa909eb95e445d5dfcef00a917ff4",
}
BUDGETS = {"g0_g1": 16, "g2": 48, "g3": 32, "order_pilot": 32,
           "main": 448, "tokenizer": 24, "shuffle": 24, "review": 16, "reserve": 32}


def digest(value) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


@dataclass(frozen=True)
class Protocol:
    version: str = "v4.0"
    vocab_size: int = 4096
    microsteps: int = 2
    embed_dim: int = 32
    effective_batch: int = 256
    eval_interval: int = 10240
    mono_cap: int = 200000
    total_cap: int = 900000
    threshold: float = .8
    panel_per_task: int = 1000
    recurrent_lr: float = .0003
    adapter_lr: float = .001
    weight_decay: float = .01
    grad_clip: float = 1.
    main_seeds: tuple[int, ...] = tuple(range(1, 21))
    pilot_seeds: tuple[int, ...] = (10001, 10002)
    budgets: dict = field(default_factory=lambda: dict(BUDGETS))

    def validate(self, production=True):
        if self.eval_interval <= 0 or self.mono_cap <= 0 or self.total_cap < self.mono_cap:
            raise ValueError("Invalid exposure schedule")
        if not 0 < self.threshold <= 1 or self.panel_per_task < 1:
            raise ValueError("Invalid mastery definition")
        if set(self.main_seeds) & set(self.pilot_seeds):
            raise ValueError("Pilot and main seeds overlap")
        if len(set(self.main_seeds)) != len(self.main_seeds):
            raise ValueError("Duplicate main seeds")
        if sum(self.budgets.values()) != 672:
            raise ValueError("The ledger must reserve exactly 672 GPU hours")
        if production and (self.eval_interval not in INTERVALS or self.threshold != .8 or
                           self.panel_per_task != 1000 or self.effective_batch != 256 or
                           self.mono_cap != 200000 or self.total_cap != 900000):
            raise ValueError("Non-production/smoke settings cannot launch the study")
        return self

    def validate_primary_model(self):
        fixed = dict(vocab_size=4096,microsteps=2,embed_dim=32,recurrent_lr=.0003,
                     adapter_lr=.001,weight_decay=.01,grad_clip=1.)
        if any(getattr(self,key) != value for key,value in fixed.items()):
            raise ValueError('Primary/pilot model hyperparameters differ from protocol v4')
        return self

    @property
    def hash(self):
        return digest(asdict(self))

    @property
    def training_hash(self):
        return digest({k: v for k, v in asdict(self).items() if k not in ("main_seeds", "pilot_seeds", "budgets")})

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("main_seeds", "pilot_seeds"):
            if key in obj:
                obj[key] = tuple(obj[key])
        return cls(**obj)


def run_matrix(seeds, cohort="main", topology="real"):
    for seed in seeds:
        for language in LANGUAGES:
            yield dict(run_id=f"{cohort}-{topology}-{seed}-mono-{language}", cohort=cohort,
                       topology=topology, seed=seed, mode="mono", order=[language])
        for order in ORDERS:
            yield dict(run_id=f"{cohort}-{topology}-{seed}-seq-{'-'.join(order)}", cohort=cohort,
                       topology=topology, seed=seed, mode="sequential", order=list(order))
        yield dict(run_id=f"{cohort}-{topology}-{seed}-mixed", cohort=cohort,
                   topology=topology, seed=seed, mode="mixed", order=list(LANGUAGES))

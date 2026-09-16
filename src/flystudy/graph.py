from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import urllib.request
import numpy as np

from .protocol import UPSTREAM, digest, file_hash, write_json


@dataclass
class Graph:
    src: np.ndarray
    dst: np.ndarray
    inputs: np.ndarray
    outputs: np.ndarray
    n: int
    provenance: dict

    def validate(self):
        for values in (self.src, self.dst, self.inputs, self.outputs):
            if values.ndim != 1 or len(values) == 0 or values.min() < 0 or values.max() >= self.n:
                raise ValueError("Invalid graph indices")
        if len(self.src) != len(self.dst) or len(np.unique(self.src * self.n + self.dst)) != len(self.src):
            raise ValueError("Duplicate/mismatched edges")
        if any(len(np.unique(x)) != len(x) for x in (self.inputs, self.outputs)):
            raise ValueError("Duplicate input/output nodes")
        return self

    @property
    def hash(self):
        return digest(dict(src=self.src.tolist(), dst=self.dst.tolist(), inputs=self.inputs.tolist(),
                           outputs=self.outputs.tolist(), n=self.n))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, src=self.src, dst=self.dst, inputs=self.inputs,
                            outputs=self.outputs, n=self.n)
        write_json(path.with_suffix(".json"), {**self.provenance, "graph_hash": self.hash, "file_hash": file_hash(path)})

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if file_hash(path) != meta["file_hash"]:
            raise ValueError("Graph artifact changed")
        with np.load(path, allow_pickle=False) as z:
            graph = cls(*(z[k].astype(np.int64) for k in ("src", "dst", "inputs", "outputs")), int(z["n"]), meta).validate()
        if graph.hash != meta["graph_hash"]:
            raise ValueError("Graph hash mismatch")
        return graph


def synthetic(n=12, seed=0):
    rng = np.random.default_rng(seed)
    edges = {(i, (i+1) % n) for i in range(n)}
    edges.update((int(a), int(b)) for a, b in rng.integers(0, n, (n*2, 2)) if a != b)
    src, dst = np.array(sorted(edges), dtype=np.int64).T
    return Graph(src, dst, np.arange(min(4, n)), np.arange(min(6, n)), n,
                 dict(source="synthetic-smoke-only", condition="synthetic")).validate()


def fetch_real(output, cache):
    """Only structural buffers are loaded. No remote Python or learned weights execute."""
    from safetensors import safe_open
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/QuixiAI/FlyGPT/resolve/{UPSTREAM['hf']}/init"
    for name in ("model.safetensors", "config.json", "graph_metadata.json"):
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(f"{base}/{name}", path.with_suffix(path.suffix + ".partial"))
            path.with_suffix(path.suffix + ".partial").replace(path)
    cfg = json.loads((cache / "config.json").read_text())
    if cfg.get("training_state", {}).get("status") != "init":
        raise ValueError("Expected the pinned initialization artifact")
    with safe_open(cache / "model.safetensors", framework="numpy") as handle:
        keys = list(handle.keys())
        def tensor(suffix):
            found = [key for key in keys if key == suffix or key.endswith("." + suffix)]
            if len(found) != 1:
                raise ValueError(f"Missing/ambiguous graph tensor {suffix}; keys={keys}")
            return handle.get_tensor(found[0]).astype(np.int64)
        src, dst = tensor("edge_index")  # HF export explicitly stores [source, destination].
        inputs, outputs = tensor("input_nodes"), tensor("output_nodes")
    graph = Graph(src, dst, inputs, outputs, cfg["num_neurons"],
                  dict(source="QuixiAI/FlyGPT/init", revision=UPSTREAM["hf"], condition="real",
                       upstream_code=UPSTREAM["flygpt"], source_file_hash=file_hash(cache / "model.safetensors"),
                       pretrained_weights_loaded=False)).validate()
    if graph.n != 5000 or len(graph.src) != 524324 or len(inputs) != 256 or len(outputs) != 512:
        raise ValueError("Wrong cb5k graph shape")
    graph.save(output)
    return graph


def scramble(graph, seed, swaps_per_edge=10):
    """Directed double-edge swaps: exact in/out degrees, no self/parallel edges."""
    src, dst = graph.src.copy(), graph.dst.copy()
    rng = np.random.default_rng(seed)
    edges = set(zip(src.tolist(), dst.tolist()))
    requested, accepted, attempted = len(src)*swaps_per_edge, 0, 0
    while accepted < requested and attempted < requested*50:
        i, j = rng.integers(len(src), size=2)
        attempted += 1
        a, b, c, d = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
        if i == j or a == c or b == d or a == d or c == b or (a, d) in edges or (c, b) in edges:
            continue
        edges.remove((a, b)); edges.remove((c, d))
        edges.add((a, d)); edges.add((c, b))
        dst[i], dst[j] = d, b
        accepted += 1
    if accepted != requested:
        raise RuntimeError("Could not complete the prespecified graph swaps")
    result = Graph(src, dst, graph.inputs.copy(), graph.outputs.copy(), graph.n,
                   dict(source=graph.hash, condition="degree_preserving", seed=seed,
                        accepted_swaps=accepted, attempted_swaps=attempted)).validate()
    for before, after in ((graph.src, result.src), (graph.dst, result.dst)):
        np.testing.assert_array_equal(np.bincount(before, minlength=graph.n), np.bincount(after, minlength=graph.n))
    return result

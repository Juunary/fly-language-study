from copy import deepcopy
from dataclasses import replace
import random
import numpy as np
import pytest
import torch
from flystudy.graph import synthetic, scramble
from flystudy.model import FlyClassifier, train_batch
from flystudy.protocol import Protocol
from flystudy.runtime import immutable_evaluation, rng_state, equal_state, restore_rng


@pytest.mark.parametrize("microsteps", [2,4])
@pytest.mark.parametrize("batch", [1,4,9])
def test_dense_sparse_outputs_all_gradients_padding(microsteps,batch):
    graph = synthetic(9)
    dense = FlyClassifier(graph,20,8,microsteps,42,"dense")
    sparse = FlyClassifier(graph,20,8,microsteps,42,"sparse")
    x = torch.randint(0,20,(batch,7)); lengths = torch.arange(batch)%7+1
    a = torch.randn(batch,9,requires_grad=True); b = a.detach().clone().requires_grad_()
    ya,yb = dense(x,lengths,a),sparse(x,lengths,b)
    torch.testing.assert_close(ya,yb,atol=1e-5,rtol=1e-5)
    ya.square().sum().backward(); yb.square().sum().backward()
    for pa,pb in zip(dense.parameters(),sparse.parameters()):
        assert pa.grad is not None and pb.grad is not None
        torch.testing.assert_close(pa.grad,pb.grad,atol=1e-5,rtol=1e-4)
    torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=1e-4)


def test_padding_cannot_change_final_logits():
    m = FlyClassifier(synthetic(),20,8,4,1,"dense")
    short = m(torch.tensor([[1,2,3]]),torch.tensor([3]))
    padded = m(torch.tensor([[1,2,3,15,19]]),torch.tensor([3]))
    torch.testing.assert_close(short,padded,atol=0,rtol=0)


def test_unequal_microbatches_match_logical_batch():
    p = replace(Protocol(),grad_clip=100.)
    a = FlyClassifier(synthetic(),20,8,2,2,"dense"); b = deepcopy(a)
    oa = torch.optim.AdamW(a.parameter_groups(p),weight_decay=p.weight_decay)
    ob = torch.optim.AdamW(b.parameter_groups(p),weight_decay=p.weight_decay)
    examples = [([1,2,3,4,5][:i%5+1],i%2) for i in range(11)]
    train_batch(a,oa,examples,0,11,p); train_batch(b,ob,examples,0,3,p)
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa,pb,atol=1e-6,rtol=1e-5)


def test_evaluation_restores_rng_and_nested_modes():
    model = FlyClassifier(synthetic(),20,8,2,1,"dense")
    model.train(); model.embed.eval()
    state = rng_state()
    with immutable_evaluation(model):
        random.random(); np.random.random(); torch.rand(4)
        model(torch.tensor([[1,2]]),torch.tensor([2]))
    assert model.training and not model.embed.training
    assert equal_state(state,rng_state())


def test_restore_normalizes_device_mapped_rng_states(monkeypatch):
    """CPU regression: both RNG APIs must receive the normalized byte tensors."""
    state = rng_state()
    cpu_bytes = state["torch"].clone()
    class MappedState:
        def cpu(self):
            return cpu_bytes
    state["torch"] = MappedState()
    state["cuda"] = [MappedState(), MappedState()]
    received = []
    def consume(value):
        assert isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.uint8
        received.append(value)
    monkeypatch.setattr(torch, "set_rng_state", consume)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: [consume(v) for v in values])
    restore_rng(state)
    assert len(received) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA checkpoint mapping")
def test_cuda_mapped_checkpoint_restores_all_random_streams(tmp_path):
    """Exercise the exact failing load path without the exploratory script's workaround."""
    before = rng_state()
    path = tmp_path / "rng.pt"
    try:
        torch.save({"rng": before}, path)
        expected = (random.random(), np.random.random(), torch.rand(12),
                    [torch.rand(12, device=f"cuda:{i}") for i in range(torch.cuda.device_count())])
        mapped = torch.load(path, map_location="cuda:0", weights_only=True)["rng"]
        assert mapped["torch"].device.type == "cuda"
        assert all(v.device.type == "cuda" for v in mapped["cuda"])
        restore_rng(mapped)
        actual = (random.random(), np.random.random(), torch.rand(12),
                  [torch.rand(12, device=f"cuda:{i}") for i in range(torch.cuda.device_count())])
        assert actual[:2] == expected[:2]
        torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
        for a, b in zip(actual[3], expected[3]):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    finally:
        restore_rng(before)


def test_mutating_evaluation_fails():
    model = FlyClassifier(synthetic(),20,8,2,1,"dense")
    with pytest.raises(RuntimeError,match="mutated model"):
        with immutable_evaluation(model): model.bias.add_(1)


def test_explicit_cuda_does_not_fall_back():
    model = FlyClassifier(synthetic(),20,8,2,1,"cuda")
    with pytest.raises(RuntimeError,match="no fallback"):
        model(torch.tensor([[1,2]]),torch.tensor([2]))


def test_shuffle_preserves_both_degrees_and_io():
    graph = synthetic(20,2)
    shuffled = scramble(graph,7,swaps_per_edge=2)
    for before,after in ((graph.src,shuffled.src),(graph.dst,shuffled.dst)):
        np.testing.assert_array_equal(np.bincount(before,minlength=graph.n),np.bincount(after,minlength=graph.n))
    np.testing.assert_array_equal(graph.inputs,shuffled.inputs)
    np.testing.assert_array_equal(graph.outputs,shuffled.outputs)
    assert shuffled.hash != graph.hash

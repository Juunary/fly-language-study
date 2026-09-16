"""Trainable scalar-neuron recurrence, adapted to final-state classification.

Architecture follows the MIT-licensed QuixiAI/FlyGPT pinned in protocol.py.
CUDA is explicit: backend failures never silently fall back to reference code.
"""
from __future__ import annotations

import torch
from torch import nn
from .graph import Graph


class FlyClassifier(nn.Module):
    def __init__(self, graph: Graph, vocab_size, embed_dim=32, microsteps=2,
                 seed=1, backend="sparse", frozen_embeddings=None):
        super().__init__()
        if backend not in ("sparse", "dense", "cuda") or microsteps < 1:
            raise ValueError("Invalid backend/microsteps")
        graph.validate()
        self.n, self.microsteps, self.backend = graph.n, microsteps, backend
        self.register_buffer("src", torch.as_tensor(graph.src.copy()))
        self.register_buffer("dst", torch.as_tensor(graph.dst.copy()))
        self.register_buffer("inputs", torch.as_tensor(graph.inputs.copy()))
        self.register_buffer("outputs", torch.as_tensor(graph.outputs.copy()))
        scale = torch.bincount(self.dst, minlength=graph.n).clamp(min=1).float().rsqrt()[self.dst]
        self.register_buffer("edge_scale", scale)
        self.edge_values = nn.Parameter(torch.randn(len(graph.src), generator=torch.Generator().manual_seed(seed)))
        self.bias = nn.Parameter(torch.zeros(graph.n))
        self.raw_leak = nn.Parameter(torch.zeros(graph.n))
        # Independent initialization streams keep adapters paired across graph conditions.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100000)
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.inp = nn.Linear(embed_dim, len(graph.inputs))
            self.head = nn.Linear(len(graph.outputs), 2)
        if frozen_embeddings is not None:
            self.embed.weight.data.copy_(frozen_embeddings)
            self.embed.weight.requires_grad_(False)
        self._csr = None

    def parameter_groups(self, protocol):
        return [{"params": [self.edge_values, self.bias, self.raw_leak], "lr": protocol.recurrent_lr},
                {"params": [p for m in (self.embed, self.inp, self.head) for p in m.parameters() if p.requires_grad],
                 "lr": protocol.adapter_lr}]

    def forward(self, tokens, lengths, initial=None):
        if tokens.ndim != 2 or lengths.shape != (tokens.shape[0],) or torch.any(lengths < 1) or torch.any(lengths > tokens.shape[1]):
            raise ValueError("Invalid token lengths")
        b, t = tokens.shape
        state = torch.zeros(b, self.n, device=tokens.device) if initial is None else initial
        values, leak = self.edge_values * self.edge_scale, self.raw_leak.sigmoid()
        drives = self.inp(self.embed(tokens)).float()
        if self.backend == "cuda":
            if tokens.device.type != "cuda":
                raise RuntimeError("CUDA backend requires a CUDA device; no fallback allowed")
            from connectome_kernels import SparseGraph, sparse_recurrence, available
            if not available():
                raise RuntimeError("Compiled connectome kernels unavailable; G0 has not passed")
            if self._csr is None:
                self._csr = SparseGraph(self.src, self.dst, self.n, self.inputs)
            states = sparse_recurrence(values, leak, self.bias,
                                       drives.permute(1, 2, 0).contiguous(), state, self._csr, self.microsteps)
            # Right padding cannot affect earlier causal states. Gather before padding.
            final = states[lengths-1, torch.arange(b, device=tokens.device)]
            return self.head(final[:, self.outputs])
        weights = torch.sparse_coo_tensor(torch.stack([self.dst, self.src]), values, (self.n, self.n), check_invariants=True).coalesce()
        if self.backend == "dense":
            weights = weights.to_dense()
        for time in range(t):
            drive = torch.zeros(b, self.n, device=tokens.device).index_copy(1, self.inputs, drives[:, time])
            active = (time < lengths).unsqueeze(1)
            for _ in range(self.microsteps):
                incoming = state @ weights.T if self.backend == "dense" else torch.sparse.mm(weights, state.T).T
                proposal = torch.tanh(incoming + drive + self.bias)
                state = torch.where(active, (1-leak)*state + leak*proposal, state)
        return self.head(state[:, self.outputs])


def collate(examples, pad_id, device):
    lengths = torch.tensor([len(x[0]) for x in examples], device=device)
    tokens = torch.full((len(examples), int(lengths.max())), pad_id, dtype=torch.long, device=device)
    for row, (ids, _) in enumerate(examples):
        tokens[row, :len(ids)] = torch.tensor(ids, device=device)
    labels = torch.tensor([x[1] for x in examples], dtype=torch.long, device=device)
    return tokens, lengths, labels


def train_batch(model, optimizer, examples, pad_id, microbatch, protocol):
    """Sum losses / logical batch size; never average microbatch means equally."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ordered = sorted(examples, key=lambda x: len(x[0]))
    total_loss, padded_tokens = 0., 0
    for start in range(0, len(ordered), microbatch):
        part = ordered[start:start+microbatch]
        x, lengths, y = collate(part, pad_id, model.edge_values.device)
        loss = nn.functional.cross_entropy(model(x, lengths), y, reduction="sum") / len(ordered)
        loss.backward()
        total_loss += float(loss.detach())
        padded_tokens += x.numel()
    norm = nn.utils.clip_grad_norm_(model.parameters(), protocol.grad_clip)
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite gradient; this is not administrative censoring")
    optimizer.step()
    return dict(loss=total_loss, grad_norm=float(norm), padded_tokens=padded_tokens,
                tokens=sum(len(x[0]) for x in examples), recurrent_updates=padded_tokens*model.microsteps)

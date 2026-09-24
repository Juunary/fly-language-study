---
license: cc-by-4.0
language:
  - en
  - de
  - ko
pretty_name: Fly Language Study Main Runs (protocol v5.1)
size_categories:
  - n<1K
tags:
  - experiment-records
  - learning-order
  - continual-learning
  - catastrophic-forgetting
  - pre-registered
  - connectome
configs:
  - config_name: default
    data_files:
      - split: runs
        path: runs.jsonl
---

# Fly Language Study Main Runs (protocol v5.1)

Complete run records of the pre-registered main study of
[fly-language-study](https://github.com/Juunary/fly-language-study): 753 training runs of a recurrent classifier whose
fixed wiring is a fly central-brain connectome (5,000 neurons, 524,324 connections, connection strengths learned),
trained on the companion *Fly Language Study Items* benchmark under ten conditions per seed: monolingual EN/DE/KO
(cap 200,000 examples), six sequential language orders (stage cap 200,000 per language, then 1:1:1 review, total cap
900,000) and a 1:1:1 mixed baseline. Every table and figure of the results report can be regenerated from this
dataset on a CPU.

## Inventory

| item | count |
|---|---|
| planned runs (launch manifest, seeds 30001–30080 × 10 conditions) | 800 |
| completed runs with a terminal state (mastered or administrative cap) | 753 |
| runs in the analysis set (complete seed blocks 30001–30075) | 750 |
| completed runs outside the analysis set (block 30076, monolingual only) | 3 |
| runs never started (operator-scheduled stop 2026-09-23T00:00Z) | 47 |
| technical failures (all recorded and recovered; no block excluded) | 6 |
| GPU hours, completed runs / failed attempts | 68.39 / 0.43 |

The study ended at 753/800 by the operator's decision after a scheduled stop unrelated to outcomes; no contrast had
been read before that decision. The analysis uses one declared rule: the longest prefix of the pre-registered seed order
whose ten-condition blocks are all complete (75 blocks). This deviation from the pre-registered N=80 is documented in
`docs/MAIN_STUDY_RESULTS_V5.1.md` §3 and `docs/MAIN_STUDY_PLAN_V5.md` §10.

## Headline results (N=75 seeds, pre-registered contrasts)

| contrast | difference (examples) | Holm p | 97.5% simultaneous CI |
|---|---|---|---|
| Korean last − Korean first | +112,230 | 2.7e-17 | [89,304, 135,157] |
| EN/DE adjacent − separated | −39,151 | 4.6e-8 | [−53,853, −24,449] |

Restricted mean cost to three-language mastery: Korean-first orders ≈325k examples, English-first 411–415k,
German-first 427–465k, mixed 576k. Conclusions are conditional on these items, this input representation, this model
and this training policy; no claim about universal language difficulty or about fly-specific wiring effects is made
(no control graph was run).

## Layout

```
README.md
runs.jsonl                     one row per completed run (see columns below)
runs/<run_id>/                 summary.json, metadata.json, events.jsonl, independent-test.json,
                               terminal-auxiliary.json, attempt-*.json
console-logs/                  console logs of the six retried runs
failed-attempts/<run>-attemptN/ attempt/failure records, kernel Xid notes, preserved run folders (no checkpoints)
manifest/                      launch-manifest-v5.1-N80.json (800 runs, frozen hashes), protocol config, retry mappings
design/                        sample-size simulation grids (5,000- and 50,000-repetition FWER tables), design decisions,
                               FWER amendment and scenario-correspondence records
ledger/                        GPU-hour ledger (every reservation, charge and status), operator stop marker
analysis/                      analysis.json (all numbers of the results report) and five aggregate figures
docs/                          results report, plan and execution log, protocol v5, extended design, FWER amendment
MANIFEST.sha256                sha256 of every file above
```

**Checkpoints are not included.** The terminal checkpoints (`primary.pt`, 22 MB each) embed the full connectome edge
list, whose upstream redistribution terms have not been confirmed; they are available from the author on request.

## `runs.jsonl` columns

| column | meaning |
|---|---|
| `run_id`, `seed`, `mode`, `order`, `condition` | identity; `condition` ∈ `mono-en`, `mono-de`, `mono-ko`, `seq-<order>`, `mixed` |
| `stop_reason` | `mastered` (all twelve cells ≥ 0.8 on two consecutive scheduled panels) or `administrative_cap` |
| `seen` | training examples consumed at termination |
| `cap` | 200,000 (monolingual) or 900,000 (sequential, mixed) |
| `first_global` | examples at confirmed mastery of all trained cells; `null` if never confirmed |
| `restricted_cost` | `first_global`, or `cap` when censored (the pre-registered outcome) |
| `gpu_hours`, `elapsed_seconds`, `device` | actual cost and device of the successful attempt |
| `in_analysis_prefix` | true for seeds 30001–30075 |
| `code_hash`, `protocol_hash`, `dataset_hash` | frozen hashes (identical across all 753 runs) |

## Per-run files

- `summary.json`: everything in `runs.jsonl` plus `first_language` and `first_task` (examples at confirmed mastery
  per language / per cell), `stages` (sequential: language, start, end, exposures, reason `mastered` or `stage_cap`),
  `review_start`, `language_exposures`, `counters`, `clocks_seconds`, all frozen hashes and review provenance.
- `events.jsonl`: one JSON object per line. `kind: "train"` rows every 256 examples (loss, gradient norm, tokens);
  `kind: "scheduled"` rows every 5,120 examples with `panel` (`dev_a`/`dev_b`), `scores` (`{cell: [correct, 1000]}`
  for the twelve language/task cells), `strata`, `stage`, `confirmed` and the first-mastery bookkeeping. These rows
  are the source of the forgetting and recovery curves.
- `independent-test.json`: held-out test split scored once on the terminal checkpoint (`checkpoint_hash`), with
  per-cell `scores`, `strata`, and `auxiliary_scores` for the outer-frame rendering.
- `terminal-auxiliary.json`: outer-frame dev_a/dev_b scores of the terminal checkpoint (`used_for_mastery: false`).
- `metadata.json`: device, backend, environment and paths recorded at launch.
- `attempt-*.json`: ledger attempt record (reservation id, status, GPU hours).

## Regenerating the analysis

The GitHub repository's `scripts/analyze_main_truncated.py` reads exactly this layout:

```
python scripts/analyze_main_truncated.py --runs <bundle>/runs --manifest <bundle>/manifest/launch-manifest-v5.1-N80.json \
  --ledger <bundle>/ledger/gpu-ledger.json --failed <bundle>/failed-attempts \
  --design <bundle>/design/design-extended-v5.1-N2.json --stop-marker <bundle>/ledger/main-v5.1-campaign.stopped-by-operator \
  --output <somewhere-new>
```

It reproduces `analysis/analysis.json` (the bootstrap uses a fixed seed) and the figures.

## Limits

- 75 of the pre-registered 80 seed blocks; power and family-wise error guarantees are those of the frozen design
  table at N=70–80 (power lower bound 0.887–0.929, FWER upper bound ≈0.054), not recomputed for N=75.
- Stimuli certified by a Claude-only review, no native speakers (see the items dataset card).
- One hardware history: GPU 1 failed twice; 725 runs ran on `cuda:0`, 28 on `cuda:1`.
- No shuffled-graph or alternative-tokenizer control was run.

## License and citation

Records, documentation and figures: **CC BY 4.0** (copyright 2026 June woo Kang). Code is MIT-licensed on GitHub.

```
June woo Kang. Fly Language Study: pre-registered measurement of language-order effects in a
connectome-constrained recurrent classifier. 2026. https://github.com/Juunary/fly-language-study
```

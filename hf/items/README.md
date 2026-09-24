---
license: cc-by-4.0
language:
  - en
  - de
  - ko
multilinguality: multilingual
task_categories:
  - text-classification
pretty_name: Fly Language Study Items (draft-v4.6)
size_categories:
  - 10K<n<100K
tags:
  - synthetic
  - sentence-pairs
  - paraphrase
  - entailment
  - learning-order
  - continual-learning
  - pre-registered
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: dev_a
        path: data/dev_a.jsonl
      - split: dev_b
        path: data/dev_b.jsonl
      - split: test
        path: data/test.jsonl
---

# Fly Language Study Items (draft-v4.6)

Template-generated sentence pairs in **English, German and Korean** for one binary task: do the two sentences express the
same content (mutual entailment, label 1) or not (label 0)? Four phenomena are covered: **argument roles**,
**negation**, **spatial relations** and **quantities**. The same meanings are rendered in all three languages, so the
three languages share items, labels and splits exactly.

This is the frozen benchmark used by the pre-registered main study of
[fly-language-study](https://github.com/Juunary/fly-language-study) (protocol v5.1, dataset hash
`47368f15d5496aa14ba0d972ae716ecd348048cbba2de83da04029dcdf1afae9`). Results: `MAIN_STUDY_RESULTS_V5.1.md` in that
repository and the companion run-records dataset.

## What this dataset is not

- **Not a language-difficulty benchmark.** Attainment differences between the three languages measured on these items
  are properties of these templates, this tokenizer and the model that was trained; they do not measure the difficulty
  of English, German or Korean.
- **Not human-validated.** Items were certified by a Claude-only review (see *Certification and limits*). No native
  speaker reviewed them.

## Splits and sizes

| split | rows | per language × task | labels | primary rendering | auxiliary outer frame |
|---|---|---|---|---|---|
| train | 60,000 | 5,000 | 2,500 / 2,500 | declarative (`assertion`) | none |
| dev_a | 12,000 | 1,000 | 500 / 500 | declarative | `truth_question` |
| dev_b | 12,000 | 1,000 | 500 / 500 | declarative | `reported_clause` |
| test | 12,000 | 1,000 | 500 / 500 | declarative | `conditional` |

Semantic orbits, foils and translations are split-exclusive: a meaning that appears in one split never appears in
another. `test` was generated from fresh semantic orbits unused by any earlier dataset version or exploratory run
(seed 1731; overlap check 0 hits, recorded in `data/manifest.json`). In the study, `dev_a`/`dev_b` were the alternating
scheduled evaluation panels and `test` was scored once on each run's terminal checkpoint.

## Row format

```json
{"id": "<meaning_id>:<k>:<lang>", "meaning_id": "…", "foil_group": "…", "template_family": "assertion",
 "split": "train", "task": "roles", "language": "en", "label": 0, "gender_pair": "mm",
 "scene": {"task": "roles", "a": [3, 1, 3], "b": [0, 4, 3], "verb": 1, "neg": false, "axis": 1, "direction": true, "counts": [4, 2], "verb2": 0},
 "sentence_a": "The old blue lion greets the old black dog and …", "sentence_b": "The old black dog is seen by …",
 "auxiliary": {"template_family": "conditional", "sentence_a": "…", "sentence_b": "…"}}
```

| field | meaning |
|---|---|
| `meaning_id` | hash of the semantic scene; identical across the three languages |
| `foil_group` | items sharing a scene family (the label-0 foil of a scene keeps the same group) |
| `task` | `roles`, `negation`, `space`, `quantity` |
| `language` | `en`, `de`, `ko` |
| `label` | 1 = mutual entailment, 0 = otherwise (one-way entailment counts as 0) |
| `gender_pair` | German grammatical gender assignment of the two animals (`mm`, `mf`, …), kept parallel in EN/KO |
| `scene` | the abstract scene the renderer expanded (animal indices, verb, negation, axis/direction, counts) |
| `sentence_a`, `sentence_b` | the primary (declarative) rendering used for training, scheduled panels, mastery and the independent test |
| `auxiliary` | dev/test only: the same item rendered in a held-out outer frame; used only to report outer-frame transfer, never for mastery |

## Task definition

Judge whether the internal propositions of the two sentences express the same content: 1 means A implies B **and**
B implies A. Roles, negation, spatial relations and quantities are judged separately from linguistic naturalness.
Spatial items assume **one fixed external viewer** for both sentences: left/right and front/behind are viewer-relative,
above/below are gravitational, and converse relations hold (A left of B ⇔ B right of A). The full definition is in
`docs/TASK_DEFINITION_V5.1.1.md`.

## Generation

Sentences are produced by a deterministic renderer from a closed inventory (animals, colours, sizes/ages, verbs,
spatial axes, counts; `data/template-inventory.json`, `data/noun-forms.csv`) with seed 1729. German case and gender
morphology and Korean case particles are generated from the same scene, so the three languages differ only in surface
form. Animal colours form a controlled fictional domain. The renderer and the audit that verifies every file hash are in
the GitHub repository (`flystudy.data`).

`tokenizer/` holds the shared byte-level BPE used in the study (word-boundary preserving, 832 tokens; hash
`ea6e992cb22dc8d2ffd878abf995dd4c9ea2e3f19906519f7ce7ee68d78eaa9c`). Language-wise token-length statistics are in
its `.meta.json`.

## Certification and limits

- Items were certified by a **Claude-only review** (procedure `v4-ai-review-1`, rule amendment `v5.1-ai-review-2`):
  a 600-item audit sample (`data/audit-sample.csv`) judged per item, plus a 21-row checklist of constructions.
  Outcome records are in `certification/` and the procedure in `docs/AI_REVIEW_GUIDE.md`.
- Item-level semantic judgments cover the **primary rendering only**; auxiliary outer-frame renderings were checked at
  the level of construction examples.
- **No native-speaker review** was performed. The certification does not establish linguistic validity, only
  consistency with the task definition as judged by the reviewer model.
- Two review rounds flagged template defects that were fixed in this version (Korean spatial particles, German
  vertical prepositions, lexical choices); the change list is in `data/manifest.json` under `template_revision`.

## Files

```
README.md
data/                 train.jsonl, dev_a.jsonl, dev_b.jsonl, test.jsonl, manifest.json, template-inventory.json,
                      noun-forms.csv, audit.json, audit-sample.csv, construction-examples.csv, review-catalog.json,
                      cue-baselines*.json, nuisance-baselines*.json
tokenizer/            tokenizer-wordbound-v4.6.json (+ .meta.json)
docs/                 TASK_DEFINITION_V5.1.1.md, AI_REVIEW_GUIDE.md
certification/        ai-review-*.json outcome and rule-amendment records
MANIFEST.sha256       sha256 of every file above
```

## License and citation

Data, documentation and certification records: **CC BY 4.0** (copyright 2026 June woo Kang). The generating code is
MIT-licensed on GitHub. The connectome graph used by the study is **not** part of this dataset.

Please cite the repository until the paper is available:

```
June woo Kang. Fly Language Study: pre-registered measurement of language-order effects in a
connectome-constrained recurrent classifier. 2026. https://github.com/Juunary/fly-language-study
```

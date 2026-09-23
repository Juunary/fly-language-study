#!/usr/bin/env python
"""Final analysis of main study v5.1 after the operator-scheduled stop (753 of 800 runs; user decision of 2026-09-23 to
end the runs and analyze).

The pre-registered analysis (flystudy.reporting.analyze) requires every one of the 80 seed blocks and refuses to select
cases. This script leaves the frozen package untouched and applies one declared, outcome-blind rule instead: the analysis
set is the longest prefix of the pre-registered seed order whose 10-condition blocks (3 monolingual, 6 sequential,
1 mixed) all reached a terminal state. On that prefix the pre-registered summary (statistics.summarize, same seed and
bootstrap) runs unchanged; everything else is descriptive, cap-censored and reported per condition. Nothing is excluded
for non-attainment: a run that never mastered contributes its cap.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flystudy.protocol import ORDERS, file_hash, write_json  # noqa: E402
from flystudy.statistics import summarize, wilson  # noqa: E402

LANGS = ("en", "de", "ko")
TASKS = ("roles", "negation", "space", "quantity")
CELLS = [f"{lang}/{task}" for lang in LANGS for task in TASKS]
TERMINAL = ("mastered", "administrative_cap")
HASH_KEYS = ("protocol_hash", "dataset_hash", "graph_hash", "tokenizer_hash", "code_hash", "review_mode", "protocol_amendment")
CONDITIONS = [("mono", (lang,)) for lang in LANGS] + [("sequential", o) for o in ORDERS] + [("mixed", LANGS)]
DEVIATION_ID = "v5.1-main-truncated-prefix"


def condition_name(mode, order):
    return {"mono": f"mono-{order[0]}", "sequential": "seq-" + "-".join(order), "mixed": "mixed"}[mode]


@dataclass
class Loaded:
    manifest: dict
    records: dict = field(default_factory=dict)      # run_id -> summary (+ _scheduled, _test, _aux)
    by_key: dict = field(default_factory=dict)       # (seed, mode, order tuple) -> record
    missing: list = field(default_factory=list)      # manifest runs without summary.json (not started / interrupted)
    nonterminal: list = field(default_factory=list)  # summaries whose stop_reason is not terminal

    def block(self, seed):
        return [self.by_key.get((seed, mode, tuple(order))) for mode, order in CONDITIONS]


def _scheduled_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if '"scheduled"' in line:
            row = json.loads(line)
            if row.get("kind") == "scheduled":
                rows.append(row)
    return rows


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_records(root, manifest_path):
    root = Path(root)
    loaded = Loaded(manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8")))
    for entry in loaded.manifest["runs"]:
        folder = root / entry["run_id"]
        summary = _load_json(folder / "summary.json")
        if summary is None:
            loaded.missing.append(entry["run_id"])
            continue
        if summary.get("cohort") != "main" or summary.get("synthetic") or summary.get("stop_reason") not in TERMINAL:
            loaded.nonterminal.append(dict(run_id=entry["run_id"], stop_reason=summary.get("stop_reason")))
            continue
        summary["_scheduled"] = _scheduled_rows(folder / "events.jsonl")
        summary["_test"] = _load_json(folder / "independent-test.json")
        summary["_aux"] = _load_json(folder / "terminal-auxiliary.json")
        summary["_device"] = (_load_json(folder / "metadata.json") or {}).get("device", summary.get("device"))
        loaded.records[entry["run_id"]] = summary
        loaded.by_key[(summary["seed"], summary["mode"], tuple(summary["order"]))] = summary
    if loaded.records:
        for key in HASH_KEYS:
            if len({r.get(key) for r in loaded.records.values()}) != 1:
                raise ValueError(f"Incompatible main-study records: {key}")
    return loaded


def select_prefix(loaded, main_seeds):
    expected = defaultdict(list)
    for entry in loaded.manifest["runs"]:
        expected[entry["seed"]].append(entry["run_id"])
    included, excluded, broken = [], {}, False
    for seed in main_seeds:
        ids = expected[seed]
        present = sum(1 for i in ids if i in loaded.records)
        complete = len(ids) == len(CONDITIONS) and present == len(ids)
        if broken:
            excluded[str(seed)] = dict(reason="after the first incomplete block", present=present, missing=len(ids) - present)
        elif complete:
            included.append(seed)
        else:
            broken = True
            excluded[str(seed)] = dict(reason="incomplete block", present=present, missing=len(ids) - present)
    return dict(rule="longest prefix of the pre-registered seed order whose 10-condition blocks (3 mono, 6 sequential, 1 mixed) "
                     "all reached a terminal state (mastered or administrative_cap); later complete blocks are not used",
                pre_registered_n=len(main_seeds), n_included=len(included), included_seeds=included, excluded=excluded,
                runs_in_analysis=len(included) * len(CONDITIONS))


def _cost(record, key=None, source="first_global"):
    if source == "first_global":
        value = record.get("first_global")
    else:
        value = record.get(source, {}).get(key)
    return record["cap"] if value is None else value


def _stats(values):
    arr = np.asarray(values, dtype=float)
    out = dict(mean=float(arr.mean()), sd=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, median=float(np.median(arr)), n=int(len(arr)))
    if len(arr) > 1 and out["sd"] > 0:
        from scipy import stats
        half = stats.t.ppf(.975, len(arr) - 1) * out["sd"] / np.sqrt(len(arr))
        out["t_ci95"] = [out["mean"] - float(half), out["mean"] + float(half)]
    else:
        out["t_ci95"] = None
    return out


def order_values(loaded, seeds):
    return np.array([[_cost(loaded.by_key[(s, "sequential", tuple(o))]) for o in ORDERS] for s in seeds], dtype=float)


def mono_costs(loaded, seeds):
    out = {}
    for lang in LANGS:
        runs = [loaded.by_key[(s, "mono", (lang,))] for s in seeds]
        for task in TASKS + ("all",):
            if task == "all":
                costs = [_cost(r, lang, "first_language") for r in runs]
                censored = sum(r.get("first_language", {}).get(lang) is None for r in runs)
            else:
                costs = [_cost(r, f"{lang}/{task}", "first_task") for r in runs]
                censored = sum(r.get("first_task", {}).get(f"{lang}/{task}") is None for r in runs)
            out[f"{lang}/{task}"] = dict(rmst=float(np.mean(costs)), restricted_costs=costs, censored=censored, events=len(costs) - censored,
                                         cap=runs[0]["cap"], seed_order=list(seeds), **{"summary": _stats(costs)})
    return out


def _cell_attainment(runs, cap_key="cap"):
    cells = {}
    for cell in CELLS:
        costs = [_cost(r, cell, "first_task") for r in runs]
        censored = sum(r.get("first_task", {}).get(cell) is None for r in runs)
        cells[cell] = dict(rmst=float(np.mean(costs)), censored=censored, events=len(costs) - censored, summary=_stats(costs))
    languages = {}
    for lang in LANGS:
        costs = [_cost(r, lang, "first_language") for r in runs]
        censored = sum(r.get("first_language", {}).get(lang) is None for r in runs)
        languages[lang] = dict(rmst=float(np.mean(costs)), censored=censored, events=len(costs) - censored, summary=_stats(costs))
    return dict(cells=cells, languages=languages)


def mixed_summary(loaded, seeds):
    runs = [loaded.by_key[(s, "mixed", LANGS)] for s in seeds]
    costs = [_cost(r) for r in runs]
    return dict(rmst=float(np.mean(costs)), events=sum(r["stop_reason"] == "mastered" for r in runs), restricted_costs=costs,
                cap=runs[0]["cap"], summary=_stats(costs), seed_order=list(seeds), **_cell_attainment(runs))


def per_order(values, loaded, seeds):
    out = {}
    for i, order in enumerate(ORDERS):
        runs = [loaded.by_key[(s, "sequential", tuple(order))] for s in seeds]
        costs = values[:, i].tolist()
        stages = []
        for j, lang in enumerate(order):
            rows = [r["stages"][j] for r in runs if len(r.get("stages", [])) > j]
            stages.append(dict(language=lang, n=len(rows), mastered=sum(s["reason"] == "mastered" for s in rows),
                               stage_cap=sum(s["reason"] == "stage_cap" for s in rows),
                               exposures=_stats([s["exposures"] for s in rows]) if rows else None))
        review = [r["seen"] - r["review_start"] for r in runs if r.get("review_start") is not None]
        out["-".join(order)] = dict(rmst=float(np.mean(costs)), events=sum(r["stop_reason"] == "mastered" for r in runs), restricted_costs=costs,
                                    cap=runs[0]["cap"], summary=_stats(costs), stages=stages,
                                    review=dict(n=len(review), exposures=_stats(review) if review else None,
                                                review_start=_stats([r["review_start"] for r in runs if r.get("review_start") is not None]) if review else None),
                                    **_cell_attainment(runs))
    return out


def _lang_mean(row, lang):
    return float(np.mean([row["scores"][f"{lang}/{t}"][0] / row["scores"][f"{lang}/{t}"][1] for t in TASKS]))


def _curve(points):
    """points: list of (offset, value) over runs -> sorted list of dict(since, mean, n) keyed by offset."""
    grouped = defaultdict(list)
    for offset, value in points:
        grouped[int(offset)].append(value)
    return [(k, float(np.mean(v)), len(v)) for k, v in sorted(grouped.items())]


def forgetting_recovery(loaded, seeds, threshold=.8):
    by_order = {}
    for order in ORDERS:
        runs = [loaded.by_key[(s, "sequential", tuple(order))] for s in seeds]
        entry = {}
        for stage_index, label in ((0, "first_language"), (1, "second_language")):
            lang = order[stage_index]
            at_end, min_after, at_review, at_run_end, drop_min, drop_review = [], [], [], [], [], []
            recovered, recovery_exposures, confirmed_after_review = 0, [], 0
            curve_points, recovery_points, per_cell = [], [], {f"{lang}/{t}": dict(at_stage_end=[], min_after=[], at_review_start=[]) for t in TASKS}
            n = 0
            for r in runs:
                sched = r["_scheduled"]
                if len(r.get("stages", [])) <= stage_index or r.get("review_start") is None:
                    continue
                stage = r["stages"][stage_index]
                end_panel = max((e for e in sched if e["seen"] <= stage["end"]), key=lambda e: e["seen"], default=None)
                later = [e for e in sched if e["seen"] > stage["end"]]
                if end_panel is None or not later:
                    continue
                n += 1
                a_end = _lang_mean(end_panel, lang)
                after = [_lang_mean(e, lang) for e in later]
                review_panel = max((e for e in sched if e["seen"] <= r["review_start"]), key=lambda e: e["seen"])
                a_review = _lang_mean(review_panel, lang)
                at_end.append(a_end); min_after.append(min(after)); at_review.append(a_review); at_run_end.append(after[-1])
                drop_min.append(a_end - min(after)); drop_review.append(a_end - a_review)
                for t in TASKS:
                    cell = f"{lang}/{t}"
                    per_cell[cell]["at_stage_end"].append(end_panel["scores"][cell][0] / end_panel["scores"][cell][1])
                    per_cell[cell]["min_after"].append(min(e["scores"][cell][0] / e["scores"][cell][1] for e in later))
                    per_cell[cell]["at_review_start"].append(review_panel["scores"][cell][0] / review_panel["scores"][cell][1])
                curve_points.append((0, a_end))
                curve_points.extend((e["seen"] - end_panel["seen"], _lang_mean(e, lang)) for e in later)
                review_rows = [e for e in sched if e["seen"] >= r["review_start"]]
                recovery_points.extend((e["seen"] - r["review_start"], _lang_mean(e, lang)) for e in review_rows)
                hit = next((e for e in review_rows if all(e["scores"][f"{lang}/{t}"][0] / e["scores"][f"{lang}/{t}"][1] >= threshold for t in TASKS)), None)
                if hit is not None:
                    recovered += 1
                    recovery_exposures.append(hit["seen"] - r["review_start"])
                first = r.get("first_language", {}).get(lang)
                if first is not None and first >= r["review_start"]:
                    confirmed_after_review += 1
            if n == 0:
                entry[label] = dict(language=lang, n=0)
                continue
            entry[label] = dict(
                language=lang, stage_index=stage_index, n=n,
                at_stage_end=_stats(at_end), min_after=_stats(min_after), at_review_start=_stats(at_review), at_run_end=_stats(at_run_end),
                drop_to_min=_stats(drop_min), drop_at_review_start=_stats(drop_review),
                recovered_within_review=dict(count=recovered, n=n, fraction=recovered / n,
                                             definition=f"first scheduled panel at or after review start with all four cells >= {threshold} (single panel; "
                                                        "not the confirmed mastery rule)"),
                recovery_exposures=dict(values=recovery_exposures, summary=_stats(recovery_exposures) if recovery_exposures else None),
                confirmed_mastery_after_review=dict(count=confirmed_after_review, n=n, definition="first_language recorded at or after review start"),
                per_cell={cell: {k: _stats(v) for k, v in d.items()} for cell, d in per_cell.items()},
                curve=[dict(since_stage_end=k, mean=m, n=c) for k, m, c in _curve(curve_points)],
                recovery_curve=[dict(since_review_start=k, mean=m, n=c) for k, m, c in _curve(recovery_points)])
        by_order["-".join(order)] = entry
    return dict(definition="Language accuracy = mean of its four full-panel cells (1,000 items each) on the scheduled panel; stage end panel = last panel "
                           "at or before the stage end; min_after over all later panels to run end; offsets in training examples.",
                threshold=threshold, by_order=by_order)


def _acc_table(rows):
    """rows: list of {cell: [c, n]} -> cell -> aggregate."""
    out = {}
    cells = sorted({c for r in rows for c in r}, key=lambda c: CELLS.index(c) if c in CELLS else 99)
    for cell in cells:
        pairs = [r[cell] for r in rows if cell in r]
        accs = [c / n for c, n in pairs]
        pooled = [int(sum(c for c, _ in pairs)), int(sum(n for _, n in pairs))]
        out[cell] = dict(mean=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, n=len(accs),
                         min=float(min(accs)), max=float(max(accs)), pooled=pooled, pooled_wilson=wilson(*pooled),
                         fraction_at_or_above_0_8=float(np.mean([a >= .8 for a in accs])))
    return out


def independent_test(loaded, seeds):
    primary, auxiliary = {}, {}
    for mode, order in CONDITIONS:
        runs = [loaded.by_key[(s, mode, tuple(order))] for s in seeds]
        name = condition_name(mode, order)
        tests = [r["_test"] for r in runs if r.get("_test")]
        primary[name] = dict(n_runs=len(runs), n_with_test=len(tests), cells=_acc_table([t["scores"] for t in tests])) if tests else dict(n_runs=len(runs), n_with_test=0)
        aux = {}
        aux_test = [t["auxiliary_scores"] for t in tests if t.get("auxiliary_scores")]
        if aux_test:
            aux["test"] = _acc_table(aux_test)
        for panel in ("dev_a", "dev_b"):
            rows = [r["_aux"]["scores"][panel] for r in runs if r.get("_aux") and panel in r["_aux"].get("scores", {})]
            if rows:
                aux[panel] = _acc_table(rows)
        auxiliary[name] = aux
    return dict(primary=primary, auxiliary=auxiliary,
                primary_note="Held-out test split scored once on the terminal checkpoint (primary.pt); never used for training or mastery.",
                auxiliary_note="Outer-frame auxiliary panels (test view, dev_a, dev_b) scored on the terminal checkpoint; the auxiliary frame was never "
                               "used for mastery decisions and is reported separately as transfer evidence only.")


def budget_and_failures(loaded, ledger_path, failed_root, stop_marker=None):
    records = list(loaded.records.values())
    by_mode = defaultdict(float)
    by_device = defaultdict(float)
    for r in records:
        by_mode[r["mode"]] += r.get("gpu_hours", 0.0)
        by_device[str(r.get("_device"))] += r.get("gpu_hours", 0.0)
    out = dict(completed_runs=len(records), gpu_hours_completed_runs=float(sum(r.get("gpu_hours", 0.0) for r in records)),
               gpu_hours_by_mode=dict(by_mode), gpu_hours_by_device=dict(by_device),
               wall_hours_completed_runs=float(sum(r.get("elapsed_seconds", 0.0) for r in records) / 3600))
    ledger = _load_json(Path(ledger_path)) if ledger_path else None
    if ledger:
        main = {k: v for k, v in ledger["runs"].items() if k.startswith("main-real-")}
        status = defaultdict(lambda: dict(count=0, used_hours=0.0, reserved_hours=0.0))
        for v in main.values():
            s = status[v.get("status")]
            s["count"] += 1; s["used_hours"] += v.get("used", 0.0) or 0.0; s["reserved_hours"] += v.get("reserved", 0.0) or 0.0
        out["ledger"] = dict(path=str(ledger_path), main_allocation_hours=ledger["allocations"].get("main"), main_entries=len(main),
                             by_status={k: dict(v) for k, v in status.items()},
                             used_hours_all_categories=float(sum((v.get("used", 0.0) or 0.0) for v in ledger["runs"].values())),
                             reserved_hours_all_categories=float(sum((v.get("reserved", 0.0) or 0.0) for v in ledger["runs"].values())),
                             note="Ledger has no cancellation API; the reservations of runs never started remain 'reserved' and are reported as unused.")
    failures = []
    failed_root = Path(failed_root) if failed_root else None
    if failed_root and failed_root.exists():
        for folder in sorted(p for p in failed_root.iterdir() if p.is_dir()):
            attempt = next(iter(folder.glob("attempt-*.json")), None)
            failure = _load_json(folder / "failure.json")
            failures.append(dict(folder=folder.name, attempt=_load_json(attempt) if attempt else None,
                                 failure=failure, kernel_xid=(folder / "kernel-xid.txt").exists(), preserved_run_folder=(folder / "run-folder").exists()))
    out["failed_attempts"] = failures
    if stop_marker and Path(stop_marker).exists():
        out["stop_marker"] = Path(stop_marker).read_text(encoding="utf-8")
    out["missing_runs"] = loaded.missing
    out["nonterminal_runs"] = loaded.nonterminal
    return out


def make_figures(report, loaded, seeds, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    names = ["-".join(o) for o in ORDERS]
    # 1. restricted mean cost by order (+ mixed)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = names + ["mixed"]
    means = [report["per_order"][n]["rmst"] for n in names] + [report["mixed"]["rmst"]]
    cis = [report["per_order"][n]["summary"]["t_ci95"] for n in names] + [report["mixed"]["summary"]["t_ci95"]]
    err = [[m - (ci[0] if ci else m) for m, ci in zip(means, cis)], [(ci[1] if ci else m) - m for m, ci in zip(means, cis)]]
    ax.bar(labels, means, yerr=err, capsize=3, color=["#4c72b0"] * 6 + ["#8c8c8c"])
    for i, n in enumerate(labels):
        costs = report["per_order"][n]["restricted_costs"] if n != "mixed" else report["mixed"]["restricted_costs"]
        ax.scatter(np.full(len(costs), i) + np.random.default_rng(0).uniform(-.2, .2, len(costs)), costs, s=6, color="black", alpha=.35)
    ax.axhline(900000, ls="--", lw=.8, color="black"); ax.set_ylabel("Restricted cost (examples, cap 900k)")
    ax.set_title(f"Restricted mean cost to three-language mastery, N={len(seeds)} seeds (descriptive 95% t CI)")
    fig.tight_layout(); fig.savefig(outdir / "rmst-by-order.png", dpi=150); plt.close(fig)
    # 2. monolingual attainment
    fig, ax = plt.subplots(figsize=(8, 3.5))
    cells = [f"{l}/{t}" for l in LANGS for t in TASKS + ("all",)]
    vals = [report["mono"][c]["rmst"] for c in cells]
    ax.bar(cells, vals, color=["#dd8452" if c.endswith("all") else "#55a868" for c in cells])
    for i, c in enumerate(cells):
        ax.text(i, vals[i], f"cens {report['mono'][c]['censored']}", ha="center", va="bottom", fontsize=6)
    ax.axhline(200000, ls="--", lw=.8, color="black"); ax.set_ylabel("Restricted cost (cap 200k)"); ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.set_title("Monolingual per-cell first attainment (cap-censored)")
    fig.tight_layout(); fig.savefig(outdir / "mono-attainment.png", dpi=150); plt.close(fig)
    # 3. forgetting and recovery of the first language
    fr = report["forgetting_recovery"]["by_order"]
    fig, axes = plt.subplots(2, 6, figsize=(16, 5.5), sharey=True)
    for i, n in enumerate(names):
        e = fr[n].get("first_language", {})
        for row, key, xkey, title in ((0, "curve", "since_stage_end", "after stage end"), (1, "recovery_curve", "since_review_start", "after review start")):
            ax = axes[row, i]
            pts = [p for p in e.get(key, []) if p["n"] >= max(3, len(seeds) // 10)]
            if pts:
                ax.plot([p[xkey] for p in pts], [p["mean"] for p in pts], color="#c44e52")
            ax.axhline(.8, ls="--", lw=.8, color="black"); ax.set_ylim(0.4, 1.02)
            ax.set_title(f"{n}: {e.get('language', '?')} {title}", fontsize=8); ax.set_xlabel("examples", fontsize=7); ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("First-language accuracy"); axes[1, 0].set_ylabel("First-language accuracy")
    fig.suptitle(f"First-language forgetting and recovery, mean over seeds (points with n >= {max(3, len(seeds) // 10)})")
    fig.tight_layout(); fig.savefig(outdir / "forgetting-recovery-first-language.png", dpi=150); plt.close(fig)
    # 4. independent test heatmap
    conds = [condition_name(m, o) for m, o in CONDITIONS]
    matrix = np.full((len(conds), len(CELLS)), np.nan)
    for i, c in enumerate(conds):
        for j, cell in enumerate(CELLS):
            v = report["independent_test"]["primary"][c].get("cells", {}).get(cell)
            if v:
                matrix[i, j] = v["mean"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(matrix, vmin=.5, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(CELLS))); ax.set_xticklabels(CELLS, rotation=60, fontsize=7); ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=7)
    for i in range(len(conds)):
        for j in range(len(CELLS)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if matrix[i, j] < .8 else "black")
    fig.colorbar(im, ax=ax, label="Mean test accuracy"); ax.set_title("Independent test accuracy on terminal checkpoints (mean over seeds)")
    fig.tight_layout(); fig.savefig(outdir / "independent-test.png", dpi=150); plt.close(fig)
    # 5. mean scheduled-panel trajectories per order (averaged across runs still training at each point)
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.4), sharey=True)
    for i, order in enumerate(ORDERS):
        runs = [loaded.by_key[(s, "sequential", tuple(order))] for s in seeds]
        ax = axes[i]
        for lang, color in zip(LANGS, ("#4c72b0", "#dd8452", "#55a868")):
            grouped = defaultdict(list)
            for r in runs:
                for e in r["_scheduled"]:
                    grouped[e["seen"]].append(_lang_mean(e, lang))
            pts = [(k, np.mean(v)) for k, v in sorted(grouped.items()) if len(v) >= max(3, len(seeds) // 10)]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=lang, color=color, lw=1)
        ax.axhline(.8, ls="--", lw=.8, color="black"); ax.set_title("-".join(order), fontsize=9); ax.set_xlabel("examples", fontsize=7); ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Language accuracy"); axes[0].legend(fontsize=7)
    fig.suptitle("Mean scheduled-panel accuracy per language (runs still training)")
    fig.tight_layout(); fig.savefig(outdir / "trajectories-by-order.png", dpi=150); plt.close(fig)
    return sorted(p.name for p in outdir.glob("*.png"))


def analyze(runs, manifest, output, ledger=None, failed=None, design=None, stop_marker=None, figures=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Keep previous reports; {output} exists")
    loaded = load_records(runs, manifest)
    main_seeds = loaded.manifest["protocol"]["main_seeds"]
    selection = select_prefix(loaded, main_seeds)
    seeds = selection["included_seeds"]
    if len(seeds) < 2:
        raise ValueError("Fewer than two complete seed blocks")
    values = order_values(loaded, seeds)
    cap = loaded.by_key[(seeds[0], "sequential", tuple(ORDERS[0]))]["cap"]
    first = next(iter(loaded.records.values()))
    report = dict(kind="main_v5_1_truncated_analysis", deviation_id=DEVIATION_ID, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  script_hash=file_hash(__file__), runs_root=str(runs), manifest=str(manifest),
                  hashes={k: first.get(k) for k in HASH_KEYS + ("training_hash",)},
                  selection=selection,
                  deviation=dict(pre_registered_n=len(main_seeds), analyzed_n=len(seeds), completed_runs=len(loaded.records),
                                 runs_in_analysis=selection["runs_in_analysis"], runs_not_started=len(loaded.missing),
                                 cause="operator-scheduled stop at 2026-09-23T00:00Z (KST 09:00) ordered before completion for reasons external to the "
                                       "study; on 2026-09-23 the user decided to end the runs and analyze the runs completed so far",
                                 outcome_blind="no order contrast, RMST, or per-order attainment was read before the stop or before this decision",
                                 statistical_status="the pre-registered summary is applied to a prefix of the pre-registered seed order; power and FWER "
                                                    "guarantees were computed for N=80 and are reported as brackets from the frozen design table"),
                  prespecified_summary=summarize(values, cap=cap),
                  per_order=per_order(values, loaded, seeds), mixed=mixed_summary(loaded, seeds), mono=mono_costs(loaded, seeds),
                  forgetting_recovery=forgetting_recovery(loaded, seeds),
                  independent_test=independent_test(loaded, seeds),
                  budget=budget_and_failures(loaded, ledger, failed, stop_marker),
                  interpretation="Report intervals and observed mastery. Conclusions are conditional on the frozen items, input representation, model and "
                                 "training policy; no universal language ranking, no 15 pairwise claims, no biological claim; nonsignificance is not equivalence.")
    report.update({k: first.get(k) for k in ("review_mode", "human_reviewed", "review_limitations", "protocol_amendment")})
    if design:
        d = _load_json(Path(design))
        table = d.get("primary_power_table", {})
        lower = sorted(int(k) for k in table if int(k) <= len(seeds))
        upper = sorted(int(k) for k in table if int(k) >= len(seeds))
        report["design_bracket"] = dict(design_version=d.get("design_version"), chosen=d.get("chosen"), minimum_passing_n=d.get("minimum_passing_n"),
                                        bracket={str(k): table[str(k)] for k in ([lower[-1]] if lower else []) + ([upper[0]] if upper else [])},
                                        note="Frozen pre-simulation values at the table Ns bracketing the analyzed N; no new simulation was run.")
    output.mkdir(parents=True)
    if figures:
        report["figures"] = make_figures(report, loaded, seeds, output / "figures")
    write_json(output / "analysis.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/main-v5.1")
    parser.add_argument("--manifest", default="reports/launch-manifest-v5.1-N80.json")
    parser.add_argument("--ledger", default="runs/gpu-ledger.json")
    parser.add_argument("--failed", default="runs/main-v5.1-failed-attempts")
    parser.add_argument("--design", default="reports/design-extended-v5.1-N2.json")
    parser.add_argument("--stop-marker", default="runs/main-v5.1-campaign.stopped-by-operator")
    parser.add_argument("--output", default="reports/main-v5.1-analysis-truncated")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    report = analyze(args.runs, args.manifest, args.output, args.ledger, args.failed, args.design, args.stop_marker, figures=not args.no_figures)
    sel = report["selection"]
    print(f"analyzed seeds: {sel['n_included']} of {sel['pre_registered_n']} ({sel['runs_in_analysis']} runs); output {args.output}")


if __name__ == "__main__":
    main()

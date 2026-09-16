from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import numpy as np

from .protocol import ORDERS, write_json
from .statistics import summarize, wilson


def analyze(run_root, output):
    root, output = Path(run_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("*/summary.json")]
    if not records:
        raise ValueError("No run summaries")
    eligible = [r for r in records if r["cohort"] == "main" and not r["synthetic"] and
                r["stop_reason"] in ("mastered", "administrative_cap")]
    if not eligible:
        report = dict(status="no_primary_evidence", reason="Only smoke/pilot/interrupted runs present", observed_runs=len(records))
        write_json(output/"analysis.json", report)
        return report
    for key in ("protocol_hash", "dataset_hash", "graph_hash", "tokenizer_hash", "code_hash"):
        if len({r[key] for r in eligible}) != 1:
            raise ValueError(f"Incompatible main-study records: {key}")
    sequential = [r for r in eligible if r["mode"] == "sequential"]
    by_seed = defaultdict(dict)
    for r in sequential:
        order = tuple(r["order"])
        if order in by_seed[r["seed"]]:
            raise ValueError("Duplicate condition/seed")
        by_seed[r["seed"]][order] = r
    expected_seeds = set(eligible[0]["protocol"]["main_seeds"])
    mixed = [r for r in eligible if r['mode'] == 'mixed']
    if {r['seed'] for r in mixed} != expected_seeds or len(mixed) != len(expected_seeds):
        raise ValueError('Incomplete mixed baseline blocks')
    if set(by_seed) != expected_seeds or any(set(g) != set(ORDERS) for g in by_seed.values()):
        raise ValueError("Incomplete prespecified seed blocks; cannot silently select complete/successful cases")
    values = np.array([[by_seed[s][o]["first_global"] or by_seed[s][o]["cap"] for o in ORDERS] for s in sorted(by_seed)])
    report = summarize(values)
    report["status"] = "analyzed"
    report['mixed'] = dict(rmst=float(np.mean([r['first_global'] or r['cap'] for r in mixed])),
                           events=sum(r['stop_reason'] == 'mastered' for r in mixed))
    report["events_by_order"] = {"-".join(o): sum(by_seed[s][o]["stop_reason"] == "mastered" for s in by_seed) for o in ORDERS}
    report["mono"] = {}
    for lang in ("en", "de", "ko"):
        mono = [r for r in eligible if r["mode"] == "mono" and r["order"] == [lang]]
        if {r["seed"] for r in mono} != expected_seeds or len(mono) != len(expected_seeds):
            raise ValueError("Incomplete monolingual blocks")
        for task in ("roles", "negation", "space", "quantity", "all"):
            costs = [r["first_language"].get(lang, r["cap"]) if task == "all" else r["first_task"].get(f"{lang}/{task}", r["cap"]) for r in mono]
            report["mono"][f"{lang}/{task}"] = dict(rmst=float(np.mean(costs)), restricted_costs=costs,
                                                      seed_order=[r["seed"] for r in mono])
    report["budget"] = dict(gpu_hours=sum(r["gpu_hours"] for r in records),
                            by_status={status: sum(r["gpu_hours"] for r in records if r["stop_reason"] == status)
                                       for status in {r["stop_reason"] for r in records}})
    report["interpretation"] = "Report intervals and observed mastery. No universal language ranking or biological claim; nonsignificance is not equivalence."
    write_json(output/"analysis.json", report)
    plot_curves(root, output)
    return report


def plot_curves(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for folder in Path(root).iterdir():
        log = folder/"events.jsonl"
        if not log.exists():
            continue
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        evaluations = [r for r in records if r["kind"] == "scheduled"]
        if not evaluations:
            continue
        peaks, forgetting = {}, []
        for row in evaluations:
            for cell,(correct,total) in row['scores'].items():
                accuracy = correct/total
                peaks[cell] = max(peaks.get(cell,accuracy),accuracy)
                forgetting.append(dict(seen=row['seen'],cell=cell,accuracy=accuracy,
                                       drop_from_observed_peak=peaks[cell]-accuracy))
        write_json(output/f'{folder.name}-forgetting.json',dict(
            definition='Drop from best previous scheduled full-panel accuracy; A/B sampling noise remains.',points=forgetting))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
        for ax, lang in zip(axes, ("en", "de", "ko")):
            for task in ("roles", "negation", "space", "quantity"):
                points = [(r["seen"], r["scores"][f"{lang}/{task}"]) for r in evaluations if f"{lang}/{task}" in r["scores"]]
                if points:
                    ax.plot([x for x,_ in points], [a/b for _,(a,b) in points], label=task)
            ax.axhline(.8, ls="--", color="black", lw=.8)
            ax.set(title=lang, xlabel="Training examples", ylim=(0,1.02))
        axes[0].set_ylabel("Full-panel accuracy")
        axes[0].legend(fontsize=7)
        fig.suptitle(folder.name)
        fig.tight_layout()
        fig.savefig(output/f"{folder.name}.png", dpi=150)
        plt.close(fig)


def power_summary(path, output):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    summary = dict(records=len(records), replications_per_scenario=sorted({r["repetitions"] for r in records}),
                   evaluated_design_replicates=sum(r["repetitions"] for r in records),
                   generated_max_n_datasets=sum(r["repetitions"] for r in records if r["n"] == 20),
                   warning="A scenario map, not achieved empirical study power. Values depend on the stated DGM.", by_n={})
    for n in (10,15,20):
        nulls = [r["rate"] for r in records if r["n"] == n and r["effect"] == 0]
        target = [r for r in records if r["n"] == n and r["effect"] == .2]
        summary["by_n"][n] = dict(null_fwer_range=[min(nulls),max(nulls)],
                                  target_power_range=[min(r["rate"] for r in target),max(r["rate"] for r in target)],
                                  target_scenarios=len(target),
                                  conservative_power_gate_passes=sum(r["conservative_power_ci"][0] >= .8 for r in target))
    write_json(output, summary)
    return summary

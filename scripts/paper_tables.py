"""Generate LaTeX table bodies for the TMLR draft (paper/tables/*.tex) from the frozen analysis and design reports.
No new statistics are computed. Run from the repository root: .venv/bin/python scripts/paper_tables.py"""
"""Generate LaTeX table bodies for the TMLR draft from the frozen analysis and design reports (no new statistics)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)
A = json.loads((ROOT / "reports/main-v5.1-analysis-truncated/analysis.json").read_text())
D = json.loads((ROOT / "reports/design-extended-v5.1-N2.json").read_text())
ORDERS = ["en-de-ko", "en-ko-de", "de-en-ko", "de-ko-en", "ko-en-de", "ko-de-en"]
LANGS, TASKS = ("en", "de", "ko"), ("roles", "negation", "space", "quantity")
COND = ["mono-en", "mono-de", "mono-ko"] + ["seq-" + o for o in ORDERS] + ["mixed"]

def k(x): return f"{x:,.0f}"

# 1. per-order RMST
rows = []
for o in ORDERS:
    v = A["per_order"][o]; ci = v["summary"]["t_ci95"]
    rows.append(f"{o} & {k(v['rmst'])} & [{k(ci[0])}, {k(ci[1])}] & {k(v['summary']['sd'])} & {v['events']}/{v['summary']['n']} \\\\")
m = A["mixed"]; ci = m["summary"]["t_ci95"]
rows.append(f"mixed 1:1:1 & {k(m['rmst'])} & [{k(ci[0])}, {k(ci[1])}] & {k(m['summary']['sd'])} & {m['events']}/{m['summary']['n']} \\\\")
(OUT / "rmst.tex").write_text("\n".join(rows) + "%\n")

# 2. contrasts
c = A["prespecified_summary"]["contrasts"]
names = {"ko_last_minus_first": "KO last $-$ KO first", "en_de_adjacent_minus_separated": "EN/DE adjacent $-$ separated"}
rows = []
for key, label in names.items():
    v = c[key]
    rows.append(f"{label} & {v['difference']:+,.0f} & ${v['holm_p']:.1e}$ & [{k(v['simultaneous_ci'][0])}, {k(v['simultaneous_ci'][1])}] & [{k(v['seed_bootstrap_ci'][0])}, {k(v['seed_bootstrap_ci'][1])}] \\\\".replace("e-", r"\times 10^{-").replace("$", "$", 1))
# fix exponent formatting
def fmt_p(p):
    mant, exp = f"{p:.1e}".split("e"); return f"${mant}\\times 10^{{{int(exp)}}}$"
rows = []
for key, label in names.items():
    v = c[key]
    rows.append(f"{label} & {v['difference']:+,.0f} & {fmt_p(v['holm_p'])} & [{k(v['simultaneous_ci'][0])}, {k(v['simultaneous_ci'][1])}] & [{k(v['seed_bootstrap_ci'][0])}, {k(v['seed_bootstrap_ci'][1])}] \\\\")
(OUT / "contrasts.tex").write_text("\n".join(rows) + "%\n")

# 3. stages per order
rows = []
for o in ORDERS:
    v = A["per_order"][o]
    cells = [f"{s['language']} {s['mastered']}/{s['stage_cap']} ({k(s['exposures']['mean'])})" for s in v["stages"]]
    rows.append(f"{o} & " + " & ".join(cells) + f" & {k(v['review']['exposures']['mean'])} \\\\")
(OUT / "stages.tex").write_text("\n".join(rows) + "%\n")

# 4. mono
rows = []
for l in LANGS:
    for t in TASKS + ("all",):
        v = A["mono"][f"{l}/{t}"]; ci = v["summary"]["t_ci95"]
        label = f"\\textbf{{{l} (all four)}}" if t == "all" else f"{l}/{t}"
        rows.append(f"{label} & {k(v['rmst'])} & [{k(ci[0])}, {k(ci[1])}] & {k(v['summary']['median'])} & {v['censored']} \\\\")
(OUT / "mono.tex").write_text("\n".join(rows) + "%\n")

# 5. forgetting
rows = []
for stage in ("first_language", "second_language"):
    for o in ORDERS:
        e = A["forgetting_recovery"]["by_order"][o][stage]
        rec = e["recovery_exposures"]["summary"]["mean"]
        rows.append(f"{o} & {e['language']} ({'1st' if stage == 'first_language' else '2nd'}) & {e['at_stage_end']['mean']:.3f} & {e['min_after']['mean']:.3f} & {e['at_review_start']['mean']:.3f} & {e['at_run_end']['mean']:.3f} & {e['recovered_within_review']['count']}/{e['n']} & {k(rec)} & {e['confirmed_mastery_after_review']['count']} \\\\")
(OUT / "forgetting.tex").write_text("\n".join(rows) + "%\n")

# 6. independent test (full, 12 cells) and language means
rows, rows_lang = [], []
for cnd in COND:
    cells = A["independent_test"]["primary"][cnd]["cells"]
    vals = []
    for l in LANGS:
        for t in TASKS:
            v = cells.get(f"{l}/{t}")
            vals.append(f"{v['mean']:.3f}" if v else "--")
    rows.append(f"{cnd} & " + " & ".join(vals) + " \\\\")
    lang_means = []
    for l in LANGS:
        vs = [cells[f'{l}/{t}']['mean'] for t in TASKS if f"{l}/{t}" in cells]
        fr = [cells[f'{l}/{t}']['fraction_at_or_above_0_8'] for t in TASKS if f"{l}/{t}" in cells]
        lang_means.append(f"{sum(vs)/len(vs):.3f} ({min(vs):.3f})" if vs else "--")
    aux = A["independent_test"]["auxiliary"][cnd]
    aux_vals = [x["mean"] for p in aux.values() for x in p.values()]
    rows_lang.append(f"{cnd} & " + " & ".join(lang_means) + f" & {min(aux_vals):.2f}--{max(aux_vals):.2f} \\\\")
(OUT / "test_full.tex").write_text("\n".join(rows) + "%\n")
(OUT / "test_lang.tex").write_text("\n".join(rows_lang) + "%\n")

# 7. per-order language attainment (appendix)
rows = []
for o in ORDERS:
    v = A["per_order"][o]["languages"]
    rows.append(f"{o} & " + " & ".join(f"{k(v[l]['rmst'])} ({v[l]['censored']})" for l in LANGS) + " \\\\")
v = A["mixed"]["languages"]
rows.append("mixed & " + " & ".join(f"{k(v[l]['rmst'])} ({v[l]['censored']})" for l in LANGS) + " \\\\")
(OUT / "languages.tex").write_text("\n".join(rows) + "%\n")

# 8. design candidates
rows = []
for cnd in D["candidates"]:
    rows.append(f"{cnd['n']} & {cnd['power_lower']:.3f} & {cnd['null_fwer_upper']:.4f} & {cnd['gpu_hours']:.0f} & {cnd['calendar_hours']:.0f} \\\\")
(OUT / "design.tex").write_text("\n".join(rows) + "%\n")

# 9. budget
b = A["budget"]; L = b["ledger"]["by_status"]
rows = [f"Completed runs (753) & {b['gpu_hours_completed_runs']:.2f} & mono {b['gpu_hours_by_mode']['mono']:.2f}, sequential {b['gpu_hours_by_mode']['sequential']:.2f}, mixed {b['gpu_hours_by_mode']['mixed']:.2f} \\\\",
        f"Technical failures (6 attempts) & {L['technical_failure']['used_hours']:.2f} & all recovered without block exclusion \\\\",
        f"Reserved, never started (47 runs) & 0 & {L['reserved']['reserved_hours']:.1f} h remain reserved in the ledger \\\\",
        f"Reservation for 800 runs & -- & 409.2 h (per-run upper bound $\\times$ 1.25) \\\\",
        f"All project categories & {b['ledger']['used_hours_all_categories']:.2f} & of 672 h allocated \\\\"]
(OUT / "budget.tex").write_text("\n".join(rows) + "%\n")
print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))

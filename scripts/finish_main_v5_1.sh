#!/usr/bin/env bash
# Unattended completion step for the frozen main study (protocol v5.1, N=80, runs/main-v5.1).
# Waits for the staggered launcher to exit. If all 800 pre-specified runs have a summary, runs the pre-registered analysis
# (flystudy analyze: paired contrasts, Holm, RMST intervals; per-run curves and forgetting records), writes a run record
# (devices, stop reasons, ledger failures, GPU hours) and commits ONLY the public report files as the repository owner.
# If the launcher stopped early (a technical failure blocks dispatch), it writes a marker and exits; explicit recovery
# (scripts/main_campaign_staggered.py with a retry reservation) is required, never an automatic restart.
set -u
cd /mnt/nas4/kjune/fly-language-study || exit 1
PY=.venv/bin/python
LOG=runs/main-v5.1-finish.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "watcher started (pid $$)"
while pgrep -f "main_campaign_staggered.py --protocol configs/protocol-v5.1-main-N80.json" >/dev/null; do sleep 120; done
done_n=$(ls runs/main-v5.1/*/summary.json 2>/dev/null | wc -l)
log "launcher exited; summaries: $done_n/800"
if [ "$done_n" -ne 800 ]; then
  { echo "stopped_at_utc=$(date -u +%FT%TZ)"; echo "summaries=$done_n"; echo "recovery=run scripts/main_campaign_staggered.py again with a retry reservation for the failed run (see reports/MAIN_STUDY_PLAN_V5.md section 8)";
    for f in $(grep -lE "Traceback|^ERROR" runs/main-v5.1/*.console.log 2>/dev/null); do echo "error_log=$f: $(grep -m1 -E 'Error|ERROR' "$f" | cut -c1-200)"; done; } > runs/main-v5.1-campaign.stopped
  log "campaign incomplete; marker written; no analysis, no commit"; exit 2
fi
log "all 800 summaries present; running the pre-registered analysis"
if ! $PY -m flystudy analyze --runs runs/main-v5.1 --output reports/main-v5.1-analysis >> "$LOG" 2>&1; then log "analysis FAILED"; exit 3; fi
if ! $PY - >> "$LOG" 2>&1 <<'PYEOF'
import json, datetime
from pathlib import Path
from collections import Counter
from flystudy.protocol import file_hash, write_json
m = json.loads(Path("reports/launch-manifest-v5.1-N80.json").read_text()); out = Path("runs/main-v5.1")
L = json.load(open("runs/gpu-ledger.json"))
rows, dev, stop = [], Counter(), Counter()
for r in m["runs"]:
    s = json.loads((out/r["run_id"]/"summary.json").read_text()); md = json.loads((out/r["run_id"]/"metadata.json").read_text())
    dev[md.get("device")] += 1; stop[s["stop_reason"]] += 1
    rows.append(dict(run_id=r["run_id"], seed=r["seed"], mode=r["mode"], order=r["order"], device=md.get("device"), stop_reason=s["stop_reason"],
                     seen=s["seen"], first_global=s.get("first_global"), gpu_hours=s.get("gpu_hours"), last_reservation_id=s.get("last_reservation_id"),
                     attempts=sorted(p.name for p in (out/r["run_id"]).glob("attempt-*.json")), independent_test=s.get("independent_test")))
main_ledger = {k: v for k, v in L["runs"].items() if v["category"] == "main"}
record = dict(status="completed", completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              launch_manifest=dict(path="reports/launch-manifest-v5.1-N80.json", sha256=file_hash("reports/launch-manifest-v5.1-N80.json")),
              runs=len(rows), stop_reasons=dict(stop), device_assignment=dict(dev),
              device_note="GPU 1 fell off the bus (Xid 79) on 2026-09-17T22:00Z after 88 runs; every later run executed on cuda:0.",
              ledger=dict(statuses=dict(Counter(v["status"] for v in main_ledger.values())), used_gpu_hours=round(sum(v["used"] for v in main_ledger.values()), 3),
                          reserved_gpu_hours=round(sum(v["reserved"] for v in main_ledger.values()), 3),
                          technical_failures=[dict(reservation=k, used_hours=round(v["used"], 4)) for k, v in main_ledger.items() if v["status"] == "technical_failure"],
                          exceeded_reservation=[k for k, v in main_ledger.items() if v.get("exceeded_reservation")]),
              failed_attempt_records="runs/main-v5.1-failed-attempts/ (not in the repository)", runs_detail=rows)
write_json("reports/main-v5.1-run-record.json", record)
print("run record written:", record["stop_reasons"], record["device_assignment"], record["ledger"]["used_gpu_hours"], "h")
PYEOF
then log "run record FAILED"; exit 4; fi
date -u +%FT%TZ > runs/main-v5.1-campaign.completed
git add reports/main-v5.1-analysis/analysis.json reports/main-v5.1-run-record.json runs/gpu-ledger.json >> "$LOG" 2>&1 || { log "git add failed"; exit 5; }
git -c user.name="June woo Kang" -c user.email="133371686+Juunary@users.noreply.github.com" commit -q -m "Main study v5.1 (N=80, 800 runs) completed: pre-registered analysis and run record

Unattended completion step (scripts/finish_main_v5_1.sh): flystudy analyze on the complete frozen matrix, plus the run
record (device assignment, stop reasons, ledger failures, GPU hours). Per-run curves and forgetting records stay on
disk under reports/main-v5.1-analysis/ (not committed); interpretation limits are stated in analysis.json." >> "$LOG" 2>&1 || { log "commit failed"; exit 6; }
git fetch -q origin && git merge-base --is-ancestor origin/main HEAD && git push -q origin main >> "$LOG" 2>&1 && log "pushed $(git rev-parse --short HEAD)" || log "push not done (not fast-forward or network); commit is local"
log "finished"

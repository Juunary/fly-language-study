#!/usr/bin/env bash
# Operator-scheduled stop of the main study (protocol v5.1, runs/main-v5.1) at a fixed UTC epoch.
# Order: stop the completion watcher (no analysis on an incomplete matrix), stop the launcher (no new dispatch), send
# SIGINT to running runs (KeyboardInterrupt: the ledger is charged and an attempt record written by the accounting
# wrapper; the last periodic checkpoint stays for a later resume under a retry reservation), then write a marker.
# Nothing is restarted automatically. Recovery: reports/MAIN_STUDY_PLAN_V5.md section 8.
set -u
cd /mnt/nas4/kjune/fly-language-study || exit 1
TARGET=$1; LOG=runs/main-v5.1-stop.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "stop scheduled for $(date -u -d @"$TARGET" +%FT%TZ) (pid $$)"
while [ "$(date -u +%s)" -lt "$TARGET" ]; do sleep 30; done
log "stop time reached"
wpid=$(cat runs/main-v5.1-finish.pid 2>/dev/null); [ -n "$wpid" ] && kill "$wpid" 2>/dev/null && log "watcher $wpid stopped"
for pid in $(pgrep -f '^bash scripts/finish_main_v5_1'); do kill "$pid" 2>/dev/null; done
for pid in $(pgrep -f '^\.venv/bin/python scripts/main_campaign_staggered'); do kill -TERM "$pid" 2>/dev/null && log "launcher $pid SIGTERM"; done
runs=$(pgrep -f 'flystudy run --protocol configs/protocol-v5.1-main-N80.json' || true)
for pid in $runs; do rid=$(tr '\0' ' ' < /proc/$pid/cmdline | grep -o -- '--run-id [^ ]*'); kill -INT "$pid" 2>/dev/null && log "run $pid ($rid) SIGINT"; done
for i in $(seq 1 24); do pgrep -f 'flystudy run --protocol configs/protocol-v5.1-main-N80.json' >/dev/null || break; sleep 5; done
for pid in $(pgrep -f 'flystudy run --protocol configs/protocol-v5.1-main-N80.json'); do kill -KILL "$pid" 2>/dev/null && log "run $pid SIGKILL after 120 s"; done
for pid in $(pgrep -f '^\.venv/bin/python scripts/main_campaign_staggered'); do kill -KILL "$pid" 2>/dev/null; done
done_n=$(ls runs/main-v5.1/*/summary.json 2>/dev/null | wc -l)
{ echo "stopped_at_utc=$(date -u +%FT%TZ)"; echo "reason=operator-scheduled stop (KST 2026-09-23 09:00)"; echo "summaries=$done_n";
  for d in runs/main-v5.1/main-real-*/; do [ -f "$d/summary.json" ] || echo "interrupted_run=$(basename "$d") checkpoint=$([ -f "$d/latest.pt" ] && echo yes || echo no)"; done;
  echo "recovery=reserve a retry id for each interrupted run, map it with resume=<run>/latest.pt in configs/main-v5.1-retries.json, relaunch scripts/main_campaign_staggered.py, then scripts/finish_main_v5_1.sh"; } > runs/main-v5.1-campaign.stopped-by-operator
log "all stopped; summaries $done_n/800; marker written"

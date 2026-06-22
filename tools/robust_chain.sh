#!/bin/bash
# robust-s06 chain: windowed-life-noise training -> standardized eval -> 5 live gates.
# Detached overnight runner; everything logged, gate reports archived per replica.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN=robust-s06
ts() { date "+%F %T"; }
log() { echo "[$(ts)] $*"; }

log "=== chain start: $RUN ==="
if [ ! -f "runs/$RUN/params.msgpack" ]; then
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
  $PY -m zappy_rl.train --run-name "$RUN" \
    --width 20 --height 24 --n-agents 6 --num-envs 256 \
    --max-episode-ticks 6144 --eval-max-ticks 6144 \
    --ent-coef-token 0.01 --total-env-steps 400000000 \
    --overhead 10 --density-scale 0.85 \
    --life-noise 126 --life-noise-window 128 \
    --log-every 100 --wandb auto \
    --init-actor runs/speedrun-s04/params.msgpack
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "runs/$RUN/params.msgpack" ]; then
    log "TRAIN FAILED (rc=$rc) — aborting chain"; exit 1
  fi
else
  log "SKIP train (checkpoint exists)"
fi

log "standardized eval (noiseless)"
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY tools/eval_time_to_l8.py \
  --run "runs/$RUN" --overhead 8 --density-scale 1.0 \
  --eval-max-ticks 8192 --eval-envs 2048 || log "WARN eval failed"

log "live gates: 5 greedy replicas"
wins=0
for i in 1 2 3 4 5; do
  $PY tools/run_deploy_gate.py --params "runs/$RUN/params.msgpack" \
    --width 20 --height 24 --agents 6 --freq 100 --duration 120 --greedy \
    > "/tmp/zgate_s06_$i.console" 2>&1
  cp "runs/$RUN/deploy_gate.json" "/tmp/zgate_s06_$i.report.json" 2>/dev/null
  t=$($PY - <<PYEOF
import json
d = json.load(open("/tmp/zgate_s06_$i.report.json"))
lv = sorted(d["server_side"]["max_level_per_player"].values())
print(d["t_all_l8_ticks"] if d["t_all_l8_ticks"] else f"LOSS(levels={lv})")
PYEOF
)
  case "$t" in LOSS*) ;; *) wins=$((wins+1));; esac
  log "replica $i: t_all_l8 = $t"
done
log "=== chain done: $wins/5 live wins ==="

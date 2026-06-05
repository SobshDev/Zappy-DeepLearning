#!/bin/sh
# Overnight curriculum chain — Phase 5 stages 3→5, warm-started from ritual8x8-v1.
#
# Each segment is a separate `zappy_rl.train` process that warm-starts its actor
# from the previous segment's checkpoint (the critic is rebuilt per stage: its
# world_extra dim changes with map/agent count, so only the actor transfers).
# Per-segment process isolation = periodic checkpoints + full GPU memory release.
#
# Resilience contract (train() writes params.msgpack only AFTER the full
# training loop, then runs eval and writes eval.json):
#   * params.msgpack exists  => training finished; skip the segment on re-run
#     and advance the warm-start chain (even if the post-train eval crashed);
#   * eval.json exists       => segment fully complete including eval;
#   * neither                => (re)train; a FAILED segment is logged and the
#     chain continues — the next segment warm-starts from the last GOOD
#     checkpoint instead.
#
# Sizing (measured on the RTX 4090 with XLA_PYTHON_CLIENT_PREALLOCATE=false,
# same setting used here): 12x12/3p@1024 envs = 298k SPS, 16x16/4p@512 = 181k,
# 20x24/6p@256 = 95k (1024/512 envs OOM at 4p/6p — do not raise num_envs).
# Total planned wall time ~7.5 h.
#
# DRY=1 runs the same chain with tiny step counts and -dry run names
# (W&B disabled) to test the launch/skip/resume mechanics in ~3 min.
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1   # stdout goes to a logfile — don't buffer it

DRY="${DRY:-0}"
if [ "$DRY" = "1" ]; then
    SUFFIX="-dry"; WANDB=disabled; EVAL_ENVS=64
    S1=2000000; S2=2000000; S3=2000000; S4=2000000
else
    SUFFIX=""; WANDB=auto; EVAL_ENVS=512
    S1=800000000; S2=1200000000; S3=800000000; S4=800000000
fi

LAST_GOOD=runs/ritual8x8-v1/params.msgpack
ts() { date "+%F %T"; }

# run_segment <name> <width> <height> <agents> <num_envs> <steps>
run_segment() {
    name="$1$SUFFIX"; w="$2"; h="$3"; a="$4"; envs="$5"; steps="$6"
    if [ -f "runs/$name/params.msgpack" ]; then
        echo "[$(ts)] SKIP $name (checkpoint exists — training already finished)"
        LAST_GOOD="runs/$name/params.msgpack"
        return 0
    fi
    echo "[$(ts)] START $name: ${w}x${h} ${a} agents, $steps steps, warm-start $LAST_GOOD"
    if "$PY" -m zappy_rl.train \
        --run-name "$name" --width "$w" --height "$h" --n-agents "$a" \
        --num-envs "$envs" --max-episode-ticks 8192 --eval-max-ticks 8192 \
        --ent-coef-token 0.01 --total-env-steps "$steps" \
        --eval-envs "$EVAL_ENVS" --log-every 100 \
        --wandb "$WANDB" --init-actor "$LAST_GOOD"
    then
        if [ -f "runs/$name/params.msgpack" ]; then
            echo "[$(ts)] DONE  $name"
            LAST_GOOD="runs/$name/params.msgpack"
        else
            echo "[$(ts)] WARN  $name exited 0 but wrote no checkpoint — not advancing warm-start"
        fi
    else
        rc=$?
        if [ -f "runs/$name/params.msgpack" ]; then
            echo "[$(ts)] PARTIAL $name (exit $rc after checkpoint — post-train eval failed); advancing warm-start"
            LAST_GOOD="runs/$name/params.msgpack"
        else
            echo "[$(ts)] FAILED $name (exit $rc) — chain continues from $LAST_GOOD"
        fi
    fi
}

echo "[$(ts)] overnight chain starting (DRY=$DRY)"

#           name              W  H  agents envs  steps
run_segment ritual12x12-3p-n1 12 12 3      1024  "$S1"   # stage 3: ~45 min
run_segment ritual16x16-4p-n1 16 16 4      512   "$S2"   # stage 4: ~110 min
run_segment ritual20x24-6p-n1 20 24 6      256   "$S3"   # stage 5: ~140 min
run_segment ritual20x24-6p-n2 20 24 6      256   "$S4"   # stage 5 cont.: ~140 min

# Best-effort live deploy gate on the stage-3 checkpoint (3 adapters, 12x12).
# Writes runs/ritual12x12-3p-n1*/deploy_gate.json. Never fails the chain.
GATE_PARAMS="runs/ritual12x12-3p-n1$SUFFIX/params.msgpack"
if [ -f "$GATE_PARAMS" ]; then
    echo "[$(ts)] deploy gate on $GATE_PARAMS"
    "$PY" tools/run_deploy_gate.py --params "$GATE_PARAMS" \
        --width 12 --height 12 --agents 3 --duration 180 \
        || echo "[$(ts)] deploy gate FAILED (non-fatal, see above)"
fi

echo "[$(ts)] ================ MORNING SUMMARY ================"
for d in ritual12x12-3p-n1 ritual16x16-4p-n1 ritual20x24-6p-n1 ritual20x24-6p-n2; do
    f="runs/$d$SUFFIX/eval.json"
    if [ -f "$f" ]; then
        "$PY" - "$d$SUFFIX" "$f" <<'EOF'
import json, sys
name, path = sys.argv[1], sys.argv[2]
s = json.load(open(path))["stochastic"]
print(f"  {name:24s} max_level {s['max_level_mean']:.2f}  "
      f"reach_l3 {s['reach_l3_rate']:.2f}  survival {s['survival_ge_2000']:.2f}  "
      f"mean_ticks {s['mean_ticks']:.0f}")
EOF
    else
        echo "  $d$SUFFIX: DID NOT COMPLETE"
    fi
done
g="runs/ritual12x12-3p-n1$SUFFIX/deploy_gate.json"
[ -f "$g" ] && echo "  deploy gate: $("$PY" -c "import json;d=json.load(open('$g'));print(d.get('gate','?'))" 2>/dev/null || echo '?')"
echo "[$(ts)] overnight chain finished"

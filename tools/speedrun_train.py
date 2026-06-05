#!/usr/bin/env python3
"""Speedrun driver: fine-tune toward the time-to-L8 theoretical limit.

Loops warm-started 20x24/6-agent segments over a rotation of sim-to-real
robustness conditions (cadence ``--overhead``, resource ``--density-scale``)
and a tightening horizon, then scores every checkpoint under ONE standardized
condition (overhead=8, density=1.0, horizon=8192 — deploy-realistic cadence)
with ``tools/eval_time_to_l8.py``, on the WIN metric (all 6 agents at L8 —
``t_all6_l8``, not the any-agent proxy). A segment becomes the new BEST iff
its ``median_all`` — median time-to-win over ALL eval episodes, failures
= +inf — improves by >= 1% without the win rate dropping > 2pp (see
``beats``; the winners-only median is survivorship-biased and incomparable
across win rates). Three consecutive non-improvements = the plateau (the
theoretical-limit stop), with a hard segment cap as backstop. Segments
always warm-start from BEST (hill climb with restarts), so a bad robustness
condition can't drag the line down.

State lives in ``runs/speedrun/state.json`` (atomic writes); completed
segments (params.msgpack exists) are skipped on re-run, so the driver — like
``tools/overnight_train.sh`` — resumes after a crash or ``make speedrun-stop``.

Launch detached via ``make speedrun``; follow with ``make speedrun-watch``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")
STATE = ROOT / "runs/speedrun/state.json"
BASELINE = "runs/ritual20x24-6p-n2"

# standardized scoring condition (deploy-realistic cadence)
STD = {"overhead": 8, "density_scale": 1.0, "eval_max_ticks": 8192}
STD_JSON = "time_to_l8@ov8@ds1.0@h8192.json"

OV_ROTATION = (8, 10, 6, 12)
DS_ROTATION = (1.0, 0.85, 0.7)


def ts() -> str:
    return time.strftime("%F %T")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"segments": [], "best": None, "plateau_streak": 0}


def segment_plan(i: int, steps: int) -> dict:
    horizon = 8192 if i < 2 else 6144 if i < 4 else 4096
    return {
        "name": f"speedrun-s{i:02d}",
        "overhead": OV_ROTATION[i % len(OV_ROTATION)],
        "density_scale": DS_ROTATION[i % len(DS_ROTATION)],
        "horizon": horizon,
        "steps": steps,
    }


def train_cmd(plan: dict, init_actor: str) -> list[str]:
    return [
        PY, "-m", "zappy_rl.train",
        "--run-name", plan["name"],
        "--width", "20", "--height", "24", "--n-agents", "6",
        "--num-envs", "256",  # 512+ OOMs at 6 agents on the 4090
        "--max-episode-ticks", str(plan["horizon"]),
        "--eval-max-ticks", str(plan["horizon"]),
        "--ent-coef-token", "0.01",
        "--total-env-steps", str(plan["steps"]),
        "--overhead", str(plan["overhead"]),
        "--density-scale", str(plan["density_scale"]),
        "--log-every", "100",
        "--wandb", "auto",
        "--init-actor", init_actor,
    ]


def eval_cmd(run_name: str) -> list[str]:
    return [
        PY, "tools/eval_time_to_l8.py", "--run", f"runs/{run_name}",
        "--overhead", str(STD["overhead"]),
        "--density-scale", str(STD["density_scale"]),
        "--eval-max-ticks", str(STD["eval_max_ticks"]),
        # forward-only rollouts, so 4x the training env count fits; at p~0.97
        # this puts the binomial SE at ~0.4pp (the 2pp rate floor = ~5 SE)
        "--eval-envs", "2048",
    ]


def run(cmd: list[str]) -> int:
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "PYTHONUNBUFFERED": "1"}
    return subprocess.call(cmd, cwd=ROOT, env=env)


def standardized_eval(run_name: str) -> dict | None:
    """Run (or reuse) the standardized eval; return scoring stats or None.

    Scores the WIN metric — t_all6_l8 (all 6 agents at L8 simultaneously) —
    NOT t_any.L8 (first agent). Hill-climbing on the any-agent proxy could
    reward degenerate single-agent rushes that never win (review-pinned).
    Pre-upgrade eval files (no ``median_all``) are regenerated in place.
    """
    out = ROOT / "runs" / run_name / STD_JSON
    if out.exists():
        win = json.loads(out.read_text())["t_all6_l8 (win)"]
        if "median_all" in win:
            return {"rate": win.get("rate", 0.0),
                    "median": win.get("median"),
                    "median_all": win["median_all"]}
        out.unlink()  # stale schema — regenerate
    if run(eval_cmd(run_name)) != 0 or not out.exists():
        return None
    return standardized_eval(run_name)


def beats(cand: dict, best: dict) -> bool:
    """True iff cand improves on best on the WIN objective.

    Scores ``median_all`` — median time-to-win over ALL eval episodes with
    failures counted as +inf (finite iff rate > 50%). Unlike the winners-only
    ``median``, it is survivorship-free, so it compares across different win
    rates and inherently balances reliability against speed: more failures
    push the median up, faster wins pull it down (review-pinned: the
    winners-only comparison scored an 11.9%->92.0% rate jump as "no
    improvement" and stopped the campaign).

    Rules: a candidate in the minority-win regime (median_all = None) never
    beats a majority-win incumbent; crossing into the majority-win regime
    beats any minority-win incumbent; between two majority-win runs a >=1%
    median_all improvement decides, with a -2pp rate floor so a time gain
    cannot buy a reliability collapse ("win rate first").
    """
    cm, bm = cand.get("median_all"), best.get("median_all")
    if cm is None:  # cand wins <=50% of episodes
        return bm is None and cand["rate"] > best["rate"] + 0.02
    if bm is None:  # cand crosses into the majority-win regime
        return True
    if cand["rate"] < best["rate"] - 0.02:
        return False
    return cm < bm * 0.99


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-segments", type=int, default=30)
    ap.add_argument("--steps", type=int, default=400_000_000)
    ap.add_argument("--plateau", type=int, default=3,
                    help="stop after N consecutive non-improving segments")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the next 6 planned segment commands and exit")
    args = ap.parse_args()

    state = load_state()

    if args.dry_run:
        print(f"state: {len(state['segments'])} segments done, "
              f"best={state['best']}, streak={state['plateau_streak']}")
        start = len(state["segments"])
        best_run = (state["best"] or {}).get("run", BASELINE)
        for i in range(start, start + 6):
            plan = segment_plan(i, args.steps)
            print(" ".join(train_cmd(plan, f"{best_run}/params.msgpack")))
        return 0

    if state["best"] is None:
        log(f"establishing baseline: standardized eval of {BASELINE}")
        base = standardized_eval(Path(BASELINE).name)
        if base is None:
            log("FATAL: baseline eval failed")
            return 1
        state["best"] = {"run": BASELINE, **base}
        save_state(state)
        log(f"baseline: rate {base['rate']:.3f} median {base['median']}")

    crashes = 0
    i = len(state["segments"])
    while i < args.max_segments and state["plateau_streak"] < args.plateau:
        plan = segment_plan(i, args.steps)
        name = plan["name"]
        ckpt = ROOT / "runs" / name / "params.msgpack"
        init = f"{state['best']['run']}/params.msgpack"
        if ckpt.exists():
            log(f"SKIP {name} (checkpoint exists)")
        else:
            log(f"START {name}: ov={plan['overhead']} ds={plan['density_scale']} "
                f"horizon={plan['horizon']} warm-start={init}")
            rc = run(train_cmd(plan, init))
            if rc != 0 and not ckpt.exists():
                crashes += 1
                log(f"CRASH {name} (exit {rc}, {crashes} consecutive)")
                state["segments"].append({**plan, "status": "crashed", "eval": None})
                save_state(state)
                if crashes >= 3:
                    log("ABORT: 3 consecutive crashes")
                    return 1
                i += 1
                continue
        crashes = 0

        ev = standardized_eval(name)
        if ev is None:
            log(f"WARN {name}: standardized eval failed — not scored")
            state["segments"].append({**plan, "status": "eval-failed", "eval": None})
            save_state(state)
            i += 1
            continue

        improved = beats(ev, state["best"])
        if improved:
            state["best"] = {"run": f"runs/{name}", **ev}
            state["plateau_streak"] = 0
        else:
            state["plateau_streak"] += 1
        med = "inf" if ev["median_all"] is None else f"{ev['median_all']:.0f}"
        status = ("** NEW BEST **" if improved
                  else f"(plateau streak {state['plateau_streak']})")
        log(f"DONE {name}: rate {ev['rate']:.3f} median_all {med} {status}")
        state["segments"].append({**plan, "status": "done", "eval": ev})
        save_state(state)
        i += 1

    log("================ SPEEDRUN SUMMARY ================")
    for seg in state["segments"]:
        ev = seg.get("eval") or {}
        med = ev.get("median_all")
        med = "  -  " if med is None else f"{med:6.0f}"
        mark = "  <-- BEST" if state["best"]["run"] == f"runs/{seg['name']}" else ""
        log(f"  {seg['name']}  ov={seg['overhead']:>2} ds={seg['density_scale']:.2f} "
            f"h={seg['horizon']}  rate {ev.get('rate', 0.0):.3f}  median_all {med}{mark}")
    log(f"BEST: {state['best']['run']}  rate {state['best']['rate']:.3f} "
        f"median_all {state['best'].get('median_all')}")
    why = ("plateau reached" if state["plateau_streak"] >= args.plateau
           else "segment cap reached")
    log(f"speedrun finished ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

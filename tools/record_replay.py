#!/usr/bin/env python3
"""Record a frozen-policy sim episode as a GUI-protocol replay.

Wraps the trained actor as a ``viz.recorder.policy_fn`` (the recorder itself is
policy-agnostic) and dumps one episode to NDJSON + SQLite under the run dir —
the artifact ``replay_to_gui.py`` (Phase 5) will stream into the reference GUI.

Run:  JAX_PLATFORMS=cpu python tools/record_replay.py --gen 0 --episode 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default="runs/ritual8x8-v1/params.msgpack")
    ap.add_argument("--out", default=None, help="default: <run dir>/replays")
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1200)
    args = ap.parse_args()

    import flax.serialization
    import jax
    import jax.numpy as jnp

    from zappy_rl.algo.networks import (
        OBS_DIM, RecurrentActor, ScannedRNN, cat_sample, flatten_obs,
    )
    from zappy_rl.env import zappy_env as Z
    from zappy_rl.viz.recorder import record_episode

    run_dir = Path(args.params).parent
    cfg_path = run_dir / "config.json"
    hidden = int(json.loads(cfg_path.read_text()).get("hidden", 128)) if cfg_path.exists() else 128
    out_dir = args.out or str(run_dir / "replays")

    A = args.agents
    actor = RecurrentActor(hidden=hidden)
    h0 = ScannedRNN.initialize_carry(A, hidden)
    template = actor.init(jax.random.PRNGKey(0), h0,
                          (jnp.zeros((1, A, OBS_DIM)), jnp.zeros((1, A), bool)))
    loaded = flax.serialization.from_bytes(
        {"actor": template, "critic": None}, Path(args.params).read_bytes())
    params = loaded["actor"]
    apply = jax.jit(actor.apply)

    # mutable closure state: GRU carry + sampling key, reset on first step
    box = {"h": h0, "key": jax.random.PRNGKey(args.seed + 7), "first": True}

    def policy_fn(obs, state, t):
        box["key"], k_a, k_t = jax.random.split(box["key"], 3)
        obs_v = flatten_obs(obs)                      # [A, OBS_DIM]
        done = jnp.full((1, A), box["first"])
        box["h"], la, lt = apply(params, box["h"], (obs_v[None], done))
        box["first"] = False
        return cat_sample(k_a, la[0]), cat_sample(k_t, lt[0])  # SAMPLED

    cfg = Z.make_cfg(args.width, args.height, A)
    path = record_episode(cfg, out_dir, generation=args.gen, episode=args.episode,
                          policy_fn=policy_fn, seed=args.seed, max_steps=args.max_steps)
    n = sum(1 for _ in open(path))
    print(f"[replay] {path}: {n} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

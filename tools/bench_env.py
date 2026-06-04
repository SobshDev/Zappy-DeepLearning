#!/usr/bin/env python3
"""Throughput benchmark for the JAX env. Prints device + env-steps/sec.

    python tools/bench_env.py --envs 8192 --steps 300 --agents 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from zappy_rl.env import zappy_env as Z  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--height", type=int, default=16)
    args = ap.parse_args()

    print("jax", jax.__version__, "devices:", jax.devices())
    cfg = Z.make_cfg(args.width, args.height, args.agents, n_teams=2)
    B = args.envs
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    states, _ = jax.vmap(Z.reset, in_axes=(None, 0))(cfg, keys)
    vstep = jax.jit(jax.vmap(Z.step, in_axes=(None, 0, 0, 0, 0)), static_argnums=0)

    def run(states, key):
        for _ in range(args.steps):
            key, ka, kk = jax.random.split(key, 3)
            acts = jax.random.randint(ka, (B, args.agents), 0, Z.N_ENV_ACTIONS)
            toks = jnp.zeros((B, args.agents), jnp.int32)
            ks = jax.random.split(kk, B)
            states, _, _, _, _ = vstep(cfg, ks, states, acts, toks)
        return states

    s = run(states, jax.random.PRNGKey(1))
    jax.block_until_ready(s.now)  # warmup / compile
    t0 = time.time()
    s = run(states, jax.random.PRNGKey(2))
    jax.block_until_ready(s.now)
    dt = time.time() - t0
    env_steps = B * args.steps
    print(f"{B} envs x {args.agents} agents x {args.steps} steps in {dt:.2f}s")
    print(f"  env-steps/sec   : {env_steps / dt:,.0f}")
    print(f"  agent-steps/sec : {env_steps * args.agents / dt:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

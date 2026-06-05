#!/usr/bin/env python3
"""Time-to-level eval: how fast does a frozen squad climb L1 -> L8 in sim?

Mirrors ``mappo.evaluate`` (fresh episodes, no autoreset, SAMPLED actions)
but times every elevation tier instead of only L3:

  * ``t_any[k]``  — tick when ANY agent first reaches level k (k = 2..8)
  * ``t_all_l8``  — tick when ALL agents are L8 simultaneously (the Zappy
    win condition needs 6 players at L8; with n_agents=6 this is the win)

Usage::

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        tools/eval_time_to_l8.py --run runs/ritual20x24-6p-n2 [--eval-envs 512]

Writes ``<run>/time_to_l8.json`` and prints a per-tier table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

import zappy_rl.env.zappy_env as Z
from zappy_rl.algo.mappo import TrainConfig
from zappy_rl.algo.networks import (
    OBS_DIM, RecurrentActor, ScannedRNN, cat_sample, flatten_obs,
)

LEVELS = list(range(2, 9))  # timed tiers


def run_eval(tc: TrainConfig, cfg: Z.Cfg, actor_params, key, greedy=False):
    B, A, H = tc.eval_envs, cfg.n_agents, tc.hidden
    BA = B * A
    actor = RecurrentActor(hidden=H)
    n_steps = tc.eval_max_ticks // 7 + 64
    no_reset = jnp.zeros((1, BA), bool)

    key, k_env = jax.random.split(key)
    env_state, obs = jax.vmap(Z.reset, in_axes=(None, 0))(cfg, jax.random.split(k_env, B))
    v_step = jax.vmap(Z.step, in_axes=(None, 0, 0, 0, 0))

    def _step(carry, _):
        env_state, obs_v, done_flag, death_tick, t_any, t_all8, h, key = carry
        key, k_a, k_t, k_s = jax.random.split(key, 4)
        h2, la, lt = actor.apply(actor_params, h, (obs_v[None], no_reset))
        if greedy:  # static under jit — argmax actions, no exploration noise
            action = jnp.argmax(la[0], axis=-1).astype(jnp.int32)
            token = jnp.argmax(lt[0], axis=-1).astype(jnp.int32)
        else:
            action = cat_sample(k_a, la[0])
            token = cat_sample(k_t, lt[0])
        s2, o2, _, done, _ = v_step(
            cfg, jax.random.split(k_s, B), env_state,
            action.reshape(B, A), token.reshape(B, A),
        )
        done = done | (s2.now >= tc.eval_max_ticks)
        newly = done & ~done_flag
        death_tick = jnp.where(newly, s2.now, death_tick)
        active = ~done_flag
        lvl_any = s2.level.max(axis=1)   # [B]
        lvl_all = s2.level.min(axis=1)   # [B]
        for i, k in enumerate(LEVELS):
            t_any = t_any.at[:, i].set(jnp.where(
                active & (lvl_any >= k) & (t_any[:, i] < 0), s2.now, t_any[:, i]))
        t_all8 = jnp.where(active & (lvl_all >= 8) & (t_all8 < 0), s2.now, t_all8)
        keep = lambda old, new: jnp.where(  # noqa: E731
            done_flag.reshape((B,) + (1,) * (new.ndim - 1)), old, new)
        env_state2 = jax.tree.map(keep, env_state, s2)
        frozen = jnp.repeat(done_flag, A)[:, None]
        obs_v2 = jnp.where(frozen, obs_v, flatten_obs(o2).reshape(BA, OBS_DIM))
        return (env_state2, obs_v2, done_flag | done, death_tick,
                t_any, t_all8, h2, key), None

    carry = (
        env_state, flatten_obs(obs).reshape(BA, OBS_DIM),
        jnp.zeros(B, bool), jnp.zeros(B, jnp.int32),
        jnp.full((B, len(LEVELS)), -1, jnp.int32), jnp.full(B, -1, jnp.int32),
        ScannedRNN.initialize_carry(BA, H), key,
    )
    carry, _ = jax.lax.scan(_step, carry, None, length=n_steps)
    return jax.device_get((carry[4], carry[5]))  # t_any [B,7], t_all8 [B]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/ritual20x24-6p-n2")
    ap.add_argument("--eval-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    # standardized-condition overrides (else: the run's own config.json) —
    # the speedrun driver compares checkpoints at fixed ov/ds/horizon
    ap.add_argument("--overhead", type=int, default=None)
    ap.add_argument("--density-scale", type=float, default=None)
    ap.add_argument("--eval-max-ticks", type=int, default=None)
    ap.add_argument("--life-noise", type=float, default=None,
                    help="override life-obs noise (robustness probe; "
                         "standardized eval stays noiseless)")
    ap.add_argument("--life-noise-window", type=int, default=None,
                    help="hold each noise draw for K ticks (deploy-like "
                         "smooth drift; default per-tick)")
    ap.add_argument("--greedy", action="store_true",
                    help="argmax actions instead of sampling")
    args = ap.parse_args()

    run = Path(args.run)
    tc_d = json.loads((run / "config.json").read_text())
    tc_d = {k: v for k, v in tc_d.items()
            if k in {f.name for f in dataclasses.fields(TrainConfig)}}
    tc_d["eval_envs"] = args.eval_envs
    suffix = ""
    if args.overhead is not None:
        tc_d["overhead"] = args.overhead
        suffix += f"@ov{args.overhead}"
    if args.density_scale is not None:
        tc_d["density_scale"] = args.density_scale
        suffix += f"@ds{args.density_scale}"
    if args.eval_max_ticks is not None:
        tc_d["eval_max_ticks"] = args.eval_max_ticks
        suffix += f"@h{args.eval_max_ticks}"
    # eval is NOISELESS unless explicitly probing: a life_noise-trained run's
    # config.json must not contaminate the standardized (comparable) numbers
    if args.life_noise is not None:
        tc_d["life_noise"] = args.life_noise
        suffix += f"@ln{args.life_noise:g}"
    else:
        tc_d["life_noise"] = 0.0
    if args.life_noise_window is not None:
        tc_d["life_noise_window"] = args.life_noise_window
        suffix += f"@lw{args.life_noise_window}"
    else:
        tc_d["life_noise_window"] = 1
    if args.greedy:
        suffix += "@greedy"
    tc = TrainConfig(**tc_d)
    cfg = Z.make_cfg(tc.width, tc.height, tc.n_agents, tc.n_teams,
                     overhead=tc.overhead, density_scale=tc.density_scale,
                     life_noise=tc.life_noise,
                     life_noise_window=tc.life_noise_window)

    # actor template -> load checkpoint (actor leaves only)
    actor = RecurrentActor(hidden=tc.hidden)
    h0 = ScannedRNN.initialize_carry(1, tc.hidden)
    template = actor.init(jax.random.PRNGKey(0), h0,
                          (jnp.zeros((1, 1, OBS_DIM)), jnp.zeros((1, 1), bool)))
    loaded = flax.serialization.from_bytes(
        {"actor": template, "critic": None}, (run / "params.msgpack").read_bytes())

    t_any, t_all8 = jax.jit(run_eval, static_argnums=(0, 1, 4))(
        tc, cfg, loaded["actor"], jax.random.PRNGKey(args.seed), args.greedy)

    def stats(t):
        t = np.asarray(t)
        hit = t >= 0
        if not hit.any():
            return {"rate": 0.0, "median_all": None}
        v = t[hit]
        # survivorship-free time score: failures count as +inf and the median
        # runs over ALL episodes (finite iff rate > 50%). Comparable across
        # runs with different win rates — the conditioned `median` is not.
        med_all = float(np.median(np.where(hit, t, np.inf)))
        return {"rate": float(hit.mean()), "median": float(np.median(v)),
                "median_all": med_all if np.isfinite(med_all) else None,
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                "min": int(v.min())}

    mode = "greedy" if args.greedy else "sampled"
    out = {"run": str(run), "n": tc.eval_envs,
           "horizon_ticks": tc.eval_max_ticks,
           "overhead": tc.overhead, "density_scale": tc.density_scale,
           "policy": mode,
           "t_any": {f"L{k}": stats(t_any[:, i]) for i, k in enumerate(LEVELS)},
           "t_all6_l8 (win)": stats(t_all8)}
    print(f"{run}  n={tc.eval_envs}  horizon={tc.eval_max_ticks} ticks  "
          f"overhead={tc.overhead} density={tc.density_scale}  ({mode} policy)")
    for name, s in {**out["t_any"], "WIN (all 6 @ L8)": out["t_all6_l8 (win)"]}.items():
        if s["rate"] == 0.0:
            print(f"  {name:16s} never")
        else:
            ma = s["median_all"]
            ma = "  inf" if ma is None else f"{ma:5.0f}"
            print(f"  {name:16s} rate {s['rate']:5.1%}  median {s['median']:6.0f}  "
                  f"median_all {ma}  "
                  f"p10 {s['p10']:6.0f}  p90 {s['p90']:6.0f}  min {s['min']:5d}")
    # atomic: the speedrun driver trusts this file's existence on resume
    final = run / f"time_to_l8{suffix}.json"
    tmp = final.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

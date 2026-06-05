"""Recurrent MAPPO over the vmapped JAX Zappy env (PureJaxRL-style).

Shared-parameter recurrent PPO with a centralized critic (CTDE / MAPPO),
event-driven episodes, autoreset, potential-based shaping, and TBPTT-chunked
updates. Phase 2 trained ``n_agents=1`` foraging; Phase 3 trains cooperative
ritual squads (per-(env,agent) rows, per-agent dones, busy-action masking).

Key conventions (PureJaxRL):

* A transition stores ``done`` = the done flag that *preceded* its obs (true
  iff the obs is the first of a new episode). The same flag both resets the
  GRU carry and masks GAE across episode boundaries.
* TBPTT: the T-step rollout is split into chunks of ``tbptt_chunk`` along
  time; each chunk is a training sequence initialized from the GRU carry
  *stored during the rollout* at the chunk start (R2D2 "stored state" style).
  Minibatches shuffle whole sequences, never time within a sequence.
* The whole update iteration (rollout + GAE + epochs) is one jitted function
  driven by a Python loop, so logging/throughput measurement is trivial.

Multi-agent semantics (Phase 3):

* ``done`` is per-(env,agent) row: a row flags done when the env resets
  (all-dead autoreset or truncation) OR when the agent itself is dead at obs
  time. The death-step transition still trains (it carries the -1 and its
  bootstrap is cut by next_done); the dead rows AFTER it are excluded from
  every loss via the ``alive`` mask.
* The event clock means busy/frozen agents submit actions the env ignores
  (``info["free"]`` says whose action was consumed). Non-free rows are
  masked out of the policy/entropy loss — a 300-tick incantation freeze must
  not hand PPO credit for ~43 ignored actions — but stay in the value loss:
  the critic must price busy states because reward keeps flowing through
  them. Advantage normalization is over free rows only.

Reward = env reward (level-up +L, death -1)
       + potential shaping  F = gamma*Phi(s') - Phi(s)  (policy-invariant),
         Phi = [ phi_life   * clip(life/1260, 0, phi_clip)
               + phi_stones * (held stones needed for the next ritual)/req
               + phi_coloc  * (tile-stone progress) x (co-located same-level
                               players / required)   <- ritual assembly
               + phi_incant * level, while frozen in an incantation
               ] * alive                              (Phi = 0 at death)
       + alive_bonus per agent per step survived, FLAT (deliberately not
         scaled by dt: PPO discounts per env-step, so a dt-proportional bonus
         would pay long-dt actions — incantation's 300-tick freeze — a lump
         sum at a single discount factor, making "freeze" beat "forage" on
         reward rate. Flat-per-step keeps the survival gradient and lets the
         potential term price the life cost of long actions correctly.)

The coloc term uses smooth tile/player *progress* instead of the plan's
binary stones_ok gate: with a binary gate, carrying the 1st and 2nd stone to
the ritual tile is locally negative (held-stones potential drops, gate still
closed) — a two-step valley before any payoff. Progress terms make each
assembly step uphill while keeping the same optimum (still a potential).

Episodes truncate at ``max_episode_ticks`` (treated as terminal in GAE — the
standard, slightly biased simplification).
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, NamedTuple

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from ..env import constants as C
from ..env import zappy_env as Z
from .networks import (
    OBS_DIM,
    RecurrentActor,
    RecurrentCritic,
    ScannedRNN,
    cat_entropy,
    cat_log_prob,
    cat_sample,
    flatten_obs,
)


# ------------------------------------------------------------------ config
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # env (frozen within a stage — changing these recompiles)
    width: int = 6
    height: int = 6
    n_agents: int = 1
    n_teams: int = 1
    max_episode_ticks: int = 4096
    # sim-to-real robustness knobs (defaults oracle-exact; see env.Cfg):
    # overhead ≈ deploy perceive prefix in ticks/decision; density_scale < 1
    # trains under scarcer resources than the reference server spawns;
    # life_noise = ±ticks of uniform error on the life obs (deploy
    # dead-reckoning model — teaches food-margin slack at ritual chains).
    overhead: int = 0
    density_scale: float = 1.0
    life_noise: float = 0.0
    # scale
    num_envs: int = 1024
    rollout_steps: int = 128
    total_env_steps: int = 10_000_000
    # ppo (plan knobs: <=4 epochs, 2-4 large minibatches, TBPTT chunk 16)
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef_action: float = 0.01
    ent_coef_token: float = 0.0
    max_grad_norm: float = 0.5
    epochs: int = 4
    num_minibatches: int = 4
    tbptt_chunk: int = 16
    hidden: int = 128
    # reward shaping (potential weights; see module doc)
    phi_life: float = 1.0
    phi_clip: float = 2.0
    phi_stones: float = 0.5   # held stones needed for the next ritual
    phi_coloc: float = 1.5    # co-location on a (progressively) stocked tile
    phi_incant: float = 1.0   # frozen in an incantation attempt, x level
    alive_bonus: float = 0.01  # flat per env-step survived (see module doc)
    # misc
    seed: int = 0
    eval_envs: int = 512
    eval_max_ticks: int = 4096
    log_every: int = 5

    def __post_init__(self):
        assert self.rollout_steps % self.tbptt_chunk == 0
        n_seq = (self.rollout_steps // self.tbptt_chunk) * self.num_envs * self.n_agents
        assert n_seq % self.num_minibatches == 0


class Transition(NamedTuple):
    """One rollout step, per (env, agent) row. ``done`` precedes ``obs``."""

    done: jnp.ndarray      # [BA] bool — obs starts a new episode for this row
    alive: jnp.ndarray     # [BA] bool — agent alive at obs time (value mask)
    free: jnp.ndarray      # [BA] bool — env consumed this action (policy mask)
    action: jnp.ndarray    # [BA] int32
    token: jnp.ndarray     # [BA] int32
    value: jnp.ndarray     # [BA] f32
    reward: jnp.ndarray    # [BA] f32
    logp: jnp.ndarray      # [BA] f32 (action + masked token)
    obs: jnp.ndarray       # [BA, OBS_DIM] f32
    wextra: jnp.ndarray    # [BA, WX] f32 critic-only world features
    h_actor: jnp.ndarray   # [BA, H] carry *before* consuming obs
    h_critic: jnp.ndarray  # [BA, H]


class SeqBatch(NamedTuple):
    """TBPTT minibatch sequences ``[chunk, n_seq, ...]`` (loss-time fields)."""

    done: jnp.ndarray
    alive: jnp.ndarray
    free: jnp.ndarray
    action: jnp.ndarray
    token: jnp.ndarray
    value: jnp.ndarray
    logp: jnp.ndarray
    obs: jnp.ndarray
    wextra: jnp.ndarray


class StepMetrics(NamedTuple):
    done: jnp.ndarray   # [B] episode finished at this step
    ticks: jnp.ndarray  # [B] episode length in game ticks (valid where done)
    ret: jnp.ndarray    # [B] episode return (valid where done)
    steps: jnp.ndarray  # [B] episode length in env steps (valid where done)
    foods: jnp.ndarray  # [B] food eaten this episode (valid where done)
    l2: jnp.ndarray     # [B] agents reaching L2 this step (level-up EVENTS:
    l3: jnp.ndarray     # [B] ... reaching L3; a pair ritual counts 2, not 1)


# ------------------------------------------------------- features / reward
def world_extra_dim(cfg: Z.Cfg) -> int:
    return cfg.width * cfg.height * C.N_RESOURCES + cfg.n_agents * 16 + cfg.n_agents


def world_extra(cfg: Z.Cfg, s: Z.State) -> jnp.ndarray:
    """Privileged critic features for one env: full grid + all agents + id."""
    grid = jnp.clip(s.grid.astype(jnp.float32) / 5.0, 0.0, 2.0).reshape(-1)
    per_agent = jnp.concatenate(
        [
            s.pos[:, 0:1].astype(jnp.float32) / cfg.width,
            s.pos[:, 1:2].astype(jnp.float32) / cfg.height,
            jax.nn.one_hot(s.orient - 1, 4),
            s.level[:, None].astype(jnp.float32) / C.MAX_LEVEL,
            s.inv.astype(jnp.float32) / 10.0,
            jnp.clip(s.life[:, None].astype(jnp.float32) / Z.START_LIFE, 0.0, 2.0),
            s.alive[:, None].astype(jnp.float32),
        ],
        axis=1,
    )  # [A, 16]
    shared = jnp.concatenate([grid, per_agent.reshape(-1)])
    A = cfg.n_agents
    return jnp.concatenate([jnp.tile(shared[None], (A, 1)), jnp.eye(A)], axis=1)


def _phi(tc: TrainConfig, s: Z.State) -> jnp.ndarray:
    """Per-agent shaping potential, 0 for dead agents (proper terminal Phi).

    Pure function of STATE (never of the action taken), so F = gamma*Phi' -
    Phi is policy-invariant. Progress terms are smooth (see module doc).
    """
    life = jnp.clip(s.life.astype(jnp.float32) / Z.START_LIFE, 0.0, tc.phi_clip)
    phi = tc.phi_life * life

    lvl = jnp.clip(s.level, 0, C.MAX_LEVEL)
    req_s = Z.REQ_S[lvl].astype(jnp.float32)        # [A,6] stones for L->L+1
    req_p = Z.REQ_P[lvl].astype(jnp.float32)        # [A] players for L->L+1
    can_lvl = (req_p > 0).astype(jnp.float32)       # 0 once at MAX_LEVEL
    s_tot = jnp.maximum(req_s.sum(-1), 1.0)

    # stones held toward the next ritual (capped per resource at requirement)
    held = jnp.minimum(s.inv[:, 1:].astype(jnp.float32), req_s).sum(-1) / s_tot
    phi = phi + tc.phi_stones * held * can_lvl

    # co-location x tile stocking progress (count includes self)
    tile = s.grid[s.pos[:, 0], s.pos[:, 1], 1:].astype(jnp.float32)  # [A,6]
    tile_prog = jnp.minimum(tile, req_s).sum(-1) / s_tot
    same_tile = (s.pos[:, None, 0] == s.pos[None, :, 0]) & (
        s.pos[:, None, 1] == s.pos[None, :, 1])
    same_lvl = s.level[:, None] == s.level[None, :]
    n_here = jnp.sum(same_tile & same_lvl & s.alive[None, :], axis=1).astype(jnp.float32)
    coloc = jnp.minimum(n_here, req_p) / jnp.maximum(req_p, 1.0)
    phi = phi + tc.phi_coloc * tile_prog * coloc * can_lvl

    # frozen in an incantation attempt
    phi = phi + tc.phi_incant * s.pending.astype(jnp.float32) * lvl.astype(jnp.float32)
    return phi * s.alive.astype(jnp.float32)


def _step_env(cfg: Z.Cfg, tc: TrainConfig, key, state: Z.State, action, token):
    """Step ONE env: shaping + truncation + autoreset. vmap over envs."""
    k_step, k_reset = jax.random.split(key)
    phi0 = _phi(tc, state)
    s2, o2, r_env, done_env, info = Z.step(cfg, k_step, state, action, token)
    phi1 = _phi(tc, s2)
    alive_f = s2.alive.astype(jnp.float32)
    reward = r_env + tc.gamma * phi1 - phi0 + tc.alive_bonus * alive_f  # [A]
    decay = 0 if cfg.no_food else info["dt"]
    foods = jnp.maximum(s2.life - state.life + decay, 0) // C.FOOD_LIFE_TICKS  # [A]
    l2 = jnp.sum((info["leveled"] & (s2.level == 2)).astype(jnp.float32))
    l3 = jnp.sum((info["leveled"] & (s2.level == 3)).astype(jnp.float32))
    done = done_env | (s2.now >= tc.max_episode_ticks)
    ep_ticks = s2.now.astype(jnp.float32)
    s_r, o_r = Z.reset(cfg, k_reset)
    s3 = jax.tree.map(lambda a, b: jnp.where(done, b, a), s2, s_r)
    o3 = jax.tree.map(lambda a, b: jnp.where(done, b, a), o2, o_r)
    return (s3, o3, reward, done, ep_ticks, jnp.sum(foods).astype(jnp.float32),
            s3.alive, info["free"], l2, l3)


# ------------------------------------------------------------------- GAE
def compute_gae(tc: TrainConfig, traj: Transition, last_val, last_done):
    """Standard GAE with PureJaxRL's done convention (done precedes obs)."""

    def scan_fn(carry, t):
        gae, next_val, next_done = carry
        nonterm = 1.0 - next_done.astype(jnp.float32)
        delta = t.reward + tc.gamma * next_val * nonterm - t.value
        gae = delta + tc.gamma * tc.gae_lambda * nonterm * gae
        return (gae, t.value, t.done), gae

    _, adv = jax.lax.scan(
        scan_fn, (jnp.zeros_like(last_val), last_val, last_done), traj, reverse=True
    )
    return adv, adv + traj.value


# ------------------------------------------------------------------ train
def make_train(tc: TrainConfig):
    """Build (init_runner, update_iteration) for the jitted training loop."""
    cfg = Z.make_cfg(tc.width, tc.height, tc.n_agents, tc.n_teams,
                     overhead=tc.overhead, density_scale=tc.density_scale,
                     life_noise=tc.life_noise)
    B, A, H = tc.num_envs, tc.n_agents, tc.hidden
    BA = B * A
    T = tc.rollout_steps
    n_chunks = T // tc.tbptt_chunk
    n_seq = n_chunks * BA
    n_iters = tc.total_env_steps // (B * T)
    n_updates = n_iters * tc.epochs * tc.num_minibatches

    actor = RecurrentActor(hidden=H)
    critic = RecurrentCritic(hidden=H)
    v_reset = jax.vmap(Z.reset, in_axes=(None, 0))
    v_step = jax.vmap(_step_env, in_axes=(None, None, 0, 0, 0, 0))
    v_wx = jax.vmap(world_extra, in_axes=(None, 0))

    if tc.anneal_lr:
        lr: Any = optax.linear_schedule(tc.lr, 0.0, n_updates)
    else:
        lr = tc.lr
    tx = lambda: optax.chain(  # noqa: E731
        optax.clip_by_global_norm(tc.max_grad_norm),
        optax.adam(learning_rate=lr, eps=1e-5),
    )

    def init_runner(key):
        key, k_env, k_actor, k_critic = jax.random.split(key, 4)
        env_state, obs = v_reset(cfg, jax.random.split(k_env, B))
        obs_v = flatten_obs(obs).reshape(BA, OBS_DIM)
        wx = v_wx(cfg, env_state).reshape(BA, -1)
        h0 = ScannedRNN.initialize_carry(BA, H)
        dones0 = jnp.zeros((1, BA), bool)
        pa = actor.init(k_actor, h0, (obs_v[None], dones0))
        cin = jnp.concatenate([obs_v, wx], axis=-1)
        pc = critic.init(k_critic, h0, (cin[None], dones0))
        ts_a = TrainState.create(apply_fn=actor.apply, params=pa, tx=tx())
        ts_c = TrainState.create(apply_fn=critic.apply, params=pc, tx=tx())
        return {
            "ts_a": ts_a,
            "ts_c": ts_c,
            "env_state": env_state,
            "obs": obs_v,
            "wx": wx,
            "done": jnp.ones(BA, bool),  # first obs of first episode
            "alive": jnp.ones(BA, bool),
            "h_a": h0,
            "h_c": h0,
            "ep_ret": jnp.zeros(B),
            "ep_steps": jnp.zeros(B),
            "ep_foods": jnp.zeros(B),
            "key": key,
        }

    # ---------------------------------------------------------- rollout
    def _env_step(carry, _):
        r = carry
        key, k_act, k_tok, k_step = jax.random.split(r["key"], 4)
        h_a2, la, lt = actor.apply(r["ts_a"].params, r["h_a"], (r["obs"][None], r["done"][None]))
        la, lt = la[0], lt[0]
        action = cat_sample(k_act, la)
        token = cat_sample(k_tok, lt)
        is_b = action == Z.ENV_BROADCAST
        logp = cat_log_prob(la, action) + jnp.where(is_b, cat_log_prob(lt, token), 0.0)
        cin = jnp.concatenate([r["obs"], r["wx"]], axis=-1)
        h_c2, value = critic.apply(r["ts_c"].params, r["h_c"], (cin[None], r["done"][None]))
        value = value[0]

        env_state, obs, reward, done_env, ep_ticks, foods, alive2, free, l2, l3 = v_step(
            cfg, tc, jax.random.split(k_step, B), r["env_state"],
            action.reshape(B, A), token.reshape(B, A),
        )
        trans = Transition(
            done=r["done"], alive=r["alive"], free=free.reshape(BA),
            action=action, token=token, value=value,
            reward=reward.reshape(BA), logp=logp, obs=r["obs"], wextra=r["wx"],
            h_actor=r["h_a"], h_critic=r["h_c"],
        )
        ep_ret = r["ep_ret"] + reward.mean(axis=1)
        ep_steps = r["ep_steps"] + 1.0
        ep_foods = r["ep_foods"] + foods
        sm = StepMetrics(done=done_env, ticks=ep_ticks, ret=ep_ret, steps=ep_steps,
                         foods=ep_foods, l2=l2, l3=l3)
        keep = ~done_env
        alive_rows = alive2.reshape(BA)
        carry2 = {
            **r,
            "env_state": env_state,
            "obs": flatten_obs(obs).reshape(BA, OBS_DIM),
            "wx": v_wx(cfg, env_state).reshape(BA, -1),
            # a row restarts on env reset; dead rows stay done (masked anyway)
            "done": jnp.repeat(done_env, A) | ~alive_rows,
            "alive": alive_rows,
            "h_a": h_a2,
            "h_c": h_c2,
            "ep_ret": ep_ret * keep,
            "ep_steps": ep_steps * keep,
            "ep_foods": ep_foods * keep,
            "key": key,
        }
        return carry2, (trans, sm)

    # ----------------------------------------------------------- update
    def _loss_fn(pa, pc, mb):
        seq, h0a, h0c, adv, target = mb
        pgm = seq.free.astype(jnp.float32)    # policy credit: action consumed
        vm = seq.alive.astype(jnp.float32)    # value credit: alive at obs
        n_pg = jnp.maximum(pgm.sum(), 1.0)
        n_v = jnp.maximum(vm.sum(), 1.0)

        _, la, lt = actor.apply(pa, h0a, (seq.obs, seq.done))
        logp_a = cat_log_prob(la, seq.action)
        is_b = seq.action == Z.ENV_BROADCAST
        logp = logp_a + jnp.where(is_b, cat_log_prob(lt, seq.token), 0.0)
        ratio = jnp.exp(logp - seq.logp)
        adv_mu = (adv * pgm).sum() / n_pg
        adv_sd = jnp.sqrt((jnp.square(adv - adv_mu) * pgm).sum() / n_pg)
        adv_n = (adv - adv_mu) / (adv_sd + 1e-8)
        pg1 = ratio * adv_n
        pg2 = jnp.clip(ratio, 1.0 - tc.clip_eps, 1.0 + tc.clip_eps) * adv_n
        pg_loss = -(jnp.minimum(pg1, pg2) * pgm).sum() / n_pg
        ent_a = (cat_entropy(la) * pgm).sum() / n_pg
        bm = is_b.astype(jnp.float32) * pgm
        n_b = jnp.maximum(bm.sum(), 1.0)
        ent_t = (cat_entropy(lt) * bm).sum() / n_b

        cin = jnp.concatenate([seq.obs, seq.wextra], axis=-1)
        _, v = critic.apply(pc, h0c, (cin, seq.done))
        v_clip = seq.value + jnp.clip(v - seq.value, -tc.clip_eps, tc.clip_eps)
        v_err = jnp.maximum((v - target) ** 2, (v_clip - target) ** 2)
        v_loss = 0.5 * (v_err * vm).sum() / n_v

        total = (
            pg_loss
            - tc.ent_coef_action * ent_a
            - tc.ent_coef_token * ent_t
            + tc.vf_coef * v_loss
        )
        approx_kl = (((ratio - 1.0) - jnp.log(ratio)) * pgm).sum() / n_pg
        clip_frac = ((jnp.abs(ratio - 1.0) > tc.clip_eps) * pgm).sum() / n_pg
        return total, {
            "pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent_a,
            "entropy_tok": ent_t, "approx_kl": approx_kl, "clip_frac": clip_frac,
        }

    grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)

    def _update_minibatch(carry, mb):
        ts_a, ts_c = carry
        (_, aux), (ga, gc) = grad_fn(ts_a.params, ts_c.params, mb)
        return (ts_a.apply_gradients(grads=ga), ts_c.apply_gradients(grads=gc)), aux

    def _chunk(x):
        """[T, BA, ...] -> [chunk, n_seq, ...], seq index = chunk_id*BA + row."""
        y = x.reshape(n_chunks, tc.tbptt_chunk, BA, *x.shape[2:])
        return y.swapaxes(0, 1).reshape(tc.tbptt_chunk, n_seq, *x.shape[2:])

    def _update_epoch(carry, _):
        ts_a, ts_c, seqs, h0a, h0c, adv, target, key = carry
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, n_seq)
        m = tc.num_minibatches
        mb_seqs = jax.tree.map(
            lambda x: jnp.take(x, perm, axis=1)
            .reshape(x.shape[0], m, -1, *x.shape[2:])
            .swapaxes(0, 1),
            (seqs, adv, target),
        )
        mb_h = jax.tree.map(
            lambda x: jnp.take(x, perm, axis=0).reshape(m, -1, *x.shape[1:]),
            (h0a, h0c),
        )
        mb = (mb_seqs[0], mb_h[0], mb_h[1], mb_seqs[1], mb_seqs[2])
        (ts_a, ts_c), aux = jax.lax.scan(_update_minibatch, (ts_a, ts_c), mb)
        return (ts_a, ts_c, seqs, h0a, h0c, adv, target, key), aux

    # -------------------------------------------------------- iteration
    def update_iteration(runner):
        runner, (traj, sm) = jax.lax.scan(_env_step, runner, None, length=T)

        cin = jnp.concatenate([runner["obs"], runner["wx"]], axis=-1)
        _, last_val = critic.apply(
            runner["ts_c"].params, runner["h_c"], (cin[None], runner["done"][None])
        )
        adv, target = compute_gae(tc, traj, last_val[0], runner["done"])

        seqs = SeqBatch(
            done=_chunk(traj.done), alive=_chunk(traj.alive),
            free=_chunk(traj.free), action=_chunk(traj.action),
            token=_chunk(traj.token), value=_chunk(traj.value),
            logp=_chunk(traj.logp), obs=_chunk(traj.obs),
            wextra=_chunk(traj.wextra),
        )
        h0a = traj.h_actor[:: tc.tbptt_chunk].reshape(n_seq, H)
        h0c = traj.h_critic[:: tc.tbptt_chunk].reshape(n_seq, H)
        adv_c, target_c = _chunk(adv), _chunk(target)

        carry = (runner["ts_a"], runner["ts_c"], seqs, h0a, h0c, adv_c, target_c, runner["key"])
        carry, aux = jax.lax.scan(_update_epoch, carry, None, length=tc.epochs)
        ts_a, ts_c, key = carry[0], carry[1], carry[7]
        runner = {**runner, "ts_a": ts_a, "ts_c": ts_c, "key": key}

        # NB: ep_* stats only see episodes that FINISH inside the rollout
        # window (~7*T ticks) — once the policy outlives the window they
        # become survival-biased. The live now_* / alive_frac metrics below
        # (current env state across ALL envs) are the gate readout.
        fin = sm.done.astype(jnp.float32)
        n_fin = jnp.maximum(fin.sum(), 1.0)
        env_state = runner["env_state"]
        metrics = {
            "episodes": fin.sum(),
            "ep_ticks": (sm.ticks * fin).sum() / n_fin,
            "ep_ticks_ge_2000": ((sm.ticks >= 2000) * fin).sum() / n_fin,
            "ep_return": (sm.ret * fin).sum() / n_fin,
            "ep_steps": (sm.steps * fin).sum() / n_fin,
            "ep_foods": (sm.foods * fin).sum() / n_fin,
            "now_mean": env_state.now.astype(jnp.float32).mean(),
            "now_ge_2000": (env_state.now >= 2000).astype(jnp.float32).mean(),
            "alive_frac": env_state.alive.any(axis=1).astype(jnp.float32).mean(),
            "life_mean": env_state.life.astype(jnp.float32).mean() / Z.START_LIFE,
            # ritual progress: per-AGENT level-up events counted over the whole
            # rollout (not window-blind; an L2->L3 pair ritual contributes 2),
            # plus live level distribution across envs.
            "lvlups_to_l2": sm.l2.sum(),
            "lvlups_to_l3": sm.l3.sum(),
            "level_mean": env_state.level.astype(jnp.float32).mean(),
            "frac_l2": (env_state.level >= 2).astype(jnp.float32).mean(),
            "frac_l3": (env_state.level >= 3).astype(jnp.float32).mean(),
            "free_frac": traj.free.astype(jnp.float32).mean(),
            "alive_row_frac": traj.alive.astype(jnp.float32).mean(),
            "reward_per_step": traj.reward.mean(),
            "value_mean": traj.value.mean(),
        }
        metrics.update(jax.tree.map(jnp.mean, aux))
        return runner, metrics

    return init_runner, update_iteration, cfg, n_iters


# ------------------------------------------------------------------- eval
class EvalOut(NamedTuple):
    ticks: jnp.ndarray      # [B] ticks survived (death tick or horizon)
    max_level: jnp.ndarray  # [B] max level reached by any agent in the env
    t_l3: jnp.ndarray       # [B] tick when an agent first hit L3 (-1 = never)


def evaluate(tc: TrainConfig, cfg: Z.Cfg, actor_params, key, greedy: bool):
    """Run fresh episodes (no autoreset); per-env survival + ritual stats."""
    B, A, H = tc.eval_envs, cfg.n_agents, tc.hidden
    BA = B * A
    actor = RecurrentActor(hidden=H)
    n_steps = tc.eval_max_ticks // 7 + 64

    key, k_env = jax.random.split(key)
    env_state, obs = jax.vmap(Z.reset, in_axes=(None, 0))(cfg, jax.random.split(k_env, B))
    v_step = jax.vmap(Z.step, in_axes=(None, 0, 0, 0, 0))
    no_reset = jnp.zeros((1, BA), bool)

    def _step(carry, _):
        env_state, obs_v, done_flag, death_tick, max_lvl, t_l3, h, key = carry
        key, k_a, k_t, k_s = jax.random.split(key, 4)
        h2, la, lt = actor.apply(actor_params, h, (obs_v[None], no_reset))
        la, lt = la[0], lt[0]
        action = jnp.argmax(la, -1) if greedy else cat_sample(k_a, la)
        token = jnp.argmax(lt, -1) if greedy else cat_sample(k_t, lt)
        s2, o2, _, done, _ = v_step(
            cfg, jax.random.split(k_s, B), env_state,
            action.reshape(B, A), token.reshape(B, A),
        )
        done = done | (s2.now >= tc.eval_max_ticks)
        newly = done & ~done_flag
        death_tick = jnp.where(newly, s2.now, death_tick)
        # level stats may update on the step the env finishes (level&die)
        active = ~done_flag
        lvl_now = s2.level.max(axis=1)
        max_lvl = jnp.where(active, jnp.maximum(max_lvl, lvl_now), max_lvl)
        t_l3 = jnp.where(active & (lvl_now >= 3) & (t_l3 < 0), s2.now, t_l3)
        # freeze finished envs (their event clock would otherwise run away)
        keep = lambda old, new: jnp.where(  # noqa: E731
            done_flag.reshape((B,) + (1,) * (new.ndim - 1)), old, new
        )
        env_state2 = jax.tree.map(keep, env_state, s2)
        frozen = jnp.repeat(done_flag, A)[:, None]
        obs_v2 = jnp.where(frozen, obs_v, flatten_obs(o2).reshape(BA, OBS_DIM))
        return (env_state2, obs_v2, done_flag | done, death_tick, max_lvl, t_l3, h2, key), None

    h0 = ScannedRNN.initialize_carry(BA, H)
    carry = (
        env_state, flatten_obs(obs).reshape(BA, OBS_DIM),
        jnp.zeros(B, bool), jnp.zeros(B, jnp.int32),
        env_state.level.max(axis=1), jnp.full(B, -1, jnp.int32), h0, key,
    )
    carry, _ = jax.lax.scan(_step, carry, None, length=n_steps)
    env_state, done_flag, death_tick, max_lvl, t_l3 = (
        carry[0], carry[2], carry[3], carry[4], carry[5])
    ticks = jnp.where(done_flag, death_tick, env_state.now)
    return EvalOut(ticks=ticks, max_level=max_lvl, t_l3=t_l3)


def eval_summary(out: EvalOut) -> dict:
    t = np.asarray(out.ticks)
    ml = np.asarray(out.max_level)
    t3 = np.asarray(out.t_l3)
    return {
        "mean_ticks": float(t.mean()),
        "p10_ticks": float(np.percentile(t, 10)),
        "min_ticks": float(t.min()),
        "survival_ge_2000": float((t >= 2000).mean()),
        "max_level_mean": float(ml.mean()),
        "reach_l2_rate": float((ml >= 2).mean()),
        "reach_l3_rate": float((ml >= 3).mean()),
        "t_l3_median": float(np.median(t3[t3 >= 0])) if (t3 >= 0).any() else None,
        "n": int(t.size),
    }


# ----------------------------------------------------------------- driver
def train(tc: TrainConfig, run_name: str = "forage", wandb_mode: str = "disabled",
          init_actor: str | None = None):
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(tc), indent=2))

    wb = None
    if wandb_mode != "disabled":
        import wandb as wb  # type: ignore[no-redef]

        try:
            wb.init(project="zappy-rl", name=run_name, mode=wandb_mode,
                    config=dataclasses.asdict(tc))
        except Exception as e:  # network/auth flake must not kill a segment
            print(f"[train] wandb init ({wandb_mode}) failed: {e!r}; "
                  f"falling back to offline")
            try:
                wb.init(project="zappy-rl", name=run_name, mode="offline",
                        config=dataclasses.asdict(tc))
            except Exception:
                print("[train] offline wandb also failed; logging disabled")
                wb = None

    init_runner, update_iteration, cfg, n_iters = make_train(tc)
    update_iteration = jax.jit(update_iteration)
    runner = init_runner(jax.random.PRNGKey(tc.seed))
    if init_actor:
        # warm-start the actor only (the critic's world_extra dim changes with
        # map/agent count); requires matching OBS_DIM + hidden.
        raw = Path(init_actor).read_bytes()
        loaded = flax.serialization.from_bytes(
            {"actor": runner["ts_a"].params, "critic": None}, raw)
        # from_bytes checks tree structure but NOT leaf shapes — fail loudly
        # here instead of with an opaque jit error mid-training.
        bad = jax.tree.map(lambda t, l: t.shape != jnp.shape(l),
                           runner["ts_a"].params, loaded["actor"])
        if any(jax.tree.leaves(bad)):
            raise ValueError(
                f"--init-actor {init_actor}: actor param shapes do not match "
                f"this config (was the checkpoint trained with a different "
                f"hidden than {tc.hidden}?)")
        runner["ts_a"] = runner["ts_a"].replace(params=loaded["actor"])
        print(f"[train] warm-started actor from {init_actor}")

    steps_per_iter = tc.num_envs * tc.rollout_steps
    sps_hist = []
    print(f"[train] {n_iters} iterations x {steps_per_iter} env-steps "
          f"({tc.total_env_steps:,} total) on {jax.devices()[0]}")
    for it in range(n_iters):
        t0 = time.perf_counter()
        runner, metrics = update_iteration(runner)
        metrics = jax.tree.map(lambda x: float(jax.device_get(x)), metrics)
        dt = time.perf_counter() - t0
        sps = steps_per_iter / dt
        if it > 0:  # exclude compile iteration
            sps_hist.append(sps)
        metrics["env_steps_per_sec"] = sps
        metrics["env_steps"] = (it + 1) * steps_per_iter
        if wb is not None:
            wb.log(metrics, step=metrics["env_steps"])
        if it % tc.log_every == 0 or it == n_iters - 1:
            print(
                f"  it {it:4d}/{n_iters}  sps {sps:>10,.0f}  "
                f"now {metrics['now_mean']:>6.0f}  "
                f"alive {metrics['alive_frac']:.2f}  "
                f"L2 {metrics['frac_l2']:.2f}  L3 {metrics['frac_l3']:.2f}  "
                f"up2 {metrics['lvlups_to_l2']:5.0f}  "
                f"up3 {metrics['lvlups_to_l3']:5.0f}  "
                f"kl {metrics['approx_kl']:.4f}"
            )

    # checkpoint
    params = {"actor": runner["ts_a"].params, "critic": runner["ts_c"].params}
    # atomic: a kill mid-write must not leave a truncated checkpoint that a
    # skip-if-checkpoint resume (tools/speedrun_train.py) would then trust
    ckpt_tmp = run_dir / f"params.msgpack.{os.getpid()}.tmp"
    ckpt_tmp.write_bytes(flax.serialization.to_bytes(params))
    os.replace(ckpt_tmp, run_dir / "params.msgpack")

    # gate eval (stochastic + greedy)
    results = {"train_sps_median": float(np.median(sps_hist)) if sps_hist else 0.0}
    for name, greedy in (("stochastic", False), ("greedy", True)):
        out = jax.jit(evaluate, static_argnums=(0, 1, 4))(
            tc, cfg, runner["ts_a"].params, jax.random.PRNGKey(tc.seed + 1), greedy
        )
        results[name] = eval_summary(jax.device_get(out))
        print(f"[eval/{name}] {results[name]}")
    print(f"[train] median SPS: {results['train_sps_median']:,.0f}")
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    if wb is not None:
        wb.summary.update(results)
        wb.finish()
    return runner, results

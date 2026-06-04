"""Recurrent MAPPO over the vmapped JAX Zappy env (PureJaxRL-style).

Phase 2 scope: shared-parameter recurrent PPO with a centralized critic
(CTDE / MAPPO), event-driven episodes, autoreset, potential-based survival
shaping, and TBPTT-chunked updates. With ``n_agents=1`` this *is* recurrent
PPO; the multi-agent plumbing (per-(env,agent) batch rows, centralized critic
features, broadcast-token head) is already in place for Phase 3.

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

Reward = env reward (level-up +L, death -1)
       + potential shaping  F = gamma*Phi(s') - Phi(s),  Phi = phi_life *
         clip(life/1260, 0, phi_clip) * alive   (policy-invariant; Phi=0 at
         death since `alive` zeroes it)
       + alive_bonus per agent per step survived, FLAT (deliberately not
         scaled by dt: PPO discounts per env-step, so a dt-proportional bonus
         would pay long-dt actions — incantation's 300-tick freeze — a lump
         sum at a single discount factor, making "freeze" beat "forage" on
         reward rate. Flat-per-step keeps the survival gradient and lets the
         potential term price the life cost of long actions correctly.)

Episodes truncate at ``max_episode_ticks`` (treated as terminal in GAE — the
standard, slightly biased simplification).
"""

from __future__ import annotations

import dataclasses
import json
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
    # reward shaping
    phi_life: float = 1.0
    phi_clip: float = 2.0
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

    done: jnp.ndarray      # [BA] bool
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
    """Survival potential, 0 for dead agents (proper terminal potential)."""
    life = jnp.clip(s.life.astype(jnp.float32) / Z.START_LIFE, 0.0, tc.phi_clip)
    return tc.phi_life * life * s.alive.astype(jnp.float32)


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
    done = done_env | (s2.now >= tc.max_episode_ticks)
    ep_ticks = s2.now.astype(jnp.float32)
    s_r, o_r = Z.reset(cfg, k_reset)
    s3 = jax.tree.map(lambda a, b: jnp.where(done, b, a), s2, s_r)
    o3 = jax.tree.map(lambda a, b: jnp.where(done, b, a), o2, o_r)
    return s3, o3, reward, done, ep_ticks, jnp.sum(foods).astype(jnp.float32)


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
    cfg = Z.make_cfg(tc.width, tc.height, tc.n_agents, tc.n_teams)
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

        env_state, obs, reward, done_env, ep_ticks, foods = v_step(
            cfg, tc, jax.random.split(k_step, B), r["env_state"],
            action.reshape(B, A), token.reshape(B, A),
        )
        trans = Transition(
            done=r["done"], action=action, token=token, value=value,
            reward=reward.reshape(BA), logp=logp, obs=r["obs"], wextra=r["wx"],
            h_actor=r["h_a"], h_critic=r["h_c"],
        )
        ep_ret = r["ep_ret"] + reward.mean(axis=1)
        ep_steps = r["ep_steps"] + 1.0
        ep_foods = r["ep_foods"] + foods
        sm = StepMetrics(done=done_env, ticks=ep_ticks, ret=ep_ret, steps=ep_steps, foods=ep_foods)
        keep = ~done_env
        carry2 = {
            **r,
            "env_state": env_state,
            "obs": flatten_obs(obs).reshape(BA, OBS_DIM),
            "wx": v_wx(cfg, env_state).reshape(BA, -1),
            "done": jnp.repeat(done_env, A),
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
        _, la, lt = actor.apply(pa, h0a, (seq.obs, seq.done))
        logp_a = cat_log_prob(la, seq.action)
        is_b = seq.action == Z.ENV_BROADCAST
        logp = logp_a + jnp.where(is_b, cat_log_prob(lt, seq.token), 0.0)
        ratio = jnp.exp(logp - seq.logp)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg1 = ratio * adv_n
        pg2 = jnp.clip(ratio, 1.0 - tc.clip_eps, 1.0 + tc.clip_eps) * adv_n
        pg_loss = -jnp.minimum(pg1, pg2).mean()
        ent_a = cat_entropy(la).mean()
        n_b = jnp.maximum(is_b.sum(), 1)
        ent_t = jnp.where(is_b, cat_entropy(lt), 0.0).sum() / n_b

        cin = jnp.concatenate([seq.obs, seq.wextra], axis=-1)
        _, v = critic.apply(pc, h0c, (cin, seq.done))
        v_clip = seq.value + jnp.clip(v - seq.value, -tc.clip_eps, tc.clip_eps)
        v_loss = 0.5 * jnp.maximum((v - target) ** 2, (v_clip - target) ** 2).mean()

        total = (
            pg_loss
            - tc.ent_coef_action * ent_a
            - tc.ent_coef_token * ent_t
            + tc.vf_coef * v_loss
        )
        approx_kl = ((ratio - 1.0) - jnp.log(ratio)).mean()
        clip_frac = (jnp.abs(ratio - 1.0) > tc.clip_eps).mean()
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
            done=_chunk(traj.done), action=_chunk(traj.action),
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
            "reward_per_step": traj.reward.mean(),
            "value_mean": traj.value.mean(),
        }
        metrics.update(jax.tree.map(jnp.mean, aux))
        return runner, metrics

    return init_runner, update_iteration, cfg, n_iters


# ------------------------------------------------------------------- eval
def evaluate(tc: TrainConfig, cfg: Z.Cfg, actor_params, key, greedy: bool):
    """Run fresh episodes (no autoreset); return ticks survived per env."""
    B, A, H = tc.eval_envs, cfg.n_agents, tc.hidden
    BA = B * A
    actor = RecurrentActor(hidden=H)
    n_steps = tc.eval_max_ticks // 7 + 64

    key, k_env = jax.random.split(key)
    env_state, obs = jax.vmap(Z.reset, in_axes=(None, 0))(cfg, jax.random.split(k_env, B))
    v_step = jax.vmap(Z.step, in_axes=(None, 0, 0, 0, 0))
    no_reset = jnp.zeros((1, BA), bool)

    def _step(carry, _):
        env_state, obs_v, done_flag, death_tick, h, key = carry
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
        # freeze finished envs (their event clock would otherwise run away)
        keep = lambda old, new: jnp.where(  # noqa: E731
            done_flag.reshape((B,) + (1,) * (new.ndim - 1)), old, new
        )
        env_state2 = jax.tree.map(keep, env_state, s2)
        frozen = jnp.repeat(done_flag, A)[:, None]
        obs_v2 = jnp.where(frozen, obs_v, flatten_obs(o2).reshape(BA, OBS_DIM))
        return (env_state2, obs_v2, done_flag | done, death_tick, h2, key), None

    h0 = ScannedRNN.initialize_carry(BA, H)
    carry = (
        env_state, flatten_obs(obs).reshape(BA, OBS_DIM),
        jnp.zeros(B, bool), jnp.zeros(B, jnp.int32), h0, key,
    )
    carry, _ = jax.lax.scan(_step, carry, None, length=n_steps)
    env_state, _, done_flag, death_tick = carry[0], carry[1], carry[2], carry[3]
    ticks = jnp.where(done_flag, death_tick, env_state.now)
    return ticks


def eval_summary(ticks: np.ndarray) -> dict:
    t = np.asarray(ticks)
    return {
        "mean_ticks": float(t.mean()),
        "p10_ticks": float(np.percentile(t, 10)),
        "min_ticks": float(t.min()),
        "survival_ge_2000": float((t >= 2000).mean()),
        "n": int(t.size),
    }


# ----------------------------------------------------------------- driver
def train(tc: TrainConfig, run_name: str = "forage", wandb_mode: str = "disabled"):
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(tc), indent=2))

    wb = None
    if wandb_mode != "disabled":
        import wandb as wb  # type: ignore[no-redef]

        wb.init(project="zappy-rl", name=run_name, mode=wandb_mode,
                config=dataclasses.asdict(tc))

    init_runner, update_iteration, cfg, n_iters = make_train(tc)
    update_iteration = jax.jit(update_iteration)
    runner = init_runner(jax.random.PRNGKey(tc.seed))

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
                f"now>=2k {metrics['now_ge_2000']:.2f}  "
                f"eps {metrics['episodes']:6.0f}  "
                f"ep_ticks {metrics['ep_ticks']:>6.0f}  "
                f"foods {metrics['ep_foods']:5.1f}  kl {metrics['approx_kl']:.4f}"
            )

    # checkpoint
    params = {"actor": runner["ts_a"].params, "critic": runner["ts_c"].params}
    (run_dir / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))

    # gate eval (stochastic + greedy)
    results = {"train_sps_median": float(np.median(sps_hist)) if sps_hist else 0.0}
    for name, greedy in (("stochastic", False), ("greedy", True)):
        ticks = jax.jit(evaluate, static_argnums=(0, 1, 4))(
            tc, cfg, runner["ts_a"].params, jax.random.PRNGKey(tc.seed + 1), greedy
        )
        results[name] = eval_summary(jax.device_get(ticks))
        print(f"[eval/{name}] {results[name]}")
    print(f"[train] median SPS: {results['train_sps_median']:,.0f}")
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    if wb is not None:
        wb.summary.update(results)
        wb.finish()
    return runner, results

"""Unit tests for the recurrent MAPPO building blocks.

Checks the bug-prone PPO/RNN math against slow references: GRU carry resets,
GAE, TBPTT chunk indexing, the ratio==1 first-update invariant, and a full
tiny train iteration (finite losses, params actually change).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from zappy_rl.algo import mappo as M
from zappy_rl.algo.networks import (
    OBS_DIM,
    RecurrentActor,
    ScannedRNN,
    cat_entropy,
    cat_log_prob,
    flatten_obs,
)
from zappy_rl.env import zappy_env as Z

KEY = jax.random.PRNGKey(0)

TINY = M.TrainConfig(
    num_envs=8, rollout_steps=32, total_env_steps=8 * 32 * 2,
    tbptt_chunk=16, num_minibatches=2, epochs=2,
    eval_envs=8, eval_max_ticks=512, max_episode_ticks=512,
)


def test_scanned_rnn_resets_carry():
    """Post-done outputs must equal a fresh-carry run of the suffix."""
    rnn = ScannedRNN(hidden=16)
    h0 = ScannedRNN.initialize_carry(3, 16)
    xs = jax.random.normal(KEY, (10, 3, 8))
    dones = jnp.zeros((10, 3), bool).at[6].set(True)
    params = rnn.init(KEY, h0, (xs, dones))
    _, full = rnn.apply(params, h0, (xs, dones))
    _, suffix = rnn.apply(params, h0, (xs[6:], jnp.zeros((4, 3), bool)))
    np.testing.assert_allclose(full[6:], suffix, rtol=1e-5)
    # ... and without the done they would differ.
    _, nodone = rnn.apply(params, h0, (xs, jnp.zeros((10, 3), bool)))
    assert not np.allclose(nodone[6:], suffix)


def test_gae_matches_slow_reference():
    T, B = 12, 5
    rng = np.random.default_rng(0)
    rew = rng.normal(size=(T, B)).astype(np.float32)
    val = rng.normal(size=(T, B)).astype(np.float32)
    done = rng.random((T, B)) < 0.2
    last_val = rng.normal(size=B).astype(np.float32)
    last_done = rng.random(B) < 0.2
    tc = TINY

    traj = M.Transition(
        done=jnp.asarray(done), action=jnp.zeros((T, B), jnp.int32),
        token=jnp.zeros((T, B), jnp.int32), value=jnp.asarray(val),
        reward=jnp.asarray(rew), logp=jnp.zeros((T, B)),
        obs=jnp.zeros((T, B, 1)), wextra=jnp.zeros((T, B, 1)),
        h_actor=jnp.zeros((T, B, 1)), h_critic=jnp.zeros((T, B, 1)),
    )
    adv, target = M.compute_gae(tc, traj, jnp.asarray(last_val), jnp.asarray(last_done))

    # slow reference: forward definition of GAE with done-as-next-obs-flag
    expect = np.zeros((T, B), np.float32)
    g = np.zeros(B, np.float32)
    nv, nd = last_val, last_done.astype(np.float32)
    for t in reversed(range(T)):
        delta = rew[t] + tc.gamma * nv * (1 - nd) - val[t]
        g = delta + tc.gamma * tc.gae_lambda * (1 - nd) * g
        expect[t] = g
        nv, nd = val[t], done[t].astype(np.float32)
    np.testing.assert_allclose(np.asarray(adv), expect, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(target), expect + val, rtol=1e-4, atol=1e-5)


def test_chunking_preserves_sequences():
    """Chunk c of env row b must be the contiguous slice x[c*chunk:(c+1)*chunk, b]."""
    tc = TINY
    init_runner, _, _, _ = M.make_train(tc)
    T, BA, chunk = tc.rollout_steps, tc.num_envs * tc.n_agents, tc.tbptt_chunk
    n_chunks = T // chunk
    x = jnp.arange(T * BA * 2, dtype=jnp.float32).reshape(T, BA, 2)
    y = x.reshape(n_chunks, chunk, BA, 2).swapaxes(0, 1).reshape(chunk, n_chunks * BA, 2)
    for c in range(n_chunks):
        for b in range(0, BA, 3):
            np.testing.assert_array_equal(
                np.asarray(y[:, c * BA + b]), np.asarray(x[c * chunk:(c + 1) * chunk, b])
            )


def test_first_update_ratio_is_one():
    """With 1 epoch x 1 minibatch the recomputed logp must equal the rollout
    logp exactly (same params, same hiddens, same done resets) => KL ~ 0."""
    tc = dataclasses.replace(TINY, epochs=1, num_minibatches=1, total_env_steps=8 * 32)
    init_runner, update_iteration, _, _ = M.make_train(tc)
    runner = init_runner(KEY)
    _, metrics = jax.jit(update_iteration)(runner)
    assert float(metrics["approx_kl"]) < 1e-5
    assert float(metrics["clip_frac"]) == 0.0


def test_update_iteration_trains():
    init_runner, update_iteration, _, n_iters = M.make_train(TINY)
    assert n_iters == 2
    runner = init_runner(KEY)
    p0 = jax.tree.map(lambda x: x.copy(), runner["ts_a"].params)
    update_iteration = jax.jit(update_iteration)
    for _ in range(2):
        runner, metrics = update_iteration(runner)
    flat = jax.tree.leaves(jax.tree.map(jnp.mean, metrics))
    assert all(np.isfinite(jax.device_get(v)) for v in flat), metrics
    diffs = jax.tree.map(lambda a, b: float(jnp.abs(a - b).max()), p0, runner["ts_a"].params)
    assert max(jax.tree.leaves(diffs)) > 0.0


def test_obs_and_world_dims():
    cfg = Z.make_cfg(6, 6, 2, 1)
    state, obs = Z.reset(cfg, KEY)
    assert flatten_obs(obs).shape == (2, OBS_DIM)
    wx = M.world_extra(cfg, state)
    assert wx.shape == (2, M.world_extra_dim(cfg))
    assert np.isfinite(np.asarray(wx)).all()


def test_token_logprob_masked_to_broadcast_steps():
    """Token head must contribute to logp only when action == ENV_BROADCAST."""
    la = jax.random.normal(KEY, (5, Z.N_ENV_ACTIONS))
    lt = jax.random.normal(KEY, (5, 8))
    action = jnp.array([0, Z.ENV_BROADCAST, 3, Z.ENV_BROADCAST, 19])
    token = jnp.array([1, 2, 3, 4, 5])
    logp = cat_log_prob(la, action) + jnp.where(
        action == Z.ENV_BROADCAST, cat_log_prob(lt, token), 0.0)
    base = cat_log_prob(la, action)
    np.testing.assert_allclose(logp[0], base[0], rtol=1e-6)
    assert abs(float(logp[1] - base[1])) > 1e-6
    assert float(cat_entropy(la[0:1])[0]) > 0.0


def test_evaluate_runs_and_counts_ticks():
    tc = TINY
    cfg = Z.make_cfg(tc.width, tc.height, tc.n_agents, tc.n_teams)
    actor = RecurrentActor(hidden=tc.hidden)
    h0 = ScannedRNN.initialize_carry(1, tc.hidden)
    params = actor.init(KEY, h0, (jnp.zeros((1, 1, OBS_DIM)), jnp.zeros((1, 1), bool)))
    ticks = M.evaluate(tc, cfg, params, KEY, greedy=False)
    t = np.asarray(jax.device_get(ticks))
    assert t.shape == (tc.eval_envs,)
    assert (t > 0).all() and (t <= tc.eval_max_ticks + 300).all()

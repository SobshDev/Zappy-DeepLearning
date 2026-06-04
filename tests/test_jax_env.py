"""JAX env tests: jit/vmap smoke + tile-for-tile cross-check vs the NumPy oracle."""

import jax
import jax.numpy as jnp
import numpy as np

from zappy_rl.env import constants as C
from zappy_rl.env import zappy_env as Z
from zappy_rl.env.reference_env import ZappyReferenceEnv

KEY = jax.random.PRNGKey(0)


def mk_state(cfg, pos, orient, level=None, grid=None):
    A = len(pos)
    level = level or [1] * A
    g = jnp.zeros((cfg.width, cfg.height, C.N_RESOURCES), jnp.int32) if grid is None else jnp.asarray(grid, jnp.int32)
    return Z.State(
        grid=g, pos=jnp.asarray(pos, jnp.int32), orient=jnp.asarray(orient, jnp.int32),
        level=jnp.asarray(level, jnp.int32), inv=jnp.zeros((A, C.N_RESOURCES), jnp.int32),
        life=jnp.full(A, Z.START_LIFE, jnp.int32), alive=jnp.ones(A, bool),
        busy_until=jnp.zeros(A, jnp.int32), pending=jnp.zeros(A, bool),
        incant_level=jnp.zeros(A, jnp.int32), initiator=jnp.zeros(A, bool),
        team=jnp.zeros(A, jnp.int32), now=jnp.int32(0), key=KEY,
        last_dir=jnp.full(A, -1, jnp.int32), last_tok=jnp.full(A, -1, jnp.int32),
    )


def step1(cfg, state, actions, tokens=None):
    A = cfg.n_agents
    tokens = jnp.zeros(A, jnp.int32) if tokens is None else jnp.asarray(tokens, jnp.int32)
    s2, obs, r, d, info = Z.step(cfg, KEY, state, jnp.asarray(actions, jnp.int32), tokens)
    return s2, obs, r, d, info


# --------------------------------------------------------------- smoke
def test_jit_and_vmap_shapes():
    cfg = Z.make_cfg(8, 8, 4, n_teams=2)
    jreset = jax.jit(Z.reset, static_argnums=0)
    jstep = jax.jit(Z.step, static_argnums=0)
    keys = jax.random.split(KEY, 16)
    states, obs = jax.vmap(jreset, in_axes=(None, 0))(cfg, keys)
    assert obs.vision.shape == (16, 4, Z.MAX_VISION_TILES, 8)
    assert obs.self_feat.shape == (16, 4, Z.SELF_DIM)
    acts = jnp.zeros((16, 4), jnp.int32)
    toks = jnp.zeros((16, 4), jnp.int32)
    s2, obs2, r, d, info = jax.vmap(jstep, in_axes=(None, 0, 0, 0, 0))(cfg, keys, states, acts, toks)
    assert r.shape == (16, 4) and d.shape == (16,)
    assert s2.now.shape == (16,)


# ---------------------------------------------------- cross-check vs oracle
def oracle(width, height, no_food=True):
    env = ZappyReferenceEnv(width, height, teams=("T1",), no_food=no_food, no_refill=True, seed=0)
    env.grid[:] = 0
    return env


def test_forward_matches_oracle():
    cfg = Z.make_cfg(11, 11, 1, no_food=True, no_refill=True)
    for orient in (C.NORTH, C.EAST, C.SOUTH, C.WEST):
        s = mk_state(cfg, [[5, 5]], [orient])
        s2, *_ = step1(cfg, s, [Z.ENV_FORWARD])
        env = oracle(11, 11)
        pid = env.spawn("T1", 5, 5, orientation=orient)
        env.do(pid, C.A_FORWARD)
        assert tuple(np.asarray(s2.pos[0])) == (env.players[pid].x, env.players[pid].y)
        assert int(s2.now) == env.now == 7


def test_rotation_matches_oracle():
    cfg = Z.make_cfg(11, 11, 1, no_food=True, no_refill=True)
    s = mk_state(cfg, [[5, 5]], [C.NORTH])
    s, *_ = step1(cfg, s, [Z.ENV_RIGHT])
    assert int(s.orient[0]) == C.EAST
    s, *_ = step1(cfg, s, [Z.ENV_LEFT])
    s, *_ = step1(cfg, s, [Z.ENV_LEFT])
    assert int(s.orient[0]) == C.WEST


def test_take_and_set_match_oracle():
    cfg = Z.make_cfg(11, 11, 1, no_food=True, no_refill=True)
    grid = np.zeros((11, 11, 7), np.int32)
    grid[5, 5, C.LINEMATE] = 2
    s = mk_state(cfg, [[5, 5]], [C.NORTH], grid=grid)
    s, *_ = step1(cfg, s, [Z.ENV_TAKE0 + C.LINEMATE])
    assert int(s.inv[0, C.LINEMATE]) == 1 and int(s.grid[5, 5, C.LINEMATE]) == 1
    s, *_ = step1(cfg, s, [Z.ENV_SET0 + C.LINEMATE])
    assert int(s.inv[0, C.LINEMATE]) == 0 and int(s.grid[5, 5, C.LINEMATE]) == 2


def test_food_decay_matches_oracle():
    cfg = Z.make_cfg(11, 11, 1, no_food=False, no_refill=True)
    s = mk_state(cfg, [[5, 5]], [C.NORTH])
    s, *_ = step1(cfg, s, [Z.ENV_FORWARD])  # cost 7 -> dt 7
    env = ZappyReferenceEnv(11, 11, teams=("T1",), no_food=False, no_refill=True, seed=0)
    env.grid[:] = 0
    pid = env.spawn("T1", 5, 5, orientation=C.NORTH)
    env.do(pid, C.A_FORWARD)
    assert int(s.life[0]) == env.players[pid].life_ticks == Z.START_LIFE - 7


def test_incantation_l1_to_l2_matches_oracle():
    cfg = Z.make_cfg(11, 11, 1, no_food=True, no_refill=True)
    grid = np.zeros((11, 11, 7), np.int32)
    grid[5, 5, C.LINEMATE] = 1
    s = mk_state(cfg, [[5, 5]], [C.NORTH], grid=grid)
    s, *_ = step1(cfg, s, [Z.ENV_INCANT])      # start: freeze 300, not yet leveled
    assert bool(s.pending[0]) and int(s.level[0]) == 1 and int(s.now) == C.COST_INCANTATION
    s, *_ = step1(cfg, s, [Z.ENV_IDLE])        # completion resolved at top of step
    assert int(s.level[0]) == 2
    assert int(s.grid[5, 5, C.LINEMATE]) == 0

    env = oracle(11, 11)
    pid = env.spawn("T1", 5, 5, level=1)
    env.grid[5, 5, C.LINEMATE] = 1
    assert env.do(pid, C.A_INCANTATION) == 2
    assert env.players[pid].level == 2 and env.grid[5, 5, C.LINEMATE] == 0


def test_incantation_needs_two_players_for_l2_to_l3():
    cfg = Z.make_cfg(11, 11, 2, no_food=True, no_refill=True)
    grid = np.zeros((11, 11, 7), np.int32)
    grid[5, 5, C.LINEMATE] = 1
    grid[5, 5, C.DERAUMERE] = 1
    grid[5, 5, C.SIBUR] = 1
    # Only agent 0 on the tile -> start-check fails, no freeze.
    s = mk_state(cfg, [[5, 5], [0, 0]], [C.NORTH, C.NORTH], level=[2, 2], grid=grid)
    s1, *_ = step1(cfg, s, [Z.ENV_INCANT, Z.ENV_IDLE])
    assert not bool(s1.pending[0])             # ritual not started
    # Both agents on the tile -> starts and (after completion) both reach level 3.
    s = mk_state(cfg, [[5, 5], [5, 5]], [C.NORTH, C.NORTH], level=[2, 2], grid=grid)
    s, *_ = step1(cfg, s, [Z.ENV_INCANT, Z.ENV_IDLE])
    assert bool(s.pending[0]) and bool(s.pending[1])
    s, *_ = step1(cfg, s, [Z.ENV_IDLE, Z.ENV_IDLE])
    assert int(s.level[0]) == 3 and int(s.level[1]) == 3


def test_incantation_double_initiator_consumes_stones_once():
    """Two agents starting the SAME ritual in the same step = one ritual:
    stones must be consumed once per tile, not once per initiator."""
    cfg = Z.make_cfg(11, 11, 2, no_food=True, no_refill=True)
    grid = np.zeros((11, 11, 7), np.int32)
    grid[5, 5, C.LINEMATE] = 2
    grid[5, 5, C.DERAUMERE] = 2
    grid[5, 5, C.SIBUR] = 2
    s = mk_state(cfg, [[5, 5], [5, 5]], [C.NORTH, C.NORTH], level=[2, 2], grid=grid)
    s, *_ = step1(cfg, s, [Z.ENV_INCANT, Z.ENV_INCANT])
    assert bool(s.pending[0]) and bool(s.pending[1])
    s, *_ = step1(cfg, s, [Z.ENV_IDLE, Z.ENV_IDLE])
    assert int(s.level[0]) == 3 and int(s.level[1]) == 3
    for res in (C.LINEMATE, C.DERAUMERE, C.SIBUR):
        assert int(s.grid[5, 5, res]) == 1, f"res {res} double-consumed"


def test_dead_during_freeze_participant_does_not_level():
    """A surplus participant that starves during the 300-tick freeze must not
    level up or be paid the level-up reward (oracle levels alive players only).
    The ritual itself still succeeds: 2 alive L2 players remain (req_p=2)."""
    cfg = Z.make_cfg(11, 11, 3, no_food=False, no_refill=True)
    grid = np.zeros((11, 11, 7), np.int32)
    grid[5, 5, C.LINEMATE] = 1
    grid[5, 5, C.DERAUMERE] = 1
    grid[5, 5, C.SIBUR] = 1
    s = mk_state(cfg, [[5, 5]] * 3, [C.NORTH] * 3, level=[2, 2, 2], grid=grid)
    s = s._replace(life=jnp.array([Z.START_LIFE, Z.START_LIFE, 100], jnp.int32))
    s, *_ = step1(cfg, s, [Z.ENV_INCANT, Z.ENV_IDLE, Z.ENV_IDLE])
    assert bool(s.pending.all())            # all 3 co-located L2s get frozen
    assert not bool(s.alive[2])             # starved during the 300-tick dt
    s, _, r, d, _ = step1(cfg, s, [Z.ENV_IDLE] * 3)
    assert int(s.level[0]) == 3 and int(s.level[1]) == 3
    assert int(s.level[2]) == 2             # dead participant must NOT level
    assert float(r[2]) <= 0.0               # ...nor be paid for it
    assert not bool(d)


def test_broadcast_direction_in_obs():
    cfg = Z.make_cfg(11, 11, 2, no_food=True, no_refill=True)
    # agent0 emitter north of agent1; agent1 faces north -> hears K=1 (front).
    s = mk_state(cfg, [[5, 4], [5, 5]], [C.NORTH, C.NORTH])
    s2, obs, *_ = step1(cfg, s, [Z.ENV_BROADCAST, Z.ENV_IDLE])
    assert int(np.argmax(np.asarray(obs.msg_dir[1]))) == 1   # one-hot K=1
    assert float(np.asarray(obs.msg_dir[0]).sum()) == 0.0    # emitter hears nothing

    env = oracle(11, 11)
    e = env.spawn("T1", 5, 4)
    r = env.spawn("T1", 5, 5, orientation=C.NORTH)
    assert env.broadcast(e) == {r: 1}

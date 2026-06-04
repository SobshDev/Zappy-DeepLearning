"""Recorder tests: pure state-diff -> GUI wire lines + end-to-end persistence.

The pure ``events_from_step`` rules are exercised against hand-built ``State``
snapshots (the ``mk_state`` pattern from ``tests/test_jax_env.py``) and scripted
actions, so each emit is checked in isolation without a policy. The driver test
runs a tiny real episode and asserts the NDJSON / SQLite contract.
"""

import glob
import json
import os
import sqlite3

import jax
import jax.numpy as jnp
import numpy as np

from zappy_rl.env import constants as C
from zappy_rl.env import zappy_env as Z
from zappy_rl.viz import recorder as R

KEY = jax.random.PRNGKey(0)


def mk_state(cfg, pos, orient, level=None, grid=None):
    A = len(pos)
    level = level or [1] * A
    g = (jnp.zeros((cfg.width, cfg.height, C.N_RESOURCES), jnp.int32)
         if grid is None else jnp.asarray(grid, jnp.int32))
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
    return Z.step(cfg, KEY, state, jnp.asarray(actions, jnp.int32), tokens)


def _tiny_cfg(n_agents=2, no_food=True):
    return Z.make_cfg(6, 6, n_agents, no_food=no_food, no_refill=True)


def _events(cfg, s, s2, actions, info, pids=None, tokens=None):
    pids = pids if pids is not None else list(range(cfg.n_agents))
    toks = (jnp.zeros(cfg.n_agents, jnp.int32) if tokens is None
            else jnp.asarray(tokens, jnp.int32))
    return R.events_from_step(cfg, s, s2, jnp.asarray(actions, jnp.int32),
                              toks, info, pids)


# --------------------------------------------------------------- ppo (move)
def test_forward_move_emits_single_ppo_with_new_pos_and_orient():
    cfg = _tiny_cfg()
    s = mk_state(cfg, [[2, 3], [0, 0]], [C.NORTH, C.NORTH])
    s2, _o, _r, _d, info = step1(cfg, s, [Z.ENV_FORWARD, Z.ENV_IDLE])
    lines = _events(cfg, s, s2, [Z.ENV_FORWARD, Z.ENV_IDLE], info)
    ppos = [ln for ln in lines if ln.startswith("ppo #0")]
    assert len(ppos) == 1
    # NORTH = (0,-1): y goes 3 -> 2, x unchanged, orient unchanged.
    nx, ny = int(s2.pos[0, 0]), int(s2.pos[0, 1])
    assert ppos[0] == f"ppo #0 {nx} {ny} {C.NORTH}"
    # the idle agent emits no ppo.
    assert not any(ln.startswith("ppo #1") for ln in lines)


def test_right_turn_emits_ppo_same_pos_new_orient():
    cfg = _tiny_cfg()
    s = mk_state(cfg, [[2, 3], [0, 0]], [C.NORTH, C.NORTH])
    s2, _o, _r, _d, info = step1(cfg, s, [Z.ENV_RIGHT, Z.ENV_IDLE])
    lines = _events(cfg, s, s2, [Z.ENV_RIGHT, Z.ENV_IDLE], info)
    ppos = [ln for ln in lines if ln.startswith("ppo #0")]
    assert len(ppos) == 1
    assert ppos[0] == f"ppo #0 2 3 {C.EAST}"  # NORTH right-turns to EAST, pos kept


# ----------------------------------------------------------- pin + bct (take)
def test_take_linemate_emits_pin_and_bct():
    cfg = _tiny_cfg()
    grid = np.zeros((6, 6, 7), np.int32)
    grid[2, 3, C.LINEMATE] = 2
    s = mk_state(cfg, [[2, 3], [0, 0]], [C.NORTH, C.NORTH], grid=grid)
    s2, _o, _r, _d, info = step1(cfg, s, [Z.ENV_TAKE0 + C.LINEMATE, Z.ENV_IDLE])
    lines = _events(cfg, s, s2, [Z.ENV_TAKE0 + C.LINEMATE, Z.ENV_IDLE], info)

    pins = [ln for ln in lines if ln.startswith("pin #0")]
    assert len(pins) == 1
    # inventory now holds one linemate (q1), tile reported at the player pos.
    inv = np.asarray(s2.inv[0])
    tail = " ".join(str(int(inv[q])) for q in range(7))
    assert pins[0] == f"pin #0 2 3 {tail}"
    assert inv[C.LINEMATE] == 1

    bcts = [ln for ln in lines if ln.startswith("bct 2 3")]
    assert len(bcts) == 1
    # tile linemate dropped 2 -> 1.
    assert bcts[0] == f"bct 2 3 {R._tile_str(np.asarray(s2.grid), 2, 3)}"
    assert int(s2.grid[2, 3, C.LINEMATE]) == 1


# ------------------------------------------------- pic / pie / plv (incant)
def test_incantation_l1_to_l2_emits_pic_then_pie_r1_and_plv():
    cfg = _tiny_cfg(n_agents=1)
    grid = np.zeros((6, 6, 7), np.int32)
    grid[2, 3, C.LINEMATE] = 1
    s = mk_state(cfg, [[2, 3]], [C.NORTH], grid=grid)

    # START: ENV_INCANT freezes the agent -> pending False->True -> one pic.
    s_start, _o, _r, _d, info_start = step1(cfg, s, [Z.ENV_INCANT])
    start_lines = _events(cfg, s, s_start, [Z.ENV_INCANT], info_start)
    pics = [ln for ln in start_lines if ln.startswith("pic")]
    assert len(pics) == 1
    assert pics[0] == "pic 2 3 1 #0"          # tile X Y, level 1, participant #0
    assert bool(s_start.pending[0]) and int(s_start.level[0]) == 1

    # COMPLETION: next step resolves the freeze -> pending True->False, leveled.
    s_done, _o, _r, _d, info_done = step1(cfg, s_start, [Z.ENV_IDLE])
    done_lines = _events(cfg, s_start, s_done, [Z.ENV_IDLE], info_done)
    pies = [ln for ln in done_lines if ln.startswith("pie")]
    plvs = [ln for ln in done_lines if ln.startswith("plv")]
    assert len(pies) == 1
    assert pies[0] == "pie 2 3 1"             # R=1: a participant leveled
    assert len(plvs) == 1
    assert plvs[0] == "plv #0 2"
    assert int(s_done.level[0]) == 2
    # stone consumed -> a bct for the tile too.
    assert any(ln.startswith("bct 2 3") for ln in done_lines)


# --------------------------------------------------------------- pbc (bcast)
def test_broadcast_emits_pbc_with_token():
    cfg = _tiny_cfg()
    s = mk_state(cfg, [[2, 3], [2, 2]], [C.NORTH, C.NORTH])
    s2, _o, _r, _d, info = step1(cfg, s, [Z.ENV_BROADCAST, Z.ENV_IDLE], tokens=[5, 0])
    lines = _events(cfg, s, s2, [Z.ENV_BROADCAST, Z.ENV_IDLE], info, tokens=[5, 0])
    pbcs = [ln for ln in lines if ln.startswith("pbc")]
    assert pbcs == ["pbc #0 5"]


def test_double_broadcast_emits_single_pbc_for_delivered_emitter():
    """zappy_env v1 delivers only the lowest-index broadcaster; the recorder
    must not fabricate a ripple (or a wrong token) for the dropped one."""
    cfg = Z.make_cfg(6, 6, 3, no_food=True, no_refill=True)
    s = mk_state(cfg, [[2, 3], [2, 2], [4, 4]], [C.NORTH] * 3)
    acts = [Z.ENV_BROADCAST, Z.ENV_IDLE, Z.ENV_BROADCAST]
    s2, _o, _r, _d, info = step1(cfg, s, acts, tokens=[5, 0, 7])
    lines = _events(cfg, s, s2, acts, info, tokens=[5, 0, 7])
    pbcs = [ln for ln in lines if ln.startswith("pbc")]
    assert pbcs == ["pbc #0 5"]  # agent 2's broadcast was dropped by the env


def test_lone_broadcaster_pbc_carries_true_token():
    """With no receivers, last_tok carries nothing — the token must come from
    the emitter's own submitted symbol (was: fell back to 0)."""
    cfg = _tiny_cfg(n_agents=1)
    s = mk_state(cfg, [[2, 3]], [C.NORTH])
    s2, _o, _r, _d, info = step1(cfg, s, [Z.ENV_BROADCAST], tokens=[6])
    lines = _events(cfg, s, s2, [Z.ENV_BROADCAST], info, tokens=[6])
    assert [ln for ln in lines if ln.startswith("pbc")] == ["pbc #0 6"]


# --------------------------------------------------------------- pdi (death)
def test_death_emits_pdi():
    cfg = _tiny_cfg(n_agents=1, no_food=False)
    s = mk_state(cfg, [[2, 3]], [C.NORTH])
    # one food tick from death: the dt=7 forward step starves the agent.
    s = s._replace(life=jnp.asarray([5], jnp.int32))
    s2, _o, _r, _d, info = step1(cfg, s, [Z.ENV_FORWARD])
    assert not bool(s2.alive[0])
    lines = _events(cfg, s, s2, [Z.ENV_FORWARD], info)
    assert "pdi #0" in lines


# --------------------------------------------------------------- header
def test_episode_header_contents():
    cfg = _tiny_cfg(n_agents=2)
    grid = np.zeros((6, 6, 7), np.int32)
    grid[1, 1, C.FOOD] = 3
    s = mk_state(cfg, [[1, 1], [4, 4]], [C.NORTH, C.EAST], grid=grid)
    pids = [0, 1]
    lines = R.episode_header(cfg, s, pids, team_names=("T1",), freq=100)
    assert lines[0] == "msz 6 6"
    assert "sgt 100" in lines
    assert "tna T1" in lines
    # one bct per tile, plus the food tile reflects its count.
    bcts = [ln for ln in lines if ln.startswith("bct")]
    assert len(bcts) == 6 * 6
    assert "bct 1 1 3 0 0 0 0 0 0" in lines
    # one pnw per agent with X Y O L team.
    assert "pnw #0 1 1 1 1 T1" in lines
    assert "pnw #1 4 4 2 1 T1" in lines


# --------------------------------------------------- end-to-end driver
def test_record_episode_end_to_end(tmp_path):
    cfg = _tiny_cfg(n_agents=2, no_food=True)

    # scripted policy: alternate forward / turn so we get steady ppo emits.
    def policy_fn(obs, state, t):
        a = Z.ENV_FORWARD if (t % 2 == 0) else Z.ENV_RIGHT
        actions = jnp.full((cfg.n_agents,), a, jnp.int32)
        tokens = jnp.zeros((cfg.n_agents,), jnp.int32)
        return actions, tokens

    out_dir = str(tmp_path / "replays")
    ndjson_path = R.record_episode(
        cfg, out_dir, generation=3, episode=7, policy_fn=policy_fn,
        seed=0, max_steps=12, team_names=("T1",),
    )

    # NDJSON exists and every line parses.
    assert os.path.exists(ndjson_path)
    assert os.path.basename(ndjson_path) == "replay_g3_e7.ndjson"
    with open(ndjson_path) as fh:
        raw = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert raw, "expected at least the header lines"
    recs = [json.loads(ln) for ln in raw]
    for rec in recs:
        assert set(rec.keys()) == {"gen", "ep", "tick", "line"}
        assert rec["gen"] == 3 and rec["ep"] == 7
        assert isinstance(rec["line"], str) and rec["line"]

    # ticks are monotonically non-decreasing (commit-time clock only advances).
    ticks = [rec["tick"] for rec in recs]
    assert ticks == sorted(ticks)

    # header present.
    lines = [rec["line"] for rec in recs]
    assert any(ln.startswith("msz") for ln in lines)
    assert any(ln.startswith("pnw") for ln in lines)
    assert any(ln.startswith("bct") for ln in lines)

    # SQLite row count == NDJSON line count, indexed by (gen, episode, tick).
    db_path = os.path.join(out_dir, "replays.sqlite")
    assert os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    try:
        n_rows = conn.execute(
            "SELECT COUNT(*) FROM events WHERE gen=? AND episode=?", (3, 7)
        ).fetchone()[0]
        assert n_rows == len(recs)
        # the index exists.
        idx = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_events'"
        ).fetchone()
        assert idx is not None
        # rows round-trip in tick order.
        db_lines = conn.execute(
            "SELECT line FROM events WHERE gen=? AND episode=? ORDER BY tick", (3, 7)
        ).fetchall()
        assert len(db_lines) == len(recs)
    finally:
        conn.close()


def test_record_episode_only_my_files(tmp_path):
    """Two episodes -> two distinct NDJSON files, one shared DB."""
    cfg = _tiny_cfg(n_agents=1, no_food=True)

    def policy_fn(obs, state, t):
        return jnp.full((1,), Z.ENV_IDLE, jnp.int32), jnp.zeros((1,), jnp.int32)

    out_dir = str(tmp_path / "r")
    R.record_episode(cfg, out_dir, generation=0, episode=0, policy_fn=policy_fn,
                     seed=1, max_steps=3)
    R.record_episode(cfg, out_dir, generation=0, episode=1, policy_fn=policy_fn,
                     seed=2, max_steps=3)
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "*.ndjson")))
    assert files == ["replay_g0_e0.ndjson", "replay_g0_e1.ndjson"]
    conn = sqlite3.connect(os.path.join(out_dir, "replays.sqlite"))
    try:
        eps = conn.execute("SELECT DISTINCT episode FROM events ORDER BY episode").fetchall()
        assert [e[0] for e in eps] == [0, 1]
    finally:
        conn.close()

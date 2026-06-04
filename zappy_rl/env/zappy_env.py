"""Vectorized JAX Zappy environment (training env).

Mirrors the NumPy oracle (``reference_env.py``) but as a JIT/vmap-friendly pure
state machine, for high-throughput RL on the GPU. The oracle is the correctness
reference: ``tests/test_jax_env.py`` cross-checks this env tile-for-tile.

Design
------
* State is a fixed-shape ``NamedTuple`` of arrays (no Python control flow over
  agents); batch with ``jax.vmap`` over keys/states.
* **Event-driven clock**: each ``step`` advances the global tick clock by
  ``dt = min remaining cooldown`` over alive agents (≥1). Free agents
  (``now >= busy_until``) act and acquire a cooldown of their action's tick cost
  (7 / 300 / ...); busy agents no-op. This matches the oracle's per-action
  timeline (the single-agent case advances by exactly the action cost).
* **Incantation** freezes co-located same-level agents for 300 ticks; the
  start-check happens at submission and the end-check is resolved at the top of
  the step in which the freeze elapses (consume stones once per initiator tile,
  level up every frozen participant on a satisfied tile).

v1 scope / deferred (see TODOs): fork + egg population growth, eject, broadcast
aggregation beyond one emitter/step, and take-contention when several agents
grab the same scarce tile in one step. The cross-check uses scenarios that avoid
these. ``Look``/``Inventory`` are not learned actions — the policy always
receives a fresh observation (the deploy adapter issues ``Look``/``Inventory``
every decision cycle), so an explicit ``IDLE`` action stands in for "perceive".
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import constants as C
from .vision import vision_offsets

# --- env-level flat action layout -----------------------------------------
ENV_FORWARD = 0
ENV_RIGHT = 1
ENV_LEFT = 2
ENV_TAKE0 = 3        # ENV_TAKE0 + res, res in 0..6
ENV_SET0 = 10        # ENV_SET0 + res
ENV_BROADCAST = 17
ENV_INCANT = 18
ENV_IDLE = 19        # perceive / no-op (costs a Look)
N_ENV_ACTIONS = 20

MAX_VISION_TILES = (C.MAX_LEVEL + 1) ** 2          # 81
SELF_DIM = C.MAX_LEVEL + C.N_RESOURCES + 4 + 2     # level1h + inv + orient1h + food + busy = 21

_COST = np.full(N_ENV_ACTIONS, 7, dtype=np.int32)
_COST[ENV_INCANT] = C.COST_INCANTATION            # 300
COST_ENV = jnp.asarray(_COST)

# Geometry constants (precomputed).
FWD = jnp.asarray([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)   # index orient-1
RGT = jnp.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=jnp.int32)
OFFS = jnp.asarray(np.stack([vision_offsets(C.MAX_LEVEL, o) for o in (1, 2, 3, 4)]), dtype=jnp.int32)  # [4,81,2]

# Elevation requirement tables indexed by *source* level (1..7).
_REQ_P = np.zeros(C.MAX_LEVEL + 1, dtype=np.int32)
_REQ_S = np.zeros((C.MAX_LEVEL + 1, C.N_STONES), dtype=np.int32)
for _L in range(1, C.MAX_LEVEL):
    _REQ_P[_L] = C.ELEVATION[_L + 1]["players"]
    _REQ_S[_L] = C.ELEVATION[_L + 1]["stones"]
REQ_P = jnp.asarray(_REQ_P)
REQ_S = jnp.asarray(_REQ_S)

START_LIFE = C.START_FOOD * C.FOOD_LIFE_TICKS
_BIG = jnp.int32(1 << 30)


class Cfg(NamedTuple):
    width: int
    height: int
    n_agents: int
    n_teams: int
    no_food: bool
    no_refill: bool


class State(NamedTuple):
    grid: jnp.ndarray        # [W,H,7] int32
    pos: jnp.ndarray         # [A,2] int32
    orient: jnp.ndarray      # [A] int32 (1..4)
    level: jnp.ndarray       # [A] int32
    inv: jnp.ndarray         # [A,7] int32
    life: jnp.ndarray        # [A] int32 (life ticks)
    alive: jnp.ndarray       # [A] bool
    busy_until: jnp.ndarray  # [A] int32
    pending: jnp.ndarray     # [A] bool (frozen in an incantation)
    incant_level: jnp.ndarray  # [A] int32
    initiator: jnp.ndarray   # [A] bool
    team: jnp.ndarray        # [A] int32
    now: jnp.ndarray         # scalar int32
    key: jnp.ndarray
    last_dir: jnp.ndarray    # [A] int32 (heard broadcast K, -1 none)
    last_tok: jnp.ndarray    # [A] int32 (heard token, -1 none)


class Obs(NamedTuple):
    vision: jnp.ndarray      # [A,81,8] float32
    vision_mask: jnp.ndarray  # [A,81] float32
    self_feat: jnp.ndarray   # [A,SELF_DIM] float32
    msg_dir: jnp.ndarray     # [A,9] float32 (one-hot K)
    msg_tok: jnp.ndarray     # [A,8] float32


def make_cfg(width, height, n_agents, n_teams=1, no_food=False, no_refill=False) -> Cfg:
    return Cfg(int(width), int(height), int(n_agents), int(n_teams), bool(no_food), bool(no_refill))


def _target_qty(cfg: Cfg, res: int) -> int:
    return max(1, int(cfg.width * cfg.height * C.DENSITY[res]))


def _scatter_grid(cfg: Cfg, key, grid, res: int, n: int):
    kx, ky = jax.random.split(key)
    xs = jax.random.randint(kx, (n,), 0, cfg.width)
    ys = jax.random.randint(ky, (n,), 0, cfg.height)
    return grid.at[xs, ys, res].add(1)


def reset(cfg: Cfg, key):
    key, kgrid = jax.random.split(key)
    grid = jnp.zeros((cfg.width, cfg.height, C.N_RESOURCES), dtype=jnp.int32)
    for res in range(C.N_RESOURCES):
        kgrid, ksub = jax.random.split(kgrid)
        grid = _scatter_grid(cfg, ksub, grid, res, _target_qty(cfg, res))

    A = cfg.n_agents
    key, kx, ky, ko = jax.random.split(key, 4)
    pos = jnp.stack([jax.random.randint(kx, (A,), 0, cfg.width),
                     jax.random.randint(ky, (A,), 0, cfg.height)], axis=1).astype(jnp.int32)
    orient = jax.random.randint(ko, (A,), 1, 5).astype(jnp.int32)
    team = (jnp.arange(A) * cfg.n_teams // A).astype(jnp.int32)
    state = State(
        grid=grid, pos=pos, orient=orient,
        level=jnp.ones(A, jnp.int32), inv=jnp.zeros((A, C.N_RESOURCES), jnp.int32),
        life=jnp.full(A, START_LIFE, jnp.int32), alive=jnp.ones(A, bool),
        busy_until=jnp.zeros(A, jnp.int32), pending=jnp.zeros(A, bool),
        incant_level=jnp.zeros(A, jnp.int32), initiator=jnp.zeros(A, bool),
        team=team, now=jnp.int32(0), key=key,
        last_dir=jnp.full(A, -1, jnp.int32), last_tok=jnp.full(A, -1, jnp.int32),
    )
    return state, observe(cfg, state)


# ------------------------------------------------------------------ observe
def observe(cfg: Cfg, s: State) -> Obs:
    W, H = cfg.width, cfg.height
    pcount = jnp.zeros((W, H), jnp.int32).at[s.pos[:, 0], s.pos[:, 1]].add(s.alive.astype(jnp.int32))
    offs = OFFS[s.orient - 1]                          # [A,81,2]
    tiles = (s.pos[:, None, :] + offs) % jnp.array([W, H])
    tx, ty = tiles[..., 0], tiles[..., 1]
    res_feat = s.grid[tx, ty].astype(jnp.float32)      # [A,81,7]
    ppl = pcount[tx, ty][..., None].astype(jnp.float32)
    vision = jnp.concatenate([res_feat, ppl], axis=-1)  # [A,81,8]
    mask = (jnp.arange(MAX_VISION_TILES)[None, :] < (s.level[:, None] + 1) ** 2).astype(jnp.float32)
    vision = vision * mask[..., None]

    self_feat = jnp.concatenate([
        jax.nn.one_hot(s.level - 1, C.MAX_LEVEL),
        (s.inv.astype(jnp.float32) / 10.0),
        jax.nn.one_hot(s.orient - 1, 4),
        jnp.clip(s.life[:, None].astype(jnp.float32) / START_LIFE, 0, 1),
        (s.now < s.busy_until)[:, None].astype(jnp.float32),
    ], axis=1)

    has = s.last_dir >= 0
    msg_dir = jnp.where(has[:, None], jax.nn.one_hot(jnp.clip(s.last_dir, 0, 8), 9), 0.0)
    msg_tok = jnp.where(has[:, None], jax.nn.one_hot(jnp.clip(s.last_tok, 0, C.BROADCAST_VOCAB - 1), C.BROADCAST_VOCAB), 0.0)
    return Obs(vision, mask, self_feat, msg_dir, msg_tok)


# --------------------------------------------------------------- geometry
def _tor(d, n):
    return ((d + n // 2) % n) - n // 2


def _dir_from_to(epos, rpos, rorient, W, H):
    """Direction K (0..8) a receiver at rpos/rorient hears an emitter at epos."""
    dx = _tor(epos[0] - rpos[:, 0], W).astype(jnp.float32)
    dy = _tor(epos[1] - rpos[:, 1], H).astype(jnp.float32)
    f = FWD[rorient - 1].astype(jnp.float32)
    r = RGT[rorient - 1].astype(jnp.float32)
    front = dx * f[:, 0] + dy * f[:, 1]
    right = dx * r[:, 0] + dy * r[:, 1]
    deg = jnp.degrees(jnp.arctan2(-right, front)) % 360.0
    sector = jnp.floor((deg + 22.5) / 45.0).astype(jnp.int32) % 8
    same = (dx == 0) & (dy == 0)
    return jnp.where(same, 0, sector + 1)


def _ready(cfg: Cfg, s: State, level):
    """Per-agent: can a `level`->level+1 ritual succeed at the agent's tile?"""
    same_tile = (s.pos[:, None, 0] == s.pos[None, :, 0]) & (s.pos[:, None, 1] == s.pos[None, :, 1])
    same_lvl = level[:, None] == level[None, :]
    count = jnp.sum(same_tile & same_lvl & s.alive[None, :], axis=1)
    req_p = REQ_P[jnp.clip(level, 0, C.MAX_LEVEL)]
    grid_at = s.grid[s.pos[:, 0], s.pos[:, 1]]          # [A,7]
    req_s = REQ_S[jnp.clip(level, 0, C.MAX_LEVEL)]      # [A,6]
    stones_ok = jnp.all(grid_at[:, 1:] >= req_s, axis=1)
    return (count >= req_p) & stones_ok & (level >= 1) & (level < C.MAX_LEVEL) & (req_p > 0)


# ------------------------------------------------------------------- step
def step(cfg: Cfg, key, s: State, actions, tokens):
    W, H = cfg.width, cfg.height
    A = cfg.n_agents
    idx = jnp.arange(A)
    now = s.now
    WH = jnp.array([W, H])

    # 1) Resolve incantation completions due at `now`.
    completed = s.pending & (s.busy_until <= now)
    comp_init = completed & s.initiator
    init_ready = comp_init & _ready(cfg, s, s.incant_level)
    # success level per tile (assume <=1 initiator per tile in v1).
    succ_lvl = jnp.zeros((W, H), jnp.int32).at[s.pos[:, 0], s.pos[:, 1]].max(
        jnp.where(init_ready, s.incant_level, 0))
    # consume stones ONCE per successful tile (keyed by succ_lvl): two agents
    # initiating on the same tile in the same step is one ritual, not two —
    # the oracle/server consume per ritual, not per initiator.
    tile_req = REQ_S[jnp.clip(succ_lvl, 0, C.MAX_LEVEL)]      # [W,H,6]; REQ_S[0]=0
    grid = jnp.maximum(s.grid.at[:, :, 1:].add(-tile_req), 0)
    # only ALIVE participants level (oracle: reference_env._incantation) — a
    # surplus participant that starved during the freeze must not level/score.
    leveled = (completed & (succ_lvl[s.pos[:, 0], s.pos[:, 1]] == s.incant_level)
               & (s.incant_level > 0) & s.alive)
    level = s.level + leveled.astype(jnp.int32)
    pending = s.pending & ~completed
    initiator = s.initiator & ~completed
    incant_level = jnp.where(completed, 0, s.incant_level)

    # 2) Free agents act.
    free = s.alive & (now >= s.busy_until) & ~pending
    act = actions
    is_fwd = free & (act == ENV_FORWARD)
    is_right = free & (act == ENV_RIGHT)
    is_left = free & (act == ENV_LEFT)
    is_take = free & (act >= ENV_TAKE0) & (act < ENV_TAKE0 + C.N_RESOURCES)
    take_res = jnp.clip(act - ENV_TAKE0, 0, C.N_RESOURCES - 1)
    is_set = free & (act >= ENV_SET0) & (act < ENV_SET0 + C.N_RESOURCES)
    set_res = jnp.clip(act - ENV_SET0, 0, C.N_RESOURCES - 1)
    is_bcast = free & (act == ENV_BROADCAST)
    is_incant = free & (act == ENV_INCANT)

    # movement / rotation
    pos = jnp.where(is_fwd[:, None], (s.pos + FWD[s.orient - 1]) % WH, s.pos)
    orient = s.orient
    orient = jnp.where(is_right, orient % 4 + 1, orient)
    orient = jnp.where(is_left, jnp.where(orient > 1, orient - 1, C.WEST), orient)

    # take (v1: no contention handling)
    grid_here = grid[s.pos[:, 0], s.pos[:, 1]]
    take_avail = grid_here[idx, take_res] > 0
    ok_take = is_take & take_avail
    grid = grid.at[s.pos[:, 0], s.pos[:, 1], take_res].add(-ok_take.astype(jnp.int32))
    grid = jnp.maximum(grid, 0)
    inv = s.inv.at[idx, take_res].add(ok_take.astype(jnp.int32))
    life = s.life + (ok_take & (take_res == C.FOOD)).astype(jnp.int32) * C.FOOD_LIFE_TICKS

    # set
    ok_set = is_set & (inv[idx, set_res] > 0)
    grid = grid.at[s.pos[:, 0], s.pos[:, 1], set_res].add(ok_set.astype(jnp.int32))
    inv = inv.at[idx, set_res].add(-ok_set.astype(jnp.int32))
    life = life - (ok_set & (set_res == C.FOOD)).astype(jnp.int32) * C.FOOD_LIFE_TICKS

    # broadcast delivery (v1: single emitter, lowest index)
    has_emit = jnp.any(is_bcast)
    ef = jnp.argmax(is_bcast)
    dirs = _dir_from_to(pos[ef], pos, orient, W, H)
    recv = has_emit & (idx != ef)
    last_dir = jnp.where(recv, dirs, -1).astype(jnp.int32)
    last_tok = jnp.where(recv, tokens[ef], -1).astype(jnp.int32)

    # incantation start
    start = is_incant & _ready(cfg, s._replace(pos=pos, level=level), level)
    M = ((pos[:, None, 0] == pos[None, :, 0]) & (pos[:, None, 1] == pos[None, :, 1])
         & (level[:, None] == level[None, :]))
    frozen = (M & start[None, :] & s.alive[:, None]).any(axis=1)
    initiator = initiator | start
    incant_level = jnp.where(frozen, level, incant_level)
    pending = pending | frozen

    # cooldowns
    cost = COST_ENV[act]
    busy = jnp.where(free, now + cost, s.busy_until)
    busy = jnp.where(frozen, now + C.COST_INCANTATION, busy)

    # 3) advance the clock to the next free time
    rel = jnp.where(s.alive, busy - now, _BIG)
    dt = jnp.maximum(jnp.min(rel), 1)
    new_now = now + dt

    # food decay over dt
    life = jnp.where(cfg.no_food, life, life - dt)
    alive = s.alive & (life > 0)

    # respawn (approximate; skipped under no_refill)
    key, grid = _respawn(cfg, key, grid, now, new_now)

    s2 = State(
        grid=grid, pos=pos, orient=orient, level=level, inv=inv, life=life, alive=alive,
        busy_until=busy, pending=pending, incant_level=incant_level, initiator=initiator,
        team=s.team, now=new_now, key=key, last_dir=last_dir, last_tok=last_tok,
    )

    reward = leveled.astype(jnp.float32) * level.astype(jnp.float32) \
        - (s.alive & ~alive).astype(jnp.float32)
    done = jnp.logical_not(jnp.any(alive))
    info = {"free": free, "leveled": leveled, "dt": dt}
    return s2, observe(cfg, s2), reward, done, info


def _respawn(cfg: Cfg, key, grid, now, new_now):
    if cfg.no_refill:
        return key, grid
    do = (new_now // C.RESPAWN_INTERVAL_TICKS) > (now // C.RESPAWN_INTERVAL_TICKS)

    def refill(args):
        key, grid = args
        for res in range(C.N_RESOURCES):
            target = _target_qty(cfg, res)
            cur = jnp.sum(grid[:, :, res])
            deficit = jnp.clip(target - cur, 0, target)
            key, kx, ky = jax.random.split(key, 3)
            xs = jax.random.randint(kx, (target,), 0, cfg.width)
            ys = jax.random.randint(ky, (target,), 0, cfg.height)
            add = (jnp.arange(target) < deficit).astype(jnp.int32)
            grid = grid.at[xs, ys, res].add(add)
        return key, grid

    return jax.lax.cond(do, refill, lambda a: a, (key, grid))

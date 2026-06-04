"""Replay recorder: JAX-env episodes -> reference GUI wire protocol.

Design
------
The reference ``zappy_gui`` is driven by a line protocol the *server* emits as
the world changes (``msz``/``bct``/``ppo``/``plv``/``pin``/``pic``/``pie``/
``pbc``/``pdi``/``pnw``/``sgt``/``tna`` ...). The GUI-protocol PDF *is* the
contract (PLAN.md), so the cheapest, most faithful "renderer" for a trained
policy is to make our sim *speak that protocol* and replay it back through the
real GUI (``replay_to_gui.py``, Phase 5). This module is the producer.

Why a pure ``events_from_step`` core
------------------------------------
The wire stream is a *diff* of world state: the server only sends a line when
something a client can see changed (a player moved, a tile's contents changed,
an incantation started/ended). We therefore split the recorder into:

* ``events_from_step`` — a PURE function ``(prev_state, new_state, actions,
  info) -> [wire lines]``. It diffs two ``State`` snapshots and never touches
  JAX control flow or a policy, so every emit rule is unit-testable in
  isolation with a hand-built ``State`` and scripted actions.
* ``record_episode`` — the impure driver: it jits ``Z.step`` once, runs the env
  with an injected ``policy_fn`` (the seam — tests pass scripted actions, Phase
  5 wraps the frozen actor), and *persists* the lines both as NDJSON (streamed,
  greppable, replay-friendly) and into a SQLite index keyed ``(gen, episode,
  tick)`` so the timelapse can pull "generation g, episode e, ticks a..b".

Granularity
-----------
The event clock is ``int(new_state.now)`` — *commit-time* granularity, not
per-tick. PLAN.md explicitly stores action-commit granularity (~330 MB of
replays for a 2-week run); the env is event-driven and ``now`` already jumps by
the action cost (7, or 300 across an incantation freeze), so one wire line per
state change at the commit tick is exactly what the GUI needs and keeps the DB
small. The GUI interpolates motion between ticks itself.

No checkpoint / flax dependency
-------------------------------
``policy_fn`` is the only seam to the policy: ``policy_fn(obs, state, t) ->
(actions [A] int32, tokens [A] int32)``. The recorder imports nothing from
``algo`` — Phase 5 supplies a closure over the frozen actor, tests supply a
scripted callback. This keeps the viz layer decoupled from training internals.

Note on casts: ``State`` fields are ``jnp`` arrays; we ``np.asarray`` /
``int()`` everything before string-formatting so wire lines are plain ASCII
ints (no ``DeviceArray`` reprs leaking into the protocol).
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..env import constants as C
from ..env import zappy_env as Z

# A policy seam: (obs, state, t) -> (actions [A] int32, tokens [A] int32).
PolicyFn = Callable[[Z.Obs, Z.State, int], "tuple[jnp.ndarray, jnp.ndarray]"]


# --------------------------------------------------------------- formatting
def _tile_str(grid_np, x: int, y: int) -> str:
    """``bct``/``pin`` tail: the seven resource counts q0..q6 at a tile."""
    return " ".join(str(int(grid_np[x, y, q])) for q in range(C.N_RESOURCES))


def _bct(grid_np, x: int, y: int) -> str:
    return f"bct {x} {y} {_tile_str(grid_np, x, y)}"


# ----------------------------------------------------------------- header
def episode_header(cfg: Z.Cfg, state: Z.State, pids, team_names=("T1",), freq: int = 100) -> list[str]:
    """The initial GUI burst that fully describes the world at episode start.

    Order mirrors what the reference server sends a freshly-connected GUI:
    map size, time unit, team names, the full tile map, then every player. A
    replay must lead with this so the GUI can build its scene before any diff
    (``ppo``/``bct``/...) arrives.
    """
    lines: list[str] = []
    lines.append(f"msz {cfg.width} {cfg.height}")
    lines.append(f"sgt {int(freq)}")
    for name in team_names:
        lines.append(f"tna {name}")

    grid_np = np.asarray(state.grid)
    for x in range(cfg.width):
        for y in range(cfg.height):
            lines.append(_bct(grid_np, x, y))

    pos = np.asarray(state.pos)
    orient = np.asarray(state.orient)
    level = np.asarray(state.level)
    team = np.asarray(state.team)
    for a, pid in enumerate(pids):
        # pnw #n X Y O L <team-name>: a player joins the world.
        tname = team_names[int(team[a])] if int(team[a]) < len(team_names) else team_names[0]
        lines.append(f"pnw #{pid} {int(pos[a, 0])} {int(pos[a, 1])} "
                     f"{int(orient[a])} {int(level[a])} {tname}")
    return lines


# ------------------------------------------------------ per-step state diff
def events_from_step(cfg: Z.Cfg, prev_state: Z.State, new_state: Z.State,
                     actions, tokens, info, pids, team_names=("T1",)) -> list[str]:
    """Diff ``prev_state`` -> ``new_state`` into ordered GUI wire lines.

    PURE: no policy, no I/O. Every emit below corresponds to a state field that
    a GUI client can observe changing during one env step. Event tick is
    ``int(new_state.now)`` (commit-time) — the caller stamps it onto each line
    (``pic`` is the exception: ``record_episode`` stamps it at the *previous*
    tick, when the freeze actually began — see there).

    Cosmetic, accepted: an agent that moves and starves in the same commit
    step emits its post-move ``ppo`` then ``pdi`` — a strict GUI may flash the
    sprite on the new tile before removing it.
    """
    A = cfg.n_agents
    tick = int(np.asarray(new_state.now))

    p_pos = np.asarray(prev_state.pos)
    n_pos = np.asarray(new_state.pos)
    p_or = np.asarray(prev_state.orient)
    n_or = np.asarray(new_state.orient)
    p_lvl = np.asarray(prev_state.level)
    n_lvl = np.asarray(new_state.level)
    p_inv = np.asarray(prev_state.inv)
    n_inv = np.asarray(new_state.inv)
    p_pend = np.asarray(prev_state.pending)
    n_pend = np.asarray(new_state.pending)
    p_alive = np.asarray(prev_state.alive)
    n_alive = np.asarray(new_state.alive)
    n_ilvl = np.asarray(new_state.incant_level)
    p_ilvl = np.asarray(prev_state.incant_level)

    acts = np.asarray(actions)
    free = np.asarray(info["free"])
    leveled = np.asarray(info["leveled"])

    lines: list[str] = []

    # 1) Per-player position / orientation changes.
    for a in range(A):
        pid = pids[a]
        moved = (n_pos[a, 0] != p_pos[a, 0]) or (n_pos[a, 1] != p_pos[a, 1])
        turned = n_or[a] != p_or[a]
        if moved or turned:
            lines.append(f"ppo #{pid} {int(n_pos[a, 0])} {int(n_pos[a, 1])} {int(n_or[a])}")

    # 2) Level changes (incantation success, or any other source).
    for a in range(A):
        if n_lvl[a] != p_lvl[a]:
            lines.append(f"plv #{pids[a]} {int(n_lvl[a])}")

    # 3) Inventory changes -> pin #n X Y q0..q6 (GUI wants the carrying tile).
    for a in range(A):
        if np.any(n_inv[a] != p_inv[a]):
            inv = n_inv[a]
            tail = " ".join(str(int(inv[q])) for q in range(C.N_RESOURCES))
            lines.append(f"pin #{pids[a]} {int(n_pos[a, 0])} {int(n_pos[a, 1])} {tail}")

    # 4) Broadcast: ONE pbc per step, for the emitter the env actually
    #    delivered. zappy_env v1 delivers only the LOWEST-index free
    #    broadcaster (ef = argmax(is_bcast)); any other same-step broadcaster
    #    is silently dropped by the sim, so emitting lines for them would
    #    fabricate ripples that never happened (review-pinned). The token is
    #    the emitter's own submitted symbol — receivers' last_tok can't carry
    #    it when there's no receiver.
    toks = np.asarray(tokens)
    is_bcast = [int(acts[a]) == Z.ENV_BROADCAST and bool(free[a]) for a in range(A)]
    if any(is_bcast):
        ef = is_bcast.index(True)
        lines.append(f"pbc #{pids[ef]} {int(toks[ef])}")

    # 5) Incantation START: pending False -> True. One ``pic`` per *tile*, with
    #    that tile's level and the participant ids (all agents frozen there).
    started = (~p_pend) & n_pend
    emitted_tiles: set[tuple[int, int]] = set()
    for a in range(A):
        if not started[a]:
            continue
        key = (int(n_pos[a, 0]), int(n_pos[a, 1]))
        if key in emitted_tiles:
            continue
        emitted_tiles.add(key)
        lvl = int(n_ilvl[a])
        # participants: every agent that started on this exact tile.
        members = [pids[b] for b in range(A)
                   if started[b] and int(n_pos[b, 0]) == key[0] and int(n_pos[b, 1]) == key[1]]
        members_str = " ".join(f"#{m}" for m in members)
        lines.append(f"pic {key[0]} {key[1]} {lvl} {members_str}")

    # 6) Incantation END: pending True -> False. R=1 if any participant on that
    #    tile leveled this step, else 0. The freeze tile is read from prev pos
    #    (it cannot have moved while frozen).
    ended = p_pend & (~n_pend)
    ended_tiles: set[tuple[int, int]] = set()
    for a in range(A):
        if not ended[a]:
            continue
        key = (int(p_pos[a, 0]), int(p_pos[a, 1]))
        if key in ended_tiles:
            continue
        ended_tiles.add(key)
        result = 0
        for b in range(A):
            if ended[b] and int(p_pos[b, 0]) == key[0] and int(p_pos[b, 1]) == key[1] and bool(leveled[b]):
                result = 1
                break
        lines.append(f"pie {key[0]} {key[1]} {result}")

    # 7) Deaths: alive True -> False.
    for a in range(A):
        if bool(p_alive[a]) and not bool(n_alive[a]):
            lines.append(f"pdi #{pids[a]}")

    # 8) Tile-content changes -> bct for every changed tile. The grid is the
    #    authoritative ground truth (take/set/incant-consume/respawn all land
    #    here); diffing it catches every change without enumerating causes.
    p_grid = np.asarray(prev_state.grid)
    n_grid = np.asarray(new_state.grid)
    if not np.array_equal(p_grid, n_grid):
        diff = np.any(n_grid != p_grid, axis=2)            # [W,H] changed mask
        xs, ys = np.nonzero(diff)
        for x, y in zip(xs.tolist(), ys.tolist()):
            lines.append(_bct(n_grid, int(x), int(y)))

    return lines


# ------------------------------------------------------------- DB plumbing
def _open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    # seq (rowid alias) makes within-tick ordering EXPLICIT: many lines share
    # a tick (the whole header is tick 0) and correct replay needs emit order;
    # relying on SQLite's implicit rowid tiebreak under ORDER BY tick is an
    # implementation detail. Consumers: ORDER BY tick, seq.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events "
        "(seq INTEGER PRIMARY KEY, gen INT, episode INT, tick INT, line TEXT)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events ON events (gen, episode, tick)"
    )
    return conn


# ------------------------------------------------------------- the driver
def record_episode(cfg: Z.Cfg, out_dir: str, *, generation: int, episode: int,
                   policy_fn: PolicyFn, seed: int, max_steps: int,
                   team_names=("T1",)) -> str:
    """Drive one (unbatched) episode and persist its GUI wire stream.

    Streams ``{"gen","ep","tick","line"}`` NDJSON to
    ``<out_dir>/replay_g<g>_e<e>.ndjson`` *and* inserts every line into the
    ``events`` table of ``<out_dir>/replays.sqlite``. Stops at ``done`` or
    ``max_steps`` (whichever first). Returns the NDJSON path.

    ``policy_fn`` is the seam: it receives the current ``obs``, ``state`` and
    step index ``t`` and returns ``(actions [A] int32, tokens [A] int32)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    pids = list(range(cfg.n_agents))

    key = jax.random.PRNGKey(seed)
    key, kreset = jax.random.split(key)
    state, obs = Z.reset(cfg, kreset)

    # jit the hot loop body once (cfg is static; the recorder is unbatched).
    jstep = jax.jit(Z.step, static_argnums=0)

    ndjson_path = os.path.join(out_dir, f"replay_g{generation}_e{episode}.ndjson")
    db_path = os.path.join(out_dir, "replays.sqlite")
    conn = _open_db(db_path)

    def _persist(tick: int, line: str, fh):
        rec = {"gen": generation, "ep": episode, "tick": tick, "line": line}
        fh.write(json.dumps(rec) + "\n")
        conn.execute(
            "INSERT INTO events (gen, episode, tick, line) VALUES (?, ?, ?, ?)",
            (generation, episode, tick, line),
        )

    try:
        with open(ndjson_path, "w") as fh:
            # Header at the episode's opening tick (now == 0 after reset).
            t0 = int(np.asarray(state.now))
            for line in episode_header(cfg, state, pids, team_names, freq=C.DEFAULT_FREQ):
                _persist(t0, line, fh)

            for t in range(max_steps):
                actions, tokens = policy_fn(obs, state, t)
                actions = jnp.asarray(actions, jnp.int32)
                tokens = jnp.asarray(tokens, jnp.int32)
                key, kstep = jax.random.split(key)
                new_state, new_obs, _r, done, info = jstep(cfg, kstep, state, actions, tokens)

                prev_tick = int(np.asarray(state.now))
                tick = int(np.asarray(new_state.now))
                step_lines = events_from_step(cfg, state, new_state, actions,
                                              tokens, info, pids, team_names)
                # pic at the FREEZE-START tick, persisted first: the event
                # clock jumps the whole 300-tick freeze inside the submission
                # step, so commit-time stamping would render the ritual glow
                # ~7 ticks long. The ritual began when the INCANT was consumed
                # (= prev now); emitting it first keeps ticks monotone.
                for line in step_lines:
                    if line.startswith("pic "):
                        _persist(prev_tick, line, fh)
                for line in step_lines:
                    if not line.startswith("pic "):
                        _persist(tick, line, fh)

                state, obs = new_state, new_obs
                if bool(done):
                    break
        conn.commit()
    finally:
        conn.close()

    return ndjson_path

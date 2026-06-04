"""Rule tests for the NumPy reference env (offline, no server needed)."""

import numpy as np

from zappy_rl.env import constants as C
from zappy_rl.env.reference_env import ZappyReferenceEnv


def fresh(width=11, height=11, **kw):
    kw.setdefault("no_food", True)
    kw.setdefault("no_refill", True)
    return ZappyReferenceEnv(width, height, teams=("T1", "T2"), clients_nb=6, seed=1, **kw)


# --- resources -------------------------------------------------------------
def test_spawn_density_targets():
    env = ZappyReferenceEnv(10, 10, seed=3)  # 10x10 => food .5 -> 50, thystame .05 -> 5
    assert int(env.grid[:, :, C.FOOD].sum()) == 50
    assert int(env.grid[:, :, C.THYSTAME].sum()) == 5
    for res in range(C.N_RESOURCES):  # at least one of each
        assert env.grid[:, :, res].sum() >= 1


# --- movement / rotation ---------------------------------------------------
def test_forward_per_orientation_and_wrap():
    env = fresh()
    for orient, (dx, dy) in C.FORWARD_DELTA.items():
        pid = env.spawn("T1", 0, 0, orientation=orient)
        env.do(pid, C.A_FORWARD)
        p = env.players[pid]
        assert (p.x, p.y) == (dx % env.width, dy % env.height)  # wraps from 0


def test_rotation_cycles():
    env = fresh()
    pid = env.spawn("T1", 5, 5, orientation=C.NORTH)
    seq = []
    for _ in range(4):
        env.do(pid, C.A_RIGHT)
        seq.append(env.players[pid].orientation)
    assert seq == [C.EAST, C.SOUTH, C.WEST, C.NORTH]
    seq = []
    for _ in range(4):
        env.do(pid, C.A_LEFT)
        seq.append(env.players[pid].orientation)
    assert seq == [C.WEST, C.SOUTH, C.EAST, C.NORTH]


# --- take / set ------------------------------------------------------------
def test_take_and_set():
    env = fresh()
    pid = env.spawn("T1", 5, 5)
    env.grid[5, 5, C.LINEMATE] = 2
    assert env.do(pid, C.A_TAKE, C.LINEMATE) == "ok"
    assert env.players[pid].inv[C.LINEMATE] == 1 and env.grid[5, 5, C.LINEMATE] == 1
    assert env.do(pid, C.A_SET, C.LINEMATE) == "ok"
    assert env.players[pid].inv[C.LINEMATE] == 0 and env.grid[5, 5, C.LINEMATE] == 2
    assert env.do(pid, C.A_TAKE, C.SIBUR) == "ko"  # none present


def test_take_food_extends_life():
    env = fresh()
    pid = env.spawn("T1", 5, 5)
    env.grid[5, 5, C.FOOD] = 1
    before = env.players[pid].life_ticks
    assert env.do(pid, C.A_TAKE, C.FOOD) == "ok"
    assert env.players[pid].life_ticks == before + C.FOOD_LIFE_TICKS


# --- food decay ------------------------------------------------------------
def test_food_decay_and_death():
    env = ZappyReferenceEnv(11, 11, no_food=False, no_refill=True, seed=2)
    pid = env.spawn("T1", 5, 5)
    assert env.players[pid].food_units() == C.START_FOOD
    env.tick(C.FOOD_LIFE_TICKS)            # one unit consumed
    assert env.players[pid].food_units() == C.START_FOOD - 1
    env.tick(C.START_FOOD * C.FOOD_LIFE_TICKS)  # well past total life
    assert not env.players[pid].alive


# --- look ------------------------------------------------------------------
def test_look_contents_and_self():
    env = fresh()
    pid = env.spawn("T1", 5, 5, orientation=C.NORTH, level=1)
    env.grid[:] = 0
    tiles = env.look(pid)
    assert len(tiles) == 4                 # level 1 => (1+1)**2
    assert "player" in tiles[0]            # self on tile 0
    # plant a sibur on the front-centre tile (look index 2) and re-look
    fx, fy = C.FORWARD_DELTA[C.NORTH]
    env.grid[(5 + fx) % 11, (5 + fy) % 11, C.SIBUR] = 1
    assert "sibur" in env.look(pid)[2]


# --- incantation -----------------------------------------------------------
def test_incantation_requires_prereqs():
    env = fresh()
    pid = env.spawn("T1", 5, 5, level=1)
    assert env.do(pid, C.A_INCANTATION) == "ko"      # no linemate
    assert env.players[pid].level == 1


def test_incantation_solo_level1_to_2():
    env = fresh()
    pid = env.spawn("T1", 5, 5, level=1)
    env.grid[5, 5, C.LINEMATE] = 1                    # 1->2 needs 1 linemate, 1 player
    assert env.do(pid, C.A_INCANTATION) == 2
    assert env.players[pid].level == 2
    assert env.grid[5, 5, C.LINEMATE] == 0            # consumed


def test_incantation_needs_enough_same_level_players():
    env = fresh()
    a = env.spawn("T1", 5, 5, level=2)
    env.grid[5, 5, C.LINEMATE] = 1
    env.grid[5, 5, C.DERAUMERE] = 1
    env.grid[5, 5, C.SIBUR] = 1                        # 2->3 needs 2 players
    assert env.do(a, C.A_INCANTATION) == "ko"
    b = env.spawn("T1", 5, 5, level=2)                # add the 2nd same-level player
    assert env.do(a, C.A_INCANTATION) == 3
    assert env.players[a].level == 3 and env.players[b].level == 3


# --- fork / connect_nbr ----------------------------------------------------
def test_fork_adds_slot():
    env = fresh()
    pid = env.spawn("T1", 5, 5)
    before = env.do(pid, C.A_CONNECT_NBR)             # 6 initial eggs
    assert before == 6
    env.do(pid, C.A_FORK)
    assert env.do(pid, C.A_CONNECT_NBR) == before + 1


# --- eject -----------------------------------------------------------------
def test_eject_pushes_and_destroys_eggs():
    env = fresh()
    a = env.spawn("T1", 5, 5, orientation=C.EAST)     # pushes toward +x
    b = env.spawn("T2", 5, 5)
    env._add_egg("T2", parent=-1, x=5, y=5)
    eggs_before = len(env.eggs)
    assert env.do(a, C.A_EJECT) == "ok"
    assert (env.players[b].x, env.players[b].y) == (6, 5)
    assert len(env.eggs) == eggs_before - 1
    assert env.do(a, C.A_EJECT) == "ko"               # nobody left to push


# --- broadcast -------------------------------------------------------------
def test_broadcast_direction_via_env():
    env = fresh()
    e = env.spawn("T1", 5, 4)                          # north of receiver
    r = env.spawn("T1", 5, 5, orientation=C.NORTH)
    assert env.broadcast(e) == {r: 1}                  # directly in front


# --- win condition ---------------------------------------------------------
def test_six_players_to_level8_wins():
    env = fresh()
    players = [env.spawn("T1", 5, 5, level=7) for _ in range(6)]
    req = C.ELEVATION[8]["stones"]                      # (2,2,2,2,2,1)
    for j, need in enumerate(req):
        env.grid[5, 5, 1 + j] = need
    assert env.do(players[0], C.A_INCANTATION) == 8
    assert all(env.players[p].level == 8 for p in players)
    assert env.winner == "T1"

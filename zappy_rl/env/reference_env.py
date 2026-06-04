"""NumPy reference Zappy environment — the readable rule *oracle*.

This is the source of truth for game rules. It is intentionally plain Python /
NumPy (no JAX tracing constraints) so the rules are easy to read and verify. It
serves three roles:

  1. validated against the reference server for the rules a live server can be
     coaxed into exercising (``tools/validate_against_server.py``),
  2. the cross-check oracle for the vectorized JAX env (``env/zappy_env.py``),
  3. a slow but correct fallback environment.

Time model: a global integer **tick** clock. Most actions cost 7 ticks, Fork 42,
Incantation 300, Inventory 1 (``constants.ACTION_COST``); real seconds = ticks/f.
Effects land at action *completion*. ``do()`` runs one action for one free
player and advances the clock by its cost (the faithful single-player timeline);
multi-agent interleaving / frame-skip stepping is layered on top in the JAX env.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .broadcast import broadcast_direction
from .vision import format_look, vision_tiles


@dataclass
class Player:
    pid: int
    team: str
    x: int
    y: int
    orientation: int
    level: int = 1
    inv: np.ndarray = field(default_factory=lambda: np.zeros(C.N_RESOURCES, dtype=np.int64))
    life_ticks: int = C.START_FOOD * C.FOOD_LIFE_TICKS
    alive: bool = True
    frozen_until: int = -1  # tick at which an incantation completes (busy/frozen)

    def food_units(self) -> int:
        return max(0, -(-self.life_ticks // C.FOOD_LIFE_TICKS))  # ceil division

    def sync_food(self) -> None:
        self.inv[C.FOOD] = self.food_units()


@dataclass
class Egg:
    eid: int
    team: str
    x: int
    y: int
    parent: int = -1
    hatched: bool = False


class ZappyReferenceEnv:
    def __init__(
        self,
        width: int,
        height: int,
        teams=("T1",),
        clients_nb: int = 6,
        freq: int = C.DEFAULT_FREQ,
        seed: int = 0,
        no_food: bool = False,
        no_refill: bool = False,
    ):
        self.width = width
        self.height = height
        self.teams = tuple(teams)
        self.clients_nb = clients_nb
        self.freq = freq
        self.no_food = no_food
        self.no_refill = no_refill
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.now = 0
        self.grid = np.zeros((self.width, self.height, C.N_RESOURCES), dtype=np.int64)
        self.players: dict[int, Player] = {}
        self.eggs: dict[int, Egg] = {}
        self._next_pid = 0
        self._next_eid = 0
        self.winner: str | None = None
        self.spawn_resources()
        for team in self.teams:
            for _ in range(self.clients_nb):
                self._add_egg(team, parent=-1)

    def spawn_resources(self) -> None:
        """Place each resource to its density target (at least one of each)."""
        for res in range(C.N_RESOURCES):
            target = self._target_qty(res)
            self._scatter(res, target)

    def _target_qty(self, res: int) -> int:
        return max(1, int(self.width * self.height * C.DENSITY[res]))

    def _scatter(self, res: int, n: int) -> None:
        for _ in range(n):
            x = int(self.rng.integers(self.width))
            y = int(self.rng.integers(self.height))
            self.grid[x, y, res] += 1

    # ----------------------------------------------------------- eggs/players
    def _add_egg(self, team: str, parent: int, x: int | None = None, y: int | None = None) -> int:
        eid = self._next_eid
        self._next_eid += 1
        if x is None:
            x = int(self.rng.integers(self.width))
            y = int(self.rng.integers(self.height))
        self.eggs[eid] = Egg(eid, team, x, y, parent=parent)
        return eid

    def unused_slots(self, team: str) -> int:
        return sum(1 for e in self.eggs.values() if e.team == team and not e.hatched)

    def connect(self, team: str) -> int | None:
        """A client connects: hatch a random available egg of ``team``."""
        avail = [e for e in self.eggs.values() if e.team == team and not e.hatched]
        if not avail:
            return None
        egg = self.rng.choice(np.array(avail, dtype=object))  # type: ignore[arg-type]
        egg.hatched = True
        return self.spawn(team, egg.x, egg.y, orientation=int(self.rng.integers(1, 5)))

    def spawn(self, team: str, x: int, y: int, orientation: int = C.NORTH, level: int = 1) -> int:
        """Directly create a player (test/setup helper, bypasses egg bookkeeping)."""
        pid = self._next_pid
        self._next_pid += 1
        p = Player(pid, team, x % self.width, y % self.height, orientation, level=level)
        p.sync_food()
        self.players[pid] = p
        return pid

    def players_on(self, x: int, y: int, alive_only: bool = True):
        return [
            p for p in self.players.values()
            if p.x == x and p.y == y and (p.alive or not alive_only)
        ]

    # ------------------------------------------------------------- time / life
    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.now += 1
            if not self.no_food:
                for p in self.players.values():
                    if not p.alive:
                        continue
                    p.life_ticks -= 1
                    if p.life_ticks <= 0:
                        p.alive = False
                    else:
                        p.sync_food()
            if not self.no_refill and self.now % C.RESPAWN_INTERVAL_TICKS == 0:
                self._refill()

    def _refill(self) -> None:
        for res in range(C.N_RESOURCES):
            deficit = self._target_qty(res) - int(self.grid[:, :, res].sum())
            if deficit > 0:
                self._scatter(res, deficit)

    # --------------------------------------------------------------- observ.
    def tile_objects(self, x: int, y: int, looker: int | None = None) -> list[str]:
        objs: list[str] = []
        for p in self.players_on(x, y):
            objs.append("player")
        for res in range(C.N_RESOURCES):
            objs.extend([C.RESOURCE_NAMES[res]] * int(self.grid[x, y, res]))
        return objs

    def look(self, pid: int) -> list[list[str]]:
        p = self.players[pid]
        tiles = vision_tiles(p.x, p.y, p.level, p.orientation, self.width, self.height)
        return [self.tile_objects(int(tx), int(ty)) for tx, ty in tiles]

    def look_str(self, pid: int) -> str:
        return format_look(self.look(pid))

    def inventory(self, pid: int) -> dict[str, int]:
        p = self.players[pid]
        p.sync_food()
        return {C.RESOURCE_NAMES[i]: int(p.inv[i]) for i in range(C.N_RESOURCES)}

    def broadcast(self, pid: int) -> dict[int, int]:
        """Return {receiver_pid: direction_K} for a broadcast emitted by ``pid``."""
        e = self.players[pid]
        out: dict[int, int] = {}
        for r in self.players.values():
            if r.pid == pid or not r.alive:
                continue
            out[r.pid] = broadcast_direction(
                e.x, e.y, r.x, r.y, r.orientation, self.width, self.height
            )
        return out

    # ----------------------------------------------------------- core actions
    def _move(self, p: Player, dx: int, dy: int) -> None:
        p.x = (p.x + dx) % self.width
        p.y = (p.y + dy) % self.height

    def do(self, pid: int, action: int, arg: int | None = None):
        """Execute one action for a free, living player; advance the clock.

        Returns ``"ok"``/``"ko"``, a look/inventory result, or a count, mirroring
        the reference server's responses.
        """
        p = self.players[pid]
        if not p.alive:
            return "dead"
        if self.now < p.frozen_until:
            return "ko"  # busy/frozen

        # Incantation has start- and end-checks separated by 300 ticks.
        if action == C.A_INCANTATION:
            return self._incantation(p)

        cost = C.ACTION_COST[action]
        result = "ok"

        if action == C.A_FORWARD:
            dx, dy = C.FORWARD_DELTA[p.orientation]
            self._move(p, dx, dy)
        elif action == C.A_RIGHT:
            p.orientation = p.orientation % 4 + 1
        elif action == C.A_LEFT:
            p.orientation = p.orientation - 1 if p.orientation > 1 else C.WEST
        elif action == C.A_LOOK:
            self.tick(cost)
            return self.look(pid)
        elif action == C.A_INVENTORY:
            self.tick(cost)
            return self.inventory(pid)
        elif action == C.A_CONNECT_NBR:
            return self.unused_slots(p.team)
        elif action == C.A_BROADCAST:
            res = self.broadcast(pid)
            self.tick(cost)
            return res
        elif action == C.A_FORK:
            self.tick(cost)
            self._add_egg(p.team, parent=pid, x=p.x, y=p.y)
            return "ok"
        elif action == C.A_EJECT:
            return self._eject(p)
        elif action == C.A_TAKE:
            assert arg is not None
            if self.grid[p.x, p.y, arg] > 0:
                self.grid[p.x, p.y, arg] -= 1
                p.inv[arg] += 1
                if arg == C.FOOD:
                    p.life_ticks += C.FOOD_LIFE_TICKS
            else:
                result = "ko"
        elif action == C.A_SET:
            assert arg is not None
            if p.inv[arg] > 0:
                p.inv[arg] -= 1
                self.grid[p.x, p.y, arg] += 1
                if arg == C.FOOD:
                    p.life_ticks = max(0, p.life_ticks - C.FOOD_LIFE_TICKS)
            else:
                result = "ko"
        else:
            result = "ko"

        self.tick(cost)
        return result

    def _eject(self, p: Player):
        others = [o for o in self.players_on(p.x, p.y) if o.pid != p.pid]
        eggs_here = [e for e in self.eggs.values() if e.x == p.x and e.y == p.y and not e.hatched]
        if not others and not eggs_here:
            self.tick(C.ACTION_COST[C.A_EJECT])
            return "ko"
        dx, dy = C.FORWARD_DELTA[p.orientation]
        for o in others:
            self._move(o, dx, dy)
        for e in eggs_here:  # eject destroys eggs on the tile
            del self.eggs[e.eid]
        self.tick(C.ACTION_COST[C.A_EJECT])
        return "ok"

    # --------------------------------------------------------- incantation
    def incantation_ready(self, x: int, y: int, level: int) -> bool:
        """Whether a level->level+1 ritual can start/succeed on this tile."""
        if level >= C.MAX_LEVEL:
            return False
        req = C.ELEVATION[level + 1]
        here = self.players_on(x, y)
        same = [q for q in here if q.level == level]
        if len(same) < req["players"]:
            return False
        for j, need in enumerate(req["stones"]):  # stones index 0..5 -> resource 1..6
            if self.grid[x, y, 1 + j] < need:
                return False
        return True

    def _incantation(self, p: Player):
        if not self.incantation_ready(p.x, p.y, p.level):
            return "ko"  # instant ko on the start check
        level = p.level
        participants = [q for q in self.players_on(p.x, p.y) if q.level == level]
        end = self.now + C.ACTION_COST[C.A_INCANTATION]
        for q in participants:
            q.frozen_until = end
        self.tick(C.ACTION_COST[C.A_INCANTATION])
        # End check (participants may have died; tile must still satisfy reqs).
        if not self.incantation_ready(p.x, p.y, level):
            return "ko"
        req = C.ELEVATION[level + 1]
        for j, need in enumerate(req["stones"]):
            self.grid[p.x, p.y, 1 + j] -= need
        for q in [q for q in self.players_on(p.x, p.y) if q.level == level and q.alive]:
            q.level += 1
            q.frozen_until = -1
        self._check_win()
        return level + 1

    def _check_win(self) -> None:
        counts: dict[str, int] = {}
        for q in self.players.values():
            if q.alive and q.level >= C.WIN_LEVEL:
                counts[q.team] = counts.get(q.team, 0) + 1
        for team, n in counts.items():
            if n >= C.WIN_PLAYERS_AT_MAX:
                self.winner = team

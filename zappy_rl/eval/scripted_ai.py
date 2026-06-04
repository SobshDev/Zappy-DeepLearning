"""A minimal greedy scripted Zappy AI client.

Purposes:
  * generate activity on the reference server for golden-trace capture, and
  * serve as a baseline opponent for the eval harness (Phase 5).

The policy is deliberately simple (survive + wander + grab stones): Look, eat
food on the current tile, otherwise step forward with occasional turns, and
opportunistically Take any stone seen on the current tile. It is *not* meant to
be strong — it is the floor the RL agent must clear.
"""

from __future__ import annotations

import random
import time

from ..deploy.protocol import LineSocket, parse_look

# Async server notifications that can arrive between a command and its response.
_ASYNC_PREFIXES = ("message ", "eject:")


class ScriptedAI:
    def __init__(self, host: str, port: int, team: str, seed: int = 0):
        self.team = team
        self.rng = random.Random(seed)
        self.sock = LineSocket.connect(host, port)
        self.alive = True
        self.width = self.height = 0
        self.slots = 0
        self._handshake()

    def _handshake(self) -> None:
        welcome = self.sock.recv_line(timeout=5)
        if welcome != "WELCOME":
            raise RuntimeError(f"expected WELCOME, got {welcome!r}")
        self.sock.send(self.team)
        self.slots = int(self.sock.recv_line(timeout=5))
        x, y = self.sock.recv_line(timeout=5).split()
        self.width, self.height = int(x), int(y)

    def _response(self, timeout: float = 5.0) -> str | None:
        """Next command response, routing async notifications aside."""
        while True:
            line = self.sock.recv_line(timeout=timeout)
            if line is None:
                return None
            if line == "dead":
                self.alive = False
                return None
            if any(line.startswith(p) for p in _ASYNC_PREFIXES):
                continue  # baseline ignores broadcasts/ejections
            return line

    def step(self) -> None:
        """One perceive-act cycle."""
        self.sock.send("Look")
        resp = self._response()
        if not self.alive or resp is None:
            return
        try:
            tiles = parse_look(resp)
        except ValueError:
            return
        here = tiles[0] if tiles else []

        if "food" in here:
            self.sock.send("Take food")
            self._response()
            return
        for stone in ("linemate", "deraumere", "sibur", "mendiane", "phiras", "thystame"):
            if stone in here:
                self.sock.send(f"Take {stone}")
                self._response()
                return

        roll = self.rng.random()
        if roll < 0.7:
            self.sock.send("Forward")
        elif roll < 0.85:
            self.sock.send("Right")
        else:
            self.sock.send("Left")
        self._response()

    def run(self, deadline: float) -> None:
        while self.alive and time.monotonic() < deadline:
            self.step()

    def close(self) -> None:
        self.sock.close()

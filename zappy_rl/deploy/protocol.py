"""Zappy wire-protocol parsing + a buffered line socket.

Formats below are confirmed against the live reference server (v3.0.1):

    AI handshake:  server ``WELCOME\\n`` -> client ``<TEAM>\\n``
                   -> server ``<slots>\\n`` then ``<X> <Y>\\n``
    Inventory:     ``[ food 9, linemate 0, deraumere 0, sibur 0, mendiane 0, phiras 0, thystame 0 ]``
    Look (lvl 1):  ``[ player food, mendiane,, food linemate ]`` (4 tiles)

The subject warns the look tile separator is "a comma followed or not by a
space", so parsing splits on ``,`` and strips — never assumes fixed spacing.
"""

from __future__ import annotations

import socket

from ..env import constants as C


def parse_look(line: str) -> list[list[str]]:
    """``[ a b, c,, d ]`` -> ``[['a','b'], ['c'], [], ['d']]`` (look-order)."""
    s = line.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"bad look response: {line!r}")
    inner = s[1:-1]
    if inner.strip() == "":
        return [[]]
    return [tile.split() for tile in inner.split(",")]


def parse_inventory(line: str) -> dict[str, int]:
    """``[ food 9, linemate 0, ... ]`` -> ``{'food': 9, 'linemate': 0, ...}``."""
    s = line.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"bad inventory response: {line!r}")
    out: dict[str, int] = {}
    for item in s[1:-1].split(","):
        parts = item.split()
        if not parts:
            continue
        name, qty = parts[0], int(parts[1])
        out[name] = qty
    return out


def inventory_to_vec(inv: dict[str, int]) -> list[int]:
    """Inventory dict -> ordered vector q0..q6 (food, stones...)."""
    return [inv.get(name, 0) for name in C.RESOURCE_NAMES]


def parse_broadcast(line: str) -> tuple[int, str]:
    """``message 3, hello world`` -> ``(3, 'hello world')``."""
    body = line.strip()
    if body.startswith("message"):
        body = body[len("message"):].strip()
    k_str, _, text = body.partition(",")
    return int(k_str.strip()), text.strip()


class LineSocket:
    """Newline-delimited message framing over a TCP socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = b""

    @classmethod
    def connect(cls, host: str, port: int, timeout: float = 5.0) -> "LineSocket":
        return cls(socket.create_connection((host, port), timeout=timeout))

    def recv_line(self, timeout: float | None = None) -> str | None:
        """Return the next ``\\n``-terminated line (without the newline).

        Returns ``None`` on timeout/EOF/connection-reset — to a line-protocol
        client an RST is the same fact as EOF ("no more data, link gone") and
        must not crash a deploy agent mid-game. Trailing ``\\r`` is stripped.
        """
        if timeout is not None:
            self.sock.settimeout(timeout)
        while b"\n" not in self._buf:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                return None
            except OSError:  # ECONNRESET & friends: abrupt server death
                return None
            if not chunk:
                return None
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line.decode(errors="replace").rstrip("\r")

    def send(self, msg: str) -> None:
        if not msg.endswith("\n"):
            msg += "\n"
        self.sock.sendall(msg.encode())

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

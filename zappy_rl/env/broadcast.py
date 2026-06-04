"""Broadcast sound-direction geometry (NumPy reference oracle).

When a player broadcasts, every other player receives ``message K, text`` where
``K`` is the tile the sound arrives from, in the *receiver's* local frame:
``K = 0`` if the emitter is on the receiver's own tile, otherwise ``1..8`` with
``1`` = directly in front and increasing counter-clockwise (trigonometric), per
``G-YEP-400_zappy.pdf`` p.5-6. The world is toroidal, so the *shortest* path is
chosen.

The orientation/handedness is CONFIRMED against reference v3.0.1
(``tools/validate_against_server.py``): with a receiver facing north, the
emitter placed at all 8 neighbouring tiles produced exactly K = 1,8,7,6,5,4,3,2.
"""

from __future__ import annotations

import math

from . import constants as C

# K index for the 8 directions, counter-clockwise from "front" = 1.
# front(1) front-left(2) left(3) back-left(4) back(5) back-right(6) right(7) front-right(8)
_FRONT = 1
_FRONT_LEFT = 2
_LEFT = 3
_BACK_LEFT = 4
_BACK = 5
_BACK_RIGHT = 6
_RIGHT = 7
_FRONT_RIGHT = 8

# Sectors centered on each direction, 45° wide, indexed by floor((deg+22.5)/45).
# Angle is measured counter-clockwise from front, with "left" as +90°.
_SECTOR = {
    0: _FRONT,
    1: _FRONT_LEFT,
    2: _LEFT,
    3: _BACK_LEFT,
    4: _BACK,
    5: _BACK_RIGHT,
    6: _RIGHT,
    7: _FRONT_RIGHT,
}


def _toroidal_delta(d: int, n: int) -> int:
    """Shortest signed displacement of ``d`` on a ring of size ``n``."""
    d %= n
    if d > n // 2:
        d -= n
    return d


def broadcast_direction(
    ex: int,
    ey: int,
    rx: int,
    ry: int,
    r_orientation: int,
    width: int,
    height: int,
) -> int:
    """Direction tile ``K`` (0..8) that the receiver hears the emitter from.

    ``(ex, ey)`` emitter tile, ``(rx, ry)`` receiver tile, ``r_orientation`` the
    receiver's facing. Returns ``0`` when emitter and receiver share a tile.
    """
    if ex % width == rx % width and ey % height == ry % height:
        return 0

    # Shortest toroidal vector from receiver -> emitter, in world axes.
    dx = _toroidal_delta(ex - rx, width)
    dy = _toroidal_delta(ey - ry, height)

    # Project onto the receiver's local frame.
    fx, fy = C.FORWARD_DELTA[r_orientation]
    rxv, ryv = C.RIGHT_DELTA[r_orientation]
    front = dx * fx + dy * fy        # +front
    right = dx * rxv + dy * ryv       # +right
    left = -right                     # +left (counter-clockwise positive)

    # Angle CCW from front; left is +90°.
    deg = math.degrees(math.atan2(left, front)) % 360.0
    sector = int(math.floor((deg + 22.5) / 45.0)) % 8
    return _SECTOR[sector]

"""Broadcast direction-K geometry tests.

Cardinal/diagonal cases are convention-independent enough to lock in the basic
mapping; the exact handedness is finally pinned by golden traces in Phase 1.
"""

from zappy_rl.env import constants as C
from zappy_rl.env.broadcast import broadcast_direction

W = H = 11
R = (5, 5)  # receiver at center, far from wrap effects


def k(ex, ey, orient=C.NORTH):
    return broadcast_direction(ex, ey, R[0], R[1], orient, W, H)


def test_same_tile_is_zero():
    assert k(5, 5) == 0


def test_cardinals_facing_north():
    # Facing north: forward = -y, right = +x.
    assert k(5, 4) == 1   # directly in front  -> K=1
    assert k(4, 5) == 3   # to the left        -> K=3
    assert k(5, 6) == 5   # directly behind    -> K=5
    assert k(6, 5) == 7   # to the right       -> K=7


def test_diagonals_facing_north():
    assert k(4, 4) == 2   # front-left
    assert k(6, 4) == 8   # front-right
    assert k(4, 6) == 4   # back-left
    assert k(6, 6) == 6   # back-right


def test_front_rotates_with_orientation():
    # The emitter that is "in front" depends on the receiver's facing; K stays 1.
    assert broadcast_direction(6, 5, *R, C.EAST, W, H) == 1   # east front = +x
    assert broadcast_direction(5, 6, *R, C.SOUTH, W, H) == 1  # south front = +y
    assert broadcast_direction(4, 5, *R, C.WEST, W, H) == 1   # west front = -x


def test_toroidal_shortest_path():
    # Receiver at (0,0) facing north; emitter at (0, H-1) is one tile "in front"
    # via the wrap (north = -y), not H-1 tiles behind.
    assert broadcast_direction(0, H - 1, 0, 0, C.NORTH, W, H) == 1

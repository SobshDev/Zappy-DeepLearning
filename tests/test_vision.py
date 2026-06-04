"""Vision-cone geometry tests against the PDF + live reference-server facts."""

import numpy as np

from zappy_rl.env import constants as C
from zappy_rl.env.vision import format_look, vision_offsets, vision_tiles


def test_tile_count_formula():
    # (L+1)**2 tiles. Level 1 => 4 (confirmed live: 4-element Look response).
    assert C.vision_tile_count(1) == 4
    assert C.vision_tile_count(2) == 9
    assert C.vision_tile_count(3) == 16
    for level in range(1, C.MAX_LEVEL + 1):
        assert vision_offsets(level, C.NORTH).shape == (
            (level + 1) ** 2,
            2,
        )


def test_self_tile_is_index_zero():
    for orient in (C.NORTH, C.EAST, C.SOUTH, C.WEST):
        off = vision_offsets(3, orient)
        assert tuple(off[0]) == (0, 0)


def test_level1_layout_north():
    # Facing north (forward = -y, right = +x): self, then row 1 = left, front, right.
    off = vision_offsets(1, C.NORTH)
    assert [tuple(o) for o in off] == [(0, 0), (-1, -1), (0, -1), (1, -1)]


def test_rows_are_centered_and_ordered():
    # Each row k has 2k+1 tiles at forward distance k, lateral -k..+k (left->right).
    off = vision_offsets(3, C.NORTH)
    idx = 0
    for k in range(4):
        row = off[idx : idx + (2 * k + 1)]
        idx += 2 * k + 1
        # forward distance k => y == -k when facing north
        assert all(o[1] == -k for o in row)
        assert [o[0] for o in row] == list(range(-k, k + 1))


def test_toroidal_wrap():
    # Player at a corner sees wrapped tiles; all coords within bounds.
    tiles = vision_tiles(0, 0, level=2, orientation=C.NORTH, width=5, height=5)
    assert tiles.shape == (9, 2)
    assert (tiles[:, 0] >= 0).all() and (tiles[:, 0] < 5).all()
    assert (tiles[:, 1] >= 0).all() and (tiles[:, 1] < 5).all()
    # Facing north from y=0 wraps to y=4, y=3 for rows 1 and 2.
    ys = {tuple(t): None for t in tiles}
    assert (0, 4) in ys  # front tile wrapped
    assert (0, 3) in ys  # second row wrapped


def test_orientation_rotation_consistency():
    # The front tile (row 1, center) must equal the forward delta for each facing.
    for orient in (C.NORTH, C.EAST, C.SOUTH, C.WEST):
        off = vision_offsets(1, orient)
        front_center = tuple(off[2])  # index 2 = center of row 1
        assert front_center == C.FORWARD_DELTA[orient]


def test_format_look_matches_reference_style():
    # Reproduces the live reference output: [ player food, mendiane,, food linemate ]
    contents = [["player", "food"], ["mendiane"], [], ["food", "linemate"]]
    assert format_look(contents) == "[ player food, mendiane,, food linemate ]"

"""Vision-cone geometry (NumPy reference oracle).

A ``Look`` at level ``L`` returns ``(L+1)**2`` tiles, in rows 0..L ahead of the
player. Row ``k`` holds ``2k+1`` tiles, numbered left-to-right; tile 0 is the
player's own tile. This is the empirically-confirmed layout (a level-1 look on
the reference server returns 4 tiles: ``[ player food, mendiane,, food linemate ]``).

The JAX env mirrors this; ``tests/test_vision.py`` and the differential test
against the reference server pin it down.
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def vision_offsets(level: int, orientation: int) -> np.ndarray:
    """Return tile offsets in look-order for a player at the origin.

    Shape ``((level+1)**2, 2)`` of integer ``(dx, dy)`` offsets, index 0 = self.
    Caller applies the player's position and the toroidal modulo.
    """
    fx, fy = C.FORWARD_DELTA[orientation]
    rx, ry = C.RIGHT_DELTA[orientation]
    offsets = []
    for k in range(level + 1):          # forward distance (row)
        for j in range(-k, k + 1):      # lateral offset, left -> right
            offsets.append((k * fx + j * rx, k * fy + j * ry))
    return np.asarray(offsets, dtype=np.int64)


def vision_tiles(
    x: int, y: int, level: int, orientation: int, width: int, height: int
) -> np.ndarray:
    """Absolute tile coordinates seen by a player, in look-order.

    Returns shape ``((level+1)**2, 2)`` of ``(tx, ty)`` already wrapped onto the
    toroidal world.
    """
    off = vision_offsets(level, orientation)
    tx = (x + off[:, 0]) % width
    ty = (y + off[:, 1]) % height
    return np.stack([tx, ty], axis=1)


def format_look(tile_contents: list[list[str]]) -> str:
    """Render a look result into the reference wire format.

    ``tile_contents`` is a list (in look-order) of lists of object name strings
    on each tile. Reproduces the reference server's exact style, e.g.
    ``[ player food, mendiane,, food linemate ]``: a space after ``[`` for the
    first tile, each later tile prefixed by ``,`` then a single space *iff*
    non-empty (so an empty tile yields ``,,``), trailing `` ]``.

    Note: the subject states the tile separator is "a comma followed or not by a
    space", so this format is *not* canonical — the parser (see
    ``deploy/protocol.py``) splits on ``,`` and strips, and differential tests
    compare parsed per-tile object multisets, not raw strings.
    """
    tiles = [" ".join(objs) for objs in tile_contents]
    out = tiles[0] if tiles else ""
    for t in tiles[1:]:
        out += "," + ((" " + t) if t else "")
    return "[ " + out + " ]"

"""Exact Zappy game constants.

Sourced from ``G-YEP-400_zappy.pdf`` (v3.3), the GUI-protocol PDF, and the
reference server README. Every number here is part of the *semantic contract*
with the reference server: the JAX env must match these so trained policies
transfer. Conventions flagged ``# VERIFY`` are pinned by differential testing
against reference-server golden traces (see ``tests/test_differential.py``).
"""

from __future__ import annotations

# --- Resources -------------------------------------------------------------
# Order matches the GUI ``bct``/``pin`` wire format: q0..q6.
FOOD = 0
LINEMATE = 1
DERAUMERE = 2
SIBUR = 3
MENDIANE = 4
PHIRAS = 5
THYSTAME = 6

RESOURCE_NAMES = (
    "food",
    "linemate",
    "deraumere",
    "sibur",
    "mendiane",
    "phiras",
    "thystame",
)
N_RESOURCES = len(RESOURCE_NAMES)
N_STONES = 6  # everything except food

# Spawn density: quantity on the map = width * height * density (PDF p.3).
DENSITY = {
    FOOD: 0.5,
    LINEMATE: 0.3,
    DERAUMERE: 0.15,
    SIBUR: 0.1,
    MENDIANE: 0.1,
    PHIRAS: 0.08,
    THYSTAME: 0.05,
}
DENSITY_VEC = tuple(DENSITY[i] for i in range(N_RESOURCES))

# --- Time / survival -------------------------------------------------------
DEFAULT_FREQ = 100          # f: reciprocal of the time unit (action_ticks / f seconds)
FOOD_LIFE_TICKS = 126       # 1 food unit = 126 time units of life (PDF p.3)
START_FOOD = 10             # players begin with 10 food => 1260 ticks (PDF p.8)
RESPAWN_INTERVAL_TICKS = 20  # server respawns resources every 20 time units (PDF p.3)

# --- Action costs (numerator; real cost in seconds is ticks / f) -----------
COST_FORWARD = 7
COST_RIGHT = 7
COST_LEFT = 7
COST_LOOK = 7
COST_INVENTORY = 1
COST_BROADCAST = 7
COST_CONNECT_NBR = 0        # immediate (PDF table: "-")
COST_FORK = 42
COST_EJECT = 7
COST_TAKE = 7
COST_SET = 7
COST_INCANTATION = 300

# --- Action space ----------------------------------------------------------
# Discrete action indices used by the policy. Take/Set are expanded per stone
# + food where relevant; Broadcast carries an 8-token symbol chosen by a
# separate head (see algo/networks.py), not enumerated here.
A_FORWARD = 0
A_RIGHT = 1
A_LEFT = 2
A_LOOK = 3
A_INVENTORY = 4
A_BROADCAST = 5
A_FORK = 6
A_EJECT = 7
A_TAKE = 8      # object selected by a companion argument head
A_SET = 9       # object selected by a companion argument head
A_INCANTATION = 10
A_CONNECT_NBR = 11
N_ACTIONS = 12

ACTION_NAMES = (
    "Forward", "Right", "Left", "Look", "Inventory", "Broadcast",
    "Fork", "Eject", "Take", "Set", "Incantation", "Connect_nbr",
)

ACTION_COST = (
    COST_FORWARD, COST_RIGHT, COST_LEFT, COST_LOOK, COST_INVENTORY,
    COST_BROADCAST, COST_FORK, COST_EJECT, COST_TAKE, COST_SET,
    COST_INCANTATION, COST_CONNECT_NBR,
)

BROADCAST_VOCAB = 8  # emergent-communication token count (8-symbol channel)

# --- Orientation -----------------------------------------------------------
# GUI ``O`` field: 1=N, 2=E, 3=S, 4=W (GUI-protocol PDF).
NORTH, EAST, SOUTH, WEST = 1, 2, 3, 4

# World uses X = column (width), Y = row (height), Y increasing downward.
# "Forward" = move one tile in the facing direction. Right = forward rotated
# 90° clockwise (screen coords, y down): (dx,dy) -> (-dy,dx).
# CONFIRMED against reference v3.0.1 (tools/validate_against_server.py): a
# level-3 Look while facing South reproduced all 16 tiles in exact order.
FORWARD_DELTA = {
    NORTH: (0, -1),
    EAST: (1, 0),
    SOUTH: (0, 1),
    WEST: (-1, 0),
}
RIGHT_DELTA = {
    NORTH: (1, 0),
    EAST: (0, 1),
    SOUTH: (-1, 0),
    WEST: (0, -1),
}

# --- Max level -------------------------------------------------------------
MIN_LEVEL = 1
MAX_LEVEL = 8

# --- Elevation requirements (PDF p.4) --------------------------------------
# Keyed by *target* level. Value = (players, linemate, deraumere, sibur,
# mendiane, phiras, thystame). Players need only be the SAME LEVEL (not team).
ELEVATION = {
    2: {"players": 1, "stones": (1, 0, 0, 0, 0, 0)},
    3: {"players": 2, "stones": (1, 1, 1, 0, 0, 0)},
    4: {"players": 2, "stones": (2, 0, 1, 0, 2, 0)},
    5: {"players": 4, "stones": (1, 1, 2, 0, 1, 0)},
    6: {"players": 4, "stones": (1, 2, 1, 3, 0, 0)},
    7: {"players": 6, "stones": (1, 2, 3, 0, 1, 0)},
    8: {"players": 6, "stones": (2, 2, 2, 2, 2, 1)},
}
# "stones" tuples are ordered (linemate, deraumere, sibur, mendiane, phiras,
# thystame) — i.e. resource indices 1..6.

WIN_LEVEL = 8
WIN_PLAYERS_AT_MAX = 6  # first team with 6 players at level 8 wins (PDF p.1)


def vision_tile_count(level: int) -> int:
    """Number of tiles a ``Look`` returns at ``level`` = (level + 1)**2."""
    return (level + 1) ** 2

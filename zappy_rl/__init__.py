"""Zappy deep-RL package.

Train multi-agent policies in a fast headless JAX clone of the Epitech Zappy
game, then deploy frozen policies to the reference server via a thin TCP
``zappy_ai`` adapter and visualize through the reference GUI.

The :mod:`zappy_rl.env` subpackage holds the game rules. Geometry helpers
(:mod:`zappy_rl.env.vision`, :mod:`zappy_rl.env.broadcast`) are implemented in
NumPy as the *reference oracle*; the production JAX env mirrors them and is
cross-checked against this oracle and against reference-server golden traces.
"""

__version__ = "0.1.0"

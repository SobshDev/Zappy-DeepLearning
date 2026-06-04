"""Offline visualization pipeline (recorder -> replay -> viewer/video).

This package turns JAX-env episodes into the *reference GUI wire protocol*, the
one contract every downstream renderer (the reference ``zappy_gui`` or our own
``viewer.py``) already understands. ``recorder.py`` is the producer; later
phases (``replay_to_gui.py``, ``viewer.py``, ``heatmaps.py``) are consumers that
read the NDJSON / SQLite it emits.
"""

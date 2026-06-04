"""PFSP (prioritized fictitious self-play) opponent pool for league training.

The plan calls for a self-play *league* rather than naive self-play because a
single learner chasing only its latest self collapses into rock-paper-scissors
cycles: policy A beats B beats C beats A, and the population never accumulates a
strict ordering. AlphaStar's fix is **prioritized fictitious self-play** — keep
a growing pool of frozen snapshots and bias the matchmaker toward opponents that
are *currently informative* — which is exactly what this module implements over
a directory of opaque parameter snapshots.

What this module is (and is NOT):

* It is a thin, dependency-light *bookkeeper*. It owns a directory, hands out
  monotonic snapshot ids, persists ``params_bytes`` to disk, and tracks the
  CURRENT learner's win/loss record against each frozen snapshot. That record
  is what PFSP needs to weight matchmaking.
* It deliberately does **not** import jax/flax. The caller serializes actor
  params however it likes (``flax.serialization.to_bytes`` in practice) and the
  league stores the resulting blob verbatim — keeping the pool decoupled from
  the network definition so old snapshots stay loadable even if the model code
  changes. numpy + stdlib only.

Matchmaking mix (from docs/PLAN.md, mirroring AlphaStar's main-agent schedule):

* **50% "pfsp"** — sample a frozen opponent with weight ``f(p) = p*(1-p)`` over
  the learner's win-rate ``p`` against it. This variance weighting peaks at
  ``p = 0.5`` (a coin-flip opponent, where each game is maximally informative)
  and vanishes at ``p -> 0`` (already crushed) and ``p -> 1`` (hopeless). It is
  the population analogue of "train on examples you're 50/50 on".
* **35% "self"** — the LATEST snapshot, i.e. (approximately) the learner's
  current self. Keeps a baseline of self-play so skills don't drift away from
  the frontier of the pool.
* **15% "exploiter"** — the snapshot with the LOWEST learner win-rate, i.e. the
  single most threatening opponent. Forces the learner to directly confront its
  worst matchup so the league can't hide a strict counter.

Win-rate is a Beta-posterior mean ``(wins + 1) / (games + 2)`` (a uniform
Beta(1,1) prior) rather than a raw frequency. Two reasons: an unplayed snapshot
reads as 0.5 instead of an undefined 0/0, so it enters PFSP at maximum weight
(``f(0.5) = 0.25``) and gets explored; and a 1/1 record reads as 0.667 instead
of a brittle 1.0, so early luck doesn't immediately exile an opponent from the
pool.

Durability: ``league.json`` (metadata + per-snapshot match records) is rewritten
on every mutation via write-tmp + fsync + ``os.replace`` + directory fsync — a
crash mid-write can never corrupt the index (atomicity: the reader always sees
either the old or the new file), and a completed mutation survives power loss
(durability: the rename's directory entry is fsynced too — fsyncing only the
file makes the rename itself volatile). Reopening a ``League`` on an existing
directory restores ids, tags, and records exactly (round-trip).

SINGLE WRITER by design: exactly one process (the training driver) mutates a
league directory. Tmp files are pid-suffixed so a stray second writer cannot
crash the first mid-replace, but two concurrent writers would still race on
``next_id`` — don't do that; readers are always safe.
"""

from __future__ import annotations

import json
import os
from typing import Literal

import numpy as np

# Snapshot ids are monotonic ints rendered as zero-padded strings ("000007").
# Zero-padding makes both the on-disk <id>.msgpack files and any directory
# listing sort in creation order, which is convenient for humans eyeballing the
# pool. Width 6 comfortably covers a 2-week run snapshotting every 4h.
_ID_WIDTH = 6

OpponentKind = Literal["pfsp", "self", "exploiter"]

# Matchmaking mix. Kept as named constants (not magic numbers in sample_*) so the
# 50/35/15 split from the plan is auditable in one place.
_P_PFSP = 0.50
_P_SELF = 0.35
_P_EXPLOITER = 0.15


def _snapshot_filename(snapshot_id: str) -> str:
    return f"{snapshot_id}.msgpack"


def _fsync_dir(path: str) -> None:
    """fsync a DIRECTORY: os.replace is atomic but the rename's directory
    entry is not durable across power loss until the dir itself is synced."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class League:
    """A directory-backed PFSP opponent pool.

    Layout under ``dir_path``::

        snapshots/<id>.msgpack   opaque serialized actor params (caller-provided)
        league.json              {next_id, snapshots:[{id, step, tag, wins, games}]}

    The in-memory ``self._records`` is the source of truth during a session and
    is flushed to ``league.json`` atomically after every mutation.
    """

    def __init__(self, dir_path: str | os.PathLike[str]) -> None:
        self.dir_path = str(dir_path)
        self.snap_dir = os.path.join(self.dir_path, "snapshots")
        self.index_path = os.path.join(self.dir_path, "league.json")

        os.makedirs(self.snap_dir, exist_ok=True)

        # next_id is the integer to assign to the NEXT snapshot; it only ever
        # grows, even across reopen, so ids are never reused after a deletion.
        self._next_id = 0
        # id -> {"step": int, "tag": str, "wins": int, "games": int}. Ordered by
        # insertion (== ascending id) because dict preserves order; "latest"
        # therefore means the last key.
        self._records: dict[str, dict] = {}

        if os.path.exists(self.index_path):
            self._load_index()

    # ---- persistence ------------------------------------------------------

    def _load_index(self) -> None:
        with open(self.index_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._next_id = int(data["next_id"])
        records: dict[str, dict] = {}
        for snap in data["snapshots"]:
            records[snap["id"]] = {
                "step": int(snap["step"]),
                "tag": snap.get("tag", ""),
                "wins": int(snap["wins"]),
                "games": int(snap["games"]),
            }
        self._records = records

    def _write_index(self) -> None:
        """Atomically rewrite league.json (tmp file + os.replace).

        os.replace is atomic on POSIX within a filesystem, so a concurrent or
        post-crash reader sees the complete old file or the complete new one.
        """
        payload = {
            "next_id": self._next_id,
            "snapshots": [
                {
                    "id": sid,
                    "step": rec["step"],
                    "tag": rec["tag"],
                    "wins": rec["wins"],
                    "games": rec["games"],
                }
                for sid, rec in self._records.items()
            ],
        }
        # pid-suffixed tmp: a stray concurrent writer can't steal/replace our
        # half-written tmp out from under us (single-writer is still the rule).
        tmp_path = f"{self.index_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())  # the bytes must hit disk before the rename
        os.replace(tmp_path, self.index_path)
        _fsync_dir(self.dir_path)  # ...and the rename must hit disk too

    # ---- snapshot management ----------------------------------------------

    def add_snapshot(self, params_bytes: bytes, step: int, tag: str = "") -> str:
        """Persist ``params_bytes`` as a new frozen opponent; return its id.

        Writes the blob first, then commits the index, so the index never
        references a snapshot file that does not yet exist on disk.
        """
        snapshot_id = format(self._next_id, f"0{_ID_WIDTH}d")
        blob_path = os.path.join(self.snap_dir, _snapshot_filename(snapshot_id))

        # Blob is also written via tmp+replace: a snapshot file is either fully
        # present or absent, never truncated.
        tmp_blob = f"{blob_path}.{os.getpid()}.tmp"
        with open(tmp_blob, "wb") as fh:
            fh.write(params_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_blob, blob_path)
        _fsync_dir(self.snap_dir)  # make the rename durable before indexing it

        self._records[snapshot_id] = {
            "step": int(step),
            "tag": tag,
            "wins": 0,
            "games": 0,
        }
        self._next_id += 1
        self._write_index()
        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> bytes:
        """Return the raw bytes stored under ``snapshot_id`` (round-trips add)."""
        if snapshot_id not in self._records:
            raise KeyError(f"unknown snapshot id: {snapshot_id!r}")
        blob_path = os.path.join(self.snap_dir, _snapshot_filename(snapshot_id))
        try:
            with open(blob_path, "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            # indexed-but-missing blob (manual deletion / partial restore):
            # keep the API's KeyError contract instead of leaking a deep
            # FileNotFoundError into the training loop.
            raise KeyError(
                f"snapshot {snapshot_id!r} is indexed but its blob is missing: "
                f"{blob_path}"
            ) from None

    # ---- match bookkeeping ------------------------------------------------

    def record_result(self, snapshot_id: str, won: bool) -> None:
        """Record one CURRENT-learner game vs ``snapshot_id``.

        ``won`` is from the learner's perspective (True == learner beat the
        frozen snapshot). Flushed atomically so the record survives a crash.
        """
        if snapshot_id not in self._records:
            raise KeyError(f"unknown snapshot id: {snapshot_id!r}")
        rec = self._records[snapshot_id]
        rec["games"] += 1
        if won:
            rec["wins"] += 1
        self._write_index()

    def win_rate(self, snapshot_id: str) -> float:
        """Beta(1,1)-posterior mean of the learner's win-rate vs ``snapshot_id``.

        ``(wins + 1) / (games + 2)`` — an unplayed snapshot is 0.5 (maximally
        informative under PFSP), avoiding both 0/0 and overconfident 0/1.
        """
        if snapshot_id not in self._records:
            raise KeyError(f"unknown snapshot id: {snapshot_id!r}")
        rec = self._records[snapshot_id]
        return (rec["wins"] + 1) / (rec["games"] + 2)

    # ---- introspection ----------------------------------------------------

    @property
    def ids(self) -> list[str]:
        """Snapshot ids in ascending (creation) order."""
        return list(self._records.keys())

    def __len__(self) -> int:
        return len(self._records)

    # ---- matchmaking ------------------------------------------------------

    def _sample_pfsp(self, rng: np.random.Generator) -> str:
        """Pick a snapshot with weight f(p)=p*(1-p) over learner win-rate p.

        Peaks at the coin-flip opponent (p=0.5); falls back to uniform when every
        weight is 0 (only possible if every snapshot reads p in {0,1}, which the
        Beta posterior makes impossible for played snapshots but a degenerate
        caller could still force by construction).
        """
        ids = self.ids
        p = np.array([self.win_rate(sid) for sid in ids], dtype=np.float64)
        weights = p * (1.0 - p)
        total = weights.sum()
        if total <= 0.0:
            weights = np.ones_like(weights)
            total = weights.sum()
        idx = rng.choice(len(ids), p=weights / total)
        return ids[idx]

    def _latest(self) -> str:
        """The most recently added snapshot (highest id)."""
        return self.ids[-1]

    def _exploiter(self) -> str:
        """The snapshot with the LOWEST learner win-rate (most threatening).

        Ties broken toward the earliest such id (np.argmin behaviour) — a stable,
        deterministic choice so the exploiter target doesn't flap between equally
        hard opponents on every call.
        """
        ids = self.ids
        p = np.array([self.win_rate(sid) for sid in ids], dtype=np.float64)
        return ids[int(np.argmin(p))]

    def sample_opponent(
        self, rng: np.random.Generator
    ) -> tuple[str, OpponentKind]:
        """Sample (snapshot_id, kind) under the 50/35/15 PFSP/self/exploiter mix.

        Edge cases: an empty league raises ValueError (nothing to play); a
        single-snapshot league always returns that snapshot — but still reports
        the *kind* drawn, since which schedule we're on is meaningful telemetry
        even when the three collapse to the same opponent.
        """
        if not self._records:
            raise ValueError("cannot sample from an empty league")

        # One draw selects the schedule; the schedule then selects the opponent.
        u = rng.random()
        if u < _P_PFSP:
            return self._sample_pfsp(rng), "pfsp"
        if u < _P_PFSP + _P_SELF:
            return self._latest(), "self"
        return self._exploiter(), "exploiter"

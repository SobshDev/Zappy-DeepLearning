"""Deploy adapter: a frozen MAPPO policy playing as a real ``zappy_ai`` client.

Bridges the two-track contract (docs/PLAN.md): everything the policy consumes
here must be *semantically identical* to what the JAX sim feeds it in training.
The flat input layout is ``networks.flatten_obs`` — vision*0.2 ‖ self ‖ msg_dir
‖ msg_tok — and ``build_obs`` below reproduces it from parsed server strings
(``tests/test_deploy_adapter.py`` pins builder == ``Z.observe`` exactly;
``tools/validate_obs_contract.py`` re-pins it against the live server).

Decision cycle (the policy never learned ``Look``/``Inventory`` — the sim hands
it a fresh observation every step, so the adapter must perceive explicitly):

    [Inventory every N cycles] -> Look -> build obs -> GRU step -> sampled
    action -> command

``Inventory`` is NOT sent every cycle (cadence: the perceive prefix is the
deploy/sim speed gap, and the live L4 stall traced to it): the inventory obs
is dead-reckoned and re-synced every ``--inv-every`` cycles (default 10), plus
after any involuntary freeze:

* stones: a ``Take``/``Set``-ok tally — which is *exactly* the sim's ``inv``
  bookkeeping, so between syncs this is the contract, not an approximation
  (a late ack can shift one tally by 1 until the next sync — tolerated);
* life: ``life_est`` ticks, charged per completed command from the known cost
  table (Look 7, moves/takes/sets/broadcast 7, Inventory 1, incantation 300 —
  incl. passive "Elevation underway" freezes), +126 per ``Take food`` ok,
  re-synced to ``food_stock*126`` at each Inventory; the signed drift is
  measured at every sync and reported (``max_drift_ticks``).

Semantic mappings that are easy to get wrong (each mirrors a sim convention):

* ``inv[food]`` in the sim ACCUMULATES successful takes and never decays (the
  canonical survival counter is ``life``, not the food stock) — so the feature
  comes from a counted ``Take food``-ok tally, while the *life* feature maps
  from the server's remaining food stock: ``clip(food/10, 0, 1)`` ==
  ``clip(life/1260, 0, 1)``.
* Orientation is dead-reckoned from an assumed ``NORTH`` spawn (the server
  never reports it). Every other input is egocentric (look-order vision,
  receiver-relative broadcast K), so a constant rotation of the one-hot is
  indistinguishable from a rotated world — harmless by symmetry.
* ``busy`` is always 0 at decision time: in training, every observation whose
  action the env actually consumed (``info["free"]``) carried ``busy=0``, and
  the adapter only queries the policy when the previous command resolved.
* ``ENV_IDLE`` maps to *no command*: the cycle's ``Look`` (7 ticks) is the
  "perceive" the sim's IDLE stands in for.
* Heard broadcasts are delivered for exactly one decision (the sim overwrites
  ``last_dir/last_tok`` every step); out-of-vocabulary message text (foreign
  AIs) is dropped entirely — the policy only ever heard tokens 0..7.
* Actions are SAMPLED by default; ``--greedy`` switches to argmax. The
  Phase-2 greedy collapse (entropy-regularized ties breaking degenerately)
  no longer applies: from speedrun-s01 onward, greedy measured ~1% faster
  median t(win) at equal win rate in the standardized sim eval.
* **Late acks** (probed live, v3.0.1): when an incantation freezes us while we
  have a command in flight — the trained behavior makes BOTH agents send
  ``Incantation`` near-simultaneously, so the loser's command is queued across
  the whole freeze — the server answers the elevation lines FIRST and the
  queued command's ``ok``/``ko`` arrives one expect-window late. These are
  well-understood responses, not desync: they are absorbed at the next
  perceive expect and counted in ``late_acks`` (reported, not a protocol
  error). A late ack CAN shift one ok/ko onto the wrong motion/take command
  in between; the only state fed by acks is the food counter / dead-reckoned
  orientation, both tolerant by design, and the gate's GUI-level cross-check
  verifies end-to-end that nothing actually desynced.

Known cadence approximation (documented, not a gate concern): one deploy cycle
costs ~14 ticks (Look 7 + action 7, Inventory amortized ~0.1) vs the sim's ~7
per decision — training with ``TrainConfig.overhead ≈ 7`` mirrors exactly this;
the GRU skips the ignored-action steps a sim agent submits while frozen in
an incantation (for co-located pair rituals — the trained behavior — the sim
also takes exactly one step across the freeze, so the carry cadence matches).

Gate accounting (Phase 4: "zero protocol errors over a full game"): every
unexpected, unparseable, or timed-out line is recorded in
``ZappyAIClient.protocol_errors`` and surfaced in the final report.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from ..env import constants as C
from ..env import zappy_env as Z
from .protocol import LineSocket, parse_broadcast, parse_inventory, parse_look

_RES_INDEX = {name: i for i, name in enumerate(C.RESOURCE_NAMES)}
_OK_KO = ("ok", "ko")
START_LIFE_TICKS = C.START_FOOD * C.FOOD_LIFE_TICKS  # 1260
# per-command tick cost for life dead-reckoning (server charges the time
# whether the action ok's or ko's; instant-ko Incantation charges nothing)
_TICK_COST = {"Forward": 7, "Right": 7, "Left": 7, "Look": 7, "Inventory": 1,
              "Broadcast": 7, "Take": 7, "Set": 7}


# ----------------------------------------------------------- obs (contract)
def build_obs(
    tiles: list[list[str]],
    inv: dict[str, int],
    level: int,
    orient: int,
    food_taken: int,
    msg: tuple[int, int] | None = None,
    busy: bool = False,
    life_ticks: float | None = None,
) -> np.ndarray:
    """Parsed server state -> the exact ``flatten_obs`` policy input.

    ``tiles`` is ``parse_look`` output (look-order word lists), ``inv`` is
    ``parse_inventory`` output (or the dead-reckoned stone tally in the same
    dict shape), ``food_taken`` the cumulative successful ``Take food`` count
    (sim ``inv[food]`` semantics), ``msg`` an optional heard ``(K, token)``.
    ``life_ticks``, when given, feeds the life feature directly as
    ``clip(life_ticks/1260, 0, 1)`` (the sim's exact form — finer than the
    food-stock fallback ``clip(food/10, 0, 1)``, which quantizes to 126-tick
    units). Mirrors ``Z.observe`` + ``networks.flatten_obs``.
    """
    vision = np.zeros((Z.MAX_VISION_TILES, C.N_RESOURCES + 1), np.float32)
    n_vis = (level + 1) ** 2  # the sim masks tiles beyond the level's cone
    for i, words in enumerate(tiles[: min(n_vis, Z.MAX_VISION_TILES)]):
        for w in words:
            if w == "player":
                vision[i, C.N_RESOURCES] += 1.0
            elif w in _RES_INDEX:
                vision[i, _RES_INDEX[w]] += 1.0
            # unknown words (eggs, ...) are invisible to the sim -> ignored

    self_feat = np.zeros(Z.SELF_DIM, np.float32)
    self_feat[level - 1] = 1.0  # one_hot(level-1, MAX_LEVEL)
    inv_vec = np.array([inv.get(n, 0) for n in C.RESOURCE_NAMES], np.float32)
    inv_vec[C.FOOD] = float(food_taken)  # sim inv[food] = cumulative takes
    # multiply by reciprocal, NOT /10.0: XLA folds the sim's division into a
    # *0.1 multiply, which rounds one float32 ULP differently for counts
    # {9, 13, 18, ...} — *0.1 here keeps the contract bit-exact (review-pinned).
    self_feat[C.MAX_LEVEL : C.MAX_LEVEL + C.N_RESOURCES] = inv_vec * np.float32(0.1)
    self_feat[C.MAX_LEVEL + C.N_RESOURCES + (orient - 1)] = 1.0
    # life feature: dead-reckoned ticks when available, else server food
    # stock (food*126 ticks <-> sim clip(life/1260, 0, 1))
    if life_ticks is not None:
        # f32 multiply by reciprocal, NOT /1260: XLA folds the sim's division
        # into a reciprocal multiply — true division diverges by 1 ULP for
        # 162 of the 1261 integer life values (empirically pinned, like *0.1).
        self_feat[C.MAX_LEVEL + C.N_RESOURCES + 4] = np.clip(
            np.float32(life_ticks) * np.float32(1.0 / START_LIFE_TICKS), 0.0, 1.0
        )
    else:
        self_feat[C.MAX_LEVEL + C.N_RESOURCES + 4] = min(
            inv.get("food", 0) / float(C.START_FOOD), 1.0
        )
    self_feat[C.MAX_LEVEL + C.N_RESOURCES + 5] = 1.0 if busy else 0.0

    msg_dir = np.zeros(9, np.float32)
    msg_tok = np.zeros(C.BROADCAST_VOCAB, np.float32)
    if msg is not None:
        k, tok = msg
        msg_dir[int(np.clip(k, 0, 8))] = 1.0
        msg_tok[int(np.clip(tok, 0, C.BROADCAST_VOCAB - 1))] = 1.0

    from ..algo.networks import VISION_SCALE  # local: keep module import light

    return np.concatenate([vision.reshape(-1) * VISION_SCALE, self_feat, msg_dir, msg_tok])


def action_to_command(action: int, token: int) -> str | None:
    """Env action index -> wire command (``None`` = IDLE: Look already sent)."""
    if action == Z.ENV_FORWARD:
        return "Forward"
    if action == Z.ENV_RIGHT:
        return "Right"
    if action == Z.ENV_LEFT:
        return "Left"
    if Z.ENV_TAKE0 <= action < Z.ENV_TAKE0 + C.N_RESOURCES:
        return f"Take {C.RESOURCE_NAMES[action - Z.ENV_TAKE0]}"
    if Z.ENV_SET0 <= action < Z.ENV_SET0 + C.N_RESOURCES:
        return f"Set {C.RESOURCE_NAMES[action - Z.ENV_SET0]}"
    if action == Z.ENV_BROADCAST:
        return f"Broadcast {int(token)}"
    if action == Z.ENV_INCANT:
        return "Incantation"
    return None  # ENV_IDLE


# ------------------------------------------------------------------ policy
class PolicyRunner:
    """Frozen recurrent actor: loads the checkpoint, holds the GRU carry,
    SAMPLES one (action, token) per observation."""

    def __init__(self, params_path: str | Path, hidden: int = 128, seed: int = 0,
                 greedy: bool = False):
        import flax.serialization
        import jax
        import jax.numpy as jnp

        from ..algo.networks import OBS_DIM, RecurrentActor, ScannedRNN

        self._jax, self._jnp = jax, jnp
        actor = RecurrentActor(hidden=hidden)
        h0 = ScannedRNN.initialize_carry(1, hidden)
        template = actor.init(
            jax.random.PRNGKey(0), h0,
            (jnp.zeros((1, 1, OBS_DIM)), jnp.zeros((1, 1), bool)),
        )
        raw = Path(params_path).read_bytes()
        loaded = flax.serialization.from_bytes({"actor": template, "critic": None}, raw)
        # from_bytes checks tree structure but NOT leaf shapes (same guard as
        # mappo.train's --init-actor): fail loudly, not with a jit shape error.
        bad = jax.tree.map(lambda t, l: t.shape != jnp.shape(l), template, loaded["actor"])
        if any(jax.tree.leaves(bad)):
            raise ValueError(
                f"{params_path}: actor param shapes do not match hidden={hidden} "
                f"(wrong checkpoint or wrong --hidden?)"
            )
        self.params = loaded["actor"]
        self.h = h0
        self.key = jax.random.PRNGKey(seed)
        self.greedy = greedy
        self.first = True  # first obs of the episode resets the GRU carry
        self._apply = jax.jit(actor.apply)
        # pre-warm the jit BEFORE the TCP connect: compilation takes ~13s on
        # CPU, and life starts decaying at connect — uncompiled, the squad
        # joins staggered and the game clock burns ~1300 ticks for nothing
        jax.block_until_ready(self._apply(
            self.params, h0,
            (jnp.zeros((1, 1, OBS_DIM)), jnp.zeros((1, 1), bool))))

    def act(self, obs: np.ndarray) -> tuple[int, int]:
        from ..algo.networks import cat_sample

        jax, jnp = self._jax, self._jnp
        self.key, k_a, k_t = jax.random.split(self.key, 3)
        obs_j = jnp.asarray(obs, jnp.float32)[None, None]  # [T=1, B=1, OBS_DIM]
        done = jnp.full((1, 1), self.first)
        self.h, la, lt = self._apply(self.params, self.h, (obs_j, done))
        self.first = False
        if self.greedy:  # measured faster at equal win rate from s01 onward
            return int(jnp.argmax(la[0])), int(jnp.argmax(lt[0]))
        action = int(cat_sample(k_a, la[0])[0])
        token = int(cat_sample(k_t, lt[0])[0])
        return action, token


# ------------------------------------------------------------------ client
class ZappyAIClient:
    """One agent: TCP handshake, perceive-act cycles, async line routing."""

    # response timeout: must outlive an involuntary 300-tick freeze (3 s at
    # f=100) stalling our queued command, plus server scheduling slack.
    RESPONSE_TIMEOUT = 15.0

    def __init__(self, sock: LineSocket, team: str, policy: PolicyRunner,
                 inv_every: int = 10):
        self.sock = sock
        self.team = team
        self.policy = policy
        self.level = 1
        self.orient = C.NORTH  # dead-reckoned (see module doc)
        self.food_taken = 0
        self.msg: tuple[int, int] | None = None
        self.alive = True
        self.connected = True
        self.disconnected = False  # link lost while alive (NOT death/budget)
        self.late_acks = 0  # queued-across-freeze ok/ko absorbed at a perceive
        self.incant_ok = 0  # rituals we initiated that completed with a level
        self.incant_ko = 0  # instant kos (requirements unmet) + mid-ritual kos
        self.protocol_errors: list[str] = []
        self.cycles = 0
        self.commands = 0
        self.levelups: list[int] = []  # levels reached, in order
        self.msgs_heard = 0
        self.slots = 0
        self.world = (0, 0)
        # inventory dead-reckoning (see module doc): stones are a Take/Set-ok
        # tally (= the sim's own inv bookkeeping); life is charged per command
        # and re-synced to the server's food stock every `inv_every` cycles.
        self.inv_every = max(1, inv_every)
        self.stones = {name: 0 for name in C.RESOURCE_NAMES}  # food key unused
        self.life_est = float(START_LIFE_TICKS)
        self.force_inv = False   # set after involuntary freezes (drift risk)
        self.inv_syncs = 0
        self.max_drift_ticks = 0.0  # max |life_est - food*126| seen at syncs

    @classmethod
    def connect(cls, host: str, port: int, team: str, policy: PolicyRunner,
                inv_every: int = 10) -> "ZappyAIClient":
        client = cls(LineSocket.connect(host, port), team, policy, inv_every=inv_every)
        client.handshake()
        return client

    def handshake(self) -> None:
        welcome = self.sock.recv_line(timeout=5)
        if welcome != "WELCOME":
            raise RuntimeError(f"expected WELCOME, got {welcome!r}")
        self.sock.send(self.team)
        slots = self.sock.recv_line(timeout=5)
        dims = self.sock.recv_line(timeout=5)
        try:
            self.slots = int(slots)
            x, y = dims.split()
            self.world = (int(x), int(y))
        except (ValueError, AttributeError, TypeError) as e:
            # TypeError: int(None) when the slots line timed out / EOF'd
            raise RuntimeError(f"bad handshake: slots={slots!r} dims={dims!r}") from e

    # ------------------------------------------------------------- wire IO
    def _route_async(self, line: str) -> bool:
        """Consume server-initiated lines; True if ``line`` was one of them."""
        if line.startswith("message "):
            try:
                k, text = parse_broadcast(line)
            except ValueError:
                self.protocol_errors.append(f"unparseable broadcast: {line!r}")
                return True
            self.msgs_heard += 1
            try:
                tok = int(text)
            except ValueError:
                return True  # out-of-vocab text: noise, drop (see module doc)
            if 0 <= k <= 8 and 0 <= tok < C.BROADCAST_VOCAB:
                self.msg = (k, tok)  # latest wins, like the sim's one-per-step
            return True
        if line.startswith("eject:"):
            return True  # we never Fork/Eject; position is untracked anyway
        if line == "Elevation underway":
            # dragged into a co-located ritual as a passive participant; our
            # in-flight command stalls ~300 ticks (covered by the timeout).
            self._charge(C.COST_INCANTATION)  # the freeze burns life too
            self.force_inv = True             # attribution is murky: re-sync
            return True
        if line.startswith("Current level:"):
            self._note_level(line)
            return True
        return False

    def _charge(self, ticks: float) -> None:
        self.life_est -= ticks

    def _sync_inventory(self, inv: dict[str, int]) -> None:
        """Re-anchor dead-reckoned life + stones to a fresh Inventory."""
        server_life = inv.get("food", 0) * C.FOOD_LIFE_TICKS
        # the server reports food = ceil(life/126), so server_life is the TOP
        # of the 126-tick band: true life is in (server_life-126, server_life]
        # and an honestly-charged life_est sits in that same band — i.e.
        # drift in (-126, 0]. Outside it is real desync. The first sync is
        # excluded: the pre-sync estimate is the connect-time default, and
        # handshake wall-time legitimately ages it (review-pinned centering).
        drift = self.life_est - server_life
        if self.inv_syncs > 0:
            excess = max(0.0, drift) + max(0.0, -drift - C.FOOD_LIFE_TICKS)
            self.max_drift_ticks = max(self.max_drift_ticks, excess)
        self.life_est = float(server_life)
        for name in C.RESOURCE_NAMES[1:]:  # food: cumulative-takes, never sync
            self.stones[name] = inv.get(name, 0)
        self.inv_syncs += 1

    def _note_level(self, line: str) -> None:
        try:
            self.level = int(line.split(":")[1])
            self.levelups.append(self.level)
        except (ValueError, IndexError):
            self.protocol_errors.append(f"unparseable level: {line!r}")

    def _response(self, expect, what: str, timeout: float | None = None,
                  late_ack_ok: bool = False):
        """Next line matching ``expect`` (async lines routed aside).

        Unexpected lines are recorded as protocol errors and skipped; ``None``
        on timeout/disconnect/death (timeout also recorded as an error). The
        deadline is enforced at the TOP of every iteration — a steady stream
        of async lines (busy server, many broadcasters) must not starve it
        (review-pinned: deadline-only-on-idle let the client hang forever).

        ``late_ack_ok`` (perceive expects): a bare ok/ko here is the response
        to a command queued across an involuntary freeze (see module doc) —
        absorbed and counted, not an error.
        """
        deadline = time.monotonic() + (timeout or self.RESPONSE_TIMEOUT)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.protocol_errors.append(f"timeout waiting for {what}")
                return None
            line = self.sock.recv_line(timeout=remaining)
            if line is None:
                if time.monotonic() >= deadline:
                    self.protocol_errors.append(f"timeout waiting for {what}")
                else:  # EOF / reset before the deadline: the link is gone
                    self.connected = False
                    if self.alive:
                        self.disconnected = True
                return None
            if line == "dead":
                self.alive = False
                return None
            # expected-first: "Elevation underway" / "Current level:" are async
            # ONLY when not the reply being awaited (our own Incantation).
            if expect(line):
                return line
            if self._route_async(line):
                continue
            if late_ack_ok and line in _OK_KO:
                self.late_acks += 1
                continue
            self.protocol_errors.append(f"unexpected reply to {what}: {line!r}")

    def _cmd(self, cmd: str, expect, timeout: float | None = None,
             late_ack_ok: bool = False):
        try:
            self.sock.send(cmd)
        except OSError:  # broken pipe / reset: report it, don't crash the gate
            self.connected = False
            if self.alive:
                self.disconnected = True
            return None
        self.commands += 1
        return self._response(expect, cmd, timeout, late_ack_ok=late_ack_ok)

    # ------------------------------------------------------------- the loop
    def cycle(self) -> None:
        """One perceive-decide-act cycle."""
        bracketed = lambda l: l.startswith("[") and l.endswith("]")  # noqa: E731
        if self.cycles % self.inv_every == 0 or self.force_inv:
            inv_line = self._cmd("Inventory", bracketed, late_ack_ok=True)
            if not (self.alive and self.connected) or inv_line is None:
                return  # dead/disconnected: don't queue more commands
            # no charge for Inventory's own tick: the sync re-anchors life to
            # the server's stock AT RESPONSE TIME, which already includes it
            try:
                self._sync_inventory(parse_inventory(inv_line))
            except ValueError as e:
                self.protocol_errors.append(str(e))
                return
            self.force_inv = False
        look_line = self._cmd("Look", bracketed, late_ack_ok=True)
        if not (self.alive and self.connected) or look_line is None:
            return
        self._charge(_TICK_COST["Look"])
        try:
            tiles = parse_look(look_line)
        except ValueError as e:
            self.protocol_errors.append(str(e))
            return
        inv = dict(self.stones)  # dead-reckoned stones, sim-inv semantics
        obs_level = self.level
        if len(tiles) != (self.level + 1) ** 2:
            self.protocol_errors.append(
                f"look returned {len(tiles)} tiles at level {self.level}"
            )
            # the Look IS the server's statement of our level — if the count
            # is a valid cone size, trust it for THIS obs (vision mask +
            # level one-hot) so a level desync can't silently mis-mask vision.
            root = int(len(tiles) ** 0.5 + 0.5)
            if root * root == len(tiles) and 1 <= root - 1 <= C.MAX_LEVEL:
                obs_level = root - 1

        obs = build_obs(tiles, inv, obs_level, self.orient, self.food_taken,
                        self.msg, life_ticks=self.life_est)
        self.msg = None  # delivered for exactly one decision, like the sim
        action, token = self.policy.act(obs)
        self.cycles += 1

        cmd = action_to_command(action, token)
        if cmd is None:
            return  # IDLE: the Look above was the perceive

        if action == Z.ENV_INCANT:
            resp = self._cmd(cmd, lambda l: l == "Elevation underway" or l == "ko")
            if resp == "Elevation underway":
                self._charge(C.COST_INCANTATION)  # instant ko charges nothing
                # frozen 300 ticks; completion is "Current level: k" or "ko"
                done = self._response(
                    lambda l: l.startswith("Current level:") or l == "ko",
                    "incantation result",
                )
                if done is not None and done.startswith("Current level:"):
                    self._note_level(done)
                    self.incant_ok += 1
                else:
                    self.incant_ko += 1
                self.force_inv = True  # re-anchor life after the long freeze
            elif resp == "ko":
                self.incant_ko += 1
            return

        resp = self._cmd(cmd, lambda l: l in _OK_KO)
        if resp in _OK_KO:  # the server charges the time on ko too
            self._charge(_TICK_COST.get(cmd.split()[0], 7))
        if resp == "ok":
            if cmd == "Take food":
                self.food_taken += 1  # sim inv[food] semantics
                self.life_est += C.FOOD_LIFE_TICKS
            elif cmd.startswith("Take "):
                self.stones[cmd.split()[1]] += 1  # sim inv tally
            elif cmd == "Set food":
                # sim: inv[food] -= 1 and the dropped ration's life is lost
                self.food_taken = max(0, self.food_taken - 1)
                self.life_est -= C.FOOD_LIFE_TICKS
            elif cmd.startswith("Set "):
                name = cmd.split()[1]
                self.stones[name] = max(0, self.stones[name] - 1)
            elif cmd == "Right":
                self.orient = self.orient % 4 + 1  # sim rotation convention
            elif cmd == "Left":
                self.orient = self.orient - 1 if self.orient > 1 else C.WEST

    def run(self, max_cycles: int | None = None, duration: float | None = None) -> dict:
        deadline = time.monotonic() + duration if duration else None
        while self.alive and self.connected:
            if max_cycles is not None and self.cycles >= max_cycles:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            self.cycle()
        return self.report()

    def report(self) -> dict:
        return {
            "team": self.team,
            "cycles": self.cycles,
            "commands": self.commands,
            "final_level": self.level,
            "levelups": self.levelups,
            "food_taken": self.food_taken,
            "msgs_heard": self.msgs_heard,
            "alive": self.alive,
            "disconnected": self.disconnected,
            "late_acks": self.late_acks,
            "incant_ok": self.incant_ok,
            "incant_ko": self.incant_ko,
            "stones": dict(self.stones),
            "inv_syncs": self.inv_syncs,
            "max_drift_ticks": round(self.max_drift_ticks, 1),
            "n_protocol_errors": len(self.protocol_errors),
            "protocol_errors": self.protocol_errors[:50],
        }

    def close(self) -> None:
        self.sock.close()


# --------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4242)
    ap.add_argument("--team", default="T1")
    ap.add_argument("--params", default="runs/ritual8x8-v1/params.msgpack")
    ap.add_argument("--config", default=None,
                    help="run config.json (for hidden); default: alongside --params")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax actions instead of sampling")
    ap.add_argument("--inv-every", type=int, default=10,
                    help="Inventory re-sync period in cycles (life/stones are "
                         "dead-reckoned in between)")
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None, help="seconds")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args(argv)

    # Deploy inference is tiny — never grab the training GPU. JAX picks its
    # backend lazily at first use, so setting env here (pre-PolicyRunner) works.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    cfg_path = Path(args.config) if args.config else Path(args.params).parent / "config.json"
    hidden = 128
    if cfg_path.exists():
        hidden = int(json.loads(cfg_path.read_text()).get("hidden", 128))

    policy = PolicyRunner(args.params, hidden=hidden, seed=args.seed,
                          greedy=args.greedy)
    client = ZappyAIClient.connect(args.host, args.port, args.team, policy,
                                   inv_every=args.inv_every)
    print(f"[adapter] joined {args.team} (slots={client.slots}, "
          f"world={client.world[0]}x{client.world[1]})")
    try:
        report = client.run(max_cycles=args.max_cycles, duration=args.duration)
    finally:
        client.close()
    print(f"[adapter] {json.dumps({k: v for k, v in report.items() if k != 'protocol_errors'})}")
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2))
    return 0 if report["n_protocol_errors"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

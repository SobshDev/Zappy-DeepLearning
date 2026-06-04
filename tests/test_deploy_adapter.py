"""Deploy-adapter tests: the obs CONTRACT (builder == sim observe, exactly),
action->command mapping, and the client's wire-protocol routing over a fake
socket. ``tools/validate_obs_contract.py`` re-pins the same contract live."""

import json
import socket
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zappy_rl.algo.networks import OBS_DIM, flatten_obs
from zappy_rl.deploy.protocol import LineSocket, parse_look
from zappy_rl.deploy.zappy_ai_adapter import (
    PolicyRunner,
    ZappyAIClient,
    action_to_command,
    build_obs,
)
from zappy_rl.env import constants as C
from zappy_rl.env import zappy_env as Z
from zappy_rl.env.vision import format_look, vision_tiles

KEY = jax.random.PRNGKey(0)
RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "ritual8x8-v1"


def mk_state(cfg, pos, orient, level=None, grid=None, inv=None, life=None,
             last_dir=None, last_tok=None):
    A = len(pos)
    level = level or [1] * A
    g = jnp.zeros((cfg.width, cfg.height, C.N_RESOURCES), jnp.int32) if grid is None \
        else jnp.asarray(grid, jnp.int32)
    return Z.State(
        grid=g, pos=jnp.asarray(pos, jnp.int32), orient=jnp.asarray(orient, jnp.int32),
        level=jnp.asarray(level, jnp.int32),
        inv=jnp.zeros((A, C.N_RESOURCES), jnp.int32) if inv is None else jnp.asarray(inv, jnp.int32),
        life=jnp.full(A, Z.START_LIFE, jnp.int32) if life is None else jnp.asarray(life, jnp.int32),
        alive=jnp.ones(A, bool),
        busy_until=jnp.zeros(A, jnp.int32), pending=jnp.zeros(A, bool),
        incant_level=jnp.zeros(A, jnp.int32), initiator=jnp.zeros(A, bool),
        team=jnp.zeros(A, jnp.int32), now=jnp.int32(0), key=KEY,
        last_dir=jnp.full(A, -1, jnp.int32) if last_dir is None else jnp.asarray(last_dir, jnp.int32),
        last_tok=jnp.full(A, -1, jnp.int32) if last_tok is None else jnp.asarray(last_tok, jnp.int32),
    )


def server_tiles_for(cfg, s, agent: int) -> list[list[str]]:
    """What the reference server's Look would report for ``agent``: per
    look-order tile, 'player' per alive player + resource names per unit."""
    grid = np.asarray(s.grid)
    pos = np.asarray(s.pos)
    alive = np.asarray(s.alive)
    x, y = int(pos[agent, 0]), int(pos[agent, 1])
    tiles = vision_tiles(x, y, int(s.level[agent]), int(s.orient[agent]),
                         cfg.width, cfg.height)
    out = []
    for tx, ty in tiles:
        words = []
        for a in range(cfg.n_agents):
            if alive[a] and pos[a, 0] == tx and pos[a, 1] == ty:
                words.append("player")
        for res in range(C.N_RESOURCES):
            words.extend([C.RESOURCE_NAMES[res]] * int(grid[tx, ty, res]))
        out.append(words)
    return out


# ---------------------------------------------------------------- contract
@pytest.mark.parametrize("level,msg", [(1, None), (3, (4, 2)), (2, (0, 7))])
def test_build_obs_matches_sim_observe(level, msg):
    """The adapter's obs builder reproduces flatten_obs(observe(...)) EXACTLY
    when fed the equivalent server strings (the sim<->server contract)."""
    cfg = Z.make_cfg(8, 8, 2, no_food=True, no_refill=True)
    grid = np.zeros((8, 8, C.N_RESOURCES), np.int32)
    rng = np.random.default_rng(level)
    grid[:, :, :] = rng.integers(0, 3, size=grid.shape)  # busy world everywhere

    # agent1 inside agent0's cone (one tile ahead) + both on tile 0's stack
    pos0, o0 = (4, 4), C.SOUTH
    fwd = C.FORWARD_DELTA[o0]
    pos1 = ((pos0[0] + fwd[0]) % 8, (pos0[1] + fwd[1]) % 8)
    # counts 9 and 13 are float32-ULP-divergent between numpy /10.0 and XLA's
    # reciprocal multiply — they pin build_obs's *0.1 (bit-exact) choice.
    inv0 = [3, 9, 13, 2, 0, 0, 1]  # inv[food]=3 == cumulative takes (sim semantics)
    life0 = 4 * C.FOOD_LIFE_TICKS  # 504 ticks <-> server food stock of 4

    s = mk_state(
        cfg, [pos0, pos1], [o0, C.NORTH], level=[level, 1],
        grid=grid, inv=[inv0, [0] * 7], life=[life0, Z.START_LIFE],
        last_dir=[msg[0] if msg else -1, -1],
        last_tok=[msg[1] if msg else -1, -1],
    )
    want = np.asarray(flatten_obs(Z.observe(cfg, s)))[0]

    # server side: words per look tile -> wire round-trip -> builder
    tiles = parse_look(format_look(server_tiles_for(cfg, s, 0)))
    inv_dict = {n: q for n, q in zip(C.RESOURCE_NAMES, [4, 9, 13, 2, 0, 0, 1])}
    got = build_obs(tiles, inv_dict, level, o0, food_taken=3, msg=msg)

    assert got.shape == (OBS_DIM,) == want.shape
    np.testing.assert_array_equal(got, want)


def test_build_obs_life_clips_at_one():
    obs = build_obs([["player"]], {"food": 25}, 1, C.NORTH, food_taken=0)
    life_idx = 81 * 8 + C.MAX_LEVEL + C.N_RESOURCES + 4
    assert obs[life_idx] == 1.0  # sim: clip(life/1260, 0, 1)


def test_action_to_command_mapping():
    assert action_to_command(Z.ENV_FORWARD, 0) == "Forward"
    assert action_to_command(Z.ENV_RIGHT, 0) == "Right"
    assert action_to_command(Z.ENV_LEFT, 0) == "Left"
    for res in range(C.N_RESOURCES):
        assert action_to_command(Z.ENV_TAKE0 + res, 0) == f"Take {C.RESOURCE_NAMES[res]}"
        assert action_to_command(Z.ENV_SET0 + res, 0) == f"Set {C.RESOURCE_NAMES[res]}"
    assert action_to_command(Z.ENV_BROADCAST, 5) == "Broadcast 5"
    assert action_to_command(Z.ENV_INCANT, 0) == "Incantation"
    assert action_to_command(Z.ENV_IDLE, 0) is None


# ------------------------------------------------------------ client wiring
class StubPolicy:
    def __init__(self, actions):
        self.script = list(actions)  # [(action, token), ...]
        self.seen = []

    def act(self, obs):
        self.seen.append(np.array(obs))
        return self.script.pop(0)


INV_LINE = "[ food 9, linemate 0, deraumere 0, sibur 0, mendiane 0, phiras 0, thystame 0 ]\n"


def mk_client(policy, feed: bytes):
    a, b = socket.socketpair()
    b.sendall(feed)
    client = ZappyAIClient(LineSocket(a), "T1", policy)
    return client, b


def test_cycle_routes_async_consumes_msg_and_acts():
    pol = StubPolicy([(Z.ENV_FORWARD, 0)])
    feed = (
        b"message 3, 5\n"        # heard broadcast (async, before our responses)
        + INV_LINE.encode()
        + b"[ player,,, ]\n"     # level-1 look: 4 tiles (3 empty)
        + b"ok\n"                # Forward response
    )
    client, _ = mk_client(pol, feed)
    client.cycle()
    assert client.protocol_errors == []
    obs = pol.seen[0]
    msg_dir = obs[81 * 8 + Z.SELF_DIM : 81 * 8 + Z.SELF_DIM + 9]
    assert msg_dir[3] == 1.0           # K=3 delivered to the policy
    assert client.msg is None          # ...for exactly one decision
    ppl0 = obs[C.N_RESOURCES] * 5      # tile-0 player count (un-scale 0.2)
    assert ppl0 == pytest.approx(1.0)
    assert client.cycles == 1 and client.commands == 3


def test_involuntary_elevation_updates_level():
    pol = StubPolicy([(Z.ENV_IDLE, 0)])
    feed = (
        b"Elevation underway\n"   # dragged into a ritual before our Inventory
        b"Current level: 2\n"
        + INV_LINE.encode()
        + b"[ player,,,,,,,, ]\n"  # now level 2 -> 9 tiles
    )
    client, _ = mk_client(pol, feed)
    client.cycle()
    assert client.level == 2 and client.levelups == [2]
    assert client.protocol_errors == []


def test_turn_and_take_food_tracking():
    pol = StubPolicy([(Z.ENV_RIGHT, 0), (Z.ENV_TAKE0 + C.FOOD, 0)])
    cyc = INV_LINE.encode() + b"[ player,,, ]\n"
    client, _ = mk_client(pol, cyc + b"ok\n" + cyc + b"ok\n")
    client.cycle()
    assert client.orient == C.EAST     # N -> E (sim rotation convention)
    client.cycle()
    assert client.food_taken == 1
    assert client.protocol_errors == []


def test_out_of_vocab_broadcast_dropped():
    pol = StubPolicy([(Z.ENV_IDLE, 0)])
    feed = b"message 2, hello world\n" + INV_LINE.encode() + b"[ player,,, ]\n"
    client, _ = mk_client(pol, feed)
    client.cycle()
    assert client.msgs_heard == 1 and client.msg is None
    obs = pol.seen[0]
    assert obs[81 * 8 + Z.SELF_DIM :].sum() == 0.0  # no msg one-hots set


def test_unexpected_line_is_protocol_error_not_crash():
    pol = StubPolicy([(Z.ENV_IDLE, 0)])
    feed = b"garbage 42\n" + INV_LINE.encode() + b"[ player,,, ]\n"
    client, _ = mk_client(pol, feed)
    client.cycle()
    assert client.cycles == 1
    assert len(client.protocol_errors) == 1
    assert "garbage" in client.protocol_errors[0]


def test_dead_stops_client():
    pol = StubPolicy([])
    client, _ = mk_client(pol, b"dead\n")
    client.cycle()
    assert not client.alive and client.cycles == 0
    assert client.protocol_errors == []
    assert not client.disconnected  # death is a game outcome, not a link loss


def test_disconnect_is_flagged_not_crashed():
    """EOF/RST mid-game must end the client cleanly AND be distinguishable
    from 'played a full game' (gate FAILs on the disconnected flag)."""
    pol = StubPolicy([])
    client, peer = mk_client(pol, b"")
    peer.close()  # server gone
    client.cycle()
    assert not client.connected and client.disconnected
    assert client.alive  # we did not die in-game; the LINK died
    report = client.report()
    assert report["disconnected"] is True


def test_response_deadline_survives_async_flood():
    """A steady stream of async lines must not starve the deadline (was: the
    deadline was only checked when the socket went idle -> infinite hang)."""
    import threading

    pol = StubPolicy([])
    client, peer = mk_client(pol, b"")
    client.RESPONSE_TIMEOUT = 0.4  # instance override for the test

    stop = threading.Event()

    def flood():
        while not stop.is_set():
            try:
                peer.sendall(b"message 1, 5\n")
            except OSError:
                return
            time.sleep(0.01)  # well under any recv floor

    th = threading.Thread(target=flood, daemon=True)
    th.start()
    t0 = time.monotonic()
    out = client._response(lambda l: l in ("ok", "ko"), "Forward")
    elapsed = time.monotonic() - t0
    stop.set()
    th.join(timeout=1)
    assert out is None
    assert elapsed < 2.0, f"deadline starved: took {elapsed:.1f}s"
    assert any("timeout waiting for Forward" in e for e in client.protocol_errors)


def test_handshake_garbage_slots_raises_clean_runtime_error():
    a, b = socket.socketpair()
    b.sendall(b"WELCOME\nnot-a-number\n10 10\n")
    pol = StubPolicy([])
    client = ZappyAIClient(LineSocket(a), "T1", pol)
    with pytest.raises(RuntimeError, match="bad handshake"):
        client.handshake()


def test_racing_incantation_late_ack_is_not_a_protocol_error():
    """Live-probed (v3.0.1): when both agents send Incantation together, the
    loser's command is queued across the freeze and its 'ko' arrives AFTER the
    elevation lines — at the next perceive expect. Must be absorbed as a late
    ack, not a protocol error (was: cascaded into 'unexpected ok/ko')."""
    pol = StubPolicy([(Z.ENV_INCANT, 0), (Z.ENV_IDLE, 0)])
    feed = (
        INV_LINE.encode() + b"[ player,,, ]\n"
        # exact probed sequence for the racing (queued) initiator:
        b"Elevation underway\n"      # partner's ritual froze us
        b"Current level: 2\n"        # leveled via the partner's ritual
        b"ko\n"                      # OUR queued Incantation, start-check fail
        + INV_LINE.encode() + b"[ player,,,,,,,, ]\n"  # next cycle (L2: 9 tiles)
    )
    client, _ = mk_client(pol, feed)
    client.cycle()   # sends Incantation; consumes the elevation pair
    client.cycle()   # the orphaned 'ko' lands at the Inventory expect
    assert client.level == 2
    assert client.protocol_errors == []
    assert client.late_acks == 1


def test_incantation_flow():
    pol = StubPolicy([(Z.ENV_INCANT, 0)])
    feed = (
        INV_LINE.encode() + b"[ player,,, ]\n"
        b"Elevation underway\n"
        b"Current level: 2\n"
    )
    client, _ = mk_client(pol, feed)
    client.cycle()
    assert client.level == 2 and client.levelups == [2]
    assert client.protocol_errors == []


# ------------------------------------------------------------------ policy
needs_ckpt = pytest.mark.skipif(
    not (RUN_DIR / "params.msgpack").exists(), reason="no ritual8x8-v1 checkpoint"
)


@needs_ckpt
def test_policy_runner_loads_and_samples():
    hidden = json.loads((RUN_DIR / "config.json").read_text())["hidden"]
    pol = PolicyRunner(RUN_DIR / "params.msgpack", hidden=hidden, seed=0)
    h_before = np.array(pol.h)
    a, t = pol.act(np.zeros(OBS_DIM, np.float32))
    assert 0 <= a < Z.N_ENV_ACTIONS and 0 <= t < C.BROADCAST_VOCAB
    assert not np.array_equal(np.array(pol.h), h_before)  # carry advanced


@needs_ckpt
def test_policy_runner_rejects_wrong_hidden():
    with pytest.raises(ValueError, match="hidden"):
        PolicyRunner(RUN_DIR / "params.msgpack", hidden=64)

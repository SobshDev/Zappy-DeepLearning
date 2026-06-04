#!/usr/bin/env python3
"""Pin the sim<->server OBS CONTRACT against the live reference server.

The deploy adapter's ``build_obs`` (parsed Look/Inventory/message -> flat
policy input) must equal ``flatten_obs(Z.observe(...))`` on a mirrored sim
state — the unit test pins it against ``format_look`` strings; THIS tool pins
it against the *real server's* wire output, catching format quirks a
formatter round-trip can't.

Scenario (deterministic via the server stdin console):
  1. two AIs join; receiver tp'd to center, level 2 (9-tile cone), emitter
     tp'd onto a visible cone tile;
  2. every visible tile's 7 resource counts overwritten with a fixed pattern;
  3. receiver Takes one linemate (known inventory), tile pattern restored;
  4. receiver's real Inventory+Look -> build_obs  vs  sim observe() on the
     mirrored State -> must match EXACTLY (vision counts incl. both players,
     level one-hot, inv mapping incl. cumulative-food semantics, life, busy);
  5. emitter Broadcasts a token; heard ``message K, tok`` -> rebuilt obs with
     msg one-hots vs sim state with last_dir/last_tok -> exact match, and the
     server's K must equal ``broadcast.broadcast_direction``.

Run:  python tools/validate_obs_contract.py    (exit 0 = ALL PASS)
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from zappy_rl.algo.networks import flatten_obs  # noqa: E402
from zappy_rl.deploy.protocol import LineSocket, parse_broadcast, parse_inventory, parse_look  # noqa: E402
from zappy_rl.deploy.zappy_ai_adapter import build_obs  # noqa: E402
from zappy_rl.env import constants as C  # noqa: E402
from zappy_rl.env import zappy_env as Z  # noqa: E402
from zappy_rl.env.broadcast import broadcast_direction  # noqa: E402
from zappy_rl.env.vision import vision_tiles  # noqa: E402

HOST = "127.0.0.1"
PORT = 4244
W = H = 10
CX = CY = 5
LEVEL = 2  # 9-tile cone
TOKEN = 5


def wait_for_port(timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


class Server:
    def __init__(self, server_bin):
        cmd = [server_bin, "-p", str(PORT), "-x", str(W), "-y", str(H),
               "-n", "T1", "-c", "4", "-f", "100", "--auto-start", "on"]
        self.log = open("/tmp/zobs_contract_server.log", "w")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=self.log,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)

    def console(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def close(self):
        try:
            self.console("/quit")
            self.p.wait(timeout=3)
        except Exception:
            self.p.terminate()
        self.log.close()


def ai_connect(team="T1"):
    s = LineSocket.connect(HOST, PORT)
    assert s.recv_line(timeout=5) == "WELCOME"
    s.send(team)
    s.recv_line(timeout=5)  # slots
    s.recv_line(timeout=5)  # X Y
    return s


def ai_cmd(s, cmd, timeout=5.0):
    s.send(cmd)
    while True:
        line = s.recv_line(timeout=timeout)
        if line is None or line == "dead":
            return line
        if line.startswith("message ") or line.startswith("eject:"):
            continue
        return line


def gui_connect():
    g = LineSocket.connect(HOST, PORT)
    assert g.recv_line(timeout=5) == "WELCOME"
    g.send("GRAPHIC")
    return g


def drain_players(gui, secs=0.6):
    players: dict[int, dict] = {}
    end = time.monotonic() + secs
    while time.monotonic() < end:
        line = gui.recv_line(timeout=0.2)
        if line is None:
            continue
        tok = line.split()
        if tok[0] == "pnw":
            players[int(tok[1][1:])] = {"x": int(tok[2]), "y": int(tok[3]),
                                        "o": int(tok[4]), "l": int(tok[5])}
        elif tok[0] == "ppo":
            players.setdefault(int(tok[1][1:]), {}).update(
                x=int(tok[2]), y=int(tok[3]), o=int(tok[4]))
        elif tok[0] == "plv":
            players.setdefault(int(tok[1][1:]), {})["l"] = int(tok[2])
    return players


def pattern_qty(i: int, res: int) -> int:
    """Deterministic per-(visible tile, resource) count; tile 0 gets >=1
    linemate so the receiver's Take has something to grab."""
    return (i + res) % 3


def compare(name: str, got: np.ndarray, want: np.ndarray) -> bool:
    ok = bool(np.array_equal(got, want))
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        nvis = Z.MAX_VISION_TILES * (C.N_RESOURCES + 1)
        secs = {"vision": (0, nvis), "self": (nvis, nvis + Z.SELF_DIM),
                "msg_dir": (nvis + Z.SELF_DIM, nvis + Z.SELF_DIM + 9),
                "msg_tok": (nvis + Z.SELF_DIM + 9, len(want))}
        for sec, (a, b) in secs.items():
            bad = np.flatnonzero(got[a:b] != want[a:b])
            if bad.size:
                print(f"  {sec}: {bad.size} mismatches, first at +{bad[0]}: "
                      f"adapter={got[a + bad[0]]} sim={want[a + bad[0]]}")
    return ok


def main():
    plat = "linux" if platform.system() == "Linux" else "macos"
    srv = Server(f"reference/{plat}/zappy_server")
    try:
        if not wait_for_port():
            print("server never listened; see /tmp/zobs_contract_server.log")
            return 1
        gui = gui_connect()
        time.sleep(0.3)
        drain_players(gui, 0.8)

        rcv = ai_connect()
        time.sleep(0.3)
        rid = max(drain_players(gui, 0.8))
        emt = ai_connect()
        time.sleep(0.3)
        eid = max(drain_players(gui, 0.8))
        print(f"receiver #{rid}, emitter #{eid}")

        srv.console("/noFood true")
        srv.console("/noRefill true")
        srv.console(f"/tp {rid} {CX} {CY}")
        srv.console(f"/setLevel {rid} {LEVEL}")
        time.sleep(0.4)
        st = drain_players(gui, 0.8)
        o_r = st[rid]["o"]

        cone = vision_tiles(CX, CY, LEVEL, o_r, W, H)
        ex, ey = int(cone[4][0]), int(cone[4][1])  # middle of row 2
        srv.console(f"/tp {eid} {ex} {ey}")

        # overwrite every visible tile with the pattern (incl. food counts)
        for i, (tx, ty) in enumerate(cone):
            for res in range(C.N_RESOURCES):
                srv.console(f"/setTile {C.RESOURCE_NAMES[res]} {pattern_qty(i, res)} "
                            f"{int(tx)} {int(ty)}")
        time.sleep(0.6)
        drain_players(gui, 0.3)

        # known inventory: one taken linemate (tile 0 pattern has >=1), restore
        take = ai_cmd(rcv, "Take linemate")
        assert take == "ok", f"Take linemate -> {take!r}"
        srv.console(f"/setTile linemate {pattern_qty(0, C.LINEMATE)} {CX} {CY}")
        time.sleep(0.3)

        inv = parse_inventory(ai_cmd(rcv, "Inventory"))
        look_line = ai_cmd(rcv, "Look")
        tiles = parse_look(look_line)
        print(f"Inventory -> {inv}")
        print(f"Look -> {look_line}")

        # ----- mirrored sim state (the contract's other side) -----
        grid = np.zeros((W, H, C.N_RESOURCES), np.int32)
        for i, (tx, ty) in enumerate(cone):
            for res in range(C.N_RESOURCES):
                grid[int(tx), int(ty), res] = pattern_qty(i, res)
        cfg = Z.make_cfg(W, H, 2, no_food=True, no_refill=True)
        inv0 = [0] * C.N_RESOURCES
        inv0[C.LINEMATE] = 1
        food_stock = inv.get("food", 0)

        def sim_obs(last_dir=-1, last_tok=-1):
            s = Z.State(
                grid=jnp.asarray(grid),
                pos=jnp.asarray([[CX, CY], [ex, ey]], jnp.int32),
                orient=jnp.asarray([o_r, 1], jnp.int32),
                level=jnp.asarray([LEVEL, 1], jnp.int32),
                inv=jnp.asarray([inv0, [0] * 7], jnp.int32),
                life=jnp.asarray([food_stock * C.FOOD_LIFE_TICKS, Z.START_LIFE], jnp.int32),
                alive=jnp.ones(2, bool), busy_until=jnp.zeros(2, jnp.int32),
                pending=jnp.zeros(2, bool), incant_level=jnp.zeros(2, jnp.int32),
                initiator=jnp.zeros(2, bool), team=jnp.zeros(2, jnp.int32),
                now=jnp.int32(0), key=jnp.zeros(2, jnp.uint32),
                last_dir=jnp.asarray([last_dir, -1], jnp.int32),
                last_tok=jnp.asarray([last_tok, -1], jnp.int32),
            )
            return np.asarray(flatten_obs(Z.observe(cfg, s)))[0]

        ok_static = compare(
            "VISION+SELF",
            build_obs(tiles, inv, LEVEL, o_r, food_taken=0),
            sim_obs(),
        )

        # ----- broadcast leg -----
        while rcv.recv_line(timeout=0.05) is not None:
            pass  # drain stale lines
        ai_cmd(emt, f"Broadcast {TOKEN}")
        got_k = got_tok = None
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end:
            line = rcv.recv_line(timeout=0.3)
            if line and line.startswith("message "):
                got_k, text = parse_broadcast(line)
                got_tok = int(text)
                break
        pred_k = broadcast_direction(ex, ey, CX, CY, o_r, W, H)
        print(f"broadcast: server K={got_k} predicted K={pred_k} tok={got_tok}")
        ok_k = got_k == pred_k and got_tok == TOKEN
        print(f"BROADCAST K: {'PASS' if ok_k else 'FAIL'}")
        ok_msg = False
        if ok_k:
            ok_msg = compare(
                "OBS WITH MESSAGE",
                build_obs(tiles, inv, LEVEL, o_r, food_taken=0, msg=(got_k, got_tok)),
                sim_obs(last_dir=got_k, last_tok=got_tok),
            )

        rcv.close(); emt.close(); gui.close()
        allok = ok_static and ok_k and ok_msg
        print("\nSUMMARY:", "ALL PASS" if allok else "FAILURES — see above")
        return 0 if allok else 2
    finally:
        srv.close()


if __name__ == "__main__":
    raise SystemExit(main())

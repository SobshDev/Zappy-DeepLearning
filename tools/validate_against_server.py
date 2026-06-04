#!/usr/bin/env python3
"""Pin the `# VERIFY` geometry conventions against the live reference server.

Two checks, each set up deterministically via the server's stdin console:

  VISION   place one player, raise its level, tag each predicted look-tile with
           a distinct thystame count, issue `Look`, and confirm tile i reports
           exactly i thystames -> validates the vision-cone orientation/order in
           ``env/vision.py`` + ``env/constants.py`` FORWARD/RIGHT_DELTA.

  BROADCAST put a receiver at center, move an emitter around its 8 neighbours,
           broadcast, read the receiver's ``message K`` and compare to
           ``env/broadcast.py``.

Prints PASS/FAIL and the exact mismatch table so the constants can be corrected.
Run:  python tools/validate_against_server.py
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zappy_rl.deploy.protocol import LineSocket, parse_look  # noqa: E402
from zappy_rl.env import constants as C  # noqa: E402
from zappy_rl.env.broadcast import broadcast_direction  # noqa: E402
from zappy_rl.env.vision import vision_tiles  # noqa: E402

HOST = "127.0.0.1"
PORT = 4243
W = H = 21
CX = CY = 10


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
               "-n", "T1", "T2", "-c", "8", "-f", "100", "--auto-start", "on"]
        self.log = open("/tmp/zvalidate_server.log", "w")
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
    """Send a command, return the first non-async response line."""
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
    """Read GUI events for `secs`, return {pid: {'x','y','o','l'}}."""
    players: dict[int, dict] = {}
    end = time.monotonic() + secs
    while time.monotonic() < end:
        line = gui.recv_line(timeout=0.2)
        if line is None:
            continue
        tok = line.split()
        if tok[0] == "pnw":  # pnw #n X Y O L N
            pid = int(tok[1][1:])
            players[pid] = {"x": int(tok[2]), "y": int(tok[3]), "o": int(tok[4]), "l": int(tok[5])}
        elif tok[0] == "ppo":  # ppo #n X Y O
            pid = int(tok[1][1:])
            players.setdefault(pid, {}).update(x=int(tok[2]), y=int(tok[3]), o=int(tok[4]))
        elif tok[0] == "plv":  # plv #n L
            pid = int(tok[1][1:])
            players.setdefault(pid, {})["l"] = int(tok[2])
    return players


def validate_vision(srv, gui):
    print("\n=== VISION ===")
    ai = ai_connect("T1")
    time.sleep(0.3)
    st = drain_players(gui, 0.8)
    pid = max(st)  # newest player
    srv.console("/noFood true")
    srv.console("/noRefill true")
    srv.console(f"/tp {pid} {CX} {CY}")
    srv.console(f"/setLevel {pid} 3")
    time.sleep(0.4)
    st = drain_players(gui, 0.8)
    p = st[pid]
    level, o = p["l"], p["o"]
    x, y = CX, CY
    tiles = vision_tiles(x, y, level, o, W, H)
    print(f"player #{pid} at ({x},{y}) O={o} level={level} -> {len(tiles)} look tiles")

    # Tag tile i with thystame x i (skip self tile 0).
    for i, (tx, ty) in enumerate(tiles):
        srv.console(f"/setTile thystame {i} {int(tx)} {int(ty)}")
    time.sleep(0.5)
    drain_players(gui, 0.3)

    resp = ai_cmd(ai, "Look")
    parsed = parse_look(resp)
    actual = [t.count("thystame") for t in parsed]
    expected = list(range(len(tiles)))
    ok = actual == expected
    print(f"Look -> {resp}")
    print(f"thystame counts per index: actual={actual}")
    print(f"                          expected={expected}")
    if not ok:
        print("MISMATCH: vision orientation/order is wrong. Per-index (expected->actual):")
        for i, (e, a) in enumerate(zip(expected, actual)):
            flag = "" if e == a else "  <-- "
            print(f"  idx {i:2d}: {e:2d} -> {a:2d}{flag}")
    ai.close()
    print("VISION:", "PASS" if ok else "FAIL")
    return ok


def validate_broadcast(srv, gui):
    print("\n=== BROADCAST ===")
    rcv = ai_connect("T1")
    emt = ai_connect("T1")
    time.sleep(0.4)
    st = drain_players(gui, 1.0)
    ids = sorted(st)[-2:]
    rid, eid = ids[0], ids[1]
    srv.console("/noFood true")
    srv.console(f"/tp {rid} {CX} {CY}")
    time.sleep(0.3)
    st = drain_players(gui, 0.6)
    ro = st[rid]["o"]
    print(f"receiver #{rid} at ({CX},{CY}) O={ro}; emitter #{eid}")

    neighbours = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    rows = []
    all_ok = True
    for ox, oy in neighbours:
        ex, ey = (CX + ox) % W, (CY + oy) % H
        srv.console(f"/tp {eid} {ex} {ey}")
        time.sleep(0.25)
        # drain any stale messages on receiver
        while rcv.recv_line(timeout=0.05) is not None:
            pass
        ai_cmd(emt, "Broadcast ping")
        got = None
        t_end = time.monotonic() + 1.5
        while time.monotonic() < t_end:
            line = rcv.recv_line(timeout=0.3)
            if line and line.startswith("message "):
                # "message K, text"
                got = int(line.split()[1].rstrip(","))
                break
        pred = broadcast_direction(ex, ey, CX, CY, ro, W, H)
        ok = got == pred
        all_ok &= ok
        rows.append((ox, oy, pred, got, ok))

    print(f"{'offset':>10} {'predicted':>10} {'server':>8}  ok")
    for ox, oy, pred, got, ok in rows:
        print(f"  ({ox:+d},{oy:+d}) {pred:>10} {str(got):>8}  {'OK' if ok else 'XX'}")
    rcv.close(); emt.close()
    print("BROADCAST:", "PASS" if all_ok else "FAIL")
    return all_ok


def validate_incantation(srv, gui):
    print("\n=== INCANTATION (level 1 -> 2) ===")
    from zappy_rl.env.reference_env import ZappyReferenceEnv

    ai = ai_connect("T1")
    time.sleep(0.3)
    st = drain_players(gui, 0.8)
    pid = max(st)
    srv.console("/noFood true")
    srv.console(f"/tp {pid} {CX} {CY}")
    srv.console(f"/setLevel {pid} 1")
    # 1->2 needs exactly 1 linemate on the tile and 1 player.
    srv.console(f"/setTile linemate 1 {CX} {CY}")
    time.sleep(0.5)
    drain_players(gui, 0.3)

    ai.send("Incantation")
    l1 = ai.recv_line(timeout=2)
    l2 = None
    t_end = time.monotonic() + 6
    while time.monotonic() < t_end:
        line = ai.recv_line(timeout=0.5)
        if line and line.startswith("Current level"):
            l2 = line
            break
    st = drain_players(gui, 1.0)
    server_level = st.get(pid, {}).get("l")
    # AI Look should no longer show linemate on its tile (consumed).
    look = parse_look(ai_cmd(ai, "Look"))
    consumed = "linemate" not in look[0]

    # Oracle prediction.
    env = ZappyReferenceEnv(W, H, teams=("T1",), no_food=True, no_refill=True, seed=0)
    op = env.spawn("T1", CX, CY, level=1)
    env.grid[CX, CY, C.LINEMATE] = 1
    oracle_result = env.do(op, C.A_INCANTATION)

    print(f"server: start={l1!r} end={l2!r} level={server_level} linemate_consumed={consumed}")
    print(f"oracle: result={oracle_result} (level -> {env.players[op].level})")
    ok = (
        l1 == "Elevation underway"
        and l2 == "Current level: 2"
        and server_level == 2
        and consumed
        and oracle_result == 2
    )
    ai.close()
    print("INCANTATION:", "PASS" if ok else "FAIL")
    return ok


def main():
    plat = "linux" if platform.system() == "Linux" else "macos"
    server_bin = f"reference/{plat}/zappy_server"
    srv = Server(server_bin)
    try:
        if not wait_for_port():
            print("server never listened; see /tmp/zvalidate_server.log")
            return 1
        gui = gui_connect()
        time.sleep(0.3)
        drain_players(gui, 0.8)  # consume initial burst
        v = validate_vision(srv, gui)
        b = validate_broadcast(srv, gui)
        i = validate_incantation(srv, gui)
        gui.close()
        allok = v and b and i
        print("\nSUMMARY:", "ALL PASS" if allok else "FAILURES — see above")
        return 0 if allok else 2
    finally:
        srv.close()


if __name__ == "__main__":
    raise SystemExit(main())

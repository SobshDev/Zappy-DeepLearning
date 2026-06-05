#!/usr/bin/env python3
"""Phase-4 gate: a frozen policy plays a full game on the reference server
with ZERO protocol errors.

Launches the reference server (10x10, one team, 2 slots, f=100 — the server
rejects maps under 10 wide, so this is the closest legal geometry to the 8x8
training map; the policy's inputs are fully egocentric and per-tile resource
densities are identical, so the shift is mild), spawns one adapter subprocess
per slot (each a real
``zappy_ai`` TCP client running the frozen ritual8x8-v1 actor on CPU), and
watches the game through a GRAPHIC connection for independent, server-side
ground truth (levels, rituals, broadcasts, deaths).

"Full game" = both agents play until death or the time budget — with 2 slots
the 6xL8 win condition is unreachable, so the budget bounds the session.

PASS requires ALL of (the protocol-error count is self-reported by the
adapters, so the rest exists to catch what self-grading can't):
  * every agent wrote a report with n_protocol_errors == 0;
  * no agent process crashed (only exit 0/3 are adapter-controlled);
  * no agent lost its connection mid-game (``disconnected`` flag — a link
    drop after 3 cycles must not read as "played a full game");
  * the GUI-observed max level per player matches the adapters' self-reported
    final levels (an end-to-end desync — e.g. responses attributed to the
    wrong command — would show up as level disagreement here).

Note ``server_side.pdi_events``: the reference server emits ``pdi`` on ANY
disconnect including our own end-of-session close, so pdi is NOT evidence of
in-game death; the in-band ``dead`` line (agents' ``alive`` field) is.

Writes ``<run_dir>/deploy_gate.json`` next to the checkpoint.

Run:  python tools/run_deploy_gate.py --duration 180
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zappy_rl.deploy.protocol import LineSocket  # noqa: E402

HOST = "127.0.0.1"


def wait_for_port(port, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with socket.create_connection((HOST, port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


class GuiWatcher(threading.Thread):
    """GRAPHIC client: server-side ground truth, independent of the adapters."""

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.stop_flag = threading.Event()
        self.levels: dict[int, int] = {}      # pid -> max level seen
        self.t0 = time.monotonic()            # reset when the watcher connects
        self.levels_timeline: list[dict] = []  # {"t_s", "pid", "level"} per climb
        self.pdi_events: list[int] = []       # ANY disconnect fires pdi too
        self.pdi_timeline: list[dict] = []     # {"t_s", "pid"} per death
        self.rituals_log: list[dict] = []      # pic/pie events with timestamps
        self.food_timeline: list[dict] = []    # {"t_s", "pid", "food"} ~2s polls
        self.pids: set[int] = set()
        self.rituals_started = 0
        self.rituals_ok = 0
        self.broadcasts = 0
        self.game_end: str | None = None
        self.parse_errors = 0
        self.died: Exception | None = None    # watcher must not die silently

    def run(self):
        try:
            gui = LineSocket.connect(HOST, self.port)
            assert gui.recv_line(timeout=5) == "WELCOME"
            gui.send("GRAPHIC")
            self.t0 = time.monotonic()  # clock starts at the GRAPHIC handshake
            last_pin = 0.0
            while not self.stop_flag.is_set():
                # ~2s server-truth inventory polls: pin replies carry food,
                # the decisive stat for mid-freeze starvation diagnosis
                now = time.monotonic()
                if now - last_pin >= 2.0 and self.pids:
                    for pid in sorted(self.pids):
                        gui.send(f"pin #{pid}")
                    last_pin = now
                line = gui.recv_line(timeout=0.5)
                if line is None:
                    continue
                try:
                    self._handle(line)
                except (ValueError, IndexError):
                    # one malformed line must not kill the only external check
                    self.parse_errors += 1
            gui.close()
        except Exception as e:  # surfaced in the report, never silent
            self.died = e

    def _handle(self, line: str):
        tok = line.split()
        if not tok:
            return
        t_s = round(time.monotonic() - self.t0, 2)
        if tok[0] in ("pnw", "plv"):
            pid = int(tok[1][1:])
            lvl = int(tok[5] if tok[0] == "pnw" else tok[2])
            if tok[0] == "pnw":
                self.pids.add(pid)
            # timeline check vs .get(pid, 0) so the first pnw (level 1) is
            # recorded too; the `levels` floor of 1 below is unchanged.
            if lvl > self.levels.get(pid, 0):
                self.levels_timeline.append({"t_s": t_s, "pid": pid, "level": lvl})
            self.levels[pid] = max(self.levels.get(pid, 1), lvl)
        elif tok[0] == "pin":
            # pin #n X Y q0..q6 — q0 is food (server truth, not dead-reckoned)
            self.food_timeline.append(
                {"t_s": t_s, "pid": int(tok[1][1:]), "food": int(tok[4])})
        elif tok[0] == "pdi":
            pid = int(tok[1][1:])
            self.pdi_events.append(pid)
            self.pdi_timeline.append({"t_s": t_s, "pid": pid})
        elif tok[0] == "pic":
            self.rituals_started += 1
            # pic X Y L #n... — tile, ritual level, participant pids
            self.rituals_log.append(
                {"t_s": t_s, "ev": "start", "x": int(tok[1]), "y": int(tok[2]),
                 "level": int(tok[3]),
                 "pids": [int(t[1:]) for t in tok[4:] if t.startswith("#")]})
        elif tok[0] == "pie":
            ok = int(tok[3] == "1") if len(tok) > 3 else 0
            self.rituals_ok += ok
            self.rituals_log.append(
                {"t_s": t_s, "ev": "end", "x": int(tok[1]), "y": int(tok[2]),
                 "ok": bool(ok)})
        elif tok[0] == "pbc":
            self.broadcasts += 1
        elif tok[0] == "seg":
            self.game_end = line


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default="runs/ritual8x8-v1/params.msgpack")
    ap.add_argument("--port", type=int, default=4245)
    # server minimum is 10 (8x8 rejected: "Value must be between 10 and 42")
    ap.add_argument("--width", type=int, default=10)
    ap.add_argument("--height", type=int, default=10)
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--team", default="T1")
    ap.add_argument("--freq", type=int, default=100)
    ap.add_argument("--duration", type=float, default=180.0, help="seconds")
    ap.add_argument("--greedy", action="store_true",
                    help="agents use argmax actions instead of sampling")
    args = ap.parse_args()

    run_dir = Path(args.params).parent
    plat = "linux" if platform.system() == "Linux" else "macos"
    server_bin = f"reference/{plat}/zappy_server"

    cmd = [server_bin, "-p", str(args.port), "-x", str(args.width),
           "-y", str(args.height), "-n", args.team, "-c", str(args.agents),
           "-f", str(args.freq), "--auto-start", "on"]
    print(f"[gate] server: {' '.join(cmd)}")
    slog = open("/tmp/zgate_server.log", "w")
    # stdin kept open: the reference server quits on stdin EOF
    srv = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=slog,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        if not wait_for_port(args.port):
            print("[gate] server never listened; see /tmp/zgate_server.log")
            return 1
        gui = GuiWatcher(args.port)
        gui.start()
        time.sleep(0.3)

        env = {**os.environ, "JAX_PLATFORMS": "cpu",
               "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        procs, report_files, logs = [], [], []
        for i in range(args.agents):
            rep = f"/tmp/zgate_agent{i}.json"
            log = open(f"/tmp/zgate_agent{i}.log", "w")
            p = subprocess.Popen(
                [sys.executable, "-m", "zappy_rl.deploy.zappy_ai_adapter",
                 "--host", HOST, "--port", str(args.port), "--team", args.team,
                 "--params", args.params, "--seed", str(i),
                 "--duration", str(args.duration), "--report-json", rep]
                + (["--greedy"] if args.greedy else []),
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            procs.append(p)
            report_files.append(rep)
            logs.append(log)
            print(f"[gate] agent {i} started (pid {p.pid})")

        t0 = time.monotonic()
        for p in procs:
            left = max(args.duration + 90 - (time.monotonic() - t0), 10)
            try:
                p.wait(timeout=left)
            except subprocess.TimeoutExpired:
                p.terminate()
        elapsed = time.monotonic() - t0
        time.sleep(0.5)
        gui.stop_flag.set()
        gui.join(timeout=3)
        for log in logs:
            log.close()

        reports = []
        for i, rep in enumerate(report_files):
            if Path(rep).exists():
                reports.append(json.loads(Path(rep).read_text()))
            else:
                reports.append({"error": f"agent {i} produced no report",
                                "n_protocol_errors": -1})

        n_errors = sum(r.get("n_protocol_errors", -1) for r in reports)
        crashed = [i for i, p in enumerate(procs) if p.returncode not in (0, 3)]
        dropped = [i for i, r in enumerate(reports) if r.get("disconnected", True)]
        # independent cross-check: server-observed levels == self-reported.
        # (catches end-to-end desync the self-graded error count can't see)
        gui_levels = sorted(gui.levels.values())
        self_levels = sorted(r.get("final_level", -1) for r in reports)
        levels_match = gui_levels == self_levels and gui.died is None
        ok = n_errors == 0 and not crashed and not dropped and levels_match
        fail = (f"FAIL: protocol_errors={n_errors} crashed={crashed} "
                f"disconnected={dropped} levels_match={levels_match}")
        # live climb timing from the GUI timeline (same tick basis as
        # approx_ticks: t_s * freq)
        to_ticks = lambda t: None if t is None else round(t * args.freq)  # noqa: E731
        l8_times = [e["t_s"] for e in gui.levels_timeline if e["level"] >= 8]
        t_first_l8 = min(l8_times) if l8_times else None
        pid_l8 = {}  # pid -> first t_s at >=8
        for e in gui.levels_timeline:
            if e["level"] >= 8 and e["pid"] not in pid_l8:
                pid_l8[e["pid"]] = e["t_s"]
        all_pids = {e["pid"] for e in gui.levels_timeline} | set(gui.levels)
        t_all_l8 = (max(pid_l8.values())
                    if all_pids and pid_l8.keys() == all_pids else None)
        result = {
            "gate": ("PASS: frozen policy played a full game on the reference "
                     "server with zero protocol errors" if ok else fail),
            "params": args.params,
            "duration_s": round(elapsed, 1),
            "approx_ticks": int(elapsed * args.freq),
            "levels_timeline": gui.levels_timeline,
            "t_first_l8_s": t_first_l8,
            "t_first_l8_ticks": to_ticks(t_first_l8),
            "t_all_l8_s": t_all_l8,
            "t_all_l8_ticks": to_ticks(t_all_l8),
            "levels_cross_check": {"gui": gui_levels, "self_reported": self_levels,
                                   "match": levels_match},
            "agents": reports,
            "server_side": {
                "max_level_per_player": gui.levels,
                # pdi fires on ANY disconnect (incl. our end-of-session close);
                # in-game death truth is the agents' own "alive" field.
                "pdi_events": gui.pdi_events,
                "pdi_timeline": gui.pdi_timeline,
                "rituals_started": gui.rituals_started,
                "rituals_succeeded": gui.rituals_ok,
                "rituals_log": gui.rituals_log,
                "food_timeline": gui.food_timeline,
                "broadcasts": gui.broadcasts,
                "game_end": gui.game_end,
                "watcher_parse_errors": gui.parse_errors,
                "watcher_died": repr(gui.died) if gui.died else None,
            },
        }
        out = run_dir / "deploy_gate.json"
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        # compact live-climb summary (first agent per tier, tiers >= 5)
        first_at = {}
        for e in gui.levels_timeline:
            first_at.setdefault(e["level"], e["t_s"])
        climb = "  ".join(f"L{lv}@{first_at[lv]:.0f}s"
                          for lv in sorted(first_at) if lv >= 5)
        if climb:
            print(f"[gate] live climb: {climb} (first agent)")
        print(f"\n[gate] {'PASS' if ok else 'FAIL'} — report at {out}")
        return 0 if ok else 2
    finally:
        try:
            srv.stdin.write("/quit\n")
            srv.stdin.flush()
            srv.wait(timeout=3)
        except Exception:
            srv.terminate()
        slog.close()


if __name__ == "__main__":
    raise SystemExit(main())

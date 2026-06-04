#!/usr/bin/env python3
"""Capture reference-server golden traces for sim validation.

Launches the reference ``zappy_server`` (keeping its stdin open so its command
console stays alive and it doesn't shut down on EOF), connects a GRAPHIC
recorder that logs every GUI event to newline-delimited JSON, and drives N
greedy scripted AIs to create activity. Optionally scripts an incantation via
the server's stdin console so traces include ``pic``/``pie``.

Usage:
    python tools/capture_golden_traces.py \
        --server-bin reference/macos/zappy_server \
        -x 10 -y 10 -n T1 T2 -c 6 -f 100 --ai 3 --seconds 8 \
        --out traces/run1.ndjson

The output NDJSON lines are ``{"t": <seconds>, "ev": "<gui event line>"}`` and
become the ground truth for ``tests/test_differential.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zappy_rl.deploy.protocol import LineSocket  # noqa: E402
from zappy_rl.eval.scripted_ai import ScriptedAI  # noqa: E402


def wait_for_port(host: str, port: int, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def record_gui(host: str, port: int, out_path: str, t0: float, stop: threading.Event) -> None:
    gui = LineSocket.connect(host, port)
    if gui.recv_line(timeout=5) != "WELCOME":
        raise RuntimeError("GUI: no WELCOME")
    gui.send("GRAPHIC")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w") as fh:
        while not stop.is_set():
            line = gui.recv_line(timeout=0.5)
            if line is None:
                continue
            fh.write(json.dumps({"t": round(time.monotonic() - t0, 4), "ev": line}) + "\n")
            n += 1
    gui.close()
    print(f"[recorder] wrote {n} events to {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", default="reference/macos/zappy_server")
    ap.add_argument("-p", "--port", type=int, default=4242)
    ap.add_argument("-x", "--width", type=int, default=10)
    ap.add_argument("-y", "--height", type=int, default=10)
    ap.add_argument("-n", "--names", nargs="+", default=["T1", "T2"])
    ap.add_argument("-c", "--clients", type=int, default=6)
    ap.add_argument("-f", "--freq", type=int, default=100)
    ap.add_argument("--ai", type=int, default=3, help="scripted AIs on the first team")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--scenario-incantation", action="store_true",
                    help="script an L1->L2 incantation via the server console")
    ap.add_argument("--out", default="traces/golden.ndjson")
    args = ap.parse_args()

    host = "127.0.0.1"
    cmd = [
        args.server_bin, "-p", str(args.port), "-x", str(args.width),
        "-y", str(args.height), "-n", *args.names, "-c", str(args.clients),
        "-f", str(args.freq), "--auto-start", "on",
    ]
    print("[server]", " ".join(cmd))
    # stdin=PIPE keeps the console alive (server quits on stdin EOF) and lets us
    # issue console commands; line-buffered text mode.
    srv = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    try:
        if not wait_for_port(host, args.port):
            print("[error] server never listened", file=sys.stderr)
            return 1

        t0 = time.monotonic()
        stop = threading.Event()
        rec = threading.Thread(target=record_gui, args=(host, args.port, args.out, t0, stop), daemon=True)
        rec.start()

        deadline = t0 + args.seconds
        ais = [ScriptedAI(host, args.port, args.names[0], seed=i) for i in range(args.ai)]
        ai_threads = [threading.Thread(target=a.run, args=(deadline,), daemon=True) for a in ais]
        for th in ai_threads:
            th.start()

        if args.scenario_incantation and srv.stdin:
            # Give an AI level-1 prerequisites and trigger an elevation in place.
            time.sleep(min(2.0, args.seconds / 2))
            srv.stdin.write("/setInventory 0 linemate 1\n")
            srv.stdin.write("/incantate 0 0\n")
            srv.stdin.flush()

        for th in ai_threads:
            th.join()
        stop.set()
        rec.join(timeout=2)
        for a in ais:
            a.close()
        return 0
    finally:
        try:
            if srv.stdin:
                srv.stdin.write("/quit\n")
                srv.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        try:
            srv.wait(timeout=3)
        except subprocess.TimeoutExpired:
            srv.terminate()


if __name__ == "__main__":
    raise SystemExit(main())

# Zappy — Deep-RL Civilisation

Train a multi-agent "civilisation" to play the Epitech **Zappy** game by
reinforcement learning + self-play, then deploy frozen policies to the
**reference server** and render Yosh/Trackmania-style "watching it learn"
visuals through the reference GUI.

Full design + rationale: `~/.claude/plans/i-want-you-to-floofy-hammock.md`.

> **Core idea:** the real game runs over a single-threaded TCP server (~10,000×
> too slow for RL), so we train in a fast headless **JAX clone** and only touch
> the real server through a thin `zappy_ai` adapter for validation + visuals.

## Status

| Phase | State |
|---|---|
| 0 — Reference ground truth + scaffold | ✅ done & verified live |
| 1 — JAX env + validation suite | ✅ JAX `vmap` env done, cross-checked vs oracle (fork/eject deferred to v2) |
| 2 — Single-agent foraging MAPPO | 🚧 next |
| 3 — Cooperative ritual + emergent broadcast | ⬜ |
| 4 — League + telemetry + deploy adapter | ⬜ |
| 5 — 2-week run orchestration + viz | ⬜ |

**Verified so far:** 39 offline tests (geometry, protocol, all core rules, JAX↔oracle cross-check) + live server agreement on vision (16/16 tiles), broadcast (8/8 directions, 2 orientations), and the L1→L2 incantation (elevation + stone consumption). JAX env throughput: **~135k env-steps/s on Apple-Silicon CPU** (vmap × 2048 worlds) — far higher expected on the 4090.

## Setup

```sh
tar -xf zappy_ref-v3.0.1.tar -C reference     # unpack reference server + GUI
uv venv --python 3.11 .venv
uv pip install --python .venv numpy pytest    # core dev deps
# training stack (on the RTX 4090 Linux box): uv pip install -e '.[train,viz]'
```

## Run

```sh
.venv/bin/python -m pytest -q                 # geometry + protocol tests

# capture reference-server golden traces (ground truth for the JAX sim):
.venv/bin/python tools/capture_golden_traces.py --ai 3 --seconds 8 --out traces/run1.ndjson

# pin vision + broadcast geometry against the live server (must print ALL PASS):
.venv/bin/python tools/validate_against_server.py
```

## Confirmed reference protocol (v3.0.1, verified live)

**AI handshake:** `WELCOME` → client `<TEAM>` → server `<slots>` then `<X> <Y>` (each on its own line).

**Action costs** (seconds = ticks ÷ `f`, default `f=100`): Forward/Right/Left/Look/Broadcast/Eject/Take/Set = `7`, Inventory = `1`, **Fork = `42`**, **Incantation = `300`**.

**Inventory:** `[ food 9, linemate 0, deraumere 0, sibur 0, mendiane 0, phiras 0, thystame 0 ]`

**Look (level 1):** `[ player food, mendiane,, food linemate ]` — `(L+1)²` tiles, tile 0 = self, resource order `food,linemate,deraumere,sibur,mendiane,phiras,thystame`.

**GUI stream:** `msz·sgt·bct·tna·pnw·ppo·plv·pin·pex·pbc·pic·pie·pfk·pdr·pgt·pdi·enw·ebo·edi·sst·seg·smg` (eggs carry `#-1` parent when pre-seeded).

**Server stdin console** (keep stdin open or the server quits on EOF): `/setLevel`, `/setInventory`, `/tp`, `/incantate`, `/fork`, `/setTile`, `/noFood`, `/noRefill`, `/setFreq`, `/pause`, `/start`, `/quit`. Used to script deterministic validation + eval scenarios.

## Layout

```
zappy_rl/
  env/      constants.py · vision.py · broadcast.py · reference_env.py (oracle) · zappy_env.py (JAX)
  deploy/   protocol.py                                (wire parsing; only TCP code)
  eval/     scripted_ai.py                             (greedy baseline)
tools/      capture_golden_traces.py · validate_against_server.py
tests/      test_vision.py · test_broadcast.py · test_protocol.py · test_reference_env.py
reference/  unpacked zappy_ref-v3.0.1.tar
```

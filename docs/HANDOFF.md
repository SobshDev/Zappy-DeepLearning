# Zappy Deep-RL — Handoff / State

> Read this first, then `docs/PLAN.md` for the full design. This file is the
> ground truth for **what exists, what's verified, and what to do next** —
> including exact setup steps for the RTX 4090 Linux box.

## 1. What this project is

Train a multi-agent "civilisation" to play the Epitech **Zappy** game by
RL + self-play, then deploy frozen policies to the **reference server** and
render Yosh/Trackmania-style "watching it learn" visuals via the reference GUI.

**Core idea:** the real game is a single-threaded TCP server (~10⁴× too slow for
RL), so we train in a fast headless **JAX clone** and only touch the real server
for validation + visuals via a thin `zappy_ai` adapter.

**Locked decisions:** pure-JAX sim · phased framing (co-op first, then
competitive self-play) · emergent 8-token broadcast comms in scope · anchor on
the **reference binaries** (`zappy_ref-v3.0.1.tar`) · target box = **Linux + RTX
4090 (24GB) + 128GB RAM**.

## 2. Status (phase by phase)

| Phase | State |
|---|---|
| 0 — Reference ground truth + scaffold | ✅ done & verified live |
| 1 — JAX env + validation suite | ✅ JAX `vmap` env done, cross-checked vs NumPy oracle |
| 2 — Single-agent foraging MAPPO | 🚧 **NEXT** |
| 3 — Cooperative ritual + emergent broadcast | ⬜ |
| 4 — League + telemetry + deploy adapter | ⬜ |
| 5 — 2-week run orchestration + viz | ⬜ |

**Verified:** 39 pytest tests pass (geometry, protocol, all core rules,
JAX↔oracle cross-check). Live reference-server agreement on vision (16/16
tiles), broadcast (8/8 directions, 2 orientations), and L1→L2 incantation
(elevation + stone consumption). JAX env throughput ≈135k env-steps/s on
Apple-Silicon CPU (vmap×2048) — expect far higher on the 4090.

**Correctness chain that's now locked:** `JAX env (zappy_env.py)` ↔
`NumPy oracle (reference_env.py)` ↔ `live reference server`, all agreeing.

## 3. File map

```
zappy_rl/
  env/
    constants.py     exact game numbers (costs 7/42/300, food 126t, densities,
                     elevation table, orientation deltas) — CONFIRMED live
    vision.py        (L+1)^2 look cone + toroidal indexing (NumPy oracle)
    broadcast.py     toroidal shortest-path -> 8-sector direction K (oracle)
    reference_env.py NumPy rule ORACLE (readable; has fork+eject)
    zappy_env.py     vectorized JAX env (training); v1 DEFERS fork+eject
  deploy/
    protocol.py      wire parsing + LineSocket (ONLY TCP-touching code)
  eval/
    scripted_ai.py   greedy baseline AI (eval opponent + trace activity)
tools/
  capture_golden_traces.py  reference server -> NDJSON GUI event traces
  validate_against_server.py vision/broadcast/incantation pinned live (ALL PASS)
  bench_env.py               JAX env throughput benchmark
tests/                       39 tests (geometry, protocol, rules, JAX cross-check)
docs/PLAN.md                 full approved plan
reference/                   unpacked zappy_ref-v3.0.1.tar (gitignored; regenerate)
```

## 4. v1 limitations / deferred (implement in later phases)

- **Fork + egg population growth**: implemented in the *oracle* but DEFERRED in
  the JAX env. Needed for the full 6-agent L8 goal (Phase 3/4). Add a per-agent
  `status` (egg/alive/dead) lifecycle to `zappy_env.py`.
- **Eject**: in the oracle, not the JAX env yet.
- **Broadcast**: JAX env delivers only one emitter/step (lowest index). Fine for
  v1; generalize when comms matters (Phase 3).
- **Take contention**: JAX env doesn't arbitrate when several agents grab the
  same scarce tile in one step (cross-check avoids this). Resolve if it bites.
- **Respawn**: JAX respawn is an approximate top-up (positions are not
  server-faithful; the live differential test uses `--no_refill`). Totals are
  correct, which is what matters for foraging.
- **Look/Inventory not learned actions**: the policy always gets a fresh obs;
  the deploy adapter must issue `Look`+`Inventory` every decision cycle.

## 5. SETUP ON THE 4090 BOX (do these in order)

Assumes Ubuntu/Debian-like Linux with a recent NVIDIA driver. Everything below
is copy-paste.

### 5.0 Get the code onto the box
From the **Mac**, rsync the repo (includes the tarball + docs; skips the venv and
the macOS-unpacked reference, which you regenerate on the box):
```sh
rsync -av --exclude .venv --exclude reference --exclude traces \
  ~/dev/epitech/Zappy-DeepLearning/  USER@BOX:~/Zappy-DeepLearning/
```
(Or use git — but the `.tar` is untracked and `reference/`,`.venv/` are
gitignored, so still `scp` the `zappy_ref-v3.0.1.tar` over.)

### 5.1 Prereqs (on the box)
```sh
nvidia-smi                         # confirm the 4090 + driver are visible
sudo apt-get update && sudo apt-get install -y ffmpeg git
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't installed
```

### 5.2 Python env + CUDA JAX
```sh
cd ~/Zappy-DeepLearning
uv venv --python 3.11 .venv
uv pip install --python .venv -e '.[train,viz]'
uv pip install --python .venv -U "jax[cuda12]"     # CUDA build (RTX 4090)
.venv/bin/python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```
If you hit a jax/jaxlib version mismatch, pin them equal, e.g.
`uv pip install --python .venv "jax[cuda12]==0.10.1"`.

### 5.3 Unpack the reference server + GUI (Linux)
```sh
mkdir -p reference && tar -xf zappy_ref-v3.0.1.tar -C reference
chmod +x reference/linux/zappy_server reference/linux/zappy_gui.AppImage
```

### 5.4 Verify everything (should all pass)
```sh
.venv/bin/python -m pytest -q                       # expect: 39 passed
.venv/bin/python tools/validate_against_server.py   # expect: SUMMARY: ALL PASS
.venv/bin/python tools/bench_env.py --envs 8192 --steps 300   # GPU throughput
```
The tools auto-detect the Linux server binary.

### 5.5 (Optional) watch the real GUI
```sh
# terminal A — keep stdin open so the server console stays alive:
tail -f /dev/null | reference/linux/zappy_server -p 4242 -x 20 -y 20 \
  -n T1 T2 -c 6 -f 100 --auto-start on
# terminal B — the reference GUI (add --appimage-extract-and-run if FUSE is missing):
reference/linux/zappy_gui.AppImage -p 4242 -h 127.0.0.1
# terminal C — drive some scripted AIs for something to watch:
.venv/bin/python -c "from zappy_rl.eval.scripted_ai import ScriptedAI; import time; \
ais=[ScriptedAI('127.0.0.1',4242,'T1',i) for i in range(4)]; \
[a.run(time.monotonic()+60) for a in ais]"
```

## 6. WHAT TO DO NEXT (Phase 2) — for Claude Code on the box

Start Claude Code in the repo and give it this prompt:

> Read `docs/HANDOFF.md` and `docs/PLAN.md`. We're on the RTX 4090 box now.
> Continue with **Phase 2: recurrent MAPPO foraging**. Build `zappy_rl/algo/
> networks.py` (GRU actor + GRU centralized critic + 8-token broadcast head,
> separate actor/critic GRUs) and `zappy_rl/algo/mappo.py` (PureJaxRL-style
> recurrent MAPPO over the vmapped `zappy_env`). Train a single agent to forage
> on a small map. **Gate:** the agent reliably survives >2000 ticks, training at
> ≥100k env-steps/s on the GPU. Use Weights & Biases for logging. Keep building
> verifiably — smoke-test the loop on a tiny config first, then scale.

Reference implementations to mirror (named in the plan):
- PureJaxRL recurrent PPO: https://github.com/luchris429/purejaxrl
- JaxMARL MAPPO: https://github.com/FLAIROX/JaxMARL

Key knobs from the plan: GRU(128), **separate** actor/critic RNNs, TBPTT chunk
16, ≤4 PPO epochs, 2–4 large minibatches, `num_envs ≈ batch`. The env's action
space is `zappy_env.N_ENV_ACTIONS` (20); broadcast token is a separate head
feeding `step`'s `tokens` arg.

## 7. Pitfalls already discovered (save yourself the debugging)

- The reference server **quits on stdin EOF** — always keep stdin open
  (`tail -f /dev/null | ...` or a kept-open pipe).
- AI handshake: `WELCOME` → send team → server replies `<slots>\n` then `<X> <Y>\n`.
- Look tile separator is "comma optionally followed by space" — parse loosely
  (`deploy/protocol.parse_look` already does).
- Resource order everywhere is `food,linemate,deraumere,sibur,mendiane,phiras,thystame`.
- JAX env recompiles if map size / agent count / action space change — freeze
  them within a training stage; change only at stage boundaries.

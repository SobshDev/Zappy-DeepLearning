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
| 2 — Single-agent foraging MAPPO | ✅ **GATE PASSED** on the 4090 (see below) |
| 3 — Cooperative ritual + emergent broadcast | 🚧 **NEXT** |
| 4 — League + telemetry + deploy adapter | ⬜ |
| 5 — 2-week run orchestration + viz | ⬜ |

**Verified:** 47 pytest tests pass (geometry, protocol, all core rules,
JAX↔oracle cross-check, MAPPO math: GAE vs slow reference, GRU carry resets,
TBPTT chunk indexing, first-update ratio==1). Live reference-server agreement
on vision (16/16 tiles), broadcast (8/8 directions, 2 orientations), and
L1→L2 incantation (elevation + stone consumption). Raw JAX env throughput on
the 4090: **5.6M env-steps/s** (vmap×8192).

**Phase-2 result (`runs/forage6x6-v1`, 50M env-steps in ~3.5 min):**
- Training throughput **940k env-steps/s** (gate: ≥100k) at num_envs=2048.
- Stochastic policy: **512/512 eval episodes survive >2000 ticks** — every
  episode reaches the 4096-tick training cap (min 4099); at a 16384-tick
  horizon: mean 8635 / min 6880 ticks, still 100% over the gate.
- Artifacts: `runs/forage6x6-v1/{params.msgpack,config.json,eval.json}`.
- Caveats logged in eval.json: the *greedy* (argmax) policy collapses (19%
  ge-2000) — deploy/eval must **sample**, as in training; survival decays
  ~8.6k ticks out (recurrent state leaves the ≤4096-tick training
  distribution — irrelevant for the gate, retrained in Phase 3 anyway).

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
  algo/
    networks.py      GRU(128) actor (+8-token broadcast head) + SEPARATE
                     GRU(128) centralized critic; ScannedRNN; flatten_obs
    mappo.py         recurrent MAPPO: rollout/GAE/TBPTT-16 PPO, autoreset,
                     potential survival shaping, evaluate(), train() driver
  train.py           CLI (flags auto-generated from TrainConfig)
  deploy/
    protocol.py      wire parsing + LineSocket (ONLY TCP-touching code)
  eval/
    scripted_ai.py   greedy baseline AI (eval opponent + trace activity)
runs/                training artifacts (gitignored): params/config/eval per run
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

## 6. WHAT TO DO NEXT (Phase 3) — cooperative ritual + broadcast

Prompt for the next session:

> Read `docs/HANDOFF.md` and `docs/PLAN.md`. Continue with **Phase 3:
> cooperative ritual**. Enable 2 agents on 8×8 with Incantation +
> co-location + broadcast, simplified L1→L2→L3 ladder. Extend the reward per
> the plan (co-location potential gated by ritual-possible, incantation
> attempt, broadcast→coordination). **Gate:** pairs reliably complete an
> L2→L3 ritual.

What Phase 3 needs that v1 deferred (see §4 and `zappy_env.py` TODOs):
- Per-agent **death/done** handling in the trainer: today done is env-level
  (all-dead) — with 2 agents one can die while the env continues. The GAE
  mask and GRU reset need per-agent done (plumbing is per-(env,agent) row
  already; `_step_env`/`Transition.done` need the per-agent flag).
- **Busy agents**: with >1 agent the event clock means some agents are busy
  (frozen/cooldown) when others act; their submitted actions are ignored by
  the env. Consider masking their logp out of the PPO loss (use
  `info["free"]`) so ignored actions don't get credit.
- Broadcast currently delivers one emitter/step (lowest index) — fine for 2
  agents, generalize later.
- Train commands: `python -m zappy_rl.train --help` (flags auto-generated
  from `TrainConfig`); start from `runs/forage6x6-v1/params.msgpack` or
  retrain from scratch (50M steps ≈ 3.5 min at 940k SPS).

## 7. Pitfalls already discovered (save yourself the debugging)

- The reference server **quits on stdin EOF** — always keep stdin open
  (`tail -f /dev/null | ...` or a kept-open pipe).
- AI handshake: `WELCOME` → send team → server replies `<slots>\n` then `<X> <Y>\n`.
- Look tile separator is "comma optionally followed by space" — parse loosely
  (`deploy/protocol.parse_look` already does).
- Resource order everywhere is `food,linemate,deraumere,sibur,mendiane,phiras,thystame`.
- JAX env recompiles if map size / agent count / action space change — freeze
  them within a training stage; change only at stage boundaries.
- **Never scale per-step rewards by the event-clock `dt`**: PPO discounts per
  env-step, so a dt-scaled bonus pays incantation's 300-tick freeze as a lump
  at one discount factor — "stand on a linemate and freeze" beat foraging
  until the alive bonus was made flat-per-step (caught in adversarial review).
- **Eval/deploy must SAMPLE the policy**, not argmax — the greedy policy
  collapses (entropy-regularized training, ties break degenerately).
- Rollout-window episode stats go blind once episodes outlive the window
  (~7·rollout_steps ticks): a converging policy shows `ep_ticks` pinned at the
  window and `ge2000 = 0`. Read the live `now>=2k` / `alive_frac` metrics.
- W&B: the box is **not** logged in — runs use `--wandb offline` (or `auto`,
  which falls back to offline). `wandb login` then `wandb sync wandb/offline-*`
  to upload, or export `WANDB_API_KEY`.
- JAX preallocates 75% of VRAM per process — set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing the GPU between runs.

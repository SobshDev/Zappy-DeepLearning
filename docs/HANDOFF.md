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
| 3 — Cooperative ritual + emergent broadcast | ✅ **GATE PASSED** (see below) |
| 4 — League + telemetry + deploy adapter | ✅ **GATE PASSED** (see below) |
| 5 — 2-week run orchestration + viz | 🚧 **NEXT** |

**Verified:** 94 pytest tests pass (geometry, protocol, all core rules,
JAX↔oracle cross-check, MAPPO math: GAE vs slow reference, GRU carry resets,
TBPTT chunk indexing, first-update ratio==1 incl. multi-agent; Phase-3: phi
terms, per-agent alive/free masks, double-initiator + dead-during-freeze
incantation edges; Phase-4: obs-contract bit-exactness incl. float32-ULP
inventory values, adapter wire routing/late-acks/disconnects, PFSP league
math + persistence, recorder GUI-event fidelity). Live reference-server
agreement on vision (16/16 tiles), broadcast (8/8 directions, 2
orientations), L1→L2 incantation (elevation + stone consumption), and the
FULL deploy obs contract (`tools/validate_obs_contract.py`: real server
Look/Inventory/broadcast → `build_obs` == sim `flatten_obs(observe(...))`
bit-for-bit). Raw JAX env throughput on the 4090: **5.6M env-steps/s**
(vmap×8192).

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

**Phase-3 result (`runs/ritual8x8-v1`, 100M env-steps in ~3.6 min):**
- 8×8, 2 agents, recurrent MAPPO with per-agent dones + busy-action masking
  + the plan's potential terms (`phi_stones`, `phi_coloc`×tile-progress,
  `phi_incant`); token entropy 0.01. **459k env-steps/s** at num_envs=1024.
- Gate: **reach_l3_rate 1.0 — 512/512 eval episodes complete an L2→L3
  ritual** (stochastic, 8192-tick horizon), median time-to-L3 **831 ticks**;
  mean max level **4.0** (pairs chain L3→L4 rituals unprompted). Survival
  100% ≥2000 ticks. Greedy no longer collapses (98.6% L3) but eval/deploy
  still SAMPLES by convention.
- Env fixes shipped with this phase (both oracle-aligned, live-revalidated):
  stones consumed once per *tile* not per *initiator* (double-initiator
  dedup), and dead-during-freeze participants no longer level/score.
- Adversarial review (65 agents): 20 findings → 3 real (all fixed: the two
  env edges above + warm-start shape check + metric rename), 5 explicit
  verified-correct notes on shaping/GAE/masking/TBPTT math.
- Artifacts: `runs/ritual8x8-v1/{params.msgpack,config.json,eval.json}`.

**Phase-4 result (deploy adapter + league + recorder):**
- **Gate PASS** (`runs/ritual8x8-v1/deploy_gate.json`): the frozen
  ritual8x8-v1 policy, run as two real `zappy_ai` TCP clients on the
  reference server (10×10 — the server rejects maps under 10 — f=100),
  played full games with **zero protocol errors**. Canonical run: both
  agents L1→L2 with cooperative rituals, played to natural in-game death
  (~4.5k ticks); an earlier 185 s run chained **L1→L2→L3→L4** (4/4 rituals
  server-confirmed, one agent surviving the whole budget, 18.5k ticks,
  3.8k commands, 0 errors). PASS criteria are strict: zero self-reported
  protocol errors AND no crash AND no silent disconnect AND GUI-observed
  levels == self-reported levels (independent desync check).
- Obs contract pinned twice: unit (`tests/test_deploy_adapter.py`, builder ==
  `flatten_obs(observe(...))` bit-exact incl. ULP-divergent inventory counts)
  and live (`tools/validate_obs_contract.py` ALL PASS: vision pattern over the
  full cone, both players visible, inventory/life mapping, broadcast K + token
  one-hots vs a mirrored sim state).
- `algo/league.py`: PFSP opponent pool (50% p(1−p)-prioritized / 35%
  latest-self / 15% exploiter, Beta(1,1) win-rates, crash-safe atomic+durable
  persistence, single-writer). `viz/recorder.py`: sim episodes → GUI wire
  protocol (NDJSON + SQLite (gen,episode,tick,seq)); a frozen-policy replay
  artifact lives at `runs/ritual8x8-v1/replays/` (rituals render with true
  ~300-tick freezes).
- Adversarial review (80 agents): 25 findings → 20 confirmed (2 critical:
  response-deadline starvation under async floods; silent-disconnect-as-PASS
  gate blindness — both fixed + regression-tested), 5 refuted. A stricter
  re-gate then caught a REAL wire quirk live (racing co-located Incantations
  queue the loser's command across the freeze; its ok/ko lands one expect
  late) — probed on the real server, handled as counted `late_acks`, and the
  canonical PASS run exercised that exact path (late_acks=1, 0 errors).

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
    league.py        PFSP opponent pool (snapshots + win-rates + matchmaking)
  train.py           CLI (flags auto-generated from TrainConfig)
  deploy/
    protocol.py      wire parsing + LineSocket (ONLY TCP-touching code)
    zappy_ai_adapter.py  frozen policy as a real zappy_ai client: build_obs
                     (the sim<->server contract), PolicyRunner, TCP loop, CLI
  eval/
    scripted_ai.py   greedy baseline AI (eval opponent + trace activity)
  viz/
    recorder.py      sim episodes -> GUI wire protocol (NDJSON + SQLite)
runs/                training artifacts (gitignored): params/config/eval per run
tools/
  capture_golden_traces.py  reference server -> NDJSON GUI event traces
  validate_against_server.py vision/broadcast/incantation pinned live (ALL PASS)
  validate_obs_contract.py   deploy obs contract pinned live (ALL PASS)
  run_deploy_gate.py         Phase-4 gate: server + 2 adapters + GUI watcher
  record_replay.py           frozen policy -> GUI-protocol replay artifact
  bench_env.py               JAX env throughput benchmark
tests/                       94 tests (geometry, protocol, rules, JAX cross-check,
                             MAPPO math, deploy contract, league, recorder)
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

## 6. WHAT TO DO NEXT (Phase 5) — curriculum scale-up + 2-week run + viz

Prompt for the next session:

> Read `docs/HANDOFF.md` and `docs/PLAN.md`. Continue with **Phase 5**: scale
> the curriculum past 2 agents (stage 3: 12×12, 3 agents, L2→L4 — generalize
> env broadcast beyond one emitter/step first, warm-start from
> `--init-actor runs/ritual8x8-v1/params.msgpack`), wire `algo/league.py`
> snapshots + W&B `generation` grouping into the train loop, build
> `viz/replay_to_gui.py` (stream recorder NDJSON into the reference GUI) and
> the heatmap/timelapse pipeline, then set up `tools/run_2week.sh` +
> systemd crash-resume. **Gate:** stage-3 squads reliably reach L4, and a
> recorded replay renders end-to-end in the reference GUI.

Notes for Phase 5 (from the Phase-4 review + build):
- Replay consumers: read SQLite `ORDER BY tick, seq` (seq makes within-tick
  emit order explicit); `pic` is stamped at the freeze-START tick so rituals
  render their true ~300-tick glow. A ready artifact:
  `runs/ritual8x8-v1/replays/replay_g0_e0.ndjson`
  (regenerate via `tools/record_replay.py`).
- 3+ agents needs env work (deferred v1 limits): broadcast delivers ONE
  emitter/step (lowest index); take-contention unarbitrated; fork/eject
  absent (needed for the 6-agent L8 stage). `evaluate()`'s
  `n_steps = eval_max_ticks//7 + 64` buffer assumes near-lockstep — revisit
  once event clocks desync at 3+ agents.
- League integration: `League.add_snapshot(flax.serialization.to_bytes(
  {"actor": ...}), step)` every ~4h of training; `sample_opponent(rng)` for
  eval matchups; it's bytes-in/bytes-out, single-writer by design.
- Deploy survival on the real server is ~2× harsher than sim per decision
  (Inventory 1 + Look 7 + action 7 ticks/cycle vs 7 in sim) — agents reach
  L2–L4 and can survive 18k+ ticks, but if real-server survival matters for
  the showcase, consider a cadence-aware fine-tune (train with a 15-tick
  per-decision cost) before the final demo.

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
- W&B: the box IS logged in (`gabriel-brument-epi`, key in `~/.netrc`) —
  `--wandb auto` resolves to online. Phase-2/3 offline runs were synced with
  `wandb sync wandb/offline-*`.
- `lvlups_to_l2/l3` metrics count per-agent level-up EVENTS, not rituals — a
  completed pair ritual contributes 2 (and 4/6 at higher tiers). The gate
  readout is eval `reach_l3_rate` (per-env max level), which is unaffected.
- JAX preallocates 75% of VRAM per process — set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing the GPU between runs.
- The reference server rejects maps under **10×10** ("Value must be between
  10 and 42") — the 8×8-trained policy deploys fine (obs are egocentric,
  per-tile densities identical), but plan deploy geometry accordingly.
- The server emits a GUI `pdi` on ANY disconnect, including a clean client
  close — `pdi` is NOT evidence of in-game death; the in-band `dead` line is.
- Racing co-located `Incantation`s (the trained pair behavior): the loser's
  command is QUEUED across the freeze; the server answers the elevation lines
  first and the queued command's ok/ko arrives one expect-window late. The
  adapter absorbs these as `late_acks` (not protocol errors); anything
  parsing AI streams must expect them.
- When mirroring sim obs outside XLA: jnp's `/10.0` compiles to a
  multiply-by-reciprocal that rounds one float32 ULP differently from
  numpy's true divide for counts {9, 13, 18, ...} — use `* np.float32(0.1)`
  to stay bit-exact (`build_obs` does).

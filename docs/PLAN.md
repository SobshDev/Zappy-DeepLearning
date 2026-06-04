# Zappy "Deep-Learning Civilisation" — Full Implementation Plan

## Context

This is the Epitech **Zappy** project (`G-YEP-400`). Zappy is a network game: teams of bodiless agents ("Trantorians") roam a **toroidal tile world**, eat food to survive, collect 6 stone types, and perform **elevation rituals** to level up. **Win = first team with 6 agents at level 8.** The repo currently holds only the two subject PDFs and the reference binary tarball (`zappy_ref-v3.0.1.tar`, which ships a Linux + macOS `zappy_server` and a Linux `zappy_gui.AppImage`).

The goal of *this* effort is **not** the graded project itself but a parallel experiment: train agents purely by **reinforcement learning + self-play** ("unsupervised") on a dedicated **RTX 4090 (24 GB) + 128 GB RAM Linux workstation** for ~2 weeks, to (stretch) beat the hand-crafted scripted AI being developed in parallel, and (primary) produce **Yosh/Trackmania-style "watching it learn" visuals**.

**The core insight driving the whole design:** the real game runs over a single-threaded TCP server (`action/f` seconds per action, `f=100` default), which is ~10,000–30,000× too slow to feed RL. So we **train in a fast headless JAX clone of the game**, and only ever touch the real server through a thin deploy adapter for validation + visuals.

**Decisions locked with the user:**
1. **Simulator:** pure **JAX** vectorized env (`jax.vmap` over thousands of worlds on the 4090).
2. **Framing:** **phased** — cooperative race to high levels first, then add competitive self-play.
3. **Broadcast comms:** **in scope** — learn an emergent 8-token protocol.
4. **Server/GUI:** anchor on the **reference binaries** for golden-trace validation, deploy, and visuals.
5. (Assumed) workstation is **Linux** — required for the reference GUI AppImage + CUDA/JAX.

---

## Goals & Non-Goals

**Goals**
- A byte-semantically-faithful JAX reimplementation of Zappy for high-throughput RL.
- Recurrent **MAPPO** training with a faded potential-shaping **curriculum** and an emergent broadcast channel.
- A **phased** run: co-op ritual progress → competitive self-play league.
- A frozen policy that plays on the **reference server** via a `zappy_ai` TCP adapter, rendered by the **reference GUI**.
- A **telemetry + replay + video** pipeline producing red→green generation timelapses, heatmaps, broadcast-ripple animations, and W&B curves.

**Non-Goals (explicitly out of scope)**
- Building the graded `zappy_server`/`zappy_gui`/`zappy_ai` (separate track; we *use* the reference binaries here).
- Guaranteeing a full 6-agent **L8** win or guaranteeing we beat a *competent* scripted AI (treated as caveated stretch goals — see Realistic Expectations).
- Training through the real TCP server (too slow; only used for deploy/eval/visuals).

---

## Recommended Stack (one choice per layer)

| Layer | Choice | Notes |
|---|---|---|
| Sim | Pure **JAX**, gymnax-style `step/reset`, `vmap` over 1k–4k worlds, JIT | toroidal wrap + `(L+1)²` vision = modulo/gather ops |
| RL framework | **JaxMARL** (https://github.com/FLAIROX/JaxMARL) on the **PureJaxRL** MAPPO recipe (https://github.com/luchris429/purejaxrl) | Mava (https://github.com/instadeepai/Mava) is the logging-rich fallback |
| Algorithm | **Recurrent MAPPO** (centralized critic, decentralized exec) | best cooperative credit assignment; ~5–10M steps vs 50–100M for QMIX |
| Recurrence | **GRU(128)**, **separate** actor & critic GRUs, TBPTT chunk **16** | sharing actor/critic RNN causes gradient explosion |
| PPO knobs | ≤**4** epochs, **2–4 large** minibatches, `num_envs ≈ batch` | many tiny minibatches hurt MAPPO badly |
| Comms | **8-token** broadcast action head; incoming `(direction_K, token)` aggregated by small **self-attention** over last N heard msgs | matches protocol (direction only, no sender id) |
| Self-play | per-team policies + **PFSP** pool (~50% prioritized / 35% latest-self / 15% exploiter), snapshot every 4h | prevents rock-paper-scissors collapse (AlphaStar league) |
| Telemetry | **Weights & Biases**, `generation` as grouping dim for red→green | TensorBoard chokes past ~100 runs |
| Replay→GUI | sim → GUI-protocol JSON → reference `zappy_gui`; SQLite indexed `(gen, episode, team, tick)` | the GUI protocol PDF *is* the contract |
| Deploy | frozen policy → `zappy_ai` TCP adapter → reference `zappy_server` → reference `zappy_gui` | adapter is the **only** TCP-touching code |
| Video | **pygame** viewer + **matplotlib** heatmaps + **ffmpeg** (`setpts=PTS/8`, concat) | CPU-bound, zero GPU contention |

---

## System Architecture

```
        ┌──────────── TRAINING HOST (RTX 4090 / 128 GB / Linux) ────────────┐
 Hydra  │  JAX ZAPPY ENV (vmap × 1k–4k worlds)  ⇄  RECURRENT MAPPO (JAX)     │
 config │   toroidal grid · (L+1)² vision ·         GRU actor(128) +         │
 ──────►│   food 126t · respawn /20t ·              GRU centralized critic + │
        │   incantation start/end · costs 7/42/300  8-token bcast + self-attn│
        │        │ event stream            │ params / 4h snapshots           │
        │        ▼                          ▼                                │
        │  REPLAY RECORDER   OPPONENT POOL (PFSP)   CHECKPOINTER (6h/24h/best)│
        └────────┼───────────────────────────────────────┼──────────────────┘
                 ▼                                         ▼
          SQLite replay DB                          W&B dashboard
                 │
   OFFLINE VIZ ◄─┤  pygame viewer · matplotlib heatmaps · broadcast ripples · ffmpeg timelapse
                 │
   ===== DEPLOY / VALIDATION (the real game) =====
   frozen policy → zappy_ai TCP adapter → reference zappy_server → reference zappy_gui
```

**Two-track contract:** every observation/action in the JAX sim must be **semantically identical** to what `zappy_ai` would parse from the reference server. The `deploy/protocol.py` adapter parses `Look`/`Inventory` strings into the exact tensor layout the sim emits and converts action indices into commands. Training never opens a socket.

---

## Repo Layout (files to create)

```
zappy_rl/
  env/   zappy_env.py · constants.py · vision.py · broadcast.py · observation.py
  algo/  mappo.py · networks.py · league.py · curriculum.py · reward.py
  eval/  eval_harness.py · scripted_ai.py
  deploy/ zappy_ai_adapter.py · protocol.py        # ONLY TCP code
  viz/   recorder.py · replay_to_gui.py · viewer.py · heatmaps.py · make_video.py
  train.py
  configs/   # Hydra: stages, reward weights, f=100, map sizes, league
tests/   test_vision.py · test_broadcast.py · test_incantation.py · test_differential.py
tools/   capture_golden_traces.py · zappy-train.service · run_2week.sh
reference/   # unpacked zappy_ref-v3.0.1.tar (server + GUI AppImage)
```

**Game constants to encode exactly** (`env/constants.py`, from the PDFs + reference README):
- Action costs (÷f): Forward/Right/Left/Look/Broadcast/Eject/Take/Set = **7**, Inventory = **1**, **Fork = 42**, **Incantation = 300**. `f=100` fixed everywhere.
- Food: 1 unit = **126 ticks**; agents start with **10** (1260 ticks); death at 0.
- Resource respawn every **20 ticks**; quantity = `width*height*density` (food .5, linemate .3, deraumere .15, sibur .1, mendiane .1, phiras .08, thystame .05).
- Vision: `(L+1)²` tiles, rows 0..L, row *k* has `2k+1` tiles; tile 0 = own tile; numbering left→right per row going forward.
- Elevation table (players, linemate, deraumere, sibur, mendiane, phiras, thystame): `1→2`(1,1,0,0,0,0,0) `2→3`(2,1,1,1,0,0,0) `3→4`(2,2,0,1,0,2,0) `4→5`(4,1,1,2,0,1,0) `5→6`(4,1,2,1,3,0,0) `6→7`(6,1,2,3,0,1,0) `7→8`(6,2,2,2,2,2,1).
- Incantation: same-**level** players co-located (not same team), stones present; verified at **start and end**; participants **frozen**; `ko` instant if start-check fails; stones consumed at end.
- Broadcast direction `K`: shortest **toroidal** path; `K=0` if emitter on same tile. Eject pushes co-located players in facing dir & destroys eggs. Fork lays an egg → new connectable slot.

---

## Phased Build Order (each phase has a gate that de-risks the next)

- **Phase 0 — Reference ground truth (Day 0–1).** Unpack the tarball; run `reference/zappy_server -p 4242 -x 10 -y 10 -n T1 T2 -c 6 -f 100` + the GUI AppImage. With `tools/capture_golden_traces.py`, drive a hand-scripted greedy `zappy_ai` and record the full GUI event stream. **These become the golden traces.** Exploit the server's stdin console (`/setLevel`, `/setInventory`, `/tp`, `/incantate`, `/fork`, `/setTile`, `/noFood`, `/noRefill`, `/setFreq`, `/pause`) to script deterministic validation + eval scenarios.
- **Phase 1 — JAX sim + L1–L4 validation (Day 1–4).** Implement `env/zappy_env.py` as a new JaxMARL/gymnax env. **Gate:** `test_vision/broadcast/incantation` reproduce the PDF p.5–6 examples, and `test_differential.py` matches Phase-0 golden traces. *(Semantic drift here is the #1 way to waste two weeks — differential-test every build.)*
- **Phase 2 — Single-agent foraging (Day 3–5).** PureJaxRL MAPPO loop, **1 agent, no rituals**, 6×6 abundant map. **Gate:** survives >2000 ticks reliably at ≥100k steps/s.
- **Phase 3 — Cooperative ritual (Day 5–7).** Enable Incantation + co-location + broadcast on 8×8, simplified L1→L2→L3 ladder. **Gate:** pairs reliably complete an L2→L3 ritual.
- **Phase 4 — League + telemetry + deploy adapter (Day 6–8).** Wire PFSP pool, W&B, SQLite recorder, and `deploy/zappy_ai_adapter.py`. **Gate:** a frozen policy plays a full game on the **reference server** through the GUI with **zero protocol errors**.
- **Phase 5 — Full curriculum run + viz (Day 8 →).** Launch the 2-week run; pygame viewer + heatmaps + ffmpeg run in parallel.

**Reuse over bespoke:** JaxMARL (env+MAPPO), PureJaxRL (clean MAPPO), gymnax (env API), lb-foraging (https://github.com/uoe-agents/lb-foraging — closest reward/curriculum analog), Melting Pot (https://github.com/google-deepmind/meltingpot — eval patterns), Pommerman PBT (https://arxiv.org/html/2407.00662 — league pattern), EPITECH-Zappy (https://github.com/fgrimaepitech/EPITECH-Zappy — reference GUI ideas).

---

## Curriculum + Reward (`algo/curriculum.py`, `algo/reward.py`)

Potential-based shaping (Ng 1999, `F = γΦ(s') − Φ(s)`, **policy-invariant**, per-agent → preserves the multi-agent equilibrium) so shaping can be **faded without changing the optimum**.

**Stages (map / agents / target levels), advanced when success-rate crosses threshold:**
1. 6×6, 1 agent, L1→L2 — survival + forage
2. 8×8, 2 agents, L1→L3 — first co-location + Incantation
3. 12×12, 3 agents, L2→L4 — broadcast-coordinated assembly
4. 16×16, 4 agents, L3→L5 — 4-player rituals
5. 20×24, 6 agents, L6→L8 — full squad + scarcity (the stretch zone)

Use **PLR** (Prioritized Level Replay, https://www.emergentmind.com/topics/prioritized-level-replay-plr) over ~20 variants (size × density × agent-count × seed), sampled ∝ TD-error regret.

**Reward terms (rough weights):**

| Term | Form | Weight |
|---|---|---|
| Game win (team, terminal) | +1 first team to 6×L8 | **+10.0** |
| Level-up (true reward) | on successful `pie` | **+1.0 × L** |
| Survival potential | Φ = −log(ticks_left/1260) | +0.5 |
| Holds next-tier stones | Φ = count(needed) | +1.0 |
| Co-location (gated by "ritual possible") | Φ = log(1+#same-level-on-tile) | +1.5 |
| Incantation attempt (ritual-possible) | Φ = current_level | +1.0 |
| Broadcast→coordination | + if teammate moved toward a broadcast preceding a successful ritual | +0.2 |
| RND intrinsic (decay) | novelty | 0.01→0.001 |
| Broadcast entropy (bounded) | H(vocab) clamped [0,0.1] | 0.01 |
| Death | terminal | **−1.0** |

**Fade schedule (the Yosh principle — put expertise in the *reward*, then remove it):** freeze shaping at day 7, exponential decay (~200k-step half-life), <5% contribution by 20M steps; final days run with shaping ≈ 0 to prove internalization.

---

## 2-Week Run Plan (`tools/run_2week.sh`, `zappy-train.service`)

- **Throughput target:** ~100k env-steps/s (1k–4k parallel worlds). Realistically 2e7–5e7 useful gradient-steps-worth after compile stalls/eval/viz.
- **Days 8–10:** PLR-driven 1→2→3→4 agents, **cooperative**. Generation snapshot **every 4h** (~40–56 generations total → the red→green sequence).
- **Days 10–12:** push to 6-agent L6→L8 on 20×24 (stretch zone).
- **Days 12–13:** **introduce competition** — flip on the PFSP self-play league (50% prioritized / 35% latest-self / 15% exploiter).
- **Days 13–14:** freeze shaping; run the eval harness.
- **Checkpoints:** tiered — recent (6h, fast recovery) / generational (24h archive) / best-policy singleton. Async + mmap writes.
- **Crash-resume:** run as a **systemd** service with auto-restart; resume from last 6h checkpoint + last completed eval batch. Background monitor caps GPU power at **~80% (≈250 W)**, auto-pause >85 °C. Log every ~1000 steps (compressed) to bound disk. Keep only 6 recent replay-buffer snapshots (full buffers = TBs).
- **Storage:** ~330 MB JSON replays (action-commit granularity, NOT per-tick) + 0.5–3 GB checkpoints.

---

## Evaluation / "Beat the AI" Harness (`eval/eval_harness.py`)

- **Baseline:** the scripted "final AI" (or `eval/scripted_ai.py`), run as a `zappy_ai` team on the **reference server**.
- **Protocol:** every 24h, **100 head-to-head matches** vs the baseline, same map/seed set, `f=100` fixed.
- **Metrics:** win-rate with **Beta-Binomial 95% CI**; **Glicko-2** Elo; **Elo-RCC** to detect intransitive cycling; secondary: **time-to-L8**, team survival curve, ritual success-rate by tier.
- **Anti-overfit caveat:** if possible, evaluate against **2–3 scripted variants** so the RL policy isn't just exploiting one bug. Note this caveat in any results writeup.

---

## Visualization Pipeline (`viz/`) — the primary deliverable

1. **Recorder** (`recorder.py`): dump sim state in the exact GUI wire protocol (`msz/bct/mct/ppo/plv/pin/pic/pie/pbc/pfk/pdr/pgt/pdi/enw/ebo/edi/sgt/sst/seg/smg`) → newline-delimited JSON → SQLite `(generation, episode, team, tick)`.
2. **Red→green timelapse** (`replay_to_gui.py`): replay the **same map/seed** at generations 0→N through the **reference GUI** (or `viewer.py`) — instant "watching it learn." Color/tint episodes by generation.
3. **Heatmaps** (`heatmaps.py`): tile-occupancy (apply `x%W, y%H` for the torus), incantation-site clusters, per-stone gather density — evolving across generations (foraging chaos → tight ritual clusters).
4. **Broadcast ripples** (`viewer.py`): expanding rings from each `pbc` emitter, fading ~0.5 s, labeled with direction+token — lights up right before successful rituals.
5. **W&B dashboards:** reward, max-level-over-time, survival %, successful incantations, population — grouped by `generation` for red→green hue.
6. **Video** (`make_video.py`): daily best-episode summary at 8× (`ffmpeg -vf "setpts=PTS/8"`), concat-demuxer into a ~90 s 2-week timelapse at 1280×720/24fps.

---

## Realistic Expectations

**High confidence in 2 weeks:** robust foraging/survival; reliable 2–4 player rituals through ~**L4–L6**; a working interpretable broadcast protocol; a clean deploy through the reference server + GUI; excellent red→green visuals.

**Frontier-hard — do NOT promise:** full **L7→L8 with 6 synchronized same-level agents** + correct stones, verified start *and* end, while frozen. This is genuinely hard MARL, not a solved benchmark. Treat "first L8" as a single-digit-% stretch.

**Odds vs a *competent* scripted AI:** roughly **30–50%** to match/beat within 2 weeks (higher if the scripted AI is mediocre). Scripted Zappy AIs are strong because ritual logic is easy to hand-code and hard to learn; the RL edge is adaptivity/exploiting a static opponent.

**Best visuals land regardless of the L8 outcome** — that is why the visuals are framed as the primary deliverable.

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Sim/server **semantic drift** (vision off-by-one, broadcast K, incantation start-vs-end, food rate) | L1–L4 validation suite + **differential test vs golden traces every build** |
| **Ritual deadlock** / 6-agent sync never forms | 1→6 agent curriculum + PLR; gated co-location & attempt rewards; explicit broadcast→assemble incentive |
| Self-play **strategy cycling** | PFSP pool + 15% exploiters + checkpoint archive; monitor with Elo-RCC |
| **VRAM OOM** | `num_envs ≈ batch` (e.g. 512/512, mini-batch 256); mmap buffer, async I/O; shared tile-map indexing (not 224 GB naive) |
| **Recurrent gradient explosion** | separate actor/critic GRUs; ≤4 epochs; 2–4 large minibatches |
| **Silent broadcast channel** | shaped broadcast→ritual reward + bounded entropy bonus; measure downstream correlation |
| **JAX recompile stalls** | freeze map size / agent count / action space within a stage; change only at stage boundaries |
| **GPU thermal throttle** over 14 days | 80% power cap, hourly `nvidia-smi`, auto-pause >85 °C |
| **Over-promising L8** | frame L5–L6 squad play + visuals as primary; L8 / beat-the-AI as caveated stretch |

---

## Verification (end-to-end)

1. **Unit tests:** `pytest tests/test_vision.py test_broadcast.py test_incantation.py` reproduce the PDF p.5–6 examples and elevation table.
2. **Differential test:** `tests/test_differential.py` asserts JAX-sim state == reference-server golden traces for scripted L1–L4 scenarios (set up via the server stdin console).
3. **Phase gates:** single agent survives >2000 ticks; pairs complete L2→L3; **frozen policy plays a full game on the reference server through the GUI with zero protocol errors** (Phase 4 gate — the deploy contract).
4. **Throughput:** confirm ≥100k env-steps/s on the 4090 before launching the long run.
5. **Eval:** `eval/eval_harness.py` reports win-rate + 95% CI + Glicko-2 + time-to-L8 vs the scripted AI.
6. **Visuals:** red→green timelapse renders end-to-end; W&B dashboards populate; heatmaps + broadcast-ripple clips generate.

---

## Open Follow-ups (decide before/early in build, not blocking the plan)

- Confirm workstation OS = Linux + CUDA/cuDNN versions compatible with the JAX build.
- Provide (or stub) the scripted "final AI" for the eval baseline; ideally 2–3 variants.
- Choose canonical map/seed set for eval + the red→green replay (fix early for comparability).

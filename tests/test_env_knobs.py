"""Training-only robustness knobs: ``overhead``, ``density_scale``, ``life_noise``.

Defaults are oracle-exact (overhead=0, density_scale=1.0, life_noise=0) — the
rest of the suite (incl. the oracle cross-check) pins that. These tests pin
the knob semantics: overhead is added to every real action's cooldown but NOT
to ENV_IDLE, ritual initiators and frozen participants stay completion-synced
under overhead, density_scale scales spawn targets, and life_noise perturbs
ONLY the life observation feature (never the dynamics)."""

import jax
import jax.numpy as jnp
import numpy as np

from zappy_rl.env import constants as C
from zappy_rl.env import zappy_env as Z

from test_jax_env import mk_state, step1

KEY = jax.random.PRNGKey(0)
OV = 8


def test_defaults_are_oracle_exact():
    cfg = Z.make_cfg(11, 11, 1)
    assert cfg.overhead == 0 and cfg.density_scale == 1.0
    assert cfg.life_noise == 0.0


def test_overhead_added_to_action_cost():
    cfg = Z.make_cfg(11, 11, 1, no_food=True, no_refill=True, overhead=OV)
    s = mk_state(cfg, [[5, 5]], [C.NORTH])
    s2, *_ = step1(cfg, s, [Z.ENV_FORWARD])
    assert int(s2.busy_until[0]) == 7 + OV
    assert int(s2.now) == 7 + OV          # single agent: clock jumps to free


def test_idle_exempt_from_overhead():
    # an idle decision in deploy IS the Look — no extra prefix on top
    cfg = Z.make_cfg(11, 11, 1, no_food=True, no_refill=True, overhead=OV)
    s = mk_state(cfg, [[5, 5]], [C.NORTH])
    s2, *_ = step1(cfg, s, [Z.ENV_IDLE])
    assert int(s2.busy_until[0]) == 7
    assert int(s2.now) == 7


def test_overhead_charges_life():
    cfg = Z.make_cfg(11, 11, 1, no_food=False, no_refill=True, overhead=OV)
    s = mk_state(cfg, [[5, 5]], [C.NORTH])
    s2, *_ = step1(cfg, s, [Z.ENV_FORWARD])
    assert int(s2.life[0]) == Z.START_LIFE - (7 + OV)


def test_solo_ritual_completes_under_overhead():
    cfg = Z.make_cfg(11, 11, 1, no_food=True, no_refill=True, overhead=OV)
    grid = np.zeros((11, 11, 7), np.int32)
    grid[5, 5, C.LINEMATE] = 1
    s = mk_state(cfg, [[5, 5]], [C.NORTH], grid=grid)
    s, *_ = step1(cfg, s, [Z.ENV_INCANT])
    assert bool(s.pending[0]) and int(s.now) == C.COST_INCANTATION + OV
    s, *_ = step1(cfg, s, [Z.ENV_IDLE])
    assert int(s.level[0]) == 2 and int(s.grid[5, 5, C.LINEMATE]) == 0


def test_group_ritual_stays_synced_under_overhead():
    """Initiator and frozen participants must share one completion time —
    a participant completing first would unfreeze without leveling."""
    cfg = Z.make_cfg(11, 11, 2, no_food=True, no_refill=True, overhead=OV)
    grid = np.zeros((11, 11, 7), np.int32)
    grid[5, 5, C.LINEMATE] = 1
    grid[5, 5, C.DERAUMERE] = 1
    grid[5, 5, C.SIBUR] = 1
    s = mk_state(cfg, [[5, 5], [5, 5]], [C.NORTH, C.NORTH], level=[2, 2], grid=grid)
    s, *_ = step1(cfg, s, [Z.ENV_INCANT, Z.ENV_IDLE])
    assert bool(s.pending[0]) and bool(s.pending[1])
    assert int(s.busy_until[0]) == int(s.busy_until[1]) == C.COST_INCANTATION + OV
    s, *_ = step1(cfg, s, [Z.ENV_IDLE, Z.ENV_IDLE])
    assert int(s.level[0]) == 3 and int(s.level[1]) == 3


def test_density_scale_targets_and_reset():
    full = Z.make_cfg(20, 24, 1, no_refill=True)
    half = Z.make_cfg(20, 24, 1, no_refill=True, density_scale=0.5)
    for res in range(C.N_RESOURCES):
        want_full = max(1, int(20 * 24 * C.DENSITY[res]))
        want_half = max(1, int(20 * 24 * C.DENSITY[res] * 0.5))
        assert Z._target_qty(full, res) == want_full
        assert Z._target_qty(half, res) == want_half
    # reset scatters exactly the target quantity per resource (stacking ok)
    s_half, _ = Z.reset(half, KEY)
    for res in range(C.N_RESOURCES):
        assert int(jnp.sum(s_half.grid[:, :, res])) == Z._target_qty(half, res)


# ----------------------------------------------------------- life_noise
LIFE_FEAT = C.MAX_LEVEL + C.N_RESOURCES + 4  # index into self_feat columns

LN = 126.0


def test_life_noise_zero_is_bit_exact():
    # the default branch must compile to today's exact expression
    cfg0 = Z.make_cfg(11, 11, 2, no_refill=True)
    s, obs = Z.reset(cfg0, KEY)
    want = np.clip(np.asarray(s.life, np.float32)[:, None]
                   / np.float32(Z.START_LIFE), 0, 1)
    got = np.asarray(obs.self_feat[:, LIFE_FEAT:LIFE_FEAT + 1])
    assert np.array_equal(got, want)


def test_life_noise_one_sided_overstatement():
    # deploy belief error is in [0, +126): life_est anchors to the TOP of
    # the server's ceil(life/126) band — training noise must match
    cfg = Z.make_cfg(11, 11, 4, no_refill=True, life_noise=LN)
    s, _ = Z.reset(cfg, KEY)
    # away from the clip ceiling so the raw noise is observable
    s = s._replace(life=jnp.array([400, 500, 600, 700], jnp.int32))
    obs = Z.observe(cfg, s)
    feat = np.asarray(obs.self_feat[:, LIFE_FEAT]) * Z.START_LIFE
    err = feat - np.asarray(s.life, np.float32)
    assert np.all(err >= -1e-3)                    # never understates
    assert np.all(err <= LN + 1e-3)                # bounded by the knob
    assert len(np.unique(np.round(err, 3))) > 1    # per-agent independent


def test_life_noise_fresh_each_tick():
    # no_food freezes life at the clip ceiling — start below it so the raw
    # per-tick draw is observable
    cfg = Z.make_cfg(11, 11, 1, no_food=True, no_refill=True, life_noise=LN)
    s = mk_state(cfg, [[5, 5]], [C.NORTH])._replace(
        life=jnp.array([600], jnp.int32))
    errs = set()
    for _i in range(4):
        s, obs, *_ = step1(cfg, s, [Z.ENV_IDLE])
        feat = float(obs.self_feat[0, LIFE_FEAT]) * Z.START_LIFE
        errs.add(round(feat - float(s.life[0]), 2))
    assert len(errs) > 1  # resampled per tick, not frozen per episode


def test_life_noise_never_touches_dynamics():
    cfg0 = Z.make_cfg(11, 11, 2, no_refill=True)
    cfgN = Z.make_cfg(11, 11, 2, no_refill=True, life_noise=LN)
    s0, _ = Z.reset(cfg0, KEY)
    sN, _ = Z.reset(cfgN, KEY)
    k = jax.random.PRNGKey(7)
    for act in ([Z.ENV_FORWARD, Z.ENV_LEFT], [Z.ENV_TAKE0, Z.ENV_FORWARD],
                [Z.ENV_IDLE, Z.ENV_IDLE]):
        s0, *_ = Z.step(cfg0, k, s0, jnp.array(act), jnp.zeros(2, jnp.int32))
        sN, *_ = Z.step(cfgN, k, sN, jnp.array(act), jnp.zeros(2, jnp.int32))
    for name in ("life", "now", "pos", "level", "inv", "busy_until"):
        assert np.array_equal(np.asarray(getattr(s0, name)),
                              np.asarray(getattr(sN, name))), name

"""Training-only robustness knobs: cadence ``overhead`` + ``density_scale``.

Defaults are oracle-exact (overhead=0, density_scale=1.0) — the rest of the
suite (incl. the oracle cross-check) pins that. These tests pin the knob
semantics: overhead is added to every real action's cooldown but NOT to
ENV_IDLE, ritual initiators and frozen participants stay completion-synced
under overhead, and density_scale scales spawn targets.
"""

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

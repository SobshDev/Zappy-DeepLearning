"""Recurrent actor / centralized critic for Zappy MAPPO.

Mirrors PureJaxRL's recurrent PPO networks (``ScannedRNN`` over a GRU,
https://github.com/luchris429/purejaxrl) with the plan's locked knobs:

* **GRU(128)** with **separate** actor and critic RNNs (sharing one RNN
  between actor and critic destabilizes recurrent MAPPO).
* The actor emits two heads: the 20-way env action and the 8-token broadcast
  symbol. The token only takes effect when the sampled action is
  ``ENV_BROADCAST`` — ``mappo.py`` masks its log-prob/entropy accordingly so
  the token head receives no gradient noise from non-broadcast steps.
* The critic is *centralized* (CTDE / MAPPO): it sees the agent's own
  observation plus privileged world features built from the raw env state
  (``mappo.world_extra``).

All sequence inputs are time-major ``[T, B, ...]``. ``dones[t]`` flags "the
observation at step t is the first of a new episode" and resets the GRU carry
*before* consuming input t (PureJaxRL convention) — the same flag layout is
used during rollouts (length-1 sequences) and TBPTT updates (length-16
chunks), so log-probs recompute exactly.

Distributions are hand-rolled categoricals (``cat_*`` helpers) instead of
distrax to avoid the distrax/tfp ↔ new-JAX version fragility; they are exact
log-softmax math, nothing clever.
"""

from __future__ import annotations

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from ..env import constants as C
from ..env import zappy_env as Z

# Flat policy-input layout (shared with the deploy adapter — part of the
# sim<->server contract): vision tiles, self features, heard-message one-hots.
OBS_DIM = Z.MAX_VISION_TILES * (C.N_RESOURCES + 1) + Z.SELF_DIM + 9 + C.BROADCAST_VOCAB
VISION_SCALE = 0.2  # mild rescale: tile counts are small ints (0..~5)


def flatten_obs(obs: Z.Obs) -> jnp.ndarray:
    """Flatten an env ``Obs`` into the ``[..., OBS_DIM]`` policy input."""
    vis = obs.vision.reshape(*obs.vision.shape[:-2], -1) * VISION_SCALE
    return jnp.concatenate([vis, obs.self_feat, obs.msg_dir, obs.msg_tok], axis=-1)


# ------------------------------------------------------------ categoricals
def cat_sample(key, logits):
    return jax.random.categorical(key, logits)


def cat_log_prob(logits, a):
    logp = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(logp, a[..., None], axis=-1)[..., 0]


def cat_entropy(logits):
    logp = jax.nn.log_softmax(logits)
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)


# ------------------------------------------------------------------- RNNs
class ScannedRNN(nn.Module):
    """GRU scanned over time; resets the carry where ``dones`` is set."""

    hidden: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(
            resets[:, None],
            self.initialize_carry(ins.shape[0], self.hidden),
            carry,
        )
        new_carry, y = nn.GRUCell(features=self.hidden)(carry, ins)
        return new_carry, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden: int):
        return nn.GRUCell(features=hidden).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden)
        )


class RecurrentActor(nn.Module):
    """obs -> Dense(256) -> GRU(hidden) -> Dense(128) -> action + token heads."""

    n_actions: int = Z.N_ENV_ACTIONS
    n_tokens: int = C.BROADCAST_VOCAB
    hidden: int = 128

    @nn.compact
    def __call__(self, h, xs):
        obs, dones = xs  # [T,B,OBS_DIM], [T,B]
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        h, x = ScannedRNN(self.hidden)(h, (x, dones))
        x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        act_logits = nn.Dense(self.n_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        tok_logits = nn.Dense(self.n_tokens, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return h, act_logits, tok_logits


class RecurrentCritic(nn.Module):
    """centralized input -> Dense(256) -> GRU(hidden) -> Dense(128) -> V."""

    hidden: int = 128

    @nn.compact
    def __call__(self, h, xs):
        x, dones = xs  # [T,B,Dc], [T,B]
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        h, x = ScannedRNN(self.hidden)(h, (x, dones))
        x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return h, jnp.squeeze(v, axis=-1)

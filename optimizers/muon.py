"""Util functions."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from typing import NamedTuple
from omegaconf import DictConfig
from ._utils import newton_schulz


class ScaleByMuonState(NamedTuple):
    momentum: optax.Updates


def scale_by_muon(
        beta: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
) -> optax.GradientTransformation:
    def init_fn(params):
        return ScaleByMuonState(momentum=jtu.tree_map(jnp.zeros_like, params))
    
    def update_fn(updates, state, params=None):
        del params
        momentum = jtu.tree_map(
            lambda m, g: beta * m + g, state.momentum, updates
        )
        if nesterov:
            updates = jtu.tree_map(
                lambda m, g: beta * m + g, momentum, updates
            )
        else:
            updates = momentum
        updates = jtu.tree_map(
            lambda u: newton_schulz(u, steps=ns_steps), updates
        )
        updates = jtu.tree_map(
            lambda u: u * max(1, u.shape[0] / u.shape[1])**0.5, updates
        )
        return updates, ScaleByMuonState(momentum=momentum)

    return optax.GradientTransformation(init_fn, update_fn)


def muon(
        learning_rate: optax.ScalarOrSchedule,
        beta: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5
) -> optax.GradientTransformation:
    return optax.chain(
        scale_by_muon(beta=beta, nesterov=nesterov, ns_steps=ns_steps),
        optax.scale_by_learning_rate(learning_rate)
    )


def init_muon(config: DictConfig):
    return muon(
        **config.config
    )
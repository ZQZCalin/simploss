"""A stateless log function"""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from typing import Any, List, Tuple, NamedTuple, Optional, Union, Callable, Protocol, Literal
from omegaconf import DictConfig
import wandb
from simploss import tree_util


LogState = chex.ArrayTree
LogMetrics = chex.ArrayTree


class LogInitFn(Protocol):
    def __call__(self, **extra_args: Any) -> LogState:
        """The `init` function."""


class LogUpdateFn(Protocol):
    def __call__(self, state: LogState, **extra_args: Any) -> Tuple[LogMetrics, LogState]:
        """The `update` function."""


class LogFn(NamedTuple):
    """Loss function implemented by value and gradient function.
    
    As a caveat, when user customizes a LogFn please include `**extra_args`
    in `init_fn` and `update_fn`. This enables the `chain()` function to work.
    """
    init: LogInitFn
    update: LogUpdateFn


def chain(
    *args: LogFn,
) -> LogFn:
    """Applies a chain of log functions.
    
    Examples:

        Chain standard log with heat map custom log.

            >>> from simploss import log
            >>> log1 = log.standard_log()
            >>> log2 = log.heapmap_log()
            >>> chained_log = log.chain(log1, log2)
            >>> params = jnp.zeros((10,))
            >>> state = chained_log.init(params=params) # we must use keyword-only arguments
            >>> params.update(state, loss_val=..., grads=..., params=...)
    """
    init_fns, update_fns = zip(*args)

    def init_fn(**extra_args):
        return tuple(fn(**extra_args) for fn in init_fns)
    
    def update_fn(state, **extra_args):
        if len(update_fns) != len(state):
            raise ValueError("The number of chained logs must be the same as states.")
        
        new_state = []
        metrics = {}
        for s, fn in zip(state, update_fns):
            new_metrics, new_s = fn(s, **extra_args)
            new_state.append(new_s)
            metrics.update(new_metrics)
        return metrics, tuple(new_state)
    
    return LogFn(init_fn, update_fn)


class StandardLogState(NamedTuple):
    grads_prev: optax.Updates
    params_prev: optax.Params
    updates_prev: optax.Updates
    cumulatives: chex.ArrayTree


def standard_log() -> LogFn:
    """An example of a log function."""
    def init_fn(params: optax.Params, **extra_args):
        state = StandardLogState(
            grads_prev = jnp.zeros_like(params),
            params_prev = params,
            updates_prev = jnp.zeros_like(params),
            cumulatives = {
                "loss_min": jnp.array(float("inf")),
                "grad/inner_sum": jnp.zeros([]),
                "updates/cos(g, g_prev)_sum": jnp.zeros([]),
                "updates/cos(g_prev, delta)_sum": jnp.zeros([]),
                "updates/cos(g, delta)_sum": jnp.zeros([]),
            },
        )
        return state
    
    def update_fn(
            state, 
            loss_val: chex.Array, 
            grads: optax.Updates,
            params: optax.Params,
            updates: optax.Updates, 
            params_target: optax.Params,
            **extra_args
    ):
        grads_prev = state.grads_prev
        params_prev = state.params_prev
        updates_prev = state.updates_prev
        cumulatives = state.cumulatives

        updates_target = tree_util.subtract(params, params_target)  # this is the negative optimal direction x-x*

        metrics = {
            "loss": loss_val,
            "grad/norm": tree_util.norm(grads),
            "grad/cos(g, x-x*)": tree_util.cosine(grads, updates_target),
            "updates/cos(delta, x-x*)": tree_util.cosine(updates, updates_target),
        }
        summands = {
            "grad/inner": tree_util.inner(grads, grads_prev),
            "updates/cos(g, g_prev)": tree_util.cosine(grads, grads_prev),
            # NOTE: by definition, delta is a function of g_prev.
            # updates is the output of optimzier after seeing grads, 
            # and is thus paired with grads.
            "updates/cos(g_prev, delta)": tree_util.cosine(grads_prev, updates_prev),
            "updates/cos(g, delta)": tree_util.cosine(grads, updates_prev),
        }

        cumulatives["loss_min"] = jnp.minimum(cumulatives["loss_min"], loss_val)
        for k, v in summands.items():
            cumulatives[f"{k}_sum"] += v
        
        metrics.update(summands)
        metrics.update(cumulatives)

        state = StandardLogState(
            grads_prev = grads,
            params_prev = params,
            updates_prev = updates,
            cumulatives = cumulatives,
        )
        return metrics, state
    
    return LogFn(init_fn, update_fn)


class HeatmapLogState(NamedTuple):
    step: chex.Array
    params_table: chex.Array


def heatmap_log(
    num_steps: int,
) -> LogFn:
    """Logs a custom heatmap plot."""
    def init_fn(params: optax.Params, **extra_args):
        d = len(tree_util.ravel(params))
        return HeatmapLogState(
            step = jnp.zeros([], dtype=jnp.int32),
            params_table = jnp.zeros((num_steps, d)),
        )
    
    def update_fn(state, params: optax.Params, **extra_args):
        step = state.step
        params_table = state.params_table
        params = tree_util.ravel(params)

        params_table = params_table.at[step].set(params)

        state = HeatmapLogState(
            step = optax.safe_int32_increment(step),
            params_table = params_table,
        )
        return {}, state
    
    return LogFn(init_fn, update_fn)




def init_log(config: DictConfig) -> LogFn:
    return standard_log()
    # return chain(
    #     standard_log(),
    #     heatmap_log(num_steps=config.train.steps),
    # )
    # return standard_log()
    return heatmap_log(num_steps=config.train.steps, num_bins=10)
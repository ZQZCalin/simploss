"""Loss functions mapping $R^d \to R$."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from omegaconf import DictConfig
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol, Literal


PyTree = Any
LossState = optax.OptState
Grads = optax.Updates
LossVal = chex.Array  # Loss value is a singleton of `jnp.ndarrays`.


class LossValFn(Protocol):
    def __call__(self, params: optax.Params) -> LossVal:
        """The `value` function."""


class LossGradFn(Protocol):
    def __call__(self, params: optax.Params) -> optax.Updates:
        """The `grad` function."""


class LossMinimaFn(Protocol):
    def __call__(self, params: optax.Params) -> optax.Params:
        """The `minima` function."""


class LossFn(NamedTuple):
    """Loss function implemented by value and gradient function."""
    val: LossValFn
    grad: LossGradFn
    minima: LossMinimaFn
    


def valley_loss(
    x0: float,
    L: Union[float, chex.Array],
    rotation: Optional[chex.Array] = None,
) -> LossFn:
    """Valley loss.
    
    Computes loss $f(x) = \sum_{i=1}^d L_i * |\hat x_i - \hat x_{i-1}|$, where $\hat x = rotation @ x$.
    **Note:** params must be jnp.1darray.

    Args:
        x0: optimal point.
        L: scalar or array, Lipschitz constant of each valley.
        rotation: optinal rotation matrix.

    Returns:
        A `LossFn` object.
    """

    def val_fn(params: chex.Array):
        if rotation is not None:
            x = rotation @ params
        else:
            x = params
        
        if isinstance(L, float):
            return L * (jnp.sum(jnp.abs(x[1:] - x[:-1])) + jnp.abs(x[0] - x0))
        else:
            return jnp.sum(L[1:] * jnp.abs(x[1:] - x[:-1])) + L[0] * jnp.abs(x[0] - x0)
        
    def minima_fn(params: chex.Array):
        return jtu.tree_map(lambda p: x0 * p, params)
    
    return LossFn(val_fn, jax.grad(val_fn), minima_fn)


def bucket_loss(
    x0: Union[float, chex.Array],
    rotation: Optional[chex.Array] = None,
) -> LossFn:
    """Bucket loss.
    
    Computes loss $f(x) = \max |\hat x_i - x0_i|$, where $\hat x = rotation @ x$.
    **Note:** params must be jnp.1darray.

    Args:
        x0: scalar or array, optimal point in each coordinate.
        rotation: optinal rotation matrix.

    Returns:
        A `LossFn` object.
    """

    def val_fn(params: chex.Array):
        if rotation is not None:
            x = rotation @ params
        else:
            x = params
        
        return jnp.max(jnp.abs(x - x0))
    
    def minima_fn(params: chex.Array):
        del params
        return x0
    
    return LossFn(val_fn, jax.grad(val_fn), minima_fn)



def init_loss(config: DictConfig) -> LossFn:
    """Initialize loss function."""
    d = config.train.dim
    rotation_seed = config.random.rotation

    def init_rotation(config):
        if config.name is None:
            return None
        if config.name == "random":
            rotation = jr.normal(key=jr.PRNGKey(rotation_seed), shape=(d, d))
            rotation /= jnp.sqrt(d)
        else:
            raise ValueError(f"invalid config: valley_loss.rotation cannot be '{config.name}'")
        return rotation

    def init_valley_loss(config):
        if isinstance(config.L, float):
            L = config.L
        # TODO: add expression for Lipschiz values
        elif config.L == "":
            pass
        else:
            raise ValueError(f"invalid config: valley_loss.L cannot be '{config.L}'")
        
        if not isinstance(config.x0, float):
            raise ValueError(f"invalid config: valley_loss.x0 must be float")
        
        rotation = init_rotation(config.rotation)
        return valley_loss(config.x0, L, rotation)
    
    def init_bucket_loss(config):
        if isinstance(config.x0, float):
            x0 = config.x0
        elif isinstance(config.x0, str):
            func = lambda x: eval(config.x0, {}, {"i": x, "d": d})
            x0 = jnp.array([func(i) for i in range(d)])
        else:
            raise ValueError(f"invalid config: bucket_loss.x0 cannot be '{config.x0}'")
        
        rotation = init_rotation(config.rotation)
        return bucket_loss(x0, rotation)

    if config.loss.name == "valley":
        return init_valley_loss(config.loss)
    if config.loss.name == "bucket":
        return init_bucket_loss(config.loss)
    raise ValueError(f"invalid config: loss.name cannot be `{config.loss.name}'")



# Testing code below...
if __name__ == "__main__":
    d = 10
    rotation = jr.normal(key=jr.PRNGKey(42), shape=(d, d))
    rotation /= jnp.sqrt(d)
    # loss = valley_loss(x0=1.0, L=1.0, rotation=rotation)
    loss = bucket_loss(x0=jnp.arange(0, 10), rotation=None)
    x = jnp.zeros((d,))
    val = loss.val(x)
    grad = loss.grad(x)
    print(val, grad)
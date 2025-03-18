"""Util functions."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, Callable, Optional
from jaxtyping import Array, PyTree
from functools import partial


@partial(jax.jit, static_argnames=("steps"))
def newton_schulz(G: Array, steps: int) -> Array:
    """An approximate Newton-Schulz method.
    
    Adapted from:

    https://github.com/KellerJordan/Muon/blob/master/muon.py

    Given a matrix G with SVD decomposition SUV^T, this function
    approximates US'V^T where S' is diagonal with values Uniform(0.5, 1.5)
    without needing to compute any SVD decomposition.
    """
    assert G.ndim == 2

    # NOTE: unlike original repo, the operation is not in bfloat16
    a, b, c = (3.4445, -4.7750,  2.0315)
    eps = 1e-7

    X = G
    if G.shape[0] > G.shape[1]:
        X = X.T

    X /= (jnp.linalg.norm(X, ord="fro") + eps)
    def body_func(i, val):
        X = val
        A = X @ X.T
        B = b * A + c * A @ A
        return a * X + B @ X
    X = jax.lax.fori_loop(
        0, steps, body_func, X
    )

    if G.shape[0] > G.shape[1]:
        X = X.T
    return X
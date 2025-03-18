import jax
import jax.numpy as jnp
from functools import partial
from omegaconf import DictConfig


@partial(jax.jit)
def matrix_valley_loss(params):
    # params: a list of matrices, first: optimal, last: rotation
    return

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr

import optax

import chex
from typing import Any, Tuple, NamedTuple, Optional, Callable, Union

import logging
from tqdm import tqdm
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig

from simploss import loss, optim, log, tree_util


if __name__ == "__main__":
    optimizer = optim.full_matrix_adagrad(learning_rate=1.0)
    params = jnp.zeros((10,))
    opt_state = optimizer.init(params)
    grad = jnp.ones_like(params)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    print(">>>", params, updates)
    print("update norm", tree_util.norm(updates))
    print("update corr", tree_util.cosine(updates, grad))
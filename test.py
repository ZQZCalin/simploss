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
    # optimizer = optim.full_matrix_adagrad(learning_rate=1.0, eps=1e-6)

    # raise KeyboardInterrupt

    # log_fn1 = log.standard_log()
    # fn = log.chain(log_fn1, log_fn1)
    # params = jnp.zeros((10,))
    # state = fn.init(params)
    # # state = fn.init(params=params)
    # metrics, _ = fn.update(state, loss_val=0, params=params, grad=params)
    # print(metrics)
    # raise KeyboardInterrupt

    optimizer = optim.full_matrix_adagrad(learning_rate=1.0, eps=1e-6)
    params = jnp.zeros((2,))
    opt_state = optimizer.init(params)
    grads_list = [
        jnp.array([2,1]),
        jnp.array([3,1]),
        jnp.array([-1,1]),
        jnp.array([2,1]),
        jnp.array([0,1]),
    ]
    for it, grads in enumerate(grads_list):
        print("\n>>>step", it)
        print("grads", grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        print("updates", updates)
        print("update norm", tree_util.norm(updates))
        print("update corr", tree_util.cosine(updates, grads))
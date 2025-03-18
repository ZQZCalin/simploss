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

import matplotlib.pyplot as plt
import numpy as np

from simploss import loss, optim, log
from simploss import TrainState, train


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(config))
    raise KeyboardInterrupt
    d = config.train.dim

    params = jnp.zeros((d,), dtype=jnp.float32)

    optimizer = optim.init_optimizer(config)
    opt_state = optimizer.init(params)

    loss_fn = loss.init_loss(config)

    log_fn = log.init_log(config)
    log_state = log_fn.init(params=params)

    train_state = TrainState(
        params = params,
        iteration = jnp.ones([], dtype=jnp.int32),
        opt_state = opt_state,
        log_state = log_state,
    )
    train(config, train_state, optimizer, loss_fn, log_fn)


if __name__ == "__main__":
    main()
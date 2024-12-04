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

from simploss import loss, optim, log


class TrainState(NamedTuple):
    params: optax.Params
    iteration: chex.Array
    opt_state: optax.OptState
    log_state: log.LogState


def train(
    config: DictConfig, 
    train_state: TrainState,
    optimizer: optax.GradientTransformation, 
    loss_fn: loss.LossFn,
    log_fn: log.LogFn,
) -> None:
    wandb_project = config.logging.wandb_project
    wandb_name = config.logging.wandb_name
    if wandb_project:
        wandb.init(project=wandb_project, name=wandb_name)
        wandb.config.update(OmegaConf.to_container(config))

    num_steps = config.train.steps

    jit_val_grad = jax.jit(
        lambda params: (loss_fn.val(params), loss_fn.grad(params))
    )
    jit_opt_update = jax.jit(optimizer.update)
    jit_log_update = jax.jit(log_fn.update)

    pbar = tqdm(range(num_steps), total=num_steps)
    for it in pbar:
        params = train_state.params
        iteration = train_state.iteration
        opt_state = train_state.opt_state
        log_state = train_state.log_state
        
        # Training logic.
        val, grad = jit_val_grad(params)
        updates, opt_state = jit_opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Visualization.
        log_state, metric = jit_log_update(log_state, val, params, grad)

        # Update train state.
        train_state = train_state._replace(
            params = params,
            iteration = optax.safe_int32_increment(iteration),
            opt_state = opt_state,
            log_state = log_state,
        )
        pbar.set_description(f"Iteration: {iteration}, Loss: {val:.4f}")
        if wandb_project:
            wandb.log(metric, step=iteration)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(config))
    # raise KeyboardInterrupt
    d = config.train.dim

    params = jnp.zeros((d,), dtype=jnp.float32)

    optimizer = optim.init_optimizer(config)
    opt_state = optimizer.init(params)

    loss_fn = loss.init_loss(config)

    log_fn = log.init_log(config)
    log_state = log_fn.init(params)

    train_state = TrainState(
        params = params,
        iteration = jnp.ones([], dtype=jnp.int32),
        opt_state = opt_state,
        log_state = log_state,
    )
    train(config, train_state, optimizer, loss_fn, log_fn)


if __name__ == "__main__":
    main()
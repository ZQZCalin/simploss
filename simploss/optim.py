"""Benchmark optimizers."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Callable, Union
from omegaconf import DictConfig
from simploss import tree_util, utils



ScalarOrPytree = Union[float, Any]


def get_current_lr(learning_rate, count):
    if callable(learning_rate):
        return learning_rate(count)
    else:
        return learning_rate


class AdamWState(NamedTuple):
    """AdamW State."""
    count: chex.Array
    mu: Updates
    nu: Updates


def adamw(
    learning_rate: ScalarOrSchedule = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: ScalarOrPytree = 0.0,
    debias_beta1: bool = True,
    debias_beta2: bool = True,
    use_momentum: bool = True,
    use_preconditioning: bool = True,
    decouple_weight_decay: bool = False,
) -> GradientTransformation:
    """AdamW for benchmark.

    Args:
        learning_rate (ScalarOrSchedule): _description_. Defaults to 1e-4.
        beta1 (float): _description_. Defaults to 0.9.
        beta2 (float): _description_. Defaults to 0.999.
        eps (float): _description_. Defaults to 1e-8.
        weight_decay (float): _description_. Defaults to 0.0.
        debias_beta1 (bool): Defaults to True.
        debias_beta2 (bool): Defaults to True.
        use_momentum (bool): Defaults to True. If false, replace \hat m_t with the gradients.
            However, m_t will still be compated based on beta1 and stored in the opt_state.
        use_preconditioning (bool): Defaults to True. If false, use \hat m_t as the update (without dividing by v_t).
            However, v_t will still be computed based on beta2 and stored in the opt_state.
        decouple_weight_decay (bool): Defaults to False. If true, learning rate eta will not be applied to weight_decay regularization.

    Returns:
        A `GradientTransformation` object.
    """

    use_pytree_wd = type(weight_decay) != float

    def init_fn(params):
        # Checks weight_decay structure during initialization.
        if use_pytree_wd and jtu.tree_structure(weight_decay)!=jtu.tree_structure(params):
            raise ValueError("structure of weight_decay must match model structure.")
        return AdamWState(
            count=jnp.zeros([], jnp.int32),
            mu=jtu.tree_map(jnp.zeros_like, params),
            nu=jtu.tree_map(jnp.zeros_like, params)
        )
    
    def update_fn(updates, state, params):
        count_inc = optax.safe_int32_increment(state.count)
        mu = jtu.tree_map(
            lambda m, g: beta1*m + (1-beta1)*g, state.mu, updates)
        nu = jtu.tree_map(
            lambda v, g: beta2*v + (1-beta2)*g**2, state.nu, updates)
        
        # Debias to get the true weighted average.
        if debias_beta1:
            mu_hat = utils.tree_scalar_multiply(mu, 1/(1-beta1**count_inc))
        else:
            mu_hat = mu
        if debias_beta2:
            nu_hat = utils.tree_scalar_multiply(nu, 1/(1-beta2**count_inc))
        else:
            nu_hat = nu

        # Other optional features: turn off momentum and/or pre-conditioning.
        if not use_momentum:
            mu_hat = updates
        if not use_preconditioning:
            nu_hat = jtu.tree_map(jnp.ones_like, nu_hat)

        # Unpack learning rate schedule.
        # eta = scheduler.get_current_lr(learning_rate, state.count)
        eta = get_current_lr(learning_rate, state.count)

        # Weight decay regularization.
        if not use_pytree_wd:
            regularization = utils.tree_scalar_multiply(params, weight_decay)
        else:
            regularization = utils.tree_multiply(params, weight_decay)
        if not decouple_weight_decay:
            regularization = utils.tree_scalar_multiply(regularization, eta)

        # Compute one-step update: -eta * [mu / (eps+sqrt(nu)) + lam * params]
        new_updates = jtu.tree_map(
            lambda m, v, r: -(eta * m / (eps+jnp.sqrt(v)) + r),
            mu_hat, nu_hat, regularization 
        )
        return new_updates, AdamWState(
            count=count_inc, mu=mu, nu=nu)
    
    return GradientTransformation(init_fn, update_fn)


class SgdmState(NamedTuple):
    count: chex.Array
    momentum: optax.Updates


def sgdm(
    learning_rate: ScalarOrSchedule,
    beta: float=0.0,
    weight_decay: ScalarOrPytree=0.0,
) -> GradientTransformation:
    """SGD with momentum.
    
    Updates m_{t+1} = beta * m_t - (1-beta) * (g_t + mu*x_t)
        and x_{t+1} = x_t - eta_t * m_{t+1}, 
    where beta is the momentum constant and mu is the weight decay constant.

    Args:
        learning_rate: The learning rate scheduler.
        beta: The momentum constant in [0, 1]. Defaults to 0.
        weight_decay (float): The weight decay constant. Defaults to 0.

    Returns:
        A `GradientTransformation` object.
    """
    
    use_pytree_wd = type(weight_decay) != float

    def init_fn(params):
        if use_pytree_wd and jtu.tree_structure(weight_decay)!=jtu.tree_structure(params):
            raise ValueError("structure of weight_decay must match model structure.")
        return SgdmState(
            count = jnp.zeros([], jnp.int32),
            momentum = jtu.tree_map(jnp.zeros_like, params),
        )
    
    def update_fn(updates, state, params):
        # TODO: which one to implement weight decay?
        # grads = jtu.tree_map(
        #     lambda g, x: g + mu*x, updates, params)
        # eta = scheduler.get_current_lr(learning_rate, state.count)
        eta = get_current_lr(learning_rate, state.count)
        new_momentum = jtu.tree_map(
            lambda m, g: beta*m + (1-beta)*g, state.momentum, updates)
        if not use_pytree_wd:
            new_updates = jtu.tree_map(
                lambda m, x: -eta * (m + weight_decay*x), new_momentum, params)
        else:
            new_updates = jtu.tree_map(
                lambda m, x, wd: -eta * (m + wd*x), new_momentum, params, weight_decay)
        return new_updates, SgdmState(
            count = optax.safe_int32_increment(state.count),
            momentum = new_momentum
        )
    
    return GradientTransformation(init_fn, update_fn)


class ScaleByFullMatrixAdagradState(NamedTuple):
    covariance: chex.ArrayTree


def scale_by_full_matrix_adagrad(
    eps: float = 1e-6,
) -> optax.GradientTransformation:
    """Full-matrix AdaGrad pre-learning rate."""
    def init_fn(params):
        covariance = jtu.tree_map(
            jnp.zeros_like,
            tree_util.outer(params, params),
        )
        return ScaleByFullMatrixAdagradState(
            covariance = covariance,
        )

    def update_fn(updates, state, params=None):
        del params
        def matrix_inv_sqrt(M):
            # change to SVD
            # D, U = jnp.linalg.eigh(M)
            # D_inv = jnp.diag(1.0 / (jnp.sqrt(D) + eps))
            # return U @ D_inv @ U.T
            u, s, vt = jnp.linalg.svd(M)
            s_inv = jnp.diag(1.0 / (jnp.sqrt(s) + eps))
            return u @ s_inv @ vt

        covariance = state.covariance
        covariance = tree_util.add(
            covariance, tree_util.outer(updates, updates))
        # print("covariance\n", covariance)
        
        preconditioner = jtu.tree_map(matrix_inv_sqrt, covariance)
        # print("pre-conditioner\n", preconditioner)
        updates = jtu.tree_map(
            lambda P, g: P @ g, preconditioner, updates
        )
        updates = tree_util.normalize(updates)

        state = ScaleByFullMatrixAdagradState(
            covariance = covariance,
        )

        return updates, state
    
    return GradientTransformation(init_fn, update_fn)


def full_matrix_adagrad(
    learning_rate: ScalarOrSchedule,
    eps: float = 1e-8,
) -> GradientTransformation:
    """Full-matrix AdaGrad.
    
    https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

    Matrix pre-conditioner is applied to each tree leaf,
    where the covariance matrix is computed via the outer product
    of the flattened gradient array.
    """
    return optax.chain(
        scale_by_full_matrix_adagrad(eps),
        optax.scale_by_learning_rate(learning_rate),
    )


def ggt() -> GradientTransformation:
    """GGT, an efficient implementation of full-matrix AdaGrad.
    
    https://arxiv.org/pdf/1806.02958
    """
    return GradientTransformation



def init_optimizer(config: DictConfig) -> optax.GradientTransformation:
    """Initialize optimizer (and optionally scheduler)."""
    def init_adam(config):
        return adamw(
            learning_rate=config.schedule.lr,
            beta1=config.beta1,
            beta2=config.beta2,
            weight_decay=config.weight_decay,
            debias_beta1=config.debias_beta1,
            debias_beta2=config.debias_beta2,
            use_momentum=config.use_momentum,
            use_preconditioning=config.use_preconditioning,
        )
    
    def init_full_matrix_adagrad(config):
        return full_matrix_adagrad(
            learning_rate=config.schedule.lr,
            eps=config.eps,
        )

    if config.optimizer.name == "adam":
        return init_adam(config.optimizer)
    if config.optimizer.name == "full_matrix_adagrad":
        return init_full_matrix_adagrad(config.optimizer)
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from functools import partial
from tqdm import tqdm
import wandb
from optimizers import muon, trapezoid_schedule


d = 128
n = 1000
seed = 42
num_steps = 10000

use_wandb = True
wandb_project = "simploss"
wandb_name = "matrix_valley_adamw"

dist = lambda m1, m2: jnp.linalg.norm(m1-m2)**2 / (m1.shape[0]*m1.shape[0]) # rms-squared (normalized to 1)

@partial(jax.jit)
def loss_fn(params):
    x0 = jnp.eye(d)
    loss = dist(x0, params[0])
    loss += jtu.tree_reduce(
        lambda x, y: x + y,
        jtu.tree_map(
            lambda p1, p2: dist(p1, p2), params[1:], params[:-1]
        )
    )
    loss /= (2*n)
    return loss

def main():
    key = jr.PRNGKey(seed=seed)
    this_key, key = jr.split(key)
    params = [jr.normal(k, shape=(d,d)) for k in jr.split(this_key, num=n)]

    learning_rate = trapezoid_schedule(0.05, 1000, 200, 200)
    # optimizer = muon(learning_rate, beta=0.95)
    optimizer = optax.adamw(learning_rate, b1=0.95, b2=0.95)
    opt_state = optimizer.init(params)

    jit_loss = jax.jit(lambda params: (loss_fn(params), jax.grad(loss_fn)(params)))
    jit_update = jax.jit(optimizer.update)

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
        )

    iteration = 0
    pbar = tqdm(range(num_steps), total=num_steps)
    for it in pbar:
        # Training logic.
        iteration += 1
        loss, grads = jit_loss(params)
        updates, opt_state = jit_update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        pbar.set_description(f"Iteration: {iteration}, Loss: {loss:.4f}")
        if use_wandb:
            metrics = {
                "loss": loss
            }
            wandb.log(metrics, step=iteration)


if __name__ == "__main__":
    main()
"""Schedule functions."""
import optax
from omegaconf import DictConfig


def trapezoid_schedule(
        peak_value: float,
        total_steps: int,
        warmup_steps: int = 0,
        decay_steps: int = 0,
) -> optax.Schedule:
    schedules = [
        optax.linear_schedule(
            init_value=0.0,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        optax.linear_schedule(
            init_value=peak_value,
            end_value=peak_value,
            transition_steps=total_steps - warmup_steps - decay_steps,
        ),
        optax.linear_schedule(
            init_value=peak_value,
            end_value=0.0,
            transition_steps=decay_steps,
        )
    ]
    return optax.join_schedules(schedules, [warmup_steps, total_steps - decay_steps])


is_positive_int = lambda var: isinstance(var, int) and (var > 0)

def init_constant_lr(config):
    learning_rate = config.lr
    return learning_rate

def init_cosine_lr(config):
    if is_positive_int(config.warmup):
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.lr,
            warmup_steps=config.warmup,
            decay_steps=config.max_steps,
        )
    else:
        learning_rate = optax.cosine_decay_schedule(
            init_value=config.lr,
            decay_steps=config.max_steps,
        )
    return learning_rate

def init_linear_lr(config):
    warmup_steps = config.warmup if is_positive_int(config.warmup) else 0
    const_steps = config.const if is_positive_int(config.const) else 0
    learning_rate = optimizers.warmup_const_linear_decay_schedule(
        peak_value=config.lr,
        warmup_steps=warmup_steps,
        const_steps=const_steps,
        total_steps=config.max_steps,
        init_value=0.0,
        end_value=0.0,
    )
    return learning_rate

def init_trapezoid_lr(config):
    warmup_steps = config.warmup if is_positive_int(config.warmup) else 0
    decay_steps = config.decay if is_positive_int(config.decay) else 0
    learning_rate = optimizers.trapezoid_schedule(
        peak_value=config.lr,
        total_steps=config.max_steps,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )
    return learning_rate

def init_piecewise_linear_lr(config):
    learning_rate = optax.linear_schedule(
        init_value=config.lr1,
        end_value=config.lr2,
        transition_steps=config.max_steps,
        transition_begin=config.start_steps,    # NOTE: for now, we still need to specify the start iteration in config.
    )
    return learning_rate



def init_schedule(lr_config: DictConfig) -> optax.ScalarOrSchedule:
    """Parses the config and initializes a learning rate scheduler.

    Args:
        lr_config: The learning rate config.
        kargs: Additional arguments to overwrite learning rate config.

    Returns:
        A `optax.ScalarOrSchedule` object.
    """
    is_positive_int = lambda x: isinstance(x, int) and (x > 0)

    def init_constant_lr(config):
        learning_rate = config.lr
        return learning_rate
    
    def init_cosine_lr(config):
        if is_positive_int(config.warmup):
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=config.lr,
                warmup_steps=config.warmup,
                decay_steps=config.max_steps,
            )
        else:
            learning_rate = optax.cosine_decay_schedule(
                init_value=config.lr,
                decay_steps=config.max_steps,
            )
        return learning_rate
    
    def init_linear_lr(config):
        warmup_steps = config.warmup if is_positive_int(config.warmup) else 0
        const_steps = config.const if is_positive_int(config.const) else 0
        learning_rate = optimizers.warmup_const_linear_decay_schedule(
            peak_value=config.lr,
            warmup_steps=warmup_steps,
            const_steps=const_steps,
            total_steps=config.max_steps,
            init_value=0.0,
            end_value=0.0,
        )
        return learning_rate
    
    def init_trapezoid_lr(config):
        warmup_steps = config.warmup if is_positive_int(config.warmup) else 0
        decay_steps = config.decay if is_positive_int(config.decay) else 0
        learning_rate = optimizers.trapezoid_schedule(
            peak_value=config.lr,
            total_steps=config.max_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        )
        return learning_rate

    def init_piecewise_linear_lr(config):
        learning_rate = optax.linear_schedule(
            init_value=config.lr1,
            end_value=config.lr2,
            transition_steps=config.max_steps,
            transition_begin=config.start_steps,    # NOTE: for now, we still need to specify the start iteration in config.
        )
        return learning_rate

    if lr_config.schedule == "constant":
        learning_rate = init_constant_lr(lr_config)
    elif lr_config.schedule == "cosine":
        learning_rate = init_cosine_lr(lr_config)
    elif lr_config.schedule == "linear":
        learning_rate = init_linear_lr(lr_config)
    elif lr_config.schedule == "trapezoid":
        learning_rate = init_trapezoid_lr(lr_config)
    elif lr_config.schedule == "piecewise_linear":
        learning_rate = init_piecewise_linear_lr(lr_config)
    else:
        raise ValueError(f"schedule type {lr_config.schedule} is not supported.")
    return learning_rate
"""Complementary to jax.tree_util functions."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from typing import Union
import chex


PyTree = chex.ArrayTree
Scalar = chex.Array


def add(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Add two PyTrees."""
    return jtu.tree_map(lambda x,y: x+y, tree1, tree2)


def subtract(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Subtract tree1 by tree2."""
    return jtu.tree_map(lambda x,y: x-y, tree1, tree2)


def multiply(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Multiply two PyTrees."""
    return jtu.tree_map(lambda x,y: x*y, tree1, tree2)


def scalar_dot(tree: PyTree, scalar: Union[float, Scalar]) -> PyTree:
    """Multiply PyTree with a scalar."""
    return jtu.tree_map(lambda x: scalar*x, tree)


def zeros_like(tree: PyTree) -> PyTree:
    """Returns an all-zero PyTree."""
    return jtu.tree_map(jnp.zeros_like, tree)


def norm(tree: PyTree, p: float=2) -> Scalar:
    """Norm of a Pytree.
    
    p = 1, 2, inf have specific implementation, other values are generic.
    """
    if p == 2:
        return jnp.sqrt(jtu.tree_reduce(
            lambda x, y: x + y,
            jtu.tree_map(lambda x: jnp.sum(x*x), tree)
        ))
    if p == 1:
        return jtu.tree_reduce(
            lambda x, y: x + y,
            jtu.tree_map(lambda x: jnp.sum(jnp.abs(x)), tree)
        )
    if p == float("inf"):
        return jtu.tree_reduce(
            lambda x, y: jnp.maximum(x, y),
            jtu.tree_map(lambda x: jnp.max(x), tree)
        )
    return jtu.tree_reduce(
        lambda x, y: x + y,
        jtu.tree_map(lambda x: jnp.sum(jnp.abs(x)**p), tree)
    ) ** (1/p)


def normalize(tree: PyTree, p: float=2) -> PyTree:
    """Normalize a PyTree."""
    n = norm(tree, p)
    return jax.lax.cond(
        n == 0,
        true_fun=lambda _: jnp.array(tree, dtype=jnp.float32),
        false_fun=lambda _: jtu.tree_map(lambda x: x/n, tree),
        operand=None,
    )


def inner(tree1: PyTree, tree2: PyTree) -> Scalar:
    """Inner product of two PyTrees."""
    return jtu.tree_reduce(
        lambda x, y: x + y,
        jtu.tree_map(lambda x, y: jnp.sum(x*y), tree1, tree2)
    )


def cosine(tree1: PyTree, tree2: PyTree) -> Scalar:
    """Cosine similarity of two PyTrees."""
    return inner(normalize(tree1), normalize(tree2))


def outer(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Broadcast jax.numpy.outer to two PyTrees."""
    return jtu.tree_map(lambda x, y: jnp.outer(x, y), tree1, tree2)


def ravel(tree: PyTree) -> chex.Array:
    """Broadcast jax.numpy.ravel to a PyTree."""
    leaves, _ = jtu.tree_flatten(tree)
    return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])



# Testing code below...
if __name__ == "__main__":
    a = jnp.ones((2, 3))
    b = a * (jnp.arange(3)+1)
    print(a,b)
    print(norm(a, p=float("inf")))
    print(inner(a,b))
    print(cosine(a,b))
    print(cosine(a,a))
    print(normalize(jnp.zeros((3,1))))
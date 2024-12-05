"""Util functions."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from jax import Array
from optax import GradientTransformation
import chex
from typing import Tuple


PyTree = chex.ArrayTree


def matrix_inv_sqrt(M: chex.Array) -> chex.Array:
    """Compute M^{-1/2}.
    
    Args:
        M: jnp.2darray, symmetric square matrix.

    Returns:
        M^{-1/2}
    """
    D, U = jnp.linalg.eigh(M)
    D_inv = jnp.diag(1.0 / jnp.sqrt(D))
    return U @ D_inv @ U.T


# if __name__ == "__main__":

#     M = jnp.array([[4,0], [0,4]])
#     print(matrix_inv_sqrt(M))

#     raise KeyboardInterrupt



# Other util functions.
def merge_non_zero_dict(target, source):
    """Merges non-zero items in source dictionary into target dictionary.
    This is a mutable operation.
    """
    for key, value in source.items():
        if not value == 0:
            target[key] = value


# Util functions for tree manipulation. 
def zero_tree(tree):
    """Returns an all-zero tree with the same structure as the input."""
    return jtu.tree_map(jnp.zeros_like, tree)


def tree_add(tree1, tree2):
    return jtu.tree_map(lambda x,y: x+y, tree1, tree2)


def tree_subtract(tree1, tree2):
    return jtu.tree_map(lambda x,y: x-y, tree1, tree2)


def tree_multiply(tree1, tree2):
    return jtu.tree_map(lambda x,y: x*y, tree1, tree2)


def tree_dot(tree1, tree2):
    return jtu.tree_reduce(
        lambda x,y: x+y,
        jtu.tree_map(lambda x,y: jnp.dot(x,y), tree1, tree2)
    )


def negative_tree(tree):
    """A `jtu.tree_map`-broadcasted version of tree -> -tree."""
    return jtu.tree_map(lambda t: -t, tree)


def tree_scalar_multiply(tree, scalar):
    return jtu.tree_map(lambda x: scalar*x, tree)


def tree_l1_norm(tree):
    """Returns the l1 norm of the vectorized tree."""
    return jtu.tree_reduce(
        lambda x, y: x + y,
        jtu.tree_map(lambda x: jnp.sum(jnp.abs(x)), tree)
    )


def tree_l2_norm(tree):
    """Returns the l2 norm of the vectorized tree."""
    return jnp.sqrt(
        jtu.tree_reduce(
            lambda x, y: x + y, jtu.tree_map(lambda x: jnp.sum(x * x), tree)
        )
    )


# TODO: deprecated, to be removed
def tree_norm(tree):
    """Returns the l2 norm of the vectorized tree."""
    return tree_l2_norm(tree)


def is_zero_tree(tree):
    """Checks if a tree only has zero entries."""
    return jtu.tree_reduce(
        lambda x, y: x & y, jtu.tree_map(lambda x: jnp.all(x == 0), tree)
    )


def is_finite_tree(tree):
    """Returns whether a tree is finite."""
    leaves = jtu.tree_flatten(tree)[0]
    return jnp.all(
        jnp.array([jnp.all(jnp.isfinite(node)) for node in leaves]))


def tree_normalize(tree):
    # Use jax.lax.cond to avoid trace issue.
    return jax.lax.cond(
        is_zero_tree(tree),
        true_fun=lambda _: zero_tree(tree),
        false_fun=lambda _: tree_scalar_multiply(tree, 1/tree_norm(tree)),
        operand=None,
    )


def tree_inner_product(tree1, tree2):
    leaves1, _ = jtu.tree_flatten(tree1)
    leaves2, _ = jtu.tree_flatten(tree2)
    return sum(jnp.sum(a * b) for a, b in zip(leaves1, leaves2))


def tree_cosine_similarity(tree1, tree2):
    """Returns the cosine similarity of two trees."""
    return tree_inner_product(tree_normalize(tree1), tree_normalize(tree2))


def tree_norm_direction_decomposition(tree):
    """Decomposes the norm and the direction of a tree.

    Returns:
        The norm of a tree (1d array) and the normalized tree.
        If the tree is all zeros, then return 0 as the norm and an all-zero tree.
    """
    def true_fun(_):
        return jnp.zeros([], jnp.float32), tree
    def false_fun(_):
        norm = tree_norm(tree)
        return norm, tree_scalar_multiply(tree, 1/norm)
    return jax.lax.cond(
        is_zero_tree(tree), true_fun, false_fun, operand=None)
    # norm = tree_norm(tree)
    # # NOTE: we need to return jax.Array to make sure both branches returns the
    # # same data structure and thus avoid lax.cond issue
    # if norm == 0:
    #     return jnp.zeros([], jnp.float32), tree
    # return norm, tree_scalar_multiply(tree, 1/norm)


def random_unit_vector(tree, *, key):
    """Constructs a pytree of same structure as input whose leaves is a random unit vector.

    Returns:
        A uniform random vector on the unit sphere.
    """
    # Construct a pytree of random keys.
    keys = jr.split(key, num=len(jtu.tree_leaves(tree)))
    keys_tree = jtu.tree_unflatten(jtu.tree_structure(tree), keys)
    # Sample Gaussian vector.
    normal_vector = jtu.tree_map(
        lambda t, k: jr.normal(k, shape=t.shape), 
        tree, keys_tree
    )
    return tree_normalize(normal_vector)


def check_tree_structures_match(tree1, tree2):
    """Check whether tree1 and tree2 have the same tree structure. 
        Raises error when structures do not match.
    """
    if jtu.tree_structure(tree1) != jtu.tree_structure(tree2):
        raise ValueError("Input Pytrees do not have the same structure")
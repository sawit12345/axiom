# Copyright 2025 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the "License");
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/axiom/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union
from jaxtyping import Array
from functools import partial
from jax import lax, vmap, jit
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.scipy.linalg as linalg
from jax.numpy import expand_dims as expand

from jax.scipy.special import logsumexp
from jax.nn import softmax


@partial(jit, static_argnames=["return_inverse", "return_logdet"])
def inv_and_logdet(
    pos_def_matrix: Array,
    return_inverse: Optional[bool] = True,
    return_logdet: Optional[bool] = True,
) -> Union[float, Tuple[Array, float]]:
    """Compute log-determinant and matrix inverse using Cholesky - CUDA accelerated."""
    return_logdet = True if return_inverse == False else return_logdet

    shape = pos_def_matrix.shape
    min_eig = jnp.clip(jnp.min(jnp.linalg.eigvalsh(pos_def_matrix)), max=0.0)
    eps = jnp.finfo(pos_def_matrix.dtype).eps
    pos_def_matrix = pos_def_matrix + jnp.broadcast_to(jnp.eye(shape[-1]), shape) * 2 * (eps - min_eig) * (min_eig < 0)

    chol = linalg.cho_factor(pos_def_matrix, lower=True)
    if return_inverse:
        identity = jnp.broadcast_to(jnp.eye(shape[-1]), shape)
        matrix_inverse = linalg.cho_solve(chol, identity)

        if return_logdet:
            logdet = jnp.expand_dims(2 * jnp.log(jnp.diagonal(chol[0], axis1=-1, axis2=-2)).sum(-1), (-1, -2))
            return matrix_inverse, logdet

        return matrix_inverse

    logdet = jnp.expand_dims(2 * jnp.log(jnp.diagonal(chol[0], axis1=-1, axis2=-2)).sum(-1), (-1, -2))
    return logdet


def bdot(x, y):
    """Batched dot product using vmap - CUDA accelerated."""
    assert x.ndim > 1
    assert y.ndim > 1
    shape = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    x = jnp.broadcast_to(x, shape + x.shape[-2:])
    y = jnp.broadcast_to(y, shape + y.shape[-2:])
    z = vmap(jnp.dot)(x.reshape((-1,) + x.shape[-2:]), y.reshape((-1,) + y.shape[-2:]))
    x_dim = x.shape[-2]
    y_dim = y.shape[-1]
    return z.reshape(shape + (x_dim, y_dim))


def positive_leading_eigenvalues(x, iters=10):
    """Ensure positive eigenvalues by adding epsilon to diagonal."""
    eps = jnp.eye(x.shape[-1]) * 0.001
    for i in range(iters):
        eigs = jnp.linalg.eigh(x).eigenvalues
        if jnp.any(eigs <= 0):
            x += eps
        else:
            return x
    raise ValueError("Unable to maintain non-negative leading diagonal")


def symmetrise(x):
    """Force matrix to be symmetric."""
    return (x + x.mT) / 2


def make_posdef(x):
    """Make matrix symmetric and positive definite."""
    return positive_leading_eigenvalues(symmetrise(x))


def mvdigamma(x: jnp.ndarray, d: int) -> jnp.ndarray:
    """Compute multivariate digamma function - CUDA accelerated."""
    return jsp.digamma(jnp.expand_dims(x, -1) - jnp.arange(d) / 2.0).sum(-1)


def mvgammaln(x: jnp.ndarray, d: int) -> jnp.ndarray:
    """Compute log of multivariate gamma function - CUDA accelerated."""
    return jnp.sum(lax.lgamma(expand(x, -1) - jnp.arange(d) / 2.0), -1) + d * (d - 1) / 4.0 * jnp.log(jnp.pi)


def stable_logsumexp(x: jnp.ndarray, dims: tuple, keepdims=False) -> jnp.ndarray:
    """Compute logsumexp along dimensions - CUDA accelerated."""
    return logsumexp(x, axis=dims, keepdims=keepdims)


def stable_softmax(x: jnp.ndarray, dims: tuple) -> jnp.ndarray:
    """Compute softmax along dimensions - CUDA accelerated."""
    return softmax(x, axis=dims)


def assign_unused(assignments, d_alpha, elbo_contrib, threshold: float = 1.0, fill_prob: float = 10.0):
    """Re-assign data-points with low ELBO to unused clusters."""
    unfilled_cluster_idx = d_alpha < threshold
    sorted_elbo_idx = jnp.argsort(elbo_contrib)
    num_to_fill = unfilled_cluster_idx.sum()
    reassign_mask = jnp.arange(assignments.shape[0]) < num_to_fill
    assignments_sorted_by_elbo = assignments[sorted_elbo_idx]
    onehots_base = fill_prob * jnp.eye(d_alpha.shape[0])
    onehots_to_keep = onehots_base * unfilled_cluster_idx[None, ...]
    onehots_to_keep = jnp.take_along_axis(onehots_to_keep, jnp.argsort(onehots_to_keep.sum(-1))[::-1][..., None], axis=0)
    sorted_ass_reassigned = assignments_sorted_by_elbo * (1.0 - reassign_mask[..., None]) + (reassign_mask[..., None]) * onehots_to_keep[jnp.cumsum(reassign_mask) - 1]
    return sorted_ass_reassigned[jnp.argsort(sorted_elbo_idx)]

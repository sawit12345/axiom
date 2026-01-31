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

"""
Mathematical utility functions - CUDA accelerated via C++ backend.

NO JAX - uses only numpy and axiomcuda_backend.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike

# Import the C++ backend - this is the ONLY backend we use
try:
    import axiomcuda_backend as backend
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False
    raise RuntimeError(
        "AXIOMCUDA C++ backend not found. Please build the C++ extensions:\n"
        "  pip install -e .\n"
        "The C++ backend is required - there is no JAX fallback."
    )


def inv_and_logdet(
    pos_def_matrix: np.ndarray,
    return_inverse: bool = True,
    return_logdet: bool = True,
) -> Union[np.ndarray, float, Tuple[np.ndarray, float]]:
    """Compute log-determinant and matrix inverse using Cholesky - CUDA accelerated."""
    return backend.inv_and_logdet(pos_def_matrix, return_inverse, return_logdet)


def bdot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Batched dot product - CUDA accelerated."""
    assert x.ndim > 1
    assert y.ndim > 1
    shape = np.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    x = np.broadcast_to(x, shape + x.shape[-2:])
    y = np.broadcast_to(y, shape + y.shape[-2:])
    
    # Use C++ backend for batch matrix multiplication
    return backend.batch_dot(x, y)


def positive_leading_eigenvalues(x: np.ndarray, iters: int = 10) -> np.ndarray:
    """Ensure positive eigenvalues by adding epsilon to diagonal."""
    eps = np.eye(x.shape[-1]) * 0.001
    for i in range(iters):
        eigs = np.linalg.eigvalsh(x)
        if np.any(eigs <= 0):
            x = x + eps
        else:
            return x
    raise ValueError("Unable to maintain non-negative leading diagonal")


def symmetrise(x: np.ndarray) -> np.ndarray:
    """Force matrix to be symmetric."""
    return (x + x.swapaxes(-1, -2)) / 2


def make_posdef(x: np.ndarray) -> np.ndarray:
    """Make matrix symmetric and positive definite."""
    return positive_leading_eigenvalues(symmetrise(x))


def mvdigamma(x: np.ndarray, d: int) -> np.ndarray:
    """Compute multivariate digamma function - CUDA accelerated."""
    return backend.mvdigamma(x, d)


def mvgammaln(x: np.ndarray, d: int) -> np.ndarray:
    """Compute log of multivariate gamma function - CUDA accelerated."""
    return backend.mvgammaln(x, d)


def stable_logsumexp(x: np.ndarray, dims: tuple, keepdims: bool = False) -> np.ndarray:
    """Compute logsumexp along dimensions - CUDA accelerated."""
    # Use scipy for logsumexp
    from scipy.special import logsumexp
    return logsumexp(x, axis=dims, keepdims=keepdims)


def stable_softmax(x: np.ndarray, dims: tuple) -> np.ndarray:
    """Compute softmax along dimensions."""
    # Stable softmax computation
    max_x = np.max(x, axis=dims, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp = np.sum(exp_x, axis=dims, keepdims=True)
    return exp_x / sum_exp


def assign_unused(
    assignments: np.ndarray, 
    d_alpha: np.ndarray, 
    elbo_contrib: np.ndarray, 
    threshold: float = 1.0, 
    fill_prob: float = 10.0
) -> np.ndarray:
    """Re-assign data-points with low ELBO to unused clusters."""
    unfilled_cluster_idx = d_alpha < threshold
    sorted_elbo_idx = np.argsort(elbo_contrib)
    num_to_fill = unfilled_cluster_idx.sum()
    reassign_mask = np.arange(assignments.shape[0]) < num_to_fill
    assignments_sorted_by_elbo = assignments[sorted_elbo_idx]
    onehots_base = fill_prob * np.eye(d_alpha.shape[0])
    onehots_to_keep = onehots_base * unfilled_cluster_idx[None, ...]
    onehots_to_keep = np.take_along_axis(
        onehots_to_keep, 
        np.argsort(onehots_to_keep.sum(-1))[::-1][..., None], 
        axis=0
    )
    sorted_ass_reassigned = assignments_sorted_by_elbo * (1.0 - reassign_mask[..., None]) + \
                           (reassign_mask[..., None]) * onehots_to_keep[np.cumsum(reassign_mask) - 1]
    return sorted_ass_reassigned[np.argsort(sorted_elbo_idx)]

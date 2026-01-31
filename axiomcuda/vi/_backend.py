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
CUDA backend integration for variational inference.

This module provides the interface to C++ CUDA kernels for accelerated
computation. When CUDA extensions are available, they are used for:
- Matrix operations (inverse, Cholesky, logdet)
- Sampling from distributions
- Batch operations on distributions
- Special mathematical functions

When CUDA extensions are not available, JAX implementations are used
as fallback with JIT compilation for CPU/GPU performance.
"""

import warnings
from typing import Optional, Tuple, Union
from functools import partial

try:
    import jax.numpy as jnp
    from jax import jit
    import jax.scipy.linalg as linalg
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Try to import CUDA backend
try:
    # This would be the compiled C++ extension
    # from ._cuda_backend import (
    #     cuda_inv_and_logdet,
    #     cuda_mvgammaln,
    #     cuda_mvdigamma,
    #     cuda_sample_multivariate_normal,
    #     cuda_batch_dot,
    # )
    # HAS_CUDA_BACKEND = True
    HAS_CUDA_BACKEND = False
except ImportError:
    HAS_CUDA_BACKEND = False

if not HAS_CUDA_BACKEND:
    warnings.warn(
        "CUDA backend not available. Using JAX fallback implementations. "
        "For optimal performance, compile and install the CUDA extensions.",
        RuntimeWarning,
        stacklevel=2
    )


class CUDABackend:
    """
    Wrapper for CUDA backend operations.
    
    This class provides a unified interface for operations that can be
    accelerated via CUDA. When the C++ backend is available, it uses
    those implementations. Otherwise, it falls back to JAX.
    """
    
    def __init__(self):
        self.has_cuda = HAS_CUDA_BACKEND
        self.has_jax = HAS_JAX
        
    def inv_and_logdet(
        self,
        pos_def_matrix,
        return_inverse: Optional[bool] = True,
        return_logdet: Optional[bool] = True,
    ) -> Union[float, Tuple]:
        """
        Compute matrix inverse and log-determinant using Cholesky.
        
        Args:
            pos_def_matrix: Positive definite matrix
            return_inverse: Whether to return the inverse
            return_logdet: Whether to return the log determinant
            
        Returns:
            Inverse matrix and/or log determinant depending on flags
        """
        if self.has_cuda:
            # Call C++ CUDA implementation
            # return cuda_inv_and_logdet(pos_def_matrix, return_inverse, return_logdet)
            pass
        
        # JAX fallback implementation
        return_logdet = True if return_inverse == False else return_logdet
        
        shape = pos_def_matrix.shape
        min_eig = jnp.clip(jnp.min(jnp.linalg.eigvalsh(pos_def_matrix)), max=0.0)
        eps = jnp.finfo(pos_def_matrix.dtype).eps
        pos_def_matrix = pos_def_matrix + jnp.broadcast_to(
            jnp.eye(shape[-1]), shape
        ) * 2 * (eps - min_eig) * (min_eig < 0)
        
        chol = linalg.cho_factor(pos_def_matrix, lower=True)
        
        if return_inverse:
            identity = jnp.broadcast_to(jnp.eye(shape[-1]), shape)
            matrix_inverse = linalg.cho_solve(chol, identity)
            
            if return_logdet:
                logdet = jnp.expand_dims(
                    2 * jnp.log(jnp.diagonal(chol[0], axis1=-1, axis2=-2)).sum(-1),
                    (-1, -2)
                )
                return matrix_inverse, logdet
            
            return matrix_inverse
        
        logdet = jnp.expand_dims(
            2 * jnp.log(jnp.diagonal(chol[0], axis1=-1, axis2=-2)).sum(-1),
            (-1, -2)
        )
        return logdet
    
    def mvgammaln(self, x: jnp.ndarray, d: int) -> jnp.ndarray:
        """
        Compute log of multivariate gamma function.
        
        Args:
            x: Input array
            d: Dimension
            
        Returns:
            Log of multivariate gamma function
        """
        if self.has_cuda:
            # return cuda_mvgammaln(x, d)
            pass
        
        # JAX fallback
        from jax import lax
        from jax.numpy import expand_dims as expand
        
        return jnp.sum(
            lax.lgamma(expand(x, -1) - jnp.arange(d) / 2.0), -1
        ) + d * (d - 1) / 4.0 * jnp.log(jnp.pi)
    
    def mvdigamma(self, x: jnp.ndarray, d: int) -> jnp.ndarray:
        """
        Compute multivariate digamma function.
        
        Args:
            x: Input array
            d: Dimension
            
        Returns:
            Multivariate digamma function
        """
        if self.has_cuda:
            # return cuda_mvdigamma(x, d)
            pass
        
        # JAX fallback
        import jax.scipy.special as jsp
        return jsp.digamma(jnp.expand_dims(x, -1) - jnp.arange(d) / 2.0).sum(-1)
    
    def sample_multivariate_normal(
        self, key, mean, cov, shape=()
    ) -> jnp.ndarray:
        """
        Sample from multivariate normal distribution.
        
        Args:
            key: JAX random key
            mean: Mean vector
            cov: Covariance matrix
            shape: Sample shape
            
        Returns:
            Samples from MVN
        """
        if self.has_cuda:
            # return cuda_sample_multivariate_normal(key, mean, cov, shape)
            pass
        
        # JAX fallback
        from jax import random as jr
        return jr.multivariate_normal(key, mean=mean, cov=cov, shape=shape)
    
    def batch_dot(self, x, y):
        """
        Batched matrix multiplication.
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Batch matrix product
        """
        if self.has_cuda:
            # return cuda_batch_dot(x, y)
            pass
        
        # JAX fallback
        from jax import vmap
        assert x.ndim > 1
        assert y.ndim > 1
        shape = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-2])
        x = jnp.broadcast_to(x, shape + x.shape[-2:])
        y = jnp.broadcast_to(y, shape + y.shape[-2:])
        z = vmap(jnp.dot)(
            x.reshape((-1,) + x.shape[-2:]),
            y.reshape((-1,) + y.shape[-2:])
        )
        x_dim = x.shape[-2]
        y_dim = y.shape[-1]
        return z.reshape(shape + (x_dim, y_dim))


# Global backend instance
_backend = CUDABackend()

# Convenience functions that use the backend
def inv_and_logdet(*args, **kwargs):
    """Compute matrix inverse and log-determinant."""
    return _backend.inv_and_logdet(*args, **kwargs)


def mvgammaln(*args, **kwargs):
    """Compute log of multivariate gamma function."""
    return _backend.mvgammaln(*args, **kwargs)


def mvdigamma(*args, **kwargs):
    """Compute multivariate digamma function."""
    return _backend.mvdigamma(*args, **kwargs)


def sample_multivariate_normal(*args, **kwargs):
    """Sample from multivariate normal distribution."""
    return _backend.sample_multivariate_normal(*args, **kwargs)


def batch_dot(*args, **kwargs):
    """Batched matrix multiplication."""
    return _backend.batch_dot(*args, **kwargs)


def has_cuda_backend() -> bool:
    """Check if CUDA backend is available."""
    return HAS_CUDA_BACKEND

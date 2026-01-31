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
computation. The C++ backend is the ONLY computation engine - NO JAX.
"""

from typing import Optional, Tuple, Union
import numpy as np

# Import the C++ backend - this is the ONLY backend we use
try:
    import axiomcuda_backend as _backend
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False
    raise RuntimeError(
        "AXIOMCUDA C++ backend not found. Please build the C++ extensions:\n"
        "  pip install -e .\n"
        "The C++ backend is required - there is no JAX fallback."
    )


class CUDABackend:
    """
    Wrapper for CUDA/C++ backend operations.
    
    All computation goes through axiomcuda_backend - NO JAX.
    """
    
    def __init__(self):
        self.has_cuda = _backend.cuda_available() if hasattr(_backend, 'cuda_available') else False
        
    def inv_and_logdet(
        self,
        pos_def_matrix,
        return_inverse: Optional[bool] = True,
        return_logdet: Optional[bool] = True,
    ) -> Union[float, Tuple]:
        """Compute matrix inverse and log-determinant using Cholesky."""
        return _backend.inv_and_logdet(pos_def_matrix, return_inverse, return_logdet)
    
    def mvgammaln(self, x: np.ndarray, d: int) -> np.ndarray:
        """Compute log of multivariate gamma function."""
        return _backend.mvgammaln(x, d)
    
    def mvdigamma(self, x: np.ndarray, d: int) -> np.ndarray:
        """Compute multivariate digamma function."""
        return _backend.mvdigamma(x, d)
    
    def sample_multivariate_normal(
        self, key, mean, cov, shape=()
    ) -> np.ndarray:
        """Sample from multivariate normal distribution."""
        return _backend.sample_multivariate_normal(key, mean, cov, shape)
    
    def batch_dot(self, x, y):
        """Batched matrix multiplication."""
        return _backend.batch_dot(x, y)


# Global backend instance
backend = CUDABackend()

# Convenience functions that use the backend
def inv_and_logdet(*args, **kwargs):
    """Compute matrix inverse and log-determinant."""
    return backend.inv_and_logdet(*args, **kwargs)

def mvgammaln(*args, **kwargs):
    """Compute log of multivariate gamma function."""
    return backend.mvgammaln(*args, **kwargs)

def mvdigamma(*args, **kwargs):
    """Compute multivariate digamma function."""
    return backend.mvdigamma(*args, **kwargs)

def sample_multivariate_normal(*args, **kwargs):
    """Sample from multivariate normal distribution."""
    return backend.sample_multivariate_normal(*args, **kwargs)

def batch_dot(*args, **kwargs):
    """Batched matrix multiplication."""
    return backend.batch_dot(*args, **kwargs)

def has_cuda_backend() -> bool:
    """Check if CUDA backend is available."""
    return backend.has_cuda

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
Random number generation for AxiomCUDA using C++ backend.

Wraps the C++ random functions with a Pythonic interface.
"""

import numpy as np
from typing import Optional, Sequence, Tuple, Union
import axiomcuda_backend as backend
from .tensor import Tensor


class PRNGKey:
    """Pseudo-random number generator key (JAX-style).
    
    This class wraps the C++ backend PRNGKey for random number generation.
    
    Attributes:
        seed: The seed value
        
    Example:
        >>> key = PRNGKey(42)
        >>> key, subkey = split(key)
        >>> samples = normal(subkey, shape=(3, 3))
    """
    
    def __init__(self, seed: Union[int, np.ndarray, Tuple[int, ...]]):
        """Initialize a PRNGKey.
        
        Args:
            seed: Random seed (int or array of two ints)
        """
        if isinstance(seed, int):
            # Single seed
            self._key = backend.random.PRNGKey.from_seed(seed)
        elif isinstance(seed, (tuple, list)) and len(seed) == 2:
            self._key = backend.random.PRNGKey.from_ints(seed[0], seed[1])
        elif isinstance(seed, np.ndarray) and seed.shape == (2,):
            self._key = backend.random.PRNGKey.from_ints(seed[0], seed[1])
        elif isinstance(seed, backend.random.PRNGKey):
            self._key = seed
        else:
            raise ValueError(f"Invalid seed format: {seed}")
    
    @property
    def key(self) -> np.ndarray:
        """Get the underlying key array."""
        return np.array([self._key.key[0], self._key.key[1]], dtype=np.uint32)
    
    def __repr__(self) -> str:
        return f"PRNGKey({self._key.key[0]}, {self._key.key[1]})"
    
    def __hash__(self) -> int:
        return hash((self._key.key[0], self._key.key[1]))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, PRNGKey):
            return False
        return self._key.key[0] == other._key.key[0] and self._key.key[1] == other._key.key[1]


def seed(seed_value: int) -> PRNGKey:
    """Create a new PRNGKey from a seed.
    
    Args:
        seed_value: Integer seed
        
    Returns:
        New PRNGKey
        
    Example:
        >>> key = seed(42)
    """
    return PRNGKey(seed_value)


def split(key: PRNGKey, num: int = 2) -> Union[PRNGKey, Tuple[PRNGKey, ...]]:
    """Split a PRNGKey into multiple independent keys.
    
    Args:
        key: The PRNGKey to split
        num: Number of keys to generate (default 2)
        
    Returns:
        If num=2, returns (new_key, subkey) tuple.
        If num>2, returns tuple of num keys.
        
    Example:
        >>> key = seed(42)
        >>> key, subkey = split(key)  # Standard usage
        >>> key, subkey1, subkey2 = split(key, 3)
    """
    # Get output keys from backend
    output_keys = [backend.random.PRNGKey() for _ in range(num)]
    backend.random.split_key(key._key, num, output_keys)
    
    # Convert to PRNGKey objects
    python_keys = [PRNGKey(k) for k in output_keys]
    
    if num == 2:
        return python_keys[0], python_keys[1]
    
    return tuple(python_keys)


def normal(
    key: PRNGKey,
    shape: Sequence[int] = (),
    dtype: np.dtype = np.float64,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from standard normal distribution.
    
    Args:
        key: PRNGKey
        shape: Output shape
        dtype: Data type
        device: Target device for Tensor
        
    Returns:
        Samples from N(0, 1)
        
    Example:
        >>> key = seed(42)
        >>> x = normal(key, shape=(3, 3))
    """
    # Call backend normal function
    result_tensor = backend.random.normal_batch(key._key, shape)
    result = result_tensor.to_numpy()
    
    if dtype != np.float64:
        result = result.astype(dtype)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def uniform(
    key: PRNGKey,
    shape: Sequence[int] = (),
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: np.dtype = np.float64,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from uniform distribution.
    
    Args:
        key: PRNGKey
        shape: Output shape
        minval: Minimum value (inclusive)
        maxval: Maximum value (exclusive)
        dtype: Data type
        device: Target device for Tensor
        
    Returns:
        Uniform samples in [minval, maxval)
        
    Example:
        >>> key = seed(42)
        >>> x = uniform(key, shape=(3,), minval=-1.0, maxval=1.0)
    """
    # Call backend uniform function
    result_tensor = backend.random.uniform_batch(key._key, shape, minval, maxval)
    result = result_tensor.to_numpy()
    
    if dtype != np.float64:
        result = result.astype(dtype)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def randint(
    key: PRNGKey,
    shape: Sequence[int] = (),
    minval: int = 0,
    maxval: int = 100,
    dtype: np.dtype = np.int32,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample random integers.
    
    Args:
        key: PRNGKey
        shape: Output shape
        minval: Minimum value (inclusive)
        maxval: Maximum value (exclusive)
        dtype: Integer data type
        device: Target device for Tensor
        
    Returns:
        Random integers
        
    Example:
        >>> key = seed(42)
        >>> x = randint(key, shape=(3,), minval=0, maxval=10)
    """
    size = int(np.prod(shape)) if shape else 1
    
    # Use backend randint
    result = np.zeros(size, dtype=np.int32)
    backend.random.randint(key._key, size, minval, maxval, result)
    result = result.reshape(shape)
    
    if dtype != np.int32:
        result = result.astype(dtype)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def categorical(
    key: PRNGKey,
    logits: np.ndarray,
    axis: int = -1,
    shape: Optional[Sequence[int]] = None,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from categorical distribution.
    
    Args:
        key: PRNGKey
        logits: Unnormalized log probabilities
        axis: Axis along which logits are defined
        shape: Output shape (None for shape of logits without axis)
        device: Target device for Tensor
        
    Returns:
        Sampled class indices
        
    Example:
        >>> key = seed(42)
        >>> logits = np.array([0.1, 0.5, 0.4])  # 3 classes
        >>> sample = categorical(key, logits)
    """
    logits = np.asarray(logits)
    
    # Compute probabilities from logits
    max_logits = np.max(logits, axis=axis, keepdims=True)
    probs = np.exp(logits - max_logits)
    probs = probs / np.sum(probs, axis=axis, keepdims=True)
    
    # Determine output shape
    if shape is None:
        shape = logits.shape[:axis] + logits.shape[axis+1:]
    
    # Sample using backend
    size = int(np.prod(shape)) if shape else 1
    result = np.zeros(size, dtype=np.int32)
    dim = probs.shape[axis] if axis >= 0 else probs.shape[len(probs.shape) + axis]
    
    # Flatten probs for batch sampling
    probs_flat = probs.reshape(-1, dim)
    
    backend.random.categorical_batch(key._key, probs_flat.ctypes.data, dim, 
                                     min(size, probs_flat.shape[0]), result)
    
    result = result.reshape(shape)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def bernoulli(
    key: PRNGKey,
    p: Union[float, np.ndarray] = 0.5,
    shape: Optional[Sequence[int]] = None,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from Bernoulli distribution.
    
    Args:
        key: PRNGKey
        p: Probability of 1 (can be scalar or array)
        shape: Output shape (None for shape of p)
        device: Target device for Tensor
        
    Returns:
        Bernoulli samples (0 or 1)
        
    Example:
        >>> key = seed(42)
        >>> x = bernoulli(key, p=0.3, shape=(10,))
    """
    p = np.asarray(p)
    
    if shape is None:
        shape = p.shape
    
    size = int(np.prod(shape)) if shape else 1
    result = np.zeros(size, dtype=np.int32)
    
    backend.random.bernoulli(key._key, size, float(p), result)
    result = result.reshape(shape)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def choice(
    key: PRNGKey,
    a: int,
    shape: Sequence[int] = (),
    replace: bool = True,
    p: Optional[np.ndarray] = None,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Random sample from a given 1-D array.
    
    Args:
        key: PRNGKey
        a: If int, random sample from np.arange(a)
        shape: Output shape
        replace: Whether to sample with replacement
        p: Probabilities for each element
        device: Target device for Tensor
        
    Returns:
        Random samples
        
    Example:
        >>> key = seed(42)
        >>> x = choice(key, 10, shape=(3,))  # 3 random numbers from 0-9
    """
    if isinstance(a, int):
        a = np.arange(a)
    else:
        a = np.asarray(a)
    
    n = len(a)
    size = int(np.prod(shape)) if shape else 1
    
    if not replace and size > n:
        raise ValueError("Cannot take larger sample than population when replace=False")
    
    if p is None:
        # Uniform sampling
        indices = randint(key, (size,), 0, n, dtype=np.int32)
    else:
        # Weighted sampling
        logits = np.log(np.asarray(p))
        indices = categorical(key, logits, shape=(size,))
    
    result = a[indices].reshape(shape)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def permutation(key: PRNGKey, x: Union[int, np.ndarray]) -> np.ndarray:
    """Randomly permute a sequence or return a permuted range.
    
    Args:
        key: PRNGKey
        x: If int, permute np.arange(x); if array, make a copy and shuffle
        
    Returns:
        Permuted array
        
    Example:
        >>> key = seed(42)
        >>> perm = permutation(key, 10)  # Permuted 0-9
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> perm = permutation(key, arr)  # Permuted copy of arr
    """
    if isinstance(x, int):
        arr = np.arange(x, dtype=np.int32)
    else:
        arr = np.array(x, dtype=np.int32)
    
    n = len(arr)
    
    # Use backend permutation
    backend.random.permutation(key._key, n, arr)
    
    return arr


def exponential(
    key: PRNGKey,
    shape: Sequence[int] = (),
    scale: float = 1.0,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from exponential distribution.
    
    Args:
        key: PRNGKey
        shape: Output shape
        scale: Scale parameter (1/rate)
        device: Target device for Tensor
        
    Returns:
        Exponential samples
    """
    # Use uniform and transform
    u = uniform(key, shape, minval=0.0, maxval=1.0)
    if isinstance(u, Tensor):
        u = u.numpy()
    
    # Avoid log(0)
    u = np.clip(u, 1e-7, 1.0)
    result = -scale * np.log(u)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def gamma(
    key: PRNGKey,
    shape: Sequence[int] = (),
    a: float = 1.0,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from Gamma distribution.
    
    Args:
        key: PRNGKey
        shape: Output shape
        a: Shape parameter (k)
        device: Target device for Tensor
        
    Returns:
        Gamma samples
    """
    size = int(np.prod(shape)) if shape else 1
    result = np.zeros(size, dtype=np.float64)
    
    backend.random.gamma_batch(key._key, size, a, 1.0, result)
    result = result.reshape(shape)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def beta(
    key: PRNGKey,
    shape: Sequence[int] = (),
    a: float = 1.0,
    b: float = 1.0,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from Beta distribution.
    
    Args:
        key: PRNGKey
        shape: Output shape
        a: Alpha parameter
        b: Beta parameter
        device: Target device for Tensor
        
    Returns:
        Beta samples
    """
    # Beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b))
    x = gamma(key, shape, a)
    y = gamma(key, shape, b)
    
    if isinstance(x, Tensor):
        x = x.numpy()
    if isinstance(y, Tensor):
        y = y.numpy()
    
    result = x / (x + y)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def laplace(
    key: PRNGKey,
    shape: Sequence[int] = (),
    loc: float = 0.0,
    scale: float = 1.0,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from Laplace distribution.
    
    Args:
        key: PRNGKey
        shape: Output shape
        loc: Location parameter (mean)
        scale: Scale parameter
        device: Target device for Tensor
        
    Returns:
        Laplace samples
    """
    u = uniform(key, shape, minval=0.0, maxval=1.0)
    if isinstance(u, Tensor):
        u = u.numpy()
    
    # Transform uniform to Laplace
    u = u - 0.5
    result = loc - scale * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def multivariate_normal(
    key: PRNGKey,
    mean: np.ndarray,
    cov: np.ndarray,
    shape: Sequence[int] = (),
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from multivariate normal distribution.
    
    Args:
        key: PRNGKey
        mean: Mean vector
        cov: Covariance matrix
        shape: Batch shape
        device: Target device for Tensor
        
    Returns:
        Multivariate normal samples
    """
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    
    # Cholesky decomposition for sampling
    L = np.linalg.cholesky(cov)
    
    # Sample standard normal
    z = normal(key, shape + mean.shape, dtype=np.float64)
    if isinstance(z, Tensor):
        z = z.numpy()
    
    # Transform: x = mean + L @ z
    result = mean + z @ L.T
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def dirichlet(
    key: PRNGKey,
    alpha: np.ndarray,
    shape: Sequence[int] = (),
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from Dirichlet distribution.
    
    Args:
        key: PRNGKey
        alpha: Concentration parameters
        shape: Batch shape (for sampling multiple)
        device: Target device for Tensor
        
    Returns:
        Dirichlet samples
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    dim = alpha.shape[-1] if len(alpha.shape) > 0 else len(alpha)
    batch_size = int(np.prod(shape)) if shape else 1
    
    # Flatten alpha for batch processing
    if len(alpha.shape) == 1:
        alpha_flat = np.tile(alpha, (batch_size, 1))
    else:
        alpha_flat = alpha.reshape(-1, dim)
    
    result = np.zeros((batch_size, dim), dtype=np.float64)
    
    backend.random.dirichlet_batch(key._key, alpha_flat, dim, batch_size, result)
    
    result = result.reshape(shape + (dim,))
    
    if device is not None:
        return Tensor(result, device=device)
    return result


# Alias for common usage
PRNG = PRNGKey

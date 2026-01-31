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
Random number generation for AxiomCUDA.

Provides JAX-like random number generation with PRNGKey management,
supporting split, normal, uniform, categorical, and other distributions.
"""

import numpy as np
from typing import Optional, Sequence, Tuple, Union
from .tensor import Tensor


class PRNGKey:
    """Pseudo-random number generator key (JAX-style).
    
    This class manages the state of a random number generator using
    a Threefry-based counter scheme (similar to JAX).
    
    Attributes:
        seed: The seed value (counter state)
        
    Example:
        >>> key = PRNGKey(42)
        >>> key, subkey = split(key)
        >>> samples = normal(subkey, shape=(3, 3))
    """
    
    def __init__(self, seed: Union[int, np.ndarray, Tuple[int, ...]]):
        """Initialize a PRNGKey.
        
        Args:
            seed: Random seed (int or array of two ints for counter/key)
        """
        if isinstance(seed, int):
            # Single seed - expand to counter/key pair
            self._key = np.array([seed, 0], dtype=np.uint32)
        elif isinstance(seed, (tuple, list)) and len(seed) == 2:
            self._key = np.array(seed, dtype=np.uint32)
        elif isinstance(seed, np.ndarray) and seed.shape == (2,):
            self._key = seed.astype(np.uint32)
        else:
            raise ValueError(f"Invalid seed format: {seed}")
    
    @property
    def key(self) -> np.ndarray:
        """Get the underlying key array."""
        return self._key.copy()
    
    def __repr__(self) -> str:
        return f"PRNGKey({self._key[0]}, {self._key[1]})"
    
    def __hash__(self) -> int:
        return hash((self._key[0], self._key[1]))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, PRNGKey):
            return False
        return np.array_equal(self._key, other._key)


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
    
    This is the primary way to generate new keys for random number
    generation without affecting the original key's state.
    
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
    # Use a simple counter-based splitting
    base = key._key[0]
    new_keys = []
    
    for i in range(num):
        # Increment counter for each new key
        new_key = np.array([base + i + 1, key._key[1]], dtype=np.uint32)
        new_keys.append(PRNGKey(new_key))
    
    # Update the original key's counter
    new_main_key = PRNGKey(np.array([base + num, key._key[1]], dtype=np.uint32))
    
    if num == 2:
        return new_main_key, new_keys[1]
    
    # Return the new main key and all subkeys
    return tuple([new_main_key] + new_keys[1:])


def _hash_key(key: PRNGKey, salt: int = 0) -> np.ndarray:
    """Hash the key for random number generation.
    
    Uses a simple hash function (would be replaced with Threefry in full impl).
    
    Args:
        key: PRNGKey to hash
        salt: Salt value for different distributions
        
    Returns:
        Hashed key array
    """
    # Simple hash: mix the counter and key values
    k0, k1 = key._key[0], key._key[1]
    
    # XOR shift (simple PRNG)
    x = k0 ^ (k1 + salt)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    
    y = k1 ^ (k0 + salt)
    y ^= y << 17
    y ^= y >> 13
    y ^= y << 5
    
    return np.array([x, y], dtype=np.uint32)


def _key_to_float(key: np.ndarray) -> float:
    """Convert key to float in [0, 1)."""
    # Combine the two 32-bit integers into a 64-bit value
    combined = (int(key[0]) << 32) | int(key[1])
    # Map to [0, 1)
    return (combined % (2**53)) / (2**53)


def _generate_values(key: PRNGKey, shape: Tuple[int, ...], salt: int = 0) -> np.ndarray:
    """Generate an array of random values from a key.
    
    Args:
        key: PRNGKey
        shape: Output shape
        salt: Salt for different distributions
        
    Returns:
        Array of random values
    """
    size = np.prod(shape) if shape else 1
    values = np.zeros(size, dtype=np.float64)
    
    # Use a simple counter-based PRNG
    k = _hash_key(key, salt)
    
    for i in range(size):
        # Hash the key with the index
        ki = np.array([k[0] + i, k[1]], dtype=np.uint32)
        ki = _hash_key(PRNGKey(ki), salt)
        values[i] = _key_to_float(ki)
    
    return values.reshape(shape)


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
    # Box-Muller transform: convert uniform to normal
    u1 = _generate_values(key, shape, salt=1)
    u2 = _generate_values(key, shape, salt=2)
    
    # Avoid log(0)
    u1 = np.clip(u1, 1e-7, 1 - 1e-7)
    
    # Box-Muller
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    
    result = z0.astype(dtype)
    
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
    values = _generate_values(key, shape, salt=0)
    result = (minval + values * (maxval - minval)).astype(dtype)
    
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
    values = _generate_values(key, shape, salt=3)
    result = (minval + (values * (maxval - minval)).astype(np.int64)).astype(dtype)
    
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
    
    # Compute probabilities
    # Subtract max for numerical stability
    max_logits = np.max(logits, axis=axis, keepdims=True)
    probs = np.exp(logits - max_logits)
    probs = probs / np.sum(probs, axis=axis, keepdims=True)
    
    # Determine output shape
    if shape is None:
        shape = logits.shape[:axis] + logits.shape[axis+1:]
    
    # Sample using inverse CDF
    size = np.prod(shape) if shape else 1
    samples = np.zeros(size, dtype=np.int32)
    
    u = _generate_values(key, shape, salt=4)
    
    # Flatten for easier processing
    probs_flat = probs.reshape(-1, probs.shape[axis])
    
    for i in range(size):
        # Find where u falls in the CDF
        cumsum = np.cumsum(probs_flat[i % probs_flat.shape[0]])
        samples[i] = np.searchsorted(cumsum, u.flat[i])
    
    result = samples.reshape(shape)
    
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
    
    u = _generate_values(key, shape, salt=5)
    result = (u < p).astype(np.int32)
    
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
    size = np.prod(shape) if shape else 1
    
    if p is None:
        # Uniform sampling
        indices = (uniform(key, (size,), 0, n) * n).astype(np.int32) % n
    else:
        # Weighted sampling
        logits = np.log(np.asarray(p))
        indices = categorical(key, logits, shape=(size,))
    
    result = a[indices].reshape(shape)
    
    if not replace and size > n:
        raise ValueError("Cannot take larger sample than population when replace=False")
    
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
        arr = np.arange(x)
    else:
        arr = np.array(x)
    
    n = len(arr)
    
    # Fisher-Yates shuffle
    for i in range(n - 1, 0, -1):
        # Generate random index j in [0, i]
        u = _generate_values(key, (), salt=i)
        j = int(u * (i + 1))
        # Swap
        arr[i], arr[j] = arr[j], arr[i]
    
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
    u = _generate_values(key, shape, salt=6)
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
    # Marsaglia and Tsang method
    if a < 1:
        # Use property: Gamma(a) = Gamma(a+1) * U^(1/a)
        g = gamma(key, shape, a + 1)
        u = _generate_values(key, shape, salt=7)
        return g * u ** (1.0 / a)
    
    d = a - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    
    result = np.zeros(shape if shape else (1,))
    size = result.size
    
    for i in range(size):
        while True:
            z = normal(key, ())
            if z > -1.0 / c:
                break
        
        v = (1.0 + c * z) ** 3
        u = _generate_values(key, (), salt=8 + i)
        
        if u < 1.0 - 0.0331 * (z ** 2) ** 2:
            result.flat[i] = d * v
            break
        
        if np.log(u) < 0.5 * z ** 2 + d * (1.0 - v + np.log(v)):
            result.flat[i] = d * v
            break
    
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
    u = _generate_values(key, shape, salt=9)
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
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    
    # Cholesky decomposition for sampling
    L = np.linalg.cholesky(cov)
    
    # Sample standard normal
    z = normal(key, shape + mean.shape)
    
    # Transform: x = mean + L @ z
    result = mean + z @ L.T
    
    if device is not None:
        return Tensor(result, device=device)
    return result


def truncated_normal(
    key: PRNGKey,
    shape: Sequence[int] = (),
    lower: float = -2.0,
    upper: float = 2.0,
    mean: float = 0.0,
    std: float = 1.0,
    device=None,
) -> Union[np.ndarray, Tensor]:
    """Sample from truncated normal distribution.
    
    Args:
        key: PRNGKey
        shape: Output shape
        lower: Lower bound
        upper: Upper bound
        mean: Mean
        std: Standard deviation
        device: Target device for Tensor
        
    Returns:
        Truncated normal samples
    """
    # Use inverse CDF method
    u = _generate_values(key, shape, salt=10)
    
    # Standard normal CDF at bounds
    from scipy import stats
    cdf_lower = stats.norm.cdf((lower - mean) / std)
    cdf_upper = stats.norm.cdf((upper - mean) / std)
    
    # Transform uniform to truncated normal
    u_transformed = cdf_lower + u * (cdf_upper - cdf_lower)
    result = mean + std * stats.norm.ppf(u_transformed)
    
    if device is not None:
        return Tensor(result, device=device)
    return result


# Alias for common usage
PRNG = PRNGKey

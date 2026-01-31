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
Base model classes for AxiomCUDA.

Provides device management, configuration, and base model interfaces
for CUDA-accelerated models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from typing_extensions import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx


# Try to import the C++ backend module
try:
    import _axiomcuda
    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    _axiomcuda = None
    CUDA_BACKEND_AVAILABLE = False


def check_cuda_available() -> bool:
    """Check if CUDA is available for computation.
    
    Returns:
        bool: True if CUDA backend is available and JAX can access GPUs.
    """
    if not CUDA_BACKEND_AVAILABLE:
        return False
    
    try:
        gpus = jax.devices('gpu')
        return len(gpus) > 0
    except (RuntimeError, AssertionError):
        return False


def get_device(prefer_gpu: bool = True) -> jax.Device:
    """Get the default device for computation.
    
    Args:
        prefer_gpu: If True, prefer GPU if available.
        
    Returns:
        jax.Device: The default device to use.
    """
    if prefer_gpu and check_cuda_available():
        return jax.devices('gpu')[0]
    return jax.devices('cpu')[0]


def to_device(x: Any, device: Optional[jax.Device] = None) -> Any:
    """Move data to specified device.
    
    Args:
        x: Data to move (array or PyTree).
        device: Target device. If None, uses default device.
        
    Returns:
        Data moved to target device.
    """
    if device is None:
        device = get_device()
    
    def move_array(arr):
        if isinstance(arr, jnp.ndarray):
            return jax.device_put(arr, device)
        return arr
    
    return jax.tree_util.tree_map(move_array, x)


@dataclass(frozen=True)
class ModelConfig:
    """Base configuration class for all models.
    
    Attributes:
        device: Device to run computations on.
        use_cuda: Whether to use CUDA acceleration.
        seed: Random seed for reproducibility.
        dtype: Data type for computations.
    """
    device: Optional[str] = None
    use_cuda: bool = True
    seed: int = 0
    dtype: jnp.dtype = field(default_factory=lambda: jnp.float32)
    
    def get_device(self) -> jax.Device:
        """Get JAX device based on configuration."""
        if self.use_cuda and check_cuda_available():
            return jax.devices('gpu')[0]
        return jax.devices('cpu')[0]


@dataclass
class ModelState:
    """Base state container for model parameters and variables.
    
    Attributes:
        params: Model parameters (PyTree).
        state: Mutable state variables.
        step: Current optimization step.
    """
    params: Any
    state: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    
    def to_device(self, device: Optional[jax.Device] = None) -> 'ModelState':
        """Move state to device."""
        return ModelState(
            params=to_device(self.params, device),
            state={k: to_device(v, device) for k, v in self.state.items()},
            step=self.step
        )


class BaseModel(eqx.Module):
    """Base class for all CUDA-accelerated models.
    
    Provides common functionality for device management, parameter access,
    and C++ backend integration.
    
    Attributes:
        config: Model configuration.
        _cpp_handle: Handle to C++ backend object (if available).
    """
    config: ModelConfig
    _cpp_handle: Optional[Any] = eqx.field(static=True, default=None)
    
    def __post_init__(self):
        """Initialize C++ backend if available."""
        if CUDA_BACKEND_AVAILABLE and self.config.use_cuda:
            self._init_cpp_backend()
    
    def _init_cpp_backend(self):
        """Initialize C++ backend. Override in subclasses."""
        pass
    
    def to_device(self, device: Optional[jax.Device] = None) -> 'BaseModel':
        """Move model to device.
        
        Args:
            device: Target device. If None, uses model config.
            
        Returns:
            Model moved to target device.
        """
        if device is None:
            device = self.config.get_device()
        
        # Use equinox's device transfer
        return jax.device_put(self, device)
    
    @property
    def device(self) -> jax.Device:
        """Get current device."""
        return self.config.get_device()
    
    def parameters(self) -> Dict[str, Any]:
        """Get model parameters as dictionary.
        
        Returns:
            Dictionary of parameter name to value.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def state_dict(self) -> Dict[str, Any]:
        """Get full state dictionary for serialization.
        
        Returns:
            Dictionary containing all model state.
        """
        return {
            'config': self.config,
            'parameters': self.parameters(),
        }
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: File path to save to.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.state_dict(), f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk.
        
        Args:
            path: File path to load from.
            
        Returns:
            Loaded model instance.
        """
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        return cls(**state['parameters'])


def ensure_array(x: Union[Array, list, tuple, float, int], dtype=None) -> Array:
    """Ensure input is a JAX array.
    
    Args:
        x: Input data.
        dtype: Desired data type.
        
    Returns:
        JAX array.
    """
    if isinstance(x, jnp.ndarray):
        if dtype is not None:
            return x.astype(dtype)
        return x
    return jnp.array(x, dtype=dtype)


def array_to_tuple(arr: Array) -> Tuple:
    """Convert JAX array to nested tuples for C++ interop.
    
    Args:
        arr: JAX array.
        
    Returns:
        Nested tuple representation.
    """
    if arr.ndim == 0:
        return float(arr)
    elif arr.ndim == 1:
        return tuple(float(x) for x in arr)
    else:
        return tuple(array_to_tuple(arr[i]) for i in range(arr.shape[0]))


def call_cpp_backend(func_name: str, *args, **kwargs) -> Any:
    """Call a function in the C++ backend.
    
    Args:
        func_name: Name of the C++ function to call.
        *args: Positional arguments.
        **kwargs: Keyword arguments.
        
    Returns:
        Result from C++ function.
        
    Raises:
        RuntimeError: If C++ backend is not available.
    """
    if not CUDA_BACKEND_AVAILABLE:
        raise RuntimeError(f"C++ backend not available. Cannot call {func_name}")
    
    func = getattr(_axiomcuda, func_name, None)
    if func is None:
        raise AttributeError(f"C++ backend does not have function: {func_name}")
    
    return func(*args, **kwargs)


# Utility for creating device-compatible functions
def device_jit(fn, device=None, **jit_kwargs):
    """JIT compile function with device placement.
    
    Args:
        fn: Function to compile.
        device: Device to run on.
        **jit_kwargs: Additional arguments to jax.jit.
        
    Returns:
        JIT-compiled function.
    """
    jitted = jax.jit(fn, **jit_kwargs)
    return jitted

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
Device management for AxiomCUDA.

Provides GPU/CPU selection, CUDA auto-detection, memory management,
and device context managers.
"""

import os
import warnings
from contextlib import contextmanager
from typing import Optional, Union
import numpy as np

# Try to import CUDA bindings
try:
    import ctypes
    _cuda_lib = ctypes.CDLL('libcudart.so')
    _CUDA_AVAILABLE = True
except (OSError, ImportError):
    _CUDA_AVAILABLE = False


def cuda_available() -> bool:
    """Check if CUDA is available on this system.
    
    Returns:
        True if CUDA runtime is available, False otherwise
    """
    return _CUDA_AVAILABLE


def _get_gpu_memory_info(device_id: int = 0) -> tuple:
    """Get GPU memory information.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        Tuple of (free_memory, total_memory) in bytes
    """
    if not _CUDA_AVAILABLE:
        return (0, 0)
    
    try:
        # This would be replaced with actual CUDA calls
        # For now, return dummy values
        return (8 * 1024**3, 16 * 1024**3)  # 8GB free, 16GB total
    except:
        return (0, 0)


class Device:
    """Device abstraction for GPU/CPU tensor operations.
    
    This class provides a unified interface for managing device placement
    and memory for both CPU and GPU tensors.
    
    Attributes:
        type: Device type ('cpu' or 'cuda')
        id: Device ID (for CUDA devices)
    
    Example:
        >>> device = Device('cuda', 0)
        >>> print(device)
        cuda:0
        >>> device = Device('cpu')
        >>> print(device)
        cpu
    """
    
    def __init__(self, device_type: str = 'cpu', device_id: int = 0):
        """Initialize a device.
        
        Args:
            device_type: 'cpu' or 'cuda'
            device_id: Device ID for CUDA devices
            
        Raises:
            RuntimeError: If CUDA is requested but not available
            ValueError: If device_type is not 'cpu' or 'cuda'
        """
        if device_type not in ('cpu', 'cuda'):
            raise ValueError(f"Device type must be 'cpu' or 'cuda', got {device_type}")
        
        if device_type == 'cuda' and not _CUDA_AVAILABLE:
            warnings.warn("CUDA requested but not available, falling back to CPU")
            device_type = 'cpu'
        
        self._type = device_type
        self._id = device_id if device_type == 'cuda' else 0
        self._memory_stats = {}
    
    @property
    def type(self) -> str:
        """Get device type."""
        return self._type
    
    @property
    def id(self) -> int:
        """Get device ID."""
        return self._id
    
    def __str__(self) -> str:
        if self._type == 'cuda':
            return f"cuda:{self._id}"
        return "cpu"
    
    def __repr__(self) -> str:
        return f"Device(type='{self._type}', id={self._id})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Device):
            return False
        return self._type == other._type and self._id == other._id
    
    def __hash__(self) -> int:
        return hash((self._type, self._id))
    
    @property
    def is_cpu(self) -> bool:
        """Check if device is CPU."""
        return self._type == 'cpu'
    
    @property
    def is_cuda(self) -> bool:
        """Check if device is CUDA."""
        return self._type == 'cuda'
    
    def get_memory_info(self) -> dict:
        """Get memory information for this device.
        
        Returns:
            Dictionary with memory statistics
        """
        if self._type == 'cpu':
            import psutil
            mem = psutil.virtual_memory()
            return {
                'total': mem.total,
                'available': mem.available,
                'used': mem.used,
                'percent': mem.percent,
            }
        else:
            free, total = _get_gpu_memory_info(self._id)
            return {
                'total': total,
                'available': free,
                'used': total - free,
                'percent': 100 * (total - free) / total if total > 0 else 0,
            }
    
    def empty_cache(self) -> None:
        """Empty the device cache to free memory."""
        if self._type == 'cuda' and _CUDA_AVAILABLE:
            # Call CUDA runtime to empty cache
            pass  # Implementation would call actual CUDA function
    
    def synchronize(self) -> None:
        """Synchronize device operations."""
        if self._type == 'cuda' and _CUDA_AVAILABLE:
            # Call CUDA synchronize
            pass  # Implementation would call actual CUDA function
    
    def to(self, array: np.ndarray) -> 'Tensor':
        """Move a numpy array to this device.
        
        Args:
            array: Numpy array to move
            
        Returns:
            Tensor on this device
        """
        from .tensor import Tensor
        tensor = Tensor(array)
        return tensor.to(self)


# Global default device
_default_device: Optional[Device] = None


def get_device(device: Union[str, Device, None] = None) -> Device:
    """Get a device object.
    
    Args:
        device: Device specification (string, Device object, or None)
               If None, returns the default device
               If 'cpu' or 'cuda', creates appropriate Device
               
    Returns:
        Device object
    """
    global _default_device
    
    if device is None:
        if _default_device is None:
            # Auto-detect: use CUDA if available, else CPU
            if cuda_available():
                _default_device = Device('cuda', 0)
            else:
                _default_device = Device('cpu')
        return _default_device
    
    if isinstance(device, Device):
        return device
    
    if isinstance(device, str):
        if device.startswith('cuda'):
            if ':' in device:
                device_id = int(device.split(':')[1])
            else:
                device_id = 0
            return Device('cuda', device_id)
        elif device == 'cpu':
            return Device('cpu')
        else:
            raise ValueError(f"Invalid device string: {device}")
    
    raise TypeError(f"Expected str or Device, got {type(device)}")


def set_default_device(device: Union[str, Device]) -> None:
    """Set the default device for tensor operations.
    
    Args:
        device: Device to set as default
    """
    global _default_device
    _default_device = get_device(device)


@contextmanager
def device_context(device: Union[str, Device]):
    """Context manager for temporarily setting the default device.
    
    Args:
        device: Device to use within the context
        
    Example:
        >>> with device_context('cuda:0'):
        ...     # Operations use cuda:0
        ...     x = Tensor([1, 2, 3])
        >>> # Back to previous default
    """
    global _default_device
    prev_device = _default_device
    _default_device = get_device(device)
    try:
        yield _default_device
    finally:
        _default_device = prev_device


def memory_summary(device: Union[str, Device, None] = None) -> str:
    """Get a string summary of device memory.
    
    Args:
        device: Device to check (None for default)
        
    Returns:
        Formatted string with memory information
    """
    dev = get_device(device)
    info = dev.get_memory_info()
    
    total_gb = info['total'] / 1024**3
    used_gb = info['used'] / 1024**3
    avail_gb = info['available'] / 1024**3
    
    return (
        f"Memory Summary for {dev}:\n"
        f"  Total: {total_gb:.2f} GB\n"
        f"  Used: {used_gb:.2f} GB ({info['percent']:.1f}%)\n"
        f"  Available: {avail_gb:.2f} GB"
    )


def clear_memory(device: Union[str, Device, None] = None) -> None:
    """Clear memory on a device.
    
    Args:
        device: Device to clear (None for default)
    """
    dev = get_device(device)
    dev.empty_cache()
    
    if dev.is_cuda:
        # Force garbage collection
        import gc
        gc.collect()


# CUDA utility functions

def cuda_device_count() -> int:
    """Get the number of CUDA devices available.
    
    Returns:
        Number of CUDA devices (0 if CUDA not available)
    """
    if not _CUDA_AVAILABLE:
        return 0
    
    try:
        # Would call actual CUDA function
        return 1  # Placeholder
    except:
        return 0


def cuda_get_device_name(device_id: int = 0) -> str:
    """Get the name of a CUDA device.
    
    Args:
        device_id: CUDA device ID
        
    Returns:
        Device name string
    """
    if not _CUDA_AVAILABLE:
        return "No CUDA available"
    
    try:
        # Would call actual CUDA function
        return f"CUDA Device {device_id}"
    except:
        return "Unknown"


def cuda_get_capability(device_id: int = 0) -> tuple:
    """Get the compute capability of a CUDA device.
    
    Args:
        device_id: CUDA device ID
        
    Returns:
        Tuple of (major, minor) compute capability
    """
    if not _CUDA_AVAILABLE:
        return (0, 0)
    
    try:
        # Would call actual CUDA function
        return (7, 5)  # Placeholder
    except:
        return (0, 0)


# Auto-detect and set default device on import
if cuda_available():
    set_default_device('cuda:0')
else:
    set_default_device('cpu')

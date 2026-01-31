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
Tensor wrapper for AxiomCUDA using C++ backend.

Wraps the C++ Tensor class to provide a Pythonic interface.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
import axiomcuda_backend as backend
from .device import Device, get_device


class Tensor:
    """Numpy-compatible tensor with GPU acceleration via C++ backend.
    
    This class wraps the C++ backend.Tensor class to provide a
    Pythonic interface with numpy compatibility.
    
    Attributes:
        shape: Shape of the tensor
        dtype: Data type of the tensor  
        device: Device where the tensor is stored
        
    Example:
        >>> import numpy as np
        >>> from axiomcuda import Tensor
        >>> 
        >>> # Create from numpy
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]))
        >>> 
        >>> # Move to GPU
        >>> x_gpu = x.cuda()
        >>> 
        >>> # Operations
        >>> y = x * 2 + 1
        >>> 
        >>> # Back to numpy
        >>> arr = y.numpy()
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, List, float, int, 'Tensor', backend.tensor.Tensor],
        dtype: Optional[np.dtype] = None,
        device: Optional[Union[str, Device]] = None,
        requires_grad: bool = False,
    ):
        """Initialize a Tensor.
        
        Args:
            data: Input data (numpy array, list, scalar, Tensor, or backend Tensor)
            dtype: Target data type (optional)
            device: Target device (optional, defaults to current default)
            requires_grad: Whether to track gradients for autograd
        """
        # Handle different input types
        if isinstance(data, backend.tensor.Tensor):
            # Already a backend tensor
            self._tensor = data
        elif isinstance(data, Tensor):
            # Copy from another Tensor
            self._tensor = data._tensor.copy()
        elif isinstance(data, np.ndarray):
            # Create from numpy array
            self._tensor = backend.tensor.Tensor.from_numpy(data.astype(np.float64))
        else:
            # Convert to numpy first
            arr = np.array(data, dtype=np.float64)
            self._tensor = backend.tensor.Tensor.from_numpy(arr)
        
        # Move to device if specified
        if device is not None:
            dev = get_device(device)
            if dev.is_cuda:
                self._tensor = self._tensor.cuda(dev.id)
        
        self._requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self._tensor.shape
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return len(self._tensor.shape)
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type."""
        return np.float64  # C++ backend uses float64
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return int(np.prod(self._tensor.shape))
    
    @property
    def device(self) -> Device:
        """Get current device."""
        device_str = self._tensor.device
        if device_str.startswith('cuda'):
            device_id = int(device_str.split(':')[1]) if ':' in device_str else 0
            return Device('cuda', device_id)
        return Device('cpu')
    
    @property
    def requires_grad(self) -> bool:
        """Check if gradients are being tracked."""
        return self._requires_grad
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array (always returns CPU copy)."""
        return self._tensor.to_numpy()
    
    def cpu(self) -> 'Tensor':
        """Move tensor to CPU."""
        if not self._tensor.is_cuda():
            return self
        new_tensor = Tensor(self._tensor.cpu())
        new_tensor._requires_grad = self._requires_grad
        return new_tensor
    
    def cuda(self, device_id: int = 0) -> 'Tensor':
        """Move tensor to CUDA GPU.
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Tensor on GPU
        """
        if self._tensor.is_cuda():
            return self
        new_tensor = Tensor(self._tensor.cuda(device_id))
        new_tensor._requires_grad = self._requires_grad
        return new_tensor
    
    def to(self, device: Union[str, Device]) -> 'Tensor':
        """Move tensor to a specific device.
        
        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', or Device object)
            
        Returns:
            Tensor on target device
        """
        dev = get_device(device)
        if dev.is_cuda:
            return self.cuda(dev.id)
        return self.cpu()
    
    def copy(self) -> 'Tensor':
        """Create a copy of the tensor."""
        new_tensor = Tensor(self._tensor.copy())
        new_tensor._requires_grad = self._requires_grad
        if self._grad is not None:
            new_tensor._grad = self._grad.copy()
        return new_tensor
    
    def clone(self) -> 'Tensor':
        """Alias for copy()."""
        return self.copy()
    
    def detach(self) -> 'Tensor':
        """Return a new tensor detached from the computation graph."""
        new_tensor = Tensor(self._tensor)
        new_tensor._requires_grad = False
        new_tensor._grad = None
        new_tensor._grad_fn = None
        return new_tensor
    
    def backward(self, grad_output: Optional['Tensor'] = None) -> None:
        """Compute gradients via backpropagation.
        
        Args:
            grad_output: Gradient of the loss with respect to this tensor
        """
        if not self._requires_grad:
            raise RuntimeError("Tensor does not require gradients")
        
        if grad_output is None:
            grad_output = Tensor(np.ones(self.shape))
        
        self._grad = grad_output.numpy() if isinstance(grad_output, Tensor) else grad_output
    
    @property
    def grad(self) -> Optional['Tensor']:
        """Get the gradient of this tensor."""
        if self._grad is None:
            return None
        return Tensor(self._grad, device=self.device)
    
    def zero_grad(self) -> None:
        """Zero out the gradient."""
        self._grad = None
    
    # Shape operations
    
    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor."""
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return Tensor(self._tensor.reshape(new_shape))
    
    def view(self, *shape: int) -> 'Tensor':
        """View tensor with new shape."""
        return self.reshape(*shape)
    
    def transpose(self, dim0: int = -1, dim1: int = -2) -> 'Tensor':
        """Transpose tensor dimensions."""
        if dim0 == -1 and dim1 == -2:
            # Reverse all dimensions
            return Tensor(self._tensor.transpose())
        return Tensor(self._tensor.transpose(dim0, dim1))
    
    def permute(self, *dims: int) -> 'Tensor':
        """Permute tensor dimensions."""
        return Tensor(self._tensor.permute(dims))
    
    def flatten(self) -> 'Tensor':
        """Flatten to 1D."""
        return self.reshape(-1)
    
    def squeeze(self, axis: Optional[int] = None) -> 'Tensor':
        """Remove dimensions of size 1."""
        if axis is None:
            return Tensor(self._tensor.squeeze())
        return Tensor(self._tensor.squeeze(axis))
    
    def unsqueeze(self, axis: int) -> 'Tensor':
        """Add a dimension of size 1 at the specified position."""
        return Tensor(self._tensor.unsqueeze(axis))
    
    def expand_dims(self, axis: int) -> 'Tensor':
        """Alias for unsqueeze."""
        return self.unsqueeze(axis)
    
    # Reduction operations
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Sum elements along axis."""
        if axis is None:
            return Tensor(self._tensor.sum())
        return Tensor(self._tensor.sum(axis, keepdims))
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Compute mean along axis."""
        if axis is None:
            return Tensor(self._tensor.mean())
        return Tensor(self._tensor.mean(axis, keepdims))
    
    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Compute maximum along axis."""
        if axis is None:
            return Tensor(self._tensor.max())
        return Tensor(self._tensor.max(axis, keepdims))
    
    def min(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Compute minimum along axis."""
        if axis is None:
            return Tensor(self._tensor.min())
        return Tensor(self._tensor.min(axis, keepdims))
    
    def argmax(self, axis: Optional[int] = None) -> 'Tensor':
        """Return indices of maximum values."""
        if axis is None:
            return Tensor(self._tensor.argmax())
        return Tensor(self._tensor.argmax(axis))
    
    def argmin(self, axis: Optional[int] = None) -> 'Tensor':
        """Return indices of minimum values."""
        if axis is None:
            return Tensor(self._tensor.argmin())
        return Tensor(self._tensor.argmin(axis))
    
    # Linear algebra
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        """Compute dot product."""
        return Tensor(backend.tensor.dot(self._tensor, other._tensor))
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        return Tensor(backend.tensor.matmul(self._tensor, other._tensor))
    
    @property
    def T(self) -> 'Tensor':
        """Transpose property."""
        return self.transpose()
    
    # Indexing
    
    def __getitem__(self, index) -> 'Tensor':
        """Get item (supports numpy-style indexing)."""
        return Tensor(self._tensor[index])
    
    def __setitem__(self, index, value) -> None:
        """Set item."""
        if isinstance(value, Tensor):
            value = value._tensor
        self._tensor[index] = value
    
    # Arithmetic operations - delegate to backend
    
    def __add__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.add(other._tensor))
        return Tensor(self._tensor.add(float(other)))
    
    def __radd__(self, other) -> 'Tensor':
        return self.__add__(other)
    
    def __sub__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.subtract(other._tensor))
        return Tensor(self._tensor.subtract(float(other)))
    
    def __rsub__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(other._tensor.subtract(self._tensor))
        return Tensor(backend.tensor.subtract(float(other), self._tensor))
    
    def __mul__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.multiply(other._tensor))
        return Tensor(self._tensor.multiply(float(other)))
    
    def __rmul__(self, other) -> 'Tensor':
        return self.__mul__(other)
    
    def __truediv__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.divide(other._tensor))
        return Tensor(self._tensor.divide(float(other)))
    
    def __rtruediv__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(other._tensor.divide(self._tensor))
        return Tensor(backend.tensor.divide(float(other), self._tensor))
    
    def __pow__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.power(other._tensor))
        return Tensor(self._tensor.power(float(other)))
    
    def __matmul__(self, other) -> 'Tensor':
        return self.matmul(other)
    
    def __neg__(self) -> 'Tensor':
        return Tensor(self._tensor.negate())
    
    def __abs__(self) -> 'Tensor':
        return Tensor(self._tensor.abs())
    
    # Comparison operations
    
    def __eq__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.eq(other._tensor))
        return Tensor(self._tensor.eq(float(other)))
    
    def __ne__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.ne(other._tensor))
        return Tensor(self._tensor.ne(float(other)))
    
    def __lt__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.lt(other._tensor))
        return Tensor(self._tensor.lt(float(other)))
    
    def __le__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.le(other._tensor))
        return Tensor(self._tensor.le(float(other)))
    
    def __gt__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.gt(other._tensor))
        return Tensor(self._tensor.gt(float(other)))
    
    def __ge__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(self._tensor.ge(other._tensor))
        return Tensor(self._tensor.ge(float(other)))
    
    # In-place operations
    
    def __iadd__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            self._tensor = self._tensor.add(other._tensor)
        else:
            self._tensor = self._tensor.add(float(other))
        return self
    
    def __isub__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            self._tensor = self._tensor.subtract(other._tensor)
        else:
            self._tensor = self._tensor.subtract(float(other))
        return self
    
    def __imul__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            self._tensor = self._tensor.multiply(other._tensor)
        else:
            self._tensor = self._tensor.multiply(float(other))
        return self
    
    def __itruediv__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            self._tensor = self._tensor.divide(other._tensor)
        else:
            self._tensor = self._tensor.divide(float(other))
        return self
    
    # Representation
    
    def __repr__(self) -> str:
        return f"Tensor({self._tensor}, requires_grad={self._requires_grad})"
    
    def __str__(self) -> str:
        return str(self._tensor)
    
    def __len__(self) -> int:
        return self.shape[0] if len(self.shape) > 0 else 0
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    # Array protocol
    
    def __array__(self, dtype=None):
        """Numpy array protocol support."""
        arr = self.numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr


# Utility functions using backend

def zeros(shape: Tuple[int, ...], dtype: np.dtype = np.float64, device=None) -> Tensor:
    """Create a tensor of zeros."""
    device_str = device.type if isinstance(device, Device) else str(device) if device else 'cpu'
    return Tensor(backend.tensor.zeros(shape, device_str))


def ones(shape: Tuple[int, ...], dtype: np.dtype = np.float64, device=None) -> Tensor:
    """Create a tensor of ones."""
    device_str = device.type if isinstance(device, Device) else str(device) if device else 'cpu'
    return Tensor(backend.tensor.ones(shape, device_str))


def zeros_like(tensor: Tensor, dtype=None, device=None) -> Tensor:
    """Create a tensor of zeros with the same shape."""
    shape = tensor.shape
    device = device if device is not None else tensor.device
    return zeros(shape, dtype or np.float64, device)


def ones_like(tensor: Tensor, dtype=None, device=None) -> Tensor:
    """Create a tensor of ones with the same shape."""
    shape = tensor.shape
    device = device if device is not None else tensor.device
    return ones(shape, dtype or np.float64, device)


def full(shape: Tuple[int, ...], fill_value, dtype=None, device=None) -> Tensor:
    """Create a tensor filled with a value."""
    device_str = device.type if isinstance(device, Device) else str(device) if device else 'cpu'
    return Tensor(backend.tensor.full(shape, float(fill_value), device_str))


def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype=None, device=None) -> Tensor:
    """Create a range of values."""
    device_str = device.type if isinstance(device, Device) else str(device) if device else 'cpu'
    return Tensor(backend.tensor.arange(start, stop, step, device_str))


def linspace(start: float, stop: float, num: int = 50, device=None) -> Tensor:
    """Create linearly spaced values."""
    device_str = device.type if isinstance(device, Device) else str(device) if device else 'cpu'
    return Tensor(backend.tensor.linspace(start, stop, num, device_str))


def eye(n: int, m: Optional[int] = None, dtype=np.float64, device=None) -> Tensor:
    """Create an identity matrix."""
    device_str = device.type if isinstance(device, Device) else str(device) if device else 'cpu'
    return Tensor(backend.tensor.eye(n, m, device_str))


def diag(v: Union[Tensor, np.ndarray], k: int = 0) -> Tensor:
    """Extract diagonal or create diagonal matrix."""
    if isinstance(v, Tensor):
        v = v.numpy()
    return Tensor(backend.tensor.diag(v, k))


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along a new axis."""
    backend_tensors = [t._tensor for t in tensors]
    return Tensor(backend.tensor.stack(backend_tensors, axis))


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Concatenate tensors along an axis."""
    backend_tensors = [t._tensor for t in tensors]
    return Tensor(backend.tensor.concatenate(backend_tensors, axis))


def where(condition: Tensor, x, y) -> Tensor:
    """Select elements from x or y based on condition."""
    if isinstance(x, Tensor):
        x = x._tensor
    if isinstance(y, Tensor):
        y = y._tensor
    return Tensor(backend.tensor.where(condition._tensor, x, y))


def exp(tensor: Tensor) -> Tensor:
    """Exponential."""
    return Tensor(backend.tensor.exp(tensor._tensor))


def log(tensor: Tensor) -> Tensor:
    """Natural logarithm."""
    return Tensor(backend.tensor.log(tensor._tensor))


def sqrt(tensor: Tensor) -> Tensor:
    """Square root."""
    return Tensor(backend.tensor.sqrt(tensor._tensor))


def sin(tensor: Tensor) -> Tensor:
    """Sine."""
    return Tensor(backend.tensor.sin(tensor._tensor))


def cos(tensor: Tensor) -> Tensor:
    """Cosine."""
    return Tensor(backend.tensor.cos(tensor._tensor))


def tan(tensor: Tensor) -> Tensor:
    """Tangent."""
    return Tensor(backend.tensor.tan(tensor._tensor))


def clip(tensor: Tensor, min_val, max_val) -> Tensor:
    """Clip values to range."""
    return Tensor(backend.tensor.clip(tensor._tensor, min_val, max_val))

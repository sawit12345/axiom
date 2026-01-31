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
Tensor wrapper for AxiomCUDA.

Provides a numpy-compatible interface for the C++ Tensor class,
with GPU/CPU transfer methods and autograd support.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from .device import Device, get_device


class Tensor:
    """Numpy-compatible tensor with GPU acceleration support.
    
    This class wraps the underlying C++ Tensor class to provide a
    Pythonic interface while supporting:
    - Numpy-compatible operations
    - GPU/CPU memory transfer
    - Automatic differentiation (autograd)
    - Device-agnostic computation
    
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
        data: Union[np.ndarray, List, float, int, 'Tensor'],
        dtype: Optional[np.dtype] = None,
        device: Optional[Union[str, Device]] = None,
        requires_grad: bool = False,
    ):
        """Initialize a Tensor.
        
        Args:
            data: Input data (numpy array, list, scalar, or another Tensor)
            dtype: Target data type (optional)
            device: Target device (optional, defaults to current default)
            requires_grad: Whether to track gradients for autograd
        """
        # Handle different input types
        if isinstance(data, Tensor):
            self._data = data._data.copy()
            self._device = data._device
            self._grad = data._grad
            self._grad_fn = data._grad_fn
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
            self._device = Device('cpu')
            self._grad = None
            self._grad_fn = None
        else:
            self._data = np.array(data)
            self._device = Device('cpu')
            self._grad = None
            self._grad_fn = None
        
        # Apply dtype if specified
        if dtype is not None:
            self._data = self._data.astype(dtype)
        
        # Move to device if specified
        if device is not None:
            dev = get_device(device)
            if dev != self._device:
                self._data = self._to_device(self._data, dev)
                self._device = dev
        
        self._requires_grad = requires_grad
    
    def _to_device(self, data: np.ndarray, device: Device) -> np.ndarray:
        """Transfer data to a device (internal helper)."""
        if device.is_cpu:
            return data
        # For GPU, this would involve actual CUDA transfer
        # For now, just return the data
        return data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self._data.shape
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self._data.ndim
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type."""
        return self._data.dtype
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self._data.size
    
    @property
    def device(self) -> Device:
        """Get current device."""
        return self._device
    
    @property
    def requires_grad(self) -> bool:
        """Check if gradients are being tracked."""
        return self._requires_grad
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array (always returns CPU copy)."""
        if self._device.is_cuda:
            # Would transfer from GPU in real implementation
            pass
        return self._data.copy()
    
    def cpu(self) -> 'Tensor':
        """Move tensor to CPU."""
        if self._device.is_cpu:
            return self
        new_tensor = Tensor(self._data, device='cpu')
        new_tensor._requires_grad = self._requires_grad
        return new_tensor
    
    def cuda(self, device_id: int = 0) -> 'Tensor':
        """Move tensor to CUDA GPU.
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Tensor on GPU
        """
        device = Device('cuda', device_id)
        if self._device == device:
            return self
        new_tensor = Tensor(self._data, device=device)
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
        if self._device == dev:
            return self
        new_tensor = Tensor(self._data, device=dev)
        new_tensor._requires_grad = self._requires_grad
        return new_tensor
    
    def copy(self) -> 'Tensor':
        """Create a copy of the tensor."""
        new_tensor = Tensor(self._data.copy(), device=self._device)
        new_tensor._requires_grad = self._requires_grad
        if self._grad is not None:
            new_tensor._grad = self._grad.copy()
        return new_tensor
    
    def clone(self) -> 'Tensor':
        """Alias for copy()."""
        return self.copy()
    
    def detach(self) -> 'Tensor':
        """Return a new tensor detached from the computation graph."""
        new_tensor = Tensor(self._data, device=self._device)
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
            # Assume scalar output, grad is 1
            grad_output = Tensor(np.ones_like(self._data))
        
        # Store the gradient
        self._grad = grad_output._data if isinstance(grad_output, Tensor) else grad_output
        
        # In a full implementation, this would propagate gradients
        # through the computation graph via _grad_fn
    
    @property
    def grad(self) -> Optional['Tensor']:
        """Get the gradient of this tensor."""
        if self._grad is None:
            return None
        return Tensor(self._grad, device=self._device)
    
    def zero_grad(self) -> None:
        """Zero out the gradient."""
        self._grad = None
    
    # Numpy-compatible methods
    
    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor."""
        new_data = self._data.reshape(shape)
        return Tensor(new_data, device=self._device)
    
    def transpose(self, *axes: int) -> 'Tensor':
        """Transpose tensor dimensions."""
        if len(axes) == 0:
            new_data = self._data.T
        else:
            new_data = self._data.transpose(axes)
        return Tensor(new_data, device=self._device)
    
    def permute(self, *dims: int) -> 'Tensor':
        """Permute tensor dimensions (same as transpose)."""
        return self.transpose(*dims)
    
    def flatten(self) -> 'Tensor':
        """Flatten to 1D."""
        return self.reshape(-1)
    
    def squeeze(self, axis: Optional[int] = None) -> 'Tensor':
        """Remove dimensions of size 1."""
        new_data = self._data.squeeze(axis)
        return Tensor(new_data, device=self._device)
    
    def unsqueeze(self, axis: int) -> 'Tensor':
        """Add a dimension of size 1 at the specified position."""
        new_data = np.expand_dims(self._data, axis)
        return Tensor(new_data, device=self._device)
    
    def expand_dims(self, axis: int) -> 'Tensor':
        """Alias for unsqueeze."""
        return self.unsqueeze(axis)
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Sum elements along axis."""
        new_data = self._data.sum(axis=axis, keepdims=keepdims)
        return Tensor(new_data, device=self._device)
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Compute mean along axis."""
        new_data = self._data.mean(axis=axis, keepdims=keepdims)
        return Tensor(new_data, device=self._device)
    
    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Compute maximum along axis."""
        new_data = self._data.max(axis=axis, keepdims=keepdims)
        return Tensor(new_data, device=self._device)
    
    def min(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Compute minimum along axis."""
        new_data = self._data.min(axis=axis, keepdims=keepdims)
        return Tensor(new_data, device=self._device)
    
    def argmax(self, axis: Optional[int] = None) -> 'Tensor':
        """Return indices of maximum values."""
        new_data = self._data.argmax(axis=axis)
        return Tensor(new_data, device=self._device)
    
    def argmin(self, axis: Optional[int] = None) -> 'Tensor':
        """Return indices of minimum values."""
        new_data = self._data.argmin(axis=axis)
        return Tensor(new_data, device=self._device)
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        """Compute dot product."""
        new_data = np.dot(self._data, other._data)
        return Tensor(new_data, device=self._device)
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        new_data = np.matmul(self._data, other._data)
        return Tensor(new_data, device=self._device)
    
    @property
    def T(self) -> 'Tensor':
        """Transpose property."""
        return self.transpose()
    
    # Indexing
    
    def __getitem__(self, index) -> 'Tensor':
        """Get item (supports numpy-style indexing)."""
        new_data = self._data[index]
        return Tensor(new_data, device=self._device)
    
    def __setitem__(self, index, value) -> None:
        """Set item."""
        if isinstance(value, Tensor):
            value = value._data
        self._data[index] = value
    
    # Arithmetic operations
    
    def __add__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data + other, device=self._device)
    
    def __radd__(self, other) -> 'Tensor':
        return self.__add__(other)
    
    def __sub__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data - other, device=self._device)
    
    def __rsub__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(other - self._data, device=self._device)
    
    def __mul__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data * other, device=self._device)
    
    def __rmul__(self, other) -> 'Tensor':
        return self.__mul__(other)
    
    def __truediv__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data / other, device=self._device)
    
    def __rtruediv__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(other / self._data, device=self._device)
    
    def __pow__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data ** other, device=self._device)
    
    def __matmul__(self, other) -> 'Tensor':
        return self.matmul(other)
    
    def __neg__(self) -> 'Tensor':
        return Tensor(-self._data, device=self._device)
    
    def __abs__(self) -> 'Tensor':
        return Tensor(np.abs(self._data), device=self._device)
    
    # Comparison operations
    
    def __eq__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data == other, device=self._device)
    
    def __ne__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data != other, device=self._device)
    
    def __lt__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data < other, device=self._device)
    
    def __le__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data <= other, device=self._device)
    
    def __gt__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data > other, device=self._device)
    
    def __ge__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        return Tensor(self._data >= other, device=self._device)
    
    # In-place operations
    
    def __iadd__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        self._data += other
        return self
    
    def __isub__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        self._data -= other
        return self
    
    def __imul__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        self._data *= other
        return self
    
    def __itruediv__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            other = other._data
        self._data /= other
        return self
    
    # Representation
    
    def __repr__(self) -> str:
        device_str = str(self._device)
        grad_str = ", grad_fn=" + str(self._grad_fn) if self._grad_fn else ""
        return f"Tensor({self._data}, device={device_str}{grad_str})"
    
    def __str__(self) -> str:
        return str(self._data)
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self):
        for i in range(len(self._data)):
            yield self[i]
    
    def __contains__(self, item) -> bool:
        return item in self._data
    
    # Array protocol
    
    def __array__(self, dtype=None):
        """Numpy array protocol support."""
        if dtype is None:
            return self._data
        return self._data.astype(dtype)


# Utility functions

def zeros(shape: Tuple[int, ...], dtype: np.dtype = np.float64, device=None) -> Tensor:
    """Create a tensor of zeros."""
    data = np.zeros(shape, dtype=dtype)
    return Tensor(data, device=device)


def ones(shape: Tuple[int, ...], dtype: np.dtype = np.float64, device=None) -> Tensor:
    """Create a tensor of ones."""
    data = np.ones(shape, dtype=dtype)
    return Tensor(data, device=device)


def zeros_like(tensor: Tensor, dtype=None, device=None) -> Tensor:
    """Create a tensor of zeros with the same shape."""
    shape = tensor.shape
    dtype = dtype if dtype is not None else tensor.dtype
    device = device if device is not None else tensor.device
    return zeros(shape, dtype, device)


def ones_like(tensor: Tensor, dtype=None, device=None) -> Tensor:
    """Create a tensor of ones with the same shape."""
    shape = tensor.shape
    dtype = dtype if dtype is not None else tensor.dtype
    device = device if device is not None else tensor.device
    return ones(shape, dtype, device)


def full(shape: Tuple[int, ...], fill_value, dtype=None, device=None) -> Tensor:
    """Create a tensor filled with a value."""
    if dtype is None:
        dtype = np.array(fill_value).dtype
    data = np.full(shape, fill_value, dtype=dtype)
    return Tensor(data, device=device)


def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype=None, device=None) -> Tensor:
    """Create a range of values."""
    data = np.arange(start, stop, step, dtype=dtype)
    return Tensor(data, device=device)


def linspace(start: float, stop: float, num: int = 50, device=None) -> Tensor:
    """Create linearly spaced values."""
    data = np.linspace(start, stop, num)
    return Tensor(data, device=device)


def eye(n: int, m: Optional[int] = None, dtype=np.float64, device=None) -> Tensor:
    """Create an identity matrix."""
    data = np.eye(n, m, dtype=dtype)
    return Tensor(data, device=device)


def diag(v: Union[Tensor, np.ndarray], k: int = 0) -> Tensor:
    """Extract diagonal or create diagonal matrix."""
    if isinstance(v, Tensor):
        v = v._data
    data = np.diag(v, k)
    return Tensor(data)


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along a new axis."""
    arrays = [t._data for t in tensors]
    data = np.stack(arrays, axis)
    device = tensors[0].device
    return Tensor(data, device=device)


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Concatenate tensors along an axis."""
    arrays = [t._data for t in tensors]
    data = np.concatenate(arrays, axis)
    device = tensors[0].device
    return Tensor(data, device=device)


def where(condition: Tensor, x, y) -> Tensor:
    """Select elements from x or y based on condition."""
    if isinstance(x, Tensor):
        x = x._data
    if isinstance(y, Tensor):
        y = y._data
    data = np.where(condition._data, x, y)
    return Tensor(data, device=condition.device)


def exp(tensor: Tensor) -> Tensor:
    """Exponential."""
    return Tensor(np.exp(tensor._data), device=tensor.device)


def log(tensor: Tensor) -> Tensor:
    """Natural logarithm."""
    return Tensor(np.log(tensor._data), device=tensor.device)


def sqrt(tensor: Tensor) -> Tensor:
    """Square root."""
    return Tensor(np.sqrt(tensor._data), device=tensor.device)


def sin(tensor: Tensor) -> Tensor:
    """Sine."""
    return Tensor(np.sin(tensor._data), device=tensor.device)


def cos(tensor: Tensor) -> Tensor:
    """Cosine."""
    return Tensor(np.cos(tensor._data), device=tensor.device)


def tan(tensor: Tensor) -> Tensor:
    """Tangent."""
    return Tensor(np.tan(tensor._data), device=tensor.device)


def clip(tensor: Tensor, min_val, max_val) -> Tensor:
    """Clip values to range."""
    return Tensor(np.clip(tensor._data, min_val, max_val), device=tensor.device)

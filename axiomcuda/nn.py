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
Neural network utilities for AxiomCUDA.

Provides common neural network operations like softmax, activations,
and one-hot encoding with GPU acceleration support.
"""

import numpy as np
from typing import Optional, Union
from .tensor import Tensor


def softmax(x: Union[Tensor, np.ndarray], axis: int = -1) -> Union[Tensor, np.ndarray]:
    """Compute softmax activation.
    
    softmax(x)_i = exp(x_i) / sum_j exp(x_j)
    
    Uses numerical stability trick: subtract max before exponentiation.
    
    Args:
        x: Input tensor or array
        axis: Axis along which to compute softmax
        
    Returns:
        Softmax of input (same type as input)
        
    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> softmax(x)
        array([0.09003057, 0.24472847, 0.66524096])
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    
    # Subtract max for numerical stability
    max_vals = np.max(arr, axis=axis, keepdims=True)
    shifted = arr - max_vals
    exp_x = np.exp(shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    result = exp_x / sum_exp
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def log_softmax(x: Union[Tensor, np.ndarray], axis: int = -1) -> Union[Tensor, np.ndarray]:
    """Compute log-softmax activation.
    
    log_softmax(x)_i = x_i - log(sum_j exp(x_j))
    
    More numerically stable than taking log of softmax.
    
    Args:
        x: Input tensor or array
        axis: Axis along which to compute log-softmax
        
    Returns:
        Log-softmax of input (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    
    # Subtract max for numerical stability
    max_vals = np.max(arr, axis=axis, keepdims=True)
    shifted = arr - max_vals
    exp_x = np.exp(shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    log_sum_exp = np.log(sum_exp) + max_vals
    result = arr - log_sum_exp
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def one_hot(
    indices: Union[Tensor, np.ndarray],
    num_classes: int,
    dtype: np.dtype = np.float64,
    device=None,
) -> Union[Tensor, np.ndarray]:
    """Convert indices to one-hot encoding.
    
    Args:
        indices: Class indices (integers)
        num_classes: Number of classes
        dtype: Output data type
        device: Target device for Tensor output
        
    Returns:
        One-hot encoded array/tensor of shape indices.shape + (num_classes,)
        
    Example:
        >>> indices = np.array([0, 2, 1])
        >>> one_hot(indices, num_classes=4)
        array([[1., 0., 0., 0.],
               [0., 0., 1., 0.],
               [0., 1., 0., 0.]])
    """
    is_tensor = isinstance(indices, Tensor)
    arr = indices._data if is_tensor else np.asarray(indices)
    
    # Flatten for easier processing
    flat_indices = arr.reshape(-1)
    n = flat_indices.size
    
    # Create one-hot encoding
    result = np.zeros((n, num_classes), dtype=dtype)
    result[np.arange(n), flat_indices.astype(int)] = 1
    
    # Reshape back
    result = result.reshape(arr.shape + (num_classes,))
    
    if is_tensor:
        return Tensor(result, device=indices.device)
    elif device is not None:
        return Tensor(result, device=device)
    return result


# Activation functions

def relu(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Rectified Linear Unit: relu(x) = max(0, x)
    
    Args:
        x: Input tensor or array
        
    Returns:
        ReLU activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = np.maximum(0, arr)
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def gelu(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Gaussian Error Linear Unit.
    
    gelu(x) = x * Phi(x) where Phi is the CDF of standard normal.
    Approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    Args:
        x: Input tensor or array
        
    Returns:
        GELU activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    
    # Approximation from the original GELU paper
    c = 0.044715
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    result = 0.5 * arr * (1.0 + np.tanh(sqrt_2_over_pi * (arr + c * arr ** 3)))
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def sigmoid(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x))
    
    Args:
        x: Input tensor or array
        
    Returns:
        Sigmoid activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = 1.0 / (1.0 + np.exp(-arr))
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def tanh(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Hyperbolic tangent: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        x: Input tensor or array
        
    Returns:
        Tanh activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = np.tanh(arr)
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def silu(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Sigmoid Linear Unit (Swish-1): silu(x) = x * sigmoid(x)
    
    Args:
        x: Input tensor or array
        
    Returns:
        SiLU activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = arr / (1.0 + np.exp(-arr))
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def swish(x: Union[Tensor, np.ndarray], beta: float = 1.0) -> Union[Tensor, np.ndarray]:
    """Swish activation: swish(x, β) = x * sigmoid(β * x)
    
    When β=1, this is equivalent to SiLU.
    
    Args:
        x: Input tensor or array
        beta: Beta parameter (default 1.0)
        
    Returns:
        Swish activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = arr / (1.0 + np.exp(-beta * arr))
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def leaky_relu(x: Union[Tensor, np.ndarray], negative_slope: float = 0.01) -> Union[Tensor, np.ndarray]:
    """Leaky ReLU: leaky_relu(x) = max(0, x) + negative_slope * min(0, x)
    
    Args:
        x: Input tensor or array
        negative_slope: Slope for negative values (default 0.01)
        
    Returns:
        Leaky ReLU activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = np.where(arr > 0, arr, negative_slope * arr)
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def elu(x: Union[Tensor, np.ndarray], alpha: float = 1.0) -> Union[Tensor, np.ndarray]:
    """Exponential Linear Unit:
    elu(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    
    Args:
        x: Input tensor or array
        alpha: Scale for negative values (default 1.0)
        
    Returns:
        ELU activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = np.where(arr > 0, arr, alpha * (np.exp(arr) - 1))
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def selu(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Scaled Exponential Linear Unit.
    
    selu(x) = scale * elu(x, alpha) with specific alpha and scale values
    that ensure self-normalization.
    
    Args:
        x: Input tensor or array
        
    Returns:
        SELU activation (same type as input)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)


def softplus(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Softplus: softplus(x) = log(1 + exp(x))
    
    Args:
        x: Input tensor or array
        
    Returns:
        Softplus activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    # For numerical stability
    result = np.where(arr > 20, arr, np.log1p(np.exp(arr)))
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def softsign(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Softsign: softsign(x) = x / (1 + |x|)
    
    Args:
        x: Input tensor or array
        
    Returns:
        Softsign activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = arr / (1.0 + np.abs(arr))
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def hard_sigmoid(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Hard sigmoid: hard_sigmoid(x) = relu6(x + 3) / 6
    
    Piecewise linear approximation of sigmoid.
    
    Args:
        x: Input tensor or array
        
    Returns:
        Hard sigmoid activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = np.clip(arr / 6.0 + 0.5, 0.0, 1.0)
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def hard_swish(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Hard Swish: hard_swish(x) = x * hard_sigmoid(x)
    
    Args:
        x: Input tensor or array
        
    Returns:
        Hard swish activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = arr * np.clip(arr / 6.0 + 0.5, 0.0, 1.0)
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


def relu6(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """ReLU6: relu6(x) = min(max(0, x), 6)
    
    ReLU clipped at 6.0.
    
    Args:
        x: Input tensor or array
        
    Returns:
        ReLU6 activation (same type as input)
    """
    is_tensor = isinstance(x, Tensor)
    arr = x._data if is_tensor else np.asarray(x)
    result = np.clip(arr, 0.0, 6.0)
    
    if is_tensor:
        return Tensor(result, device=x.device)
    return result


# Loss functions

def cross_entropy_loss(
    logits: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    reduction: str = 'mean',
) -> Union[Tensor, np.ndarray]:
    """Cross-entropy loss for classification.
    
    Args:
        logits: Unnormalized predictions (pre-softmax)
        targets: Target class indices
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Cross-entropy loss
    """
    is_tensor = isinstance(logits, Tensor)
    logits_arr = logits._data if is_tensor else np.asarray(logits)
    targets_arr = targets._data if isinstance(targets, Tensor) else np.asarray(targets)
    
    # Log-softmax for numerical stability
    log_probs = log_softmax(logits_arr, axis=-1)
    
    # Gather log probs for target classes
    n = log_probs.shape[0]
    loss = -log_probs[np.arange(n), targets_arr.astype(int)]
    
    if reduction == 'mean':
        result = np.mean(loss)
    elif reduction == 'sum':
        result = np.sum(loss)
    else:
        result = loss
    
    if is_tensor:
        return Tensor(result, device=logits.device)
    return result


def binary_cross_entropy(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    reduction: str = 'mean',
) -> Union[Tensor, np.ndarray]:
    """Binary cross-entropy loss.
    
    Args:
        predictions: Predicted probabilities (after sigmoid)
        targets: Target binary labels (0 or 1)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Binary cross-entropy loss
    """
    is_tensor = isinstance(predictions, Tensor)
    pred_arr = predictions._data if is_tensor else np.asarray(predictions)
    targets_arr = targets._data if isinstance(targets, Tensor) else np.asarray(targets)
    
    # Clip predictions for numerical stability
    epsilon = 1e-7
    pred_arr = np.clip(pred_arr, epsilon, 1 - epsilon)
    
    loss = -(targets_arr * np.log(pred_arr) + (1 - targets_arr) * np.log(1 - pred_arr))
    
    if reduction == 'mean':
        result = np.mean(loss)
    elif reduction == 'sum':
        result = np.sum(loss)
    else:
        result = loss
    
    if is_tensor:
        return Tensor(result, device=predictions.device)
    return result


def mse_loss(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    reduction: str = 'mean',
) -> Union[Tensor, np.ndarray]:
    """Mean squared error loss.
    
    Args:
        predictions: Predicted values
        targets: Target values
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        MSE loss
    """
    is_tensor = isinstance(predictions, Tensor)
    pred_arr = predictions._data if is_tensor else np.asarray(predictions)
    targets_arr = targets._data if isinstance(targets, Tensor) else np.asarray(targets)
    
    loss = (pred_arr - targets_arr) ** 2
    
    if reduction == 'mean':
        result = np.mean(loss)
    elif reduction == 'sum':
        result = np.sum(loss)
    else:
        result = loss
    
    if is_tensor:
        return Tensor(result, device=predictions.device)
    return result


def l1_loss(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    reduction: str = 'mean',
) -> Union[Tensor, np.ndarray]:
    """L1 (mean absolute error) loss.
    
    Args:
        predictions: Predicted values
        targets: Target values
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        L1 loss
    """
    is_tensor = isinstance(predictions, Tensor)
    pred_arr = predictions._data if is_tensor else np.asarray(predictions)
    targets_arr = targets._data if isinstance(targets, Tensor) else np.asarray(targets)
    
    loss = np.abs(pred_arr - targets_arr)
    
    if reduction == 'mean':
        result = np.mean(loss)
    elif reduction == 'sum':
        result = np.sum(loss)
    else:
        result = loss
    
    if is_tensor:
        return Tensor(result, device=predictions.device)
    return result

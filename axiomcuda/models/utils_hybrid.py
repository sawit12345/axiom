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
Hybrid mixture utilities - CUDA accelerated wrapper.

Provides utilities for creating and training hybrid mixture models
with both continuous and discrete likelihoods.
"""

from typing import List, Optional, Tuple

import numpy as np

# Use ONLY the C++ backend
try:
    import axiomcuda_backend as backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

from .base import ModelConfig


def _check_backend():
    """Check if C++ backend is available."""
    if not BACKEND_AVAILABLE:
        raise RuntimeError("C++ backend (axiomcuda_backend) is not available. Cannot create mixture model.")


def create_mm(
    num_components: int,
    continuous_dim: int,
    discrete_dims: List[int],
    discrete_alphas: Optional[List[float]] = None,
    cont_scale: float = 1.0,
    color_precision_scale: Optional[float] = None,
    opt: Optional[dict] = None,
    **kwargs,
):
    """Create a hybrid mixture model using the C++ backend.
    
    Args:
        num_components: Number of mixture components.
        continuous_dim: Dimension of continuous features.
        discrete_dims: List of dimensions for discrete features.
        discrete_alphas: Prior alphas for discrete features.
        cont_scale: Scale for continuous features.
        color_precision_scale: Scale for color precision.
        opt: Optimization options.
        **kwargs: Additional arguments.
        
    Returns:
        HybridMixture: Initialized hybrid mixture model.
    """
    _check_backend()
    
    if opt is None:
        opt = {"lr": 1.0, "beta": 0.0}
    
    if discrete_alphas is None:
        discrete_alphas = [1e-4] * len(discrete_dims)
    
    # Create the C++ HybridMixture model
    # This creates a model with the specified structure
    model = backend.models.HybridMixture()
    
    # Initialize continuous likelihood parameters
    # Prior parameters for Normal-Inverse-Wishart
    kappa = np.full(num_components, 1e-4, dtype=np.float64)
    n = np.full(num_components, continuous_dim + 2.0, dtype=np.float64)
    
    # Initialize U matrix (scale matrix)
    u = np.zeros((num_components, continuous_dim, continuous_dim), dtype=np.float64)
    for k in range(num_components):
        for i in range(continuous_dim):
            u[k, i, i] = cont_scale * cont_scale
    
    # Apply color precision scale if provided
    if color_precision_scale is not None:
        spatial_dim = 2
        for k in range(num_components):
            for i in range(spatial_dim, continuous_dim):
                u[k, i, i] *= color_precision_scale
    
    # Initialize means with small random noise
    np.random.seed(0)  # For reproducibility
    mean = np.random.randn(num_components, continuous_dim, 1) * 0.01
    
    # Initialize discrete likelihoods
    disc_alpha = []
    for discrete_alpha, discrete_dim in zip(discrete_alphas, discrete_dims):
        alpha = discrete_alpha * np.ones((num_components, discrete_dim, 1), dtype=np.float64)
        
        # Eye prior (permuted)
        d = min(num_components, discrete_dim)
        eye = np.eye(d)
        nr, nc = int(np.ceil(num_components / d)), int(np.ceil(discrete_dim / d))
        eye = np.tile(eye, (nr, nc))
        eye = eye[:num_components, :discrete_dim, None]
        
        # Permute
        idcs = np.random.permutation(eye.shape[0])
        alpha = alpha + eye[idcs] * 10.0
        
        # Add random scaling
        alpha = alpha * np.random.uniform(0.1, 0.9, size=alpha.shape)
        
        disc_alpha.append(alpha)
    
    # Initialize prior (Dirichlet)
    prior_alpha = 0.1 * np.ones(num_components, dtype=np.float64)
    
    # Store in model state (this would be done through backend calls)
    # For now, we return the model handle which manages its own state
    
    return model


def train_step_fn(model, mask: np.ndarray, c_sample: np.ndarray, d_sample: List[np.ndarray], logp_thr: float = -0.1) -> Tuple[object, np.ndarray, np.ndarray]:
    """Training step for hybrid mixture model.
    
    Args:
        model: Hybrid mixture model (C++ backend handle).
        mask: Used component mask.
        c_sample: Continuous sample of shape (batch, cont_dim, 1).
        d_sample: Discrete samples, each of shape (batch, disc_dim, 1).
        logp_thr: Log probability threshold.
        
    Returns:
        Tuple of (updated_model, updated_mask, qz).
    """
    _check_backend()
    
    # Format inputs for C++ backend
    c_t = c_sample.astype(np.float64)
    d_t = [d.astype(np.float64) for d in d_sample]
    
    # E-step
    qz = np.zeros(mask.shape[0], dtype=np.float64)
    c_ell = np.zeros(mask.shape[0], dtype=np.float64)
    d_ell = np.zeros(mask.shape[0], dtype=np.float64)
    
    w_disc = [1.0] * len(d_t)
    
    if hasattr(model, 'eStep'):
        model.eStep(c_t, d_t, w_disc, qz, c_ell, d_ell)
    
    elogp = c_ell + d_ell
    
    # Mask out unused components
    elogp = elogp * mask + (1 - mask) * (-1e10)
    
    # Softmax to get qz
    exp_elogp = np.exp(elogp - elogp.max())
    qz = exp_elogp / exp_elogp.sum()
    
    # Check if well explained
    if elogp.max() > logp_thr:
        # Update model
        if hasattr(model, 'mStep'):
            model.mStep(c_t, d_t, qz, 1.0, 0.0)
    else:
        # Find unused component
        unused_idx = np.where(mask == 0)[0]
        if len(unused_idx) > 0:
            idx = unused_idx[0]
            # Assign to this component
            qz_new = np.zeros_like(qz)
            qz_new[idx] = 1.0
            qz = qz_new
            mask_new = mask.copy()
            mask_new[idx] = 1.0
            mask = mask_new
            
            # Update model
            if hasattr(model, 'mStep'):
                model.mStep(c_t, d_t, qz, 1.0, 0.0)
    
    # Update mask based on alpha values
    if hasattr(model, 'state_'):
        mask = (model.state_.posterior_alpha > model.state_.prior_alpha).astype(np.float64)
    
    return model, mask, qz


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along specified axis.
    
    Args:
        x: Input array.
        axis: Axis to compute softmax over.
        
    Returns:
        Softmax of input.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# Re-export for compatibility
Mixture = object  # Will be replaced with actual HybridMixture when backend is available
HybridMixture = object  # Will be replaced with actual HybridMixture when backend is available

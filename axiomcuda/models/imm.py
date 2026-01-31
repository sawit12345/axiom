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
Identity Mixture Model (IMM) - CUDA accelerated wrapper.

Wraps the C++ backend for GPU-accelerated object identity modeling.
"""

from dataclasses import dataclass
from typing import NamedTuple, Tuple, Optional

import numpy as np

# Use ONLY the C++ backend
try:
    import axiomcuda_backend as backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

from .base import ModelConfig


@dataclass(frozen=True)
class IMMConfig(ModelConfig):
    """
    Configuration for the IMM
    
    Attributes:
        num_object_types: Number of object identity classes.
        num_features: Number of features for identity.
        i_ell_threshold: Threshold for identity log-likelihood.
        cont_scale_identity: Scale for continuous identity features.
        color_precision_scale: Scale for color precision.
        color_only_identity: Use only color for identity.
    """

    num_object_types: int = 32
    num_features: int = 5

    i_ell_threshold: float = -500

    cont_scale_identity: float = 0.5
    color_precision_scale: float = 1.0
    color_only_identity: bool = False


class IMM(NamedTuple):
    """Identity Mixture Model container.
    
    Attributes:
        model: The underlying C++ IdentityMixtureModel handle.
        used_mask: Mask indicating which components are used.
    """
    model: object  # C++ backend model handle
    used_mask: np.ndarray


def _check_backend():
    """Check if C++ backend is available."""
    if not BACKEND_AVAILABLE:
        raise RuntimeError("C++ backend (axiomcuda_backend) is not available. Cannot create IMM model.")


def infer_identity(imm: IMM, x: np.ndarray, color_only_identity: bool = False) -> np.ndarray:
    """Infer object identity from features.
    
    Args:
        imm: Identity Mixture Model.
        x: Object features of shape (B, num_features, 1).
        color_only_identity: Use only color features.
        
    Returns:
        Array of class labels.
    """
    _check_backend()
    
    # Extract last 5 features if needed
    if x.shape[1] > 5:
        x = x[:, -5:, :]
    
    # Scale color features (indices 2+)
    object_features = x.copy()
    object_features[:, 2:, :] = object_features[:, 2:, :] * 100.0
    
    # Use only color if requested
    if color_only_identity:
        object_features = object_features[:, 2:, :]
    
    # Convert to float64 for backend
    object_features = object_features.astype(np.float64)
    
    # Infer identity using C++ backend
    class_labels = np.zeros(x.shape[0], dtype=np.float64)
    imm.model.inferIdentity(object_features, color_only_identity, class_labels)
    
    return class_labels.astype(np.int32)


def create_imm(
    num_object_types: int,
    num_features: int = 5,
    cont_scale_identity: float = 0.5,
    color_precision_scale: Optional[float] = None,
    color_only_identity: bool = False,
    **kwargs,
) -> IMM:
    """Create an Identity Mixture Model using the C++ backend.
    
    Args:
        num_object_types: Number of object types.
        num_features: Number of features.
        cont_scale_identity: Scale for continuous features.
        color_precision_scale: Scale for color precision.
        color_only_identity: Use only color for identity.
        **kwargs: Additional arguments.
        
    Returns:
        IMM: Initialized Identity Mixture Model.
    """
    _check_backend()
    
    # Set default color precision scale if not provided
    if color_precision_scale is None:
        color_precision_scale = 1.0
    
    # Create the C++ backend model
    model = backend.models.createIMM(
        num_object_types,
        num_features,
        cont_scale_identity,
        color_precision_scale,
        color_only_identity
    )
    
    # Initialize used mask
    i_used_mask = np.zeros(num_object_types, dtype=np.float64)
    
    return IMM(
        model=model,
        used_mask=i_used_mask,
    )


def infer_remapped_color_identity(
    imm: IMM,
    obs: np.ndarray,
    object_idx: int,
    num_features: int,
    ell_threshold: float = -100.0,
    **kwargs
) -> np.ndarray:
    """Infer identity with color remapping.
    
    Args:
        imm: Identity Mixture Model.
        obs: Observations.
        object_idx: Object index.
        num_features: Number of features.
        ell_threshold: Threshold for using shape-only inference.
        
    Returns:
        Identity probabilities (qz).
    """
    _check_backend()
    
    # Extract object features
    object_features = obs[object_idx:object_idx+1, obs.shape[-1] - num_features:, :]
    
    # Scale color features
    object_features = object_features.copy()
    object_features[:, 2:, :] = object_features[:, 2:, :] * 100.0
    
    # Convert to float64
    object_features = object_features.astype(np.float64)
    
    # Infer with remapped colors
    qz = np.zeros(imm.model.state_.num_object_types, dtype=np.float64)
    imm.model.inferRemappedColorIdentity(object_features, False, ell_threshold, qz)
    
    return qz


def infer_and_update_identity(
    imm: IMM,
    obs: np.ndarray,
    object_idx: int,
    num_features: int,
    i_ell_threshold: float,
    color_only_identity: bool = False,
    **kwargs,
) -> IMM:
    """Infer and update identity.
    
    Args:
        imm: Identity Mixture Model.
        obs: Observations.
        object_idx: Object index.
        num_features: Number of features.
        i_ell_threshold: Threshold for identity model.
        color_only_identity: Use only color for identity.
        
    Returns:
        Updated IMM.
    """
    _check_backend()
    
    # Extract object features
    object_features = obs[object_idx:object_idx+1, obs.shape[-1] - num_features:, :]
    
    # Scale color features
    object_features = object_features.copy()
    object_features[:, 2:, :] = object_features[:, 2:, :] * 100.0
    
    # Use only color if requested
    if color_only_identity:
        object_features = object_features[:, 2:, :]
    
    # Convert to float64
    object_features = object_features.astype(np.float64)
    
    # Training step
    qz = np.zeros(imm.model.state_.num_object_types, dtype=np.float64)
    grew_component = np.array([False])
    
    imm.model.trainStep(object_features, i_ell_threshold, qz, grew_component)
    
    # Update used mask
    used_mask = imm.model.getUsedMask().data.astype(np.float64)
    
    return IMM(
        model=imm.model,
        used_mask=used_mask,
    )


def compute_ell(imm: IMM, x: np.ndarray) -> np.ndarray:
    """Compute expected log-likelihood for identity features.
    
    Args:
        imm: Identity Mixture Model.
        x: Object features.
        
    Returns:
        ELL values for each component.
    """
    _check_backend()
    
    x_t = x.astype(np.float64)
    
    ell = np.zeros((x.shape[0], imm.model.state_.num_object_types), dtype=np.float64)
    imm.model.computeELL(x_t, ell)
    
    return ell


def get_posterior(imm: IMM, x: np.ndarray) -> np.ndarray:
    """Get posterior probabilities for identity.
    
    Args:
        imm: Identity Mixture Model.
        x: Object features.
        
    Returns:
        Posterior probabilities (batch, num_object_types).
    """
    _check_backend()
    
    x_t = x.astype(np.float64)
    
    posterior = np.zeros((x.shape[0], imm.model.state_.num_object_types), dtype=np.float64)
    imm.model.getPosterior(x_t, posterior)
    
    return posterior

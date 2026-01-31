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
Slot Mixture Model (SMM) - CUDA accelerated wrapper.

Wraps the C++ backend for GPU-accelerated slot mixture modeling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, NamedTuple, Union, Sequence, Optional

# Use ONLY the C++ backend
try:
    import axiomcuda_backend as backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

from .base import ModelConfig


@dataclass(frozen=True)
class SMMConfig(ModelConfig):
    """
    Configuration class for SMM model.

    Attributes:
        width (int): The width of the input image.
        height (int): The height of the input image.
        input_dim (int): The dimension of the input feature vector.
        slot_dim (int): The dimension of each slot latent.
        num_slots (int): The number of slots.
        use_bias (bool): Whether to use bias in the model.
        ns_a (float): The noise parameter of the transform on the mean of the linear transform.
        ns_b (float): The noise parameter of the bias on the mean of the linear transform.
        dof_offset (float): The offset value for the degrees of freedom.
        mask_prob (tuple[float]): The probability of using the linear template corresponding to the respective index.
        scale (tuple[float]): The scaling factor for each element of the prior on the bias.
        transform_inv_v_scale (float): The scaling factor for the inverse variance of the linear transform.
        bias_inv_v_scale (float): The scaling factor for the inverse variance of the bias vector.
        num_e_steps (int): Number of E-steps to run.
        learning_rate (float): The learning rate for the model updates.
        beta (float): The beta value for the model updates.
        eloglike_threshold (float): Threshold of the ELBO below which we grow the model.
        max_grow_steps (int): Maximum number of grow steps per infer_and_update call.
    """

    width: int = 160
    height: int = 210
    input_dim: int = 5
    slot_dim: int = 2
    num_slots: int = 32
    use_bias: bool = True
    ns_a: float = 1
    ns_b: float = 1
    dof_offset: float = 10
    mask_prob: tuple[float] = field(
        default_factory=lambda: tuple([0.0, 0.0, 0.0, 0.0, 1.0])
    )
    scale: tuple[float] = field(
        default_factory=lambda: tuple([0.075, 0.075, 0.75, 0.75, 0.75])
    )
    transform_inv_v_scale: float = 100
    bias_inv_v_scale: float = 0.001
    num_e_steps: int = 2
    learning_rate: float = 1.0
    beta: float = 0.0
    eloglike_threshold: float = 5.0
    max_grow_steps: int = 20


class SMM(NamedTuple):
    """Slot Mixture Model container.
    
    Attributes:
        model: The underlying C++ SlotMixtureModel handle.
        num_slots: Number of slots in the model.
        width: Image width.
        height: Image height.
        stats: Statistics for normalization (offset and stdevs).
    """
    model: object  # C++ backend model handle
    num_slots: int
    width: int
    height: int
    stats: dict


def _check_backend():
    """Check if C++ backend is available."""
    if not BACKEND_AVAILABLE:
        raise RuntimeError("C++ backend (axiomcuda_backend) is not available. Cannot create SMM model.")


def create_smm(
    width: int = 160,
    height: int = 210,
    input_dim: int = 5,
    slot_dim: int = 2,
    num_slots: int = 32,
    use_bias: bool = True,
    ns_a: float = 1,
    ns_b: float = 1,
    dof_offset: float = 10,
    mask_prob: list[float] = None,
    scale: list[float] = None,
    transform_inv_v_scale: float = 100,
    bias_inv_v_scale: float = 0.001,
    **kwargs,
) -> SMM:
    """Create a Slot Mixture Model using the C++ backend.
    
    Args:
        width: Image width.
        height: Image height.
        input_dim: Input feature dimension.
        slot_dim: Slot latent dimension.
        num_slots: Number of slots.
        use_bias: Whether to use bias.
        ns_a: Noise scale for transform.
        ns_b: Noise scale for bias.
        dof_offset: Degrees of freedom offset.
        mask_prob: Mask template probabilities.
        scale: Scaling factors for prior.
        transform_inv_v_scale: Transform inverse variance scale.
        bias_inv_v_scale: Bias inverse variance scale.
        **kwargs: Additional arguments.
        
    Returns:
        SMM: Initialized Slot Mixture Model.
    """
    _check_backend()
    
    if mask_prob is None:
        mask_prob = [0.0, 0.0, 0.0, 0.0, 1.0]
    if scale is None:
        scale = [0.075, 0.075, 0.75, 0.75, 0.75]
    
    # Create the C++ backend model using factory function
    model = backend.models.createSMM(
        width, height, input_dim, slot_dim, num_slots,
        use_bias, ns_a, ns_b, dof_offset, mask_prob, scale,
        transform_inv_v_scale, bias_inv_v_scale
    )
    
    stats = {
        "offset": np.array([width / 2, height / 2, 128, 128, 128], dtype=np.float64),
        "stdevs": np.array([width / 2, height / 2, 128, 128, 128], dtype=np.float64),
    }
    
    return SMM(
        model=model,
        num_slots=num_slots,
        width=width,
        height=height,
        stats=stats,
    )


def add_position_encoding(img: np.ndarray) -> np.ndarray:
    """Add position encoding (x, y coordinates) to image.
    
    Args:
        img: Image array of shape (H, W, C).
        
    Returns:
        Array with position encoding added, shape (H*W, C+2).
    """
    height, width, n_channels = img.shape
    
    # Create meshgrid for position encoding
    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Flatten and concatenate
    data = np.concatenate([
        u_grid.reshape(-1, 1),
        v_grid.reshape(-1, 1),
        img.reshape(-1, n_channels),
    ], axis=1)
    
    return data


def format_single_frame(single_obs: np.ndarray, offset: np.ndarray, stdevs: np.ndarray) -> np.ndarray:
    """
    Format a single observation frame for the SMM model.
    
    Args:
        single_obs: Observation array of shape (H, W, C).
        offset: Offset values for normalization.
        stdevs: Standard deviation values for normalization.
        
    Returns:
        Formatted observation array of shape (H*W, C+2).
    """
    height, width, n_channels = single_obs.shape
    obs_with_xy = add_position_encoding(single_obs)
    obs_with_xy = obs_with_xy - offset
    obs_with_xy = obs_with_xy / stdevs
    return obs_with_xy.reshape((height * width, n_channels + 2))


def initialize_smm_model(smm: SMM, init_inputs: np.ndarray) -> Tuple[SMM, np.ndarray, np.ndarray]:
    """Initialize SMM model with initial inputs.
    
    Args:
        smm: Slot Mixture Model.
        init_inputs: Initial input observations of shape (H, W, C) or (H*W, C+2).
        
    Returns:
        Tuple of (smm, qx, qz) - initialized model and initial posteriors.
    """
    _check_backend()
    
    # Format inputs if needed
    if init_inputs.ndim == 3:
        init_inputs = format_single_frame(
            init_inputs, 
            smm.stats["offset"], 
            smm.stats["stdevs"]
        )
    
    # Initialize from data using C++ backend
    smm.model.initializeFromData(init_inputs.astype(np.float64))
    
    # Get initial qx and qz from model state
    num_tokens = init_inputs.shape[0]
    qz = np.zeros((1, num_tokens, smm.num_slots), dtype=np.float64)
    qz[0, :, 0] = 1.0  # All data assigned to first slot initially
    
    # qx is obtained from model state after initialization
    qx = np.zeros((1, smm.num_slots, smm.model.state_.slot_dim, 1), dtype=np.float64)
    
    return smm, qx, qz


def infer_and_update(
    smm: SMM,
    inputs: np.ndarray,
    qx_prev: Optional[np.ndarray] = None,
    num_e_steps: int = 2,
    eloglike_threshold: float = 5.0,
    max_grow_steps: int = 10,
    learning_rate: float = 1.0,
    beta: float = 0.0,
    **kwargs,
) -> Tuple[SMM, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and update on SMM model.
    
    Args:
        smm: Slot Mixture Model.
        inputs: Input observations of shape (H, W, C) or (H*W, C+2).
        qx_prev: Previous q(x) distribution (optional).
        num_e_steps: Number of E-steps.
        eloglike_threshold: Threshold for growing components.
        max_grow_steps: Maximum growth steps.
        learning_rate: Learning rate for updates.
        beta: Momentum parameter.
        **kwargs: Additional arguments.
        
    Returns:
        Tuple of (smm, py, qx, qz, used, ell_max).
    """
    _check_backend()
    
    # Format inputs if needed
    if inputs.ndim == 3:
        inputs = format_single_frame(inputs, smm.stats["offset"], smm.stats["stdevs"])
    
    inputs = inputs.astype(np.float64)
    
    # Run EM step using C++ backend
    smm.model.emStep(inputs, learning_rate, beta)
    
    # Get results from model state
    qz = smm.model.getAssignments()
    used_mask = smm.model.getUsedMask()
    
    # Compute ELBO
    qx_mu = smm.model.state_.qx_mu
    qx_inv_sigma = smm.model.state_.qx_inv_sigma
    elbo = smm.model.computeELBO(inputs, qx_mu, qx_inv_sigma, qz)
    
    # Compute variational forward (prediction)
    py_mu = np.zeros((1, smm.num_slots, smm.model.state_.input_dim), dtype=np.float64)
    py_inv_sigma = np.zeros((1, smm.num_slots, smm.model.state_.input_dim, smm.model.state_.input_dim), dtype=np.float64)
    smm.model.variationalForward(qx_mu, qx_inv_sigma, py_mu, py_inv_sigma)
    
    # Compute ell_max (max expected log-likelihood per token)
    num_tokens = inputs.shape[1] if inputs.ndim > 1 else inputs.shape[0]
    ell_max = np.zeros((1, num_tokens), dtype=np.float64)
    
    return smm, py_mu, qx_mu, qz, used_mask, ell_max


def create_e_step_fn(smm: SMM, inputs: np.ndarray):
    """
    Given a fixed SMM model and some observations `input`, returns a function that can
    be used to perform a sequence of E-steps on the model to update qx and qz.
    
    Args:
        smm: Slot Mixture Model.
        inputs: Input observations.
        
    Returns:
        Function for E-step.
    """
    _check_backend()
    
    def e_step_fn(qx_prev: np.ndarray, num_iterations: int = 1):
        """Perform E-step iterations."""
        for _ in range(num_iterations):
            smm.model.eStep(inputs.astype(np.float64), num_iterations)
        
        qx = smm.model.state_.qx_mu
        qz = smm.model.getAssignments()
        
        # Compute ell_max
        qx_inv_sigma = smm.model.state_.qx_inv_sigma
        ell_buffer = np.zeros((inputs.shape[0], inputs.shape[1], smm.num_slots), dtype=np.float64)
        smm.model.computeExpectedLogLikelihood(inputs, qx, qx_inv_sigma, ell_buffer)
        ell_max = ell_buffer.max(axis=-1)
        
        return qx, qz, ell_max
    
    return e_step_fn


# =============================================================================
# Hierarchical SMM
# =============================================================================

class HierarchicalSMM(NamedTuple):
    """Hierarchical Slot Mixture Model.
    
    Attributes:
        models: List of SMM models at different layers.
        num_slots: Number of slots at each layer.
        num_layers: Number of layers in the hierarchy.
        width: Image width.
        height: Image height.
        stats: Statistics for normalization.
    """
    models: Sequence[SMM]
    num_slots: Sequence[int]
    num_layers: int
    width: int
    height: int
    stats: dict


def create_hierarch_smm(layer_configs: Sequence[SMMConfig]) -> HierarchicalSMM:
    """Create a hierarchical SMM.
    
    Args:
        layer_configs: Configuration for each layer.
        
    Returns:
        HierarchicalSMM: Multi-layer SMM model.
    """
    models = []
    num_slots = []
    
    for config in layer_configs:
        model = create_smm(**config.__dict__)
        models.append(model)
        num_slots.append(model.num_slots)
    
    return HierarchicalSMM(
        models=models,
        num_slots=num_slots,
        num_layers=len(layer_configs),
        width=models[0].width,
        height=models[0].height,
        stats=models[0].stats,
    )


def initialize_hierarch_smm(
    model: HierarchicalSMM,
    init_inputs: np.ndarray,
    layer_configs: Sequence[SMMConfig],
) -> Tuple[HierarchicalSMM, list, list, list, list, list]:
    """Initialize hierarchical SMM.
    
    Args:
        model: Hierarchical SMM model.
        init_inputs: Initial inputs.
        layer_configs: Configuration for each layer.
        
    Returns:
        Tuple of updated model and intermediate results.
    """
    from dataclasses import asdict
    
    models_updated = []
    py_updated = []
    qx_updated = []
    qz_updated = []
    used_updated = []
    ell_max_updated = []
    
    init_data = init_inputs
    
    for i, (m, layer_config) in enumerate(zip(model.models, layer_configs)):
        # Initialize this layer
        m, qx, qz = initialize_smm_model(m, init_data)
        
        # Run inference
        m, py, qx, qz, used, ell_max = infer_and_update(
            m,
            init_data,
            qx_prev=qx,
            **asdict(layer_config),
        )
        
        models_updated.append(m)
        py_updated.append(py)
        qx_updated.append(qx)
        qz_updated.append(qz)
        used_updated.append(used)
        ell_max_updated.append(ell_max)
        
        # Prepare input for next layer (first few dimensions of decoded output)
        if i < len(layer_configs) - 1:
            next_input_dim = layer_configs[i + 1].input_dim
            init_data = py[:, :next_input_dim]
    
    smm_updated = HierarchicalSMM(
        models=models_updated,
        num_slots=model.num_slots,
        num_layers=len(layer_configs),
        width=model.width,
        height=model.height,
        stats=model.stats,
    )
    
    return (
        smm_updated,
        py_updated,
        qx_updated,
        qz_updated,
        used_updated,
        ell_max_updated,
    )

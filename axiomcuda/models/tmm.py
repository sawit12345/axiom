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
Transition Mixture Model (TMM) - CUDA accelerated wrapper.

Wraps the C++ backend for GPU-accelerated transition dynamics modeling.
"""

from typing import NamedTuple
from dataclasses import dataclass

import numpy as np

# Use ONLY the C++ backend
try:
    import axiomcuda_backend as backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

from .base import ModelConfig


@dataclass(frozen=True)
class TMMConfig(ModelConfig):
    """
    Configuration class for TMM model
    
    Attributes:
        n_total_components: Total number of components in the TMM model
        state_dim: Dimension of the state space
        dt: Time step for the TMM model
        vu: Unused counter velocity
        use_bias: Whether to use bias
        sigma_sqr: Variance of the Gaussian likelihood
        logp_threshold: Threshold for the log probability
        position_threshold: Threshold for the position
        use_unused_counter: Whether to use unused counter
        use_velocity: Whether to use velocity components
        clip_value: Small value to clip in transition matrix
    """

    n_total_components: int = 200
    state_dim: int = 2
    dt: float = 1.0
    vu: float = 0.05
    use_bias: bool = True
    sigma_sqr: float = 2.0
    logp_threshold: float = -0.00001
    position_threshold: float = 0.15
    use_unused_counter: bool = True
    use_velocity: bool = True
    clip_value: float = 5e-4


class TMM(NamedTuple):
    """Transition Mixture Model container.
    
    Attributes:
        model: The underlying C++ TransitionMixtureModel handle.
        transitions: Transition matrices (K_max, 2*state_dim, 2*state_dim+1).
        used_mask: Boolean mask for used components.
    """
    model: object  # C++ backend model handle
    transitions: np.ndarray
    used_mask: np.ndarray


def _check_backend():
    """Check if C++ backend is available."""
    if not BACKEND_AVAILABLE:
        raise RuntimeError("C++ backend (axiomcuda_backend) is not available. Cannot create TMM model.")


def create_tmm(
    n_total_components: int,
    state_dim: int,
    dt: float = 1.0,
    vu: float = 0.1,
    use_bias: bool = True,
    use_velocity: bool = True,
    **kwargs,
) -> TMM:
    """
    Create a Transition Mixture Model using the C++ backend.
    
    Args:
        n_total_components: Total number of components.
        state_dim: State dimension.
        dt: Time step.
        vu: Unused counter velocity.
        use_bias: Whether to use bias.
        use_velocity: Whether to use velocity components.
        **kwargs: Additional arguments.
        
    Returns:
        TMM: Initialized Transition Mixture Model.
    """
    _check_backend()
    
    # Create the C++ backend model using factory function
    model = backend.models.createTMM(
        n_total_components,
        state_dim,
        dt,
        vu,
        use_bias,
        use_velocity
    )
    
    # Get initial transitions and used mask from model state
    transitions = np.zeros(
        (n_total_components, 2 * state_dim, 2 * state_dim + int(use_bias)),
        dtype=np.float64
    )
    used_mask = model.getUsedMask().data.astype(bool)
    
    # Copy initial transitions from model
    for k in range(n_total_components):
        if used_mask[k]:
            trans_k = np.zeros((2 * state_dim, 2 * state_dim + int(use_bias)), dtype=np.float64)
            model.getTransition(k, trans_k)
            transitions[k] = trans_k
    
    return TMM(
        model=model,
        transitions=transitions,
        used_mask=used_mask,
    )


def forward(transitions: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply transition model to state.
    
    Args:
        transitions: (K_max, 2*state_dim, (2*state_dim)+1) transition matrices.
        x: (2*state_dim,) state vector.
        
    Returns:
        Predicted next states (K_max, 2*state_dim).
    """
    # y = A @ x + b where transitions = [A | b]
    A = transitions[..., :-1]
    b = transitions[..., -1]
    return (A @ x).sum(axis=-1) + b


def forward_single(model, x: np.ndarray) -> np.ndarray:
    """
    Apply single best transition to state using C++ backend.
    
    Args:
        model: TMM model.
        x: (2*state_dim,) state vector.
        
    Returns:
        Predicted next state (2*state_dim,).
    """
    _check_backend()
    
    x_t = x.astype(np.float64)
    out = np.zeros_like(x_t)
    
    # Use C++ forward method
    # model.forward expects a batch of transitions but we pass single
    return model.forward(x_t)


def gaussian_loglike(y: np.ndarray, mu: np.ndarray, sigma_sqr: float = 2.0) -> np.ndarray:
    """
    Compute Gaussian log-likelihood.
    
    Args:
        y: Observed values.
        mu: Mean values.
        sigma_sqr: Variance.
        
    Returns:
        Log-likelihood values.
    """
    squared_error = ((y - mu) ** 2).sum(axis=-1)
    dim = y.shape[-1]
    return -0.5 * squared_error / sigma_sqr - 0.5 * dim * np.log(2 * np.pi * sigma_sqr)


def compute_logprobs(model, x_prev: np.ndarray, x_curr: np.ndarray, sigma_sqr: float = 2.0, use_velocity: bool = True) -> np.ndarray:
    """
    Compute log probabilities for transitions using C++ backend.
    
    Args:
        model: TMM model.
        x_prev: Previous state.
        x_curr: Current state.
        sigma_sqr: Variance.
        use_velocity: Whether to use velocity in likelihood.
        
    Returns:
        Log probabilities for each component.
    """
    _check_backend()
    
    x_prev_t = x_prev.astype(np.float64)
    x_curr_t = x_curr.astype(np.float64)
    
    out_logprobs = np.zeros(model.state_.n_total_components, dtype=np.float64)
    
    model.computeLogProbs(x_prev_t, x_curr_t, sigma_sqr, use_velocity, out_logprobs)
    
    return out_logprobs


def compute_logprobs_masked(model, x_prev: np.ndarray, x_curr: np.ndarray, sigma_sqr: float = 2.0, use_velocity: bool = True) -> np.ndarray:
    """Compute log probabilities with masking using C++ backend.
    
    Args:
        model: TMM model.
        x_prev: Previous state.
        x_curr: Current state.
        sigma_sqr: Variance.
        use_velocity: Whether to use velocity.
        
    Returns:
        Log probabilities (masked components get -inf).
    """
    _check_backend()
    
    x_prev_t = x_prev.astype(np.float64)
    x_curr_t = x_curr.astype(np.float64)
    
    out_logprobs = np.zeros(model.state_.n_total_components, dtype=np.float64)
    
    model.computeLogProbsMasked(x_prev_t, x_curr_t, sigma_sqr, use_velocity, out_logprobs)
    
    return out_logprobs


def get_best_transition(model, x_prev: np.ndarray, x_curr: np.ndarray) -> int:
    """Get best transition for state pair.
    
    Args:
        model: TMM model.
        x_prev: Previous state.
        x_curr: Current state.
        
    Returns:
        Index of best transition.
    """
    _check_backend()
    
    x_prev_t = x_prev.astype(np.float64)
    x_curr_t = x_curr.astype(np.float64)
    
    return model.getBestTransition(x_prev_t, x_curr_t)


def update_model(
    model,
    x_prev: np.ndarray,
    x_curr: np.ndarray,
    sigma_sqr: float = 2.0,
    logp_threshold: float = -0.00001,
    position_threshold: float = 0.15,
    dt: float = 1.0,
    use_unused_counter: bool = True,
    use_velocity: bool = True,
    clip_value: float = 5e-4,
    **kwargs,
) -> Tuple[object, np.ndarray]:
    """
    Update TMM model with new transition using C++ backend.
    
    Args:
        model: TMM model.
        x_prev: Previous state.
        x_curr: Current state.
        sigma_sqr: Variance.
        logp_threshold: Threshold for adding new component.
        position_threshold: Position threshold.
        dt: Time step.
        use_unused_counter: Whether to use unused counter.
        use_velocity: Whether to use velocity.
        clip_value: Small value to clip.
        
    Returns:
        Tuple of (updated_model, logprobs).
    """
    _check_backend()
    
    x_prev_t = x_prev.astype(np.float64)
    x_curr_t = x_curr.astype(np.float64)
    
    # Update the model
    model.updateModel(
        x_prev_t,
        x_curr_t,
        sigma_sqr,
        logp_threshold,
        position_threshold,
        dt,
        use_unused_counter,
        use_velocity,
        clip_value
    )
    
    # Get updated log probabilities
    logprobs = compute_logprobs_masked(model, x_prev, x_curr, sigma_sqr, use_velocity)
    
    return model, logprobs


def generate_default_dynamics_component(state_dim: int, dt: float = 1.0, use_bias: bool = True) -> np.ndarray:
    """
    Generate default dynamics component (constant velocity model).
    
    Args:
        state_dim: State dimension.
        dt: Time step.
        use_bias: Whether to use bias.
        
    Returns:
        Transition matrix of shape (2*state_dim, (2*state_dim)+1).
    """
    full_dim = 2 * state_dim
    x_dim = full_dim + int(use_bias)
    
    # Initialize to zero
    transition = np.zeros((full_dim, x_dim), dtype=np.float64)
    
    # Position identity and velocity coupling
    for i in range(state_dim):
        transition[i, i] = 1.0
        transition[state_dim + i, state_dim + i] = 1.0
        transition[i, state_dim + i] = dt
    
    # Zero out unused counter velocity
    transition[state_dim - 1, :] = 0.0
    transition[full_dim - 1, :] = 0.0
    
    return transition


def generate_default_keep_unused_component(state_dim: int, dt: float = 1.0, vu: float = 1.0, use_bias: bool = True) -> np.ndarray:
    """Generate component for keeping unused state.
    
    Args:
        state_dim: State dimension.
        dt: Time step.
        vu: Unused counter velocity.
        use_bias: Whether to use bias.
        
    Returns:
        Transition matrix.
    """
    full_dim = 2 * state_dim
    x_dim = full_dim + int(use_bias)
    
    transition = np.zeros((full_dim, x_dim), dtype=np.float64)
    
    # Keep position
    for i in range(state_dim):
        transition[i, i] = 1.0
    
    # Keep velocity (except unused counter)
    for i in range(state_dim - 1):
        transition[state_dim + i, state_dim + i] = 1.0
    
    # Bias for unused counter
    if use_bias:
        transition[state_dim - 1, -1] = dt * vu
    
    return transition


def generate_default_become_unused_component(state_dim: int, dt: float = 1.0, vu: float = 1.0, use_bias: bool = True) -> np.ndarray:
    """Generate component for becoming unused.
    
    Args:
        state_dim: State dimension.
        dt: Time step.
        vu: Unused counter velocity.
        use_bias: Whether to use bias.
        
    Returns:
        Transition matrix.
    """
    full_dim = 2 * state_dim
    x_dim = full_dim + int(use_bias)
    
    transition = np.zeros((full_dim, x_dim), dtype=np.float64)
    
    # Keep position (except last)
    for i in range(state_dim - 1):
        transition[i, i] = 1.0
    
    # Bias for unused counter
    if use_bias:
        transition[state_dim - 1, -1] = dt * vu
    
    return transition


def generate_default_stop_component(state_dim: int, use_bias: bool = True) -> np.ndarray:
    """Generate stop component (zero velocity).
    
    Args:
        state_dim: State dimension.
        use_bias: Whether to use bias.
        
    Returns:
        Transition matrix.
    """
    full_dim = 2 * state_dim
    x_dim = full_dim + int(use_bias)
    
    transition = np.zeros((full_dim, x_dim), dtype=np.float64)
    
    # Position identity for first 2 coordinates
    transition[0, 0] = 1.0
    if full_dim > 1:
        transition[1, 1] = 1.0
    
    return transition


def create_velocity_component(x_current: np.ndarray, x_next: np.ndarray, dt: float = 1.0, use_unused_counter: bool = True) -> np.ndarray:
    """Create velocity component from current and next state.
    
    Args:
        x_current: Current state.
        x_next: Next state.
        dt: Time step.
        use_unused_counter: Whether to use unused counter.
        
    Returns:
        Transition matrix.
    """
    state_dim = x_current.shape[-1] // 2
    base_dynamics = generate_default_dynamics_component(state_dim, dt, True)
    
    vel = x_next[:state_dim] - x_current[:state_dim]
    prev_vel = x_current[state_dim:]
    vel_bias = vel - prev_vel
    
    new_component = base_dynamics.copy()
    new_component[:, -1] = np.concatenate([vel_bias, vel_bias])
    
    # Zero out unused
    if use_unused_counter:
        new_component[state_dim - 1, :] = 0.0
        new_component[2 * state_dim - 1, :] = 0.0
    
    return new_component


def create_bias_component(x: np.ndarray, use_unused_counter: bool) -> np.ndarray:
    """Create bias component from state.
    
    Args:
        x: State vector.
        use_unused_counter: Whether to use unused counter.
        
    Returns:
        Transition matrix.
    """
    state_dim_with_vel = x.shape[-1]
    new_component = np.concatenate([
        np.zeros((state_dim_with_vel, state_dim_with_vel)),
        x[..., None]
    ], axis=-1)
    
    # Zero out unused
    if use_unused_counter:
        new_component[state_dim_with_vel // 2 - 1, :] = 0.0
        new_component[state_dim_with_vel - 1, :] = 0.0
    
    return new_component

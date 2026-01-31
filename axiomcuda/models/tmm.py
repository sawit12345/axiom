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

import equinox as eqx

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import lax
from jaxtyping import Array

from .base import (
    ModelConfig,
    to_device,
    check_cuda_available,
    call_cpp_backend,
)


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
        transitions: Transition matrices (K_max, 2*state_dim, 2*state_dim+1).
        used_mask: Boolean mask for used components.
    """
    transitions: Array
    used_mask: Array


def generate_default_dynamics_component(state_dim, dt=1.0, use_bias=True):
    """
    Generate default dynamics component (constant velocity model).
    
    Args:
        state_dim: State dimension.
        dt: Time step.
        use_bias: Whether to use bias.
        
    Returns:
        Transition matrix of shape (2*state_dim, (2*state_dim)+1).
    """
    # encodes the assumption that velocity is constant in time
    velocity_coupling = jnp.eye(state_dim)
    base_transitions = block_diag(jnp.eye(state_dim), velocity_coupling)
    transition_matrix = jnp.pad(base_transitions, [(0, 0), (0, 1)])
    transition_matrix = transition_matrix.at[:state_dim, state_dim:-1].set(
        jnp.diag(dt * jnp.ones(state_dim))
    )

    if not use_bias:
        transition_matrix = transition_matrix[:, :-1]

    # Remove unused velocity and bias (assume used)
    transition_matrix = transition_matrix.at[2, :].set(0)
    transition_matrix = transition_matrix.at[5, :].set(0)

    return transition_matrix


def generate_default_become_unused_component(state_dim, dt=1, vu=1, use_bias=True):
    """Generate component for becoming unused.
    
    Args:
        state_dim: State dimension.
        dt: Time step.
        vu: Unused counter velocity.
        use_bias: Whether to use bias.
        
    Returns:
        Transition matrix.
    """
    transition_matrix = jnp.zeros((state_dim * 2, state_dim * 2 + int(use_bias)))
    transition_matrix = transition_matrix.at[: state_dim - 1, : state_dim - 1].set(
        jnp.eye(state_dim - 1)
    )
    # set the unused component to dt * vu
    transition_matrix = transition_matrix.at[state_dim - 1, -1].set(dt * vu)

    if not use_bias:
        transition_matrix = transition_matrix[:, :-1]

    return transition_matrix


def generate_default_keep_unused_component(state_dim, dt=1, vu=1, use_bias=True):
    """Generate component for keeping unused state.
    
    Args:
        state_dim: State dimension.
        dt: Time step.
        vu: Unused counter velocity.
        use_bias: Whether to use bias.
        
    Returns:
        Transition matrix.
    """
    transition_matrix = jnp.zeros((state_dim * 2, state_dim * 2 + int(use_bias)))
    transition_matrix = transition_matrix.at[:state_dim, :state_dim].set(
        jnp.eye(state_dim)
    )
    transition_matrix = transition_matrix.at[
        state_dim : 2 * state_dim - 1, state_dim : 2 * state_dim - 1
    ].set(jnp.eye(state_dim - 1))

    # set the unused component to dt * vu
    transition_matrix = transition_matrix.at[state_dim - 1, -1].set(dt * vu)

    if not use_bias:
        transition_matrix = transition_matrix[:, :-1]

    return transition_matrix


def generate_default_stop_component(state_dim, use_bias=True):
    """Generate stop component (zero velocity).
    
    Args:
        state_dim: State dimension.
        use_bias: Whether to use bias.
        
    Returns:
        Transition matrix.
    """
    transition_matrix = jnp.zeros((state_dim * 2, state_dim * 2 + int(use_bias)))
    transition_matrix = transition_matrix.at[:2, :2].set(jnp.eye(2))
    return transition_matrix


def create_velocity_component(x_current, x_next, dt=1.0, use_unused_counter=True):
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
    base_dynamics = generate_default_dynamics_component(
        state_dim=state_dim, dt=dt, use_bias=True
    )
    vel = x_next[:state_dim] - x_current[:state_dim]
    prev_vel = x_current[state_dim:]
    vel_bias = vel - prev_vel

    new_component = base_dynamics.at[:, -1].set(jnp.concatenate(2 * [vel_bias]))

    # No unused here
    if use_unused_counter:
        new_component = new_component.at[state_dim - 1, :].set(0)
        new_component = new_component.at[2 * state_dim - 1, :].set(0)

    return new_component


def create_bias_component(x, use_unused_counter):
    """Create bias component from state.
    
    Args:
        x: State vector.
        use_unused_counter: Whether to use unused counter.
        
    Returns:
        Transition matrix.
    """
    state_dim_with_vel = x.shape[-1]
    new_component = jnp.concatenate(
        [jnp.zeros((state_dim_with_vel, state_dim_with_vel)), x[..., None]], axis=-1
    )

    # No unused here
    if use_unused_counter:
        new_component = new_component.at[state_dim_with_vel // 2 - 1, :].set(0)
        new_component = new_component.at[state_dim_with_vel - 1, :].set(0)
    return new_component


def create_position_velocity_component(transitions, x_prev, x_curr, use_unused_counter):
    """Create position-velocity component.
    
    Args:
        transitions: Existing transitions.
        x_prev: Previous state.
        x_curr: Current state.
        use_unused_counter: Whether to use unused counter.
        
    Returns:
        Transition matrix.
    """
    num_coords = x_curr.shape[-1] // 2 - int(use_unused_counter)

    vel = x_curr[:num_coords] - x_prev[:num_coords]
    new_component = jnp.zeros_like(transitions[0])

    new_component = new_component.at[:num_coords, :num_coords].set(jnp.eye(num_coords))
    new_component = new_component.at[:num_coords, -1].set(vel)

    new_component = new_component.at[
        num_coords + int(use_unused_counter) : 2 * num_coords + int(use_unused_counter),
        -1,
    ].set(vel)
    return new_component


def create_position_bias_component(transitions, x_prev, x_curr):
    """Create position-bias component.
    
    Args:
        transitions: Existing transitions.
        x_prev: Previous state.
        x_curr: Current state.
        
    Returns:
        Transition matrix.
    """
    new_component = jnp.zeros_like(transitions[0])
    new_component = new_component.at[:2, -1].set(x_curr[:2])
    return new_component


def add_component(existing_transitions, new_transition, used_mask):
    """
    Add a new transition component to existing transitions.
    
    Args:
        existing_transitions: (K_max, ...) transition matrices.
        new_transition: New transition matrix to add.
        used_mask: (K_max,) boolean mask for used slots.
        
    Returns:
        Updated transitions with new component added.
    """
    # Boolean for the first row that is False in `used_mask`
    first_unused_mask = jnp.logical_and(~used_mask, jnp.cumsum(~used_mask) == 1)

    # Expand dims so that it can broadcast to the shape of new_transition
    mask_expanded = first_unused_mask[..., None, None]
    update_tensor = mask_expanded * new_transition

    return existing_transitions + update_tensor


def forward(transitions, x):
    """
    Apply transition model to state.
    
    Args:
        transitions: (K_max, 2*state_dim, (2*state_dim)+1) transition matrices.
        x: (2*state_dim,) state vector.
        
    Returns:
        Predicted next states (K_max, 2*state_dim).
    """
    return (transitions[..., :-1] * x).sum(-1) + transitions[..., -1]


def gaussian_loglike(y, mu, sigma_sqr=2):
    """
    Compute Gaussian log-likelihood.
    
    Args:
        y: Observed values.
        mu: Mean values.
        sigma_sqr: Variance.
        
    Returns:
        Log-likelihood values.
    """
    squared_error = (y - mu) ** 2
    return -0.5 * squared_error.sum(axis=-1) / sigma_sqr - 0.5 * y.shape[-1] * jnp.log(
        2 * jnp.pi * sigma_sqr
    )


def compute_logprobs(transitions, x_prev, x_curr, sigma_sqr=2, use_velocity=True):
    """
    Compute log probabilities for transitions.
    
    Args:
        transitions: Transition matrices.
        x_prev: Previous state.
        x_curr: Current state.
        sigma_sqr: Variance.
        use_velocity: Whether to use velocity in likelihood.
        
    Returns:
        Log probabilities for each component.
    """
    mu = forward(transitions, x_prev)
    if use_velocity:
        return gaussian_loglike(x_curr, mu, sigma_sqr)
    else:
        return gaussian_loglike(x_curr.at[:3], mu[:, :3], sigma_sqr)


def add_vel_or_bias_component(
    transitions,
    x_prev,
    x_curr,
    used_mask,
    pos_thr,
    use_unused_counter=False,
    dt=1.0,
    use_velocity=True,
    clip_value=5e-4,
):
    """Add velocity or bias component based on teleportation check.
    
    Args:
        transitions: Existing transitions.
        x_prev: Previous state.
        x_curr: Current state.
        used_mask: Used component mask.
        pos_thr: Position threshold for teleportation.
        use_unused_counter: Whether to use unused counter.
        dt: Time step.
        use_velocity: Whether to use velocity components.
        clip_value: Small value to clip.
        
    Returns:
        Updated transitions.
    """
    state_dim = x_curr.shape[-1] // 2 - int(use_unused_counter)

    # conditions for creating a bias component
    teleport = jnp.linalg.norm(x_curr[:state_dim] - x_prev[:state_dim]) > pos_thr

    if use_velocity:
        component_to_add = lax.cond(
            teleport,
            lambda x: create_bias_component(x, use_unused_counter),
            lambda x: create_velocity_component(x_prev, x, dt, use_unused_counter),
            x_curr,
        )
    else:
        component_to_add = lax.cond(
            teleport,
            lambda x: create_position_bias_component(
                transitions, x_prev, x, use_unused_counter
            ),
            lambda x: create_position_velocity_component(
                transitions, x_prev, x, use_unused_counter
            ),
            x_curr,
        )

    # filter out noisy values below clip_value
    component_to_add = jnp.where(
        jnp.abs(component_to_add) < clip_value, 0.0, component_to_add
    )

    return add_component(transitions, component_to_add, used_mask)


def single_logprob(transition, x_prev, x_curr, sigma_sqr):
    """
    Compute log probability for single transition.
    
    Args:
        transition: Single transition matrix.
        x_prev: Previous state.
        x_curr: Current state.
        sigma_sqr: Variance.
        
    Returns:
        Log probability.
    """
    mu = forward(transition, x_prev)
    return gaussian_loglike(x_curr, mu, sigma_sqr)


def compute_logprobs_masked_vmap(transitions, used_mask, x_prev, x_curr, sigma_sqr):
    """Compute log probabilities with masking using vmap.
    
    Args:
        transitions: Transition matrices.
        used_mask: Used component mask.
        x_prev: Previous state.
        x_curr: Current state.
        sigma_sqr: Variance.
        
    Returns:
        Log probabilities (masked components get -inf).
    """
    def compute_logprob_if_used(transition, mask, x_p, x_c):
        return lax.cond(
            mask,
            lambda t: single_logprob(t, x_p, x_c, sigma_sqr),
            lambda _: -jnp.inf,
            transition,
        )

    return jax.vmap(compute_logprob_if_used, in_axes=(0, 0, None, None))(
        transitions, used_mask, x_prev, x_curr
    )


def compute_logprobs_masked_fori(transitions, used_mask, x_prev, x_curr, sigma_sqr=2):
    """
    Compute log probabilities with masking using fori_loop.
    
    Args:
        transitions: (K_max, 2*state_dim, 2*state_dim + 1) transition matrices.
        used_mask: (K_max,) boolean mask.
        x_prev, x_curr: (2*state_dim,) state vectors.
        sigma_sqr: Variance.
        
    Returns:
        Log probabilities (K_max,).
    """
    K_max = transitions.shape[0]

    def body_fn(i, logps):
        logp_i = jax.lax.cond(
            used_mask[i],
            lambda t: single_logprob(t, x_prev, x_curr, sigma_sqr),
            lambda _: -jnp.inf,
            transitions[i],
        )
        return logps.at[i].set(logp_i)

    logps_init = jnp.full((K_max,), -jnp.inf)
    logps = jax.lax.fori_loop(0, K_max, body_fn, logps_init)
    return logps


def update_transitions(
    transitions,
    x_prev,
    x_curr,
    used_mask,
    sigma_sqr=2.0,
    logp_thr=-0.001,
    pos_thr=0.5,
    dt=1.0,
    use_unused_counter=False,
    use_velocity=True,
    clip_value=5e-4,
):
    """
    Update transitions with new component if needed.
    
    Args:
        transitions: Transition matrices.
        x_prev: Previous state.
        x_curr: Current state.
        used_mask: Used component mask.
        sigma_sqr: Variance.
        logp_thr: Log probability threshold for adding new component.
        pos_thr: Position threshold.
        dt: Time step.
        use_unused_counter: Whether to use unused counter.
        use_velocity: Whether to use velocity.
        clip_value: Small value to clip.
        
    Returns:
        Tuple of (transitions, used_mask, logprobs).
    """
    # shape (K_used,)
    logprobs_all = compute_logprobs(
        transitions, x_prev, x_curr, sigma_sqr, use_velocity
    )

    # 2) Replace the "unused" transitions' logprobs with -∞
    logprobs_used = jnp.where(used_mask, logprobs_all, -jnp.inf)

    # 3) The maximum over used transitions
    max_used = logprobs_used.max()

    # 4) If the max used is < threshold, we add new component
    def add_component_case(trans):
        return add_vel_or_bias_component(
            trans,
            x_prev,
            x_curr,
            used_mask,
            pos_thr,
            dt=dt,
            use_unused_counter=use_unused_counter,
            use_velocity=use_velocity,
            clip_value=clip_value,
        )

    def no_op_case(trans):
        return trans

    transitions = jax.lax.cond(
        max_used < logp_thr, add_component_case, no_op_case, transitions
    )

    used_mask = jnp.sum(jnp.abs(transitions), axis=(-1, -2)) > 0

    # recompute logprobs with new mask
    logprobs_all = compute_logprobs(
        transitions, x_prev, x_curr, sigma_sqr, use_velocity
    )
    logprobs_used = jnp.where(used_mask, logprobs_all, -jnp.inf)

    return transitions, used_mask, logprobs_used


def create_tmm(
    key,
    n_total_components,
    state_dim,
    dt=1.0,
    vu=0.1,
    use_bias=True,
    use_velocity=True,
    device=None,
    **kwargs,
):
    """
    Create a Transition Mixture Model.
    
    Args:
        key: Random key.
        n_total_components: Total number of components.
        state_dim: State dimension.
        dt: Time step.
        vu: Unused counter velocity.
        use_bias: Whether to use bias.
        use_velocity: Whether to use velocity components.
        device: JAX device to use.
        **kwargs: Additional arguments.
        
    Returns:
        TMM: Initialized Transition Mixture Model.
    """
    transitions = jnp.zeros(
        (n_total_components, 2 * state_dim, 2 * state_dim + int(use_bias))
    )
    used_mask = jnp.zeros(n_total_components, dtype=bool)

    if use_velocity:
        # In case we use velocity, we initialize some sensible default components
        transitions = transitions.at[0].set(
            generate_default_dynamics_component(state_dim, dt=dt, use_bias=use_bias)
        )
        transitions = transitions.at[1].set(
            generate_default_keep_unused_component(
                state_dim, dt=dt, vu=vu, use_bias=use_bias
            )
        )
        transitions = transitions.at[2].set(
            generate_default_become_unused_component(
                state_dim, dt=dt, vu=vu, use_bias=use_bias
            )
        )
        transitions = transitions.at[3].set(
            generate_default_stop_component(state_dim, use_bias)
        )

        used_mask = used_mask.at[:4].set(True)
    else:
        transitions = transitions.at[0].set(
            generate_default_keep_unused_component(
                state_dim, dt=dt, vu=vu, use_bias=use_bias
            )
        )
        transitions = transitions.at[1].set(
            generate_default_become_unused_component(
                state_dim, dt=dt, vu=vu, use_bias=use_bias
            )
        )
        used_mask = used_mask.at[:2].set(True)

    tmm = TMM(transitions=transitions, used_mask=used_mask)
    
    # Move to device if specified
    if device is not None:
        tmm = to_device(tmm, device)
    
    return tmm


def update_model(
    model,
    x_prev,
    x_curr,
    state_dim=2,
    sigma_sqr=2.0,
    logp_threshold=-0.001,
    position_threshold=0.5,
    dt=1.0,
    use_unused_counter=False,
    use_velocity=True,
    clip_value=5e-4,
    **kwargs,
):
    """
    Update TMM model with new transition.
    
    Args:
        model: TMM model.
        x_prev: Previous state.
        x_curr: Current state.
        state_dim: State dimension.
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
    logp_threshold_adjust = logp_threshold * (
        1.0 / 2 * sigma_sqr
    ) - 0.5 * 2 * state_dim * jnp.log(2 * jnp.pi * sigma_sqr)

    new_transitions, new_used_mask, logprobs = update_transitions(
        model.transitions,
        x_prev,
        x_curr,
        model.used_mask,
        sigma_sqr,
        logp_threshold_adjust,
        position_threshold,
        dt=dt,
        use_unused_counter=use_unused_counter,
        use_velocity=use_velocity,
        clip_value=clip_value,
    )

    model = eqx.tree_at(
        lambda x: (x.transitions, x.used_mask), model, (new_transitions, new_used_mask)
    )

    return model, logprobs
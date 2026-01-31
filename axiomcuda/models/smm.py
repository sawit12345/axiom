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

import jax
from jax import numpy as jnp, random as jr
from jax.nn import softmax, one_hot
from jax import lax
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
from typing import Tuple, NamedTuple, Union, Sequence
from dataclasses import dataclass, field

from axiom.vi.exponential import MultivariateNormal
from axiom.vi.transforms import LinearMatrixNormalGamma
from axiom.vi import ArrayDict, Distribution
from axiom.vi.utils import bdot

from axiom.vi.models.slot_mixture_model import (
    SlotMixtureModel,
    _inputs_to_delta,
    _m_step_keep_unused,
)

from .base import (
    ModelConfig,
    BaseModel,
    check_cuda_available,
    to_device,
    ensure_array,
    call_cpp_backend,
)


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
        default_factory=lambda: tuple([0.0, 0.0, 0.0, 0.0, 1])
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
        model: The underlying SlotMixtureModel.
        num_slots: Number of slots in the model.
        width: Image width.
        height: Image height.
        stats: Statistics for normalization (offset and stdevs).
    """
    model: SlotMixtureModel
    num_slots: int
    width: int
    height: int
    stats: dict


def _create_slot_mask(
    key: PRNGKeyArray,
    input_dim: int,
    slot_dim: int,
    use_bias: bool,
    num_components: int,
    transform_dims: list[int],
    probs: list[float] = None,
):
    """Create slot mask templates for initialization."""
    m = jnp.zeros((5, input_dim, slot_dim + int(use_bias)))
    # bias component
    m = m.at[0, :, -1:].set(1)
    # scale component
    m = m.at[1, :, :-1].set(1)
    # Bias for color, scale for position component
    m = m.at[2, transform_dims, :-1].set(1)
    non_transform_dims = jnp.setdiff1d(
        jnp.arange(input_dim), jnp.asarray(transform_dims)
    )
    m = m.at[2, non_transform_dims, -1].set(1)
    # full component
    m = m.at[3, :, :].set(1)

    # Identity for top left len(transform_dims)xlen(transform_dims) matrix, bias for the bottom right column
    # Construct identity matrix elements
    identity_indices = (
        jnp.array([4] * len(transform_dims)),
        transform_dims,
        transform_dims,
    )
    identity_values = jnp.ones(len(transform_dims))

    # Set identity elements
    m = m.at[identity_indices].set(identity_values)
    m = m.at[4, non_transform_dims, -1:].set(1)

    if probs is None:
        probs = jnp.ones(m.shape[0]) / m.shape[0]
    else:
        probs = jnp.asarray(probs)
        assert probs.shape[0] == m.shape[0], (
            f"Probs shape should equal {m.shape[0]} but is instead {probs.shape[0]}"
        )

    key, subkey = jr.split(key)
    p = jr.choice(subkey, jnp.arange(m.shape[0]), p=probs, shape=(num_components,))

    return p, m[p]


def _create_lmg_params(
    key: PRNGKeyArray,
    input_dim: int,
    slot_dim: int,
    num_slots: int,
    use_bias: bool,
    ns_a: float,
    ns_b: float,
    dof_offset: float,
    mask_prob: list[float],
    scale: list[float],
    transform_inv_v_scale: float,
    bias_inv_v_scale: float,
) -> Tuple[ArrayDict, ArrayDict]:
    """
    Creates the parameters for the Linear Matrix Gamma conjugate prior

    Returns:
        Tuple[ArrayDict, ArrayDict]: The prior and initial posterior parameters.
    """

    # TODO
    if input_dim == 5:
        transform_dims = [0, 1]
    else:
        transform_dims = [0, 1, 2, 3]

    if mask_prob is not None:
        key, subkey = jr.split(key)
        _, slot_mask = _create_slot_mask(
            subkey,
            input_dim,
            slot_dim,
            use_bias,
            num_slots,
            transform_dims=transform_dims,
            probs=jnp.asarray(mask_prob).flatten(),
        )
    else:
        slot_mask = 1

    batch_shape = (1, num_slots)
    use_bias = int(use_bias)
    mu_pri = jnp.zeros((input_dim, slot_dim + use_bias))
    prior_mu = jnp.broadcast_to(mu_pri, batch_shape + (input_dim, slot_dim + use_bias))

    inv_v_scale = jnp.ones(slot_dim + use_bias)
    inv_v_scale = inv_v_scale.at[:slot_dim].set(transform_inv_v_scale)
    if use_bias:
        inv_v_scale = inv_v_scale.at[-1].set(bias_inv_v_scale)

    prior_inv_v = jnp.broadcast_to(
        jnp.eye(slot_dim + use_bias) * inv_v_scale,
        batch_shape + (slot_dim + use_bias, slot_dim + use_bias),
    )

    # shape should be (batch_shape,) + (1, 1)
    prior_a = jnp.full(batch_shape + (input_dim, 1), 1.0 + dof_offset)

    # shape should be (batch_shape,) + (y_dim, y_dim).
    if isinstance(scale, tuple):
        scale = jnp.asarray(scale)
        assert scale.shape[0] == input_dim
    else:
        scale = scale * jnp.ones((input_dim))

    prior_b = jnp.broadcast_to(
        (scale**2).reshape((input_dim, 1)), batch_shape + (input_dim, 1)
    )

    # if there is a strong prior on the slot mask template, this should be encoded in
    # the prior
    if transform_inv_v_scale > 1:
        prior_mu = prior_mu + slot_mask

    prior_params = ArrayDict(mu=prior_mu, inv_v=prior_inv_v, a=prior_a, b=prior_b)

    mu = prior_mu
    key, subkey = jr.split(key)
    mu_noise = jr.uniform(subkey, shape=prior_mu.shape, minval=1, maxval=2)

    mu = mu.at[..., :-1].set((mu[..., :-1] + ns_a * mu_noise[..., :-1]))
    mu = mu.at[..., -1].set((mu[..., -1] + ns_b * mu_noise[..., -1]))

    key, subkey = jr.split(key)
    a = prior_a * jr.uniform(subkey, shape=prior_a.shape, minval=1, maxval=3)

    inv_v = jnp.where(prior_params.inv_v > 0, 1.0, 0.0)
    initial_posterior_params = ArrayDict(mu=mu * slot_mask, inv_v=inv_v, a=a, b=prior_b)

    return prior_params, initial_posterior_params


def create_smm(
    key: PRNGKeyArray,
    width: int = 160,
    height: int = 210,
    input_dim: int = 5,
    slot_dim: int = 2,
    num_slots: int = 32,
    use_bias: bool = True,
    ns_a: float = 1,
    ns_b: float = 1,
    dof_offset: float = 10,
    mask_prob: list[float] = [0.0, 0.0, 0.0, 0.0, 1],
    scale: list[float] = [0.075, 0.075, 0.75, 0.75, 0.75],
    transform_inv_v_scale: float = 100,
    bias_inv_v_scale: float = 0.001,
    device: jax.Device = None,
    **kwargs,
):
    """Create a Slot Mixture Model.
    
    Args:
        key: Random key for initialization.
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
        device: JAX device to use.
        **kwargs: Additional arguments.
        
    Returns:
        SMM: Initialized Slot Mixture Model.
    """
    key, subkey = jr.split(key)

    prior_params, params = _create_lmg_params(
        key=subkey,
        input_dim=input_dim,
        slot_dim=slot_dim,
        num_slots=num_slots,
        use_bias=use_bias,
        ns_a=ns_a,
        ns_b=ns_b,
        dof_offset=dof_offset,
        mask_prob=mask_prob,
        scale=scale,
        transform_inv_v_scale=transform_inv_v_scale,
        bias_inv_v_scale=bias_inv_v_scale,
    )

    likelihood = LinearMatrixNormalGamma(
        params=params, prior_params=prior_params, use_bias=use_bias
    )

    smm = SlotMixtureModel(
        num_slots,
        input_dim=input_dim,
        slot_dim=slot_dim,
        multi_modality=False,
        likelihood=likelihood,
    )

    smm_result = SMM(
        model=smm,
        num_slots=num_slots,
        width=width,
        height=height,
        stats={
            "offset": jnp.array([width / 2, height / 2, 128, 128, 128]),
            "stdevs": jnp.array([width / 2, height / 2, 128, 128, 128]),
        },
    )
    
    # Move to device if specified
    if device is not None:
        smm_result = to_device(smm_result, device)
    
    # Try to create C++ backend if available
    if check_cuda_available():
        try:
            call_cpp_backend("createSMM", width, height, input_dim, slot_dim, num_slots, use_bias)
        except (RuntimeError, AttributeError):
            pass
    
    return smm_result


def add_position_encoding(img: Array) -> Array:
    """Add position encoding (x, y coordinates) to image.
    
    Args:
        img: Image array of shape (H, W, C).
        
    Returns:
        Array with position encoding added, shape (H*W, C+2).
    """
    u, v = jnp.meshgrid(jnp.arange(img.shape[1]), jnp.arange(img.shape[0]))

    data = jnp.concatenate(
        [
            (u.reshape(-1, 1)),
            (v.reshape(-1, 1)),
            img.reshape(-1, img.shape[-1]),
        ],
        axis=1,
    )
    return data


def format_single_frame(single_obs: Array, offset: Array, stdevs: Array) -> Array:
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


def _compute_qx_given_qz(linear_likelihood, prior_x, inputs, qz):
    """Compute q(x|z) given q(z) and inputs.
    
    Args:
        linear_likelihood: Linear likelihood module.
        prior_x: Prior over x.
        inputs: Input observations.
        qz: Assignment probabilities.
        
    Returns:
        Updated q(x) distribution.
    """
    bkwd_message = linear_likelihood.variational_backward(inputs)

    inv_sigma_mu = (bkwd_message.inv_sigma_mu * qz[..., None, None]).sum(-4)
    inv_sigma = (bkwd_message.inv_sigma * qz[..., None, None]).sum(-4) + 1e-8

    qx = (
        MultivariateNormal(
            nat_params=ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma),
            event_shape=prior_x.event_shape,
            event_dim=2,
        )
        * prior_x
    )
    return qx.expand_batch_shape(-2)


def _compute_qz_given_qx(linear_likelihood, prior_z_logmean, inputs, qx):
    """Compute q(z|x) given q(x) and inputs.
    
    Args:
        linear_likelihood: Linear likelihood module.
        prior_z_logmean: Prior log mean over z.
        inputs: Input observations.
        qx: Current q(x) distribution.
        
    Returns:
        Tuple of (qz, ell) - assignment probabilities and expected log-likelihood.
    """
    ell = linear_likelihood.average_energy((qx, inputs))

    return softmax(ell + prior_z_logmean, axis=-1), ell


def _compute_qx_and_qz_given_qx(
    linear_likelihood, prior_z_logmean, prior_x, inputs, qx
):
    """Compute both q(z|x) and updated q(x|z).
    
    Args:
        linear_likelihood: Linear likelihood module.
        prior_z_logmean: Prior log mean over z.
        prior_x: Prior over x.
        inputs: Input observations.
        qx: Current q(x) distribution.
        
    Returns:
        Tuple of (qx, qz, ell) - updated q(x), q(z), and expected log-likelihood.
    """
    qz, ell = _compute_qz_given_qx(linear_likelihood, prior_z_logmean, inputs, qx)

    qx = _compute_qx_given_qz(linear_likelihood, prior_x, inputs, qz)

    return qx, qz, ell


def _update_optimized(
    smm: SMM, inputs, qz, learning_rate=1.0, beta=0.0, grow_mask=None
):
    """Optimized update step with E-M iterations.
    
    Args:
        smm: Slot Mixture Model.
        inputs: Input observations.
        qz: Initial assignment probabilities.
        learning_rate: Learning rate for M-step.
        beta: Momentum parameter.
        grow_mask: Optional mask for growing components.
        
    Returns:
        Tuple of (smm, qx, qz, ell_max, used_component_idx).
    """
    linear_likelihood, prior_z, prior_x = (
        smm.model.likelihood,
        smm.model.pi,
        smm.model.px,
    )

    # first update qx given qz
    qx = _compute_qx_given_qz(linear_likelihood, prior_x, inputs, qz)

    # do an M step given the updated qx and input qz
    linear_likelihood, prior_z = _m_step_keep_unused(
        inputs,
        linear_likelihood,
        prior_z,
        qx,
        qz,
        lr=learning_rate,
        beta=beta,
        grow_mask=grow_mask,
    )

    pi_log_mean = prior_z.log_mean()

    # do e-m step
    qx, qz, ell = _compute_qx_and_qz_given_qx(
        linear_likelihood, jnp.zeros_like(pi_log_mean), prior_x, inputs, qx
    )

    linear_likelihood, prior_z = _m_step_keep_unused(
        inputs, linear_likelihood, prior_z, qx, qz, lr=learning_rate, beta=beta
    )

    # do another e-m step
    pi_log_mean = prior_z.log_mean()

    qx, qz, ell = _compute_qx_and_qz_given_qx(
        linear_likelihood, pi_log_mean, prior_x, inputs, qx
    )

    linear_likelihood, prior_z = _m_step_keep_unused(
        inputs, linear_likelihood, prior_z, qx, qz, lr=learning_rate, beta=beta
    )

    # Get the hard assignments (shape: (1, H*W))
    assignments = qz.argmax(-1)

    # Compute the histogram of assignments over the K components (shape: (K,))
    counts = jnp.bincount(
        assignments.ravel(), minlength=qz.shape[-1], length=qz.shape[-1]
    )

    # Convert to boolean (1 if component is used, 0 otherwise)
    used_component_idx = counts != 0

    smm = eqx.tree_at(
        lambda x: (x.model.likelihood, x.model.pi),
        smm,
        (linear_likelihood, prior_z),
    )

    # (1, H*W) array of integers, which component got selected for each pixel
    ell_max = ell.max(axis=-1)

    return smm, qx, qz, ell_max, used_component_idx


def initialize_smm_model(smm: SMM, init_inputs: Union[Array, Distribution]):
    """Initialize SMM model with initial inputs.
    
    Args:
        smm: Slot Mixture Model.
        init_inputs: Initial input observations.
        
    Returns:
        Tuple of (smm_model, qx, qz) - initialized model and initial posteriors.
    """
    N_tokens = init_inputs.shape[1]
    if isinstance(init_inputs, Array):
        # convert to Delta distributions as input for the model
        init_inputs = _inputs_to_delta(init_inputs)
    else:
        init_inputs = init_inputs.expand_batch_shape(-1)

    # force assign all data to the first slot
    qz = one_hot(jnp.zeros(N_tokens), smm.num_slots)[None, ...]

    smm_model, qx, qz, _, _ = _update_optimized(
        smm, init_inputs, qz, learning_rate=1.0, beta=0.0
    )

    return smm_model, qx, qz


def create_mvn(inv_sigma_mu, inv_sigma):
    """
    Helper function for quickly creating an optimus.exponential.MultivariateNormal instance
    with given natural parameters inv_sigma_mu and inv_sigma
    """
    return MultivariateNormal(
        nat_params=ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma),
        event_shape=(inv_sigma_mu.shape[-2], 1),
        event_dim=2,
    )


def _combine_mvns(model_0, model_1, select_from_1):
    """Combine two MVN distributions based on selection mask."""
    def fn(x_0, x_1):
        return jnp.where(select_from_1, x_1, x_0)

    model_0.nat_params = jtu.tree_map(
        lambda x, y: fn(x, y), model_0.nat_params, model_1.nat_params
    )
    return model_0


def _increase_position_uncertainty(qx):
    """
    Sets the covariance of the position dimensions to be very large in order
    to have more emphasis on the color features of qx.
    NOTE: not properly tested

    """
    mu, inv_si = qx.mu, qx.inv_sigma.at[:, 0, 0].set(3).at[:, 1, 1].set(3)
    return MultivariateNormal(
        nat_params=ArrayDict(inv_sigma_mu=bdot(inv_si, mu), inv_sigma=inv_si)
    )


def create_e_step_fn(smm: SMM, inputs):
    """
    Given a fixed SMM model and some observations `input`, returns a function that can
    be passed into a lax.scan to perform a sequence of E-steps on the model to update qx
    and qz
    
    Args:
        smm: Slot Mixture Model.
        inputs: Input observations.
        
    Returns:
        Function for E-step scanning.
    """

    linear_likelihood, prior_z_logmean, prior_x = (
        smm.model.likelihood,
        smm.model.pi.log_mean(),
        smm.model.px,
    )

    def e_step_scan_fn(carry, _):
        qx_prev, _, _ = carry

        qx, qz, ell = _compute_qx_and_qz_given_qx(
            linear_likelihood, prior_z_logmean, prior_x, inputs, qx_prev
        )
        ell_max = ell.max(axis=-1)

        select_from_qx = (qz[0].sum(0) > 0).reshape(1, 1, -1, 1, 1)
        qx = _combine_mvns(qx_prev, qx, select_from_qx)

        return (qx, qz, ell_max), None

    return e_step_scan_fn


def infer_and_update(
    key: PRNGKeyArray,
    smm: SMM,
    inputs: Union[Array, Distribution],
    qx_prev: MultivariateNormal,
    num_slots: int,
    num_e_steps=2,
    eloglike_threshold: float = 5.0,
    max_grow_steps: int = 10,
    learning_rate: float = 1.0,
    beta: float = 0.0,
    device: jax.Device = None,
    **kwargs,
):
    """Run inference and update on SMM model.
    
    Args:
        key: Random key.
        smm: Slot Mixture Model.
        inputs: Input observations.
        qx_prev: Previous q(x) distribution.
        num_slots: Number of slots.
        num_e_steps: Number of E-steps.
        eloglike_threshold: Threshold for growing components.
        max_grow_steps: Maximum growth steps.
        learning_rate: Learning rate for updates.
        beta: Momentum parameter.
        device: JAX device to use.
        **kwargs: Additional arguments.
        
    Returns:
        Tuple of updated (smm, py, qx, qz, used, ell_max).
    """
    if isinstance(inputs, Array):
        N_tokens = inputs.shape[0]
        # rescale and normalize
        # add a leading dimension because format_frames expects at least one sample dimension
        inputs = _inputs_to_delta(inputs[None, ...])
    else:
        N_tokens = inputs.shape[1]
        inputs = inputs.expand_batch_shape(-1)

    ### Do E steps to update qx and qz, and to compute the maximum eloglike across components
    e_step_scan_fn = create_e_step_fn(smm, inputs)

    init = (
        qx_prev,
        jnp.empty((1, N_tokens, num_slots)),
        jnp.empty((1, N_tokens)),
    )
    (qx, qz, ell_max), _ = lax.scan(e_step_scan_fn, init, jnp.arange(num_e_steps))

    # Get the hard assignments (shape: (1, H*W))
    assignments = qz.argmax(-1)

    # Compute the histogram of assignments over the K components (shape: (K,))
    counts = jnp.bincount(assignments.ravel(), minlength=num_slots, length=num_slots)

    # Convert to boolean (1 if component is used, 0 otherwise)
    used_component_idx = counts != 0

    def grow_body(carry, t):
        """
        carry = (smm, qx, qz, ell_max, ell, used, tries, done)
        """
        done_last = carry[-1]

        # ---------- Utility functions -------------
        def do_nothing(carry_in):
            # just return carry_in as is, or you might prefer setting done=True
            smm_i, qx_i, qz_i, ell_max_i, used_i, tries_i, done_i = carry_in
            # keep the rest, set done=True
            return (smm_i, qx_i, qz_i, ell_max_i, used_i, tries_i, True)

        def do_grow(carry_in):
            ssmm_i, qx_i, qz_i, ell_max_i, used_i, tries_i, done_i = carry_in
            new_tries = tries_i + 1
            cond_ok = (ell_max_i.min() < eloglike_threshold) & (~done_i)

            def skip_grow(c2):
                smm_j, qx_j, qz_j, ell_max_j, used_j, tries_j, done_j = c2
                return (smm_j, qx_j, qz_j, ell_max_j, used_j, new_tries, True)

            def really_grow(c2):
                smm_j, qx_j, qz_j, ell_max_j, used_j, tries_j, done_j = c2

                slot_candidates = jnp.where(~used_j, jnp.arange(num_slots), num_slots)
                slot_to_use = slot_candidates.min()

                def no_unused(c3):
                    # no unused => we are done
                    smm_k, qx_k, qz_k, ell_max_k, used_k, tries_k, done_k = c3
                    return (
                        smm_k,
                        qx_k,
                        qz_k,
                        ell_max_k,
                        used_k,
                        new_tries,
                        True,
                    )

                def force_assign(c3):
                    smm_k, qx_k, qz_k, ell_max_k, used_k, tries_k, done_k = c3
                    idx = jnp.argmin(ell_max_k)

                    grow_mask = jax.nn.one_hot(slot_to_use, num_slots)
                    new_qz = qz_k.at[0, idx].set(grow_mask)

                    new_smm, new_qx, newest_qz, new_ell_max, new_used = (
                        _update_optimized(
                            smm_k,
                            inputs,
                            new_qz,
                            learning_rate=learning_rate,
                            beta=beta,
                            grow_mask=grow_mask,
                        )
                    )
                    return (
                        new_smm,
                        new_qx,
                        newest_qz,
                        new_ell_max,
                        new_used,
                        new_tries,
                        False,
                    )

                return lax.cond(used_j.sum() == num_slots, no_unused, force_assign, c2)

            return lax.cond(cond_ok, really_grow, skip_grow, carry_in)

        new_carry = lax.cond(done_last, do_nothing, do_grow, carry)
        return new_carry, None

    def grow_loop(smm, qx, qz, ell_max, used, max_grow_steps):
        init_carry = (smm, qx, qz, ell_max, used, 0, False)
        final_carry, _ = jax.lax.scan(grow_body, init_carry, jnp.arange(max_grow_steps))
        return final_carry

    (
        smm_updated,
        qx_updated,
        qz_updated,
        ell_max_updated,
        used_updated,
        tries,
        done,
    ) = grow_loop(smm, qx, qz, ell_max, used_component_idx, max_grow_steps)

    # return the variational forward mean as object descriptors
    py = smm_updated.model.likelihood.variational_forward(qx_updated)

    return (
        smm_updated,
        py,
        qx_updated,
        qz_updated,
        used_updated,
        ell_max_updated,
    )


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


def create_hierarch_smm(key, layer_configs: Sequence[SMMConfig]) -> HierarchicalSMM:
    """Create a hierarchical SMM.
    
    Args:
        key: Random key.
        layer_configs: Configuration for each layer.
        
    Returns:
        HierarchicalSMM: Multi-layer SMM model.
    """
    models = []
    num_slots = []
    for config in layer_configs:
        model = create_smm(key, **config.__dict__)
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
    key: PRNGKeyArray,
    model: HierarchicalSMM,
    init_inputs: Union[Array, MultivariateNormal],
    layer_configs: Sequence[SMMConfig],
) -> Tuple[HierarchicalSMM, list, list, list, list, list]:
    """Initialize hierarchical SMM.
    
    Args:
        key: Random key.
        model: Hierarchical SMM model.
        init_inputs: Initial inputs.
        layer_configs: Configuration for each layer.
        
    Returns:
        Tuple of updated model and intermediate results.
    """
    from jax.numpy import expand_dims as expand
    from dataclasses import asdict

    (
        models_updated,
        py_updated,
        qx_updated,
        qz_updated,
        used_updated,
        ell_max_updated,
    ) = ([], [], [], [], [], [])
        
    next_layer_input_dims = [2] * (len(layer_configs))

    for i, (m, layer_config) in enumerate(zip(model.models, layer_configs)):

        if i > 0:
            init_inputs = augment_state_with_velocity(init_inputs, prev_qx=None, velocity_precision=0.1)

        m, qx, _ = initialize_smm_model(
            m,
            init_inputs=(
                expand(init_inputs, 0)
                if isinstance(init_inputs, Array)
                else init_inputs
            ),
        )
        # Run a first step of inference and growing, in order to get an initial input for the higher layer
        key, subkey = jr.split(key)

        (m_updated, py, qx, qz, used, ell_max) = infer_and_update(
            subkey,
            m,
            init_inputs,
            qx_prev=qx,
            **asdict(layer_config),
        )
        models_updated.append(m_updated)
        py_updated.append(py)
        qx_updated.append(qx)
        qz_updated.append(qz)
        used_updated.append(used)
        ell_max_updated.append(ell_max)

        # the inputs for the (i+1)-th  layer are the first (0, 1, ..., input_dim[i+1]) dimensions of the lower layer's decoded outputs
        inv_sigma_mu_i = py.nat_params.inv_sigma_mu.squeeze(0)[
            :, :, : next_layer_input_dims[i], :
        ]
        inv_sigma_i = py.nat_params.inv_sigma.squeeze(0)[
            :, :, : next_layer_input_dims[i], : next_layer_input_dims[i]
        ]
        init_inputs = MultivariateNormal(
            nat_params=ArrayDict(inv_sigma_mu=inv_sigma_mu_i, inv_sigma=inv_sigma_i),
            event_shape=(layer_config.slot_dim, 1),
            event_dim=2,
        )

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


def augment_state_with_velocity(qx: MultivariateNormal, prev_qx: MultivariateNormal = None, velocity_precision=1.0) -> MultivariateNormal:
    """
    Augment the latent state by computing a velocity term from the difference between the
    current state (qx.mu) and a previous state (prev_qx.mu), then concatenating the velocity
    to the original state. If prev_qx is None, velocity is zero.

    The latent state is assumed to be along axis 2 of qx.mu (e.g., shape (batch, components, latent_dim, 1)).
    The corresponding covariance has shape (batch, components, latent_dim, latent_dim).

    Args:
        qx: Current MultivariateNormal object with a `mu` property representing the current state.
        prev_qx: Previous MultivariateNormal object with a `mu` property representing the prior state.
        velocity_precision: Precision weight for velocity component.

    Returns:
        A new MultivariateNormal object whose state has been augmented to include the velocity component.
    """
    from jax.scipy.linalg import block_diag
    from jax import vmap

    # Determine the latent state dimension (e.g., 4 in your printed shapes).
    latent_dim = qx.nat_params.inv_sigma.shape[-1]

    # Compute velocity as the difference between current and previous means,
    # or use zeros if no previous state is provided.
    if prev_qx is None:
        velocity = jnp.zeros_like(qx.mu)
    else:
        # In your case, prev_qx.mu has an extra dimension:
        # qx.mu shape:       (1, 16, 2, 1)
        # prev_qx.mu shape:   (1,  1, 16, 4, 1)
        # We assume that the extra axis (axis=1) in prev_qx is a singleton that needs to be removed.
        prev_mu = jnp.squeeze(prev_qx.mu, axis=1)  # shape: (1, 16, 4, 1)

        prev_pos_mu = prev_mu[:, :, :2, :]  # shape: (1, 16, 2, 1)
        velocity = qx.mu - prev_pos_mu  # shape: (1, 16, 2, 1)

    # Augment the natural parameter for the mean:
    # Pad inv_sigma_mu with zeros in the new latent state dimensions.
    augmented_inv_sigma_mu = jnp.concatenate(
        [qx.nat_params.inv_sigma_mu, velocity_precision*velocity], axis=2
    )

    # For the inverse covariance (inv_sigma), create a block-diagonal matrix.
    orig_inv_sigma = (
        qx.nat_params.inv_sigma
    )  # shape: (batch, components, latent_dim, latent_dim)
    batch_shape = orig_inv_sigma.shape[:-2]
    dim_dim_shape = orig_inv_sigma.shape[-2:]

    # Create the identity matrix for the velocity block.
    velocity_inv_sigma = jnp.broadcast_to(velocity_precision*jnp.eye(latent_dim), orig_inv_sigma.shape).reshape((-1,) + dim_dim_shape)

    augmented_inv_sigma = vmap(block_diag)(orig_inv_sigma.reshape((-1,) + dim_dim_shape), velocity_inv_sigma)
    augmented_dim_dim_shape = augmented_inv_sigma.shape[-2:]
    augmented_inv_sigma = augmented_inv_sigma.reshape(batch_shape + augmented_dim_dim_shape)

    # Construct new natural parameters.
    new_nat_params = ArrayDict(
        inv_sigma_mu=augmented_inv_sigma_mu, inv_sigma=augmented_inv_sigma
    )
    event_shape = new_nat_params.inv_sigma_mu.shape[-2:]
    return MultivariateNormal(nat_params=new_nat_params, event_shape=event_shape)

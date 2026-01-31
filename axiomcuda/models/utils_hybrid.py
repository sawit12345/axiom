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

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softmax
import equinox as eqx

from axiom.vi import ArrayDict
from axiom.vi.conjugate import Multinomial, MultivariateNormal
from axiom.vi.models.hybrid_mixture_model import HybridMixture

from .base import (
    ModelConfig,
    to_device,
    check_cuda_available,
)


def create_mm(
    key,
    num_components,
    continuous_dim,
    discrete_dims,
    discrete_alphas=None,
    cont_scale=1.0,
    color_precision_scale=None,
    opt=None,
    device=None,
):
    """Create a hybrid mixture model.
    
    Args:
        key: Random key.
        num_components: Number of mixture components.
        continuous_dim: Dimension of continuous features.
        discrete_dims: List of dimensions for discrete features.
        discrete_alphas: Prior alphas for discrete features.
        cont_scale: Scale for continuous features.
        color_precision_scale: Scale for color precision.
        opt: Optimization options.
        device: JAX device to use.
        
    Returns:
        HybridMixture: Initialized hybrid mixture model.
    """
    if opt is None:
        opt = {"lr": 1, "beta": 1}

    component_shape = (num_components,)

    prior_params = MultivariateNormal.init_default_params(
        component_shape, (continuous_dim, 1), cont_scale, dof_offset=continuous_dim
    )
    prior_params = eqx.tree_at(
        lambda x: x.kappa, prior_params, prior_params.kappa * 1e-4
    )

    if color_precision_scale is not None:
        spatial_dim = 2

        u = prior_params.u
        for i in range(spatial_dim, continuous_dim):
            u = u.at[..., i, i].set(u[..., i, i] * color_precision_scale)

        prior_params = eqx.tree_at(lambda x: x.u, prior_params, u)

    key, subkey = jr.split(key)
    params = eqx.tree_at(
        lambda x: x.mean,
        prior_params,
        prior_params.mean + jr.normal(subkey, shape=prior_params.mean.shape),
    )

    key, subkey = jr.split(key)
    continuous_likelihood = MultivariateNormal(
        batch_shape=component_shape,
        event_shape=(continuous_dim, 1),
        event_dim=2,
        init_key=subkey,
        params=params,
        prior_params=prior_params,
    )

    if discrete_alphas is None:
        discrete_alphas = [1e-4] * len(discrete_dims)

    discrete_likelihoods = []
    for discrete_alpha, discrete_dim in zip(discrete_alphas, discrete_dims):
        key, subkey = jr.split(key)
        alpha = discrete_alpha * jnp.ones(component_shape + (discrete_dim, 1))
        prior_params = ArrayDict(alpha=alpha)

        # Eye prior
        d = min(num_components, discrete_dim)
        eye = jnp.eye(d)
        nr, nc = int(jnp.ceil(num_components / d)), int(jnp.ceil(discrete_dim / d))
        eye = jnp.tile(eye, (nr, nc))
        eye = eye[:num_components, :discrete_dim, None]

        key, subkey = jr.split(key)
        idcs = jr.permutation(subkey, eye.shape[0], independent=True)

        alpha = alpha + eye[idcs] * 10

        alpha = alpha * (jr.uniform(subkey, shape=alpha.shape, minval=0.1, maxval=0.9))
        discrete_likelihoods.append(
            Multinomial(
                batch_shape=component_shape,
                event_shape=(discrete_dim, 1),
                initial_count=1,
                init_key=subkey,
                params=ArrayDict(alpha=alpha),
                prior_params=prior_params,
            )
        )

    key, subkey = jr.split(key)
    alpha = 0.1 * jnp.ones(component_shape)
    prior = Multinomial(
        batch_shape=(),
        event_shape=component_shape,
        prior_params=ArrayDict(alpha=alpha),
        params=ArrayDict(alpha=alpha),
    )

    mm = HybridMixture(
        discrete_likelihoods,
        continuous_likelihood,
        prior,
        pi_opts=opt,
        likelihood_opts=opt,
    )
    
    # Move to device if specified
    if device is not None:
        mm = to_device(mm, device)
    
    return mm


def train_step_fn(model: HybridMixture, mask, c_sample, d_sample, logp_thr=-0.1):
    """Training step for hybrid mixture model.
    
    Args:
        model: Hybrid mixture model.
        mask: Used component mask.
        c_sample: Continuous sample.
        d_sample: Discrete samples.
        logp_thr: Log probability threshold.
        
    Returns:
        Tuple of (updated_model, updated_mask, qz).
    """
    qz, c_ell, d_ell = model._e_step(c_sample, d_sample)
    elogp = c_ell + d_ell

    # Mask out the contributions of unused components
    elogp = elogp * mask[None] + (1 - mask[None]) * (-1e10)

    qz = softmax(elogp, model.mix_dims)

    def true_fn(qz, mask):
        return qz, mask

    def false_fn(qz, mask):
        # Find a component that is unused
        idx = jnp.argmax(mask == 0)
        # Assign the the data point to this component
        qz_new = (qz * 0.0).at[:, idx].set(1.0)
        # Update the mask to reflect that this component is now used
        mask_new = mask.at[idx].set(1.0)
        return qz_new, mask_new

    qz, mask = jax.lax.cond(elogp.max() > logp_thr, true_fn, false_fn, qz, mask)

    model._m_step_keep_unused(c_sample, d_sample, qz)

    mask = (model.prior.alpha > model.prior.prior_alpha) * 1.0

    return model, mask, qz


# Re-export HybridMixture and create Mixture alias
Mixture = HybridMixture
HybridMixture = HybridMixture
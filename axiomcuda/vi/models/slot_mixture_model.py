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

from jax import numpy as jnp, random as jr
from jax.nn import softmax
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray
from typing import Tuple, Union, List, Dict

from axiomcuda.vi.exponential import MultivariateNormal
from axiomcuda.vi.conjugate import Multinomial as CatDirichlet
from axiomcuda.vi.transforms import LinearMatrixNormalGamma
from axiomcuda.vi.distribution import Delta
from axiomcuda.vi import ArrayDict
from axiomcuda.vi.models import Model


class SlotMixtureModel(Model):
    """Latent attention model with slots - CUDA accelerated."""

    pytree_data_fields = ("likelihood", "pi", "px")
    pytree_aux_fields = ("num_slots", "slot_dim")

    def __init__(
        self,
        num_slots: int,
        input_dim: int,
        slot_dim: int,
        multi_modality: bool,
        prior_type: str = "mng",
        linear_hyperparams: Dict = None,
        likelihood=None,
    ):
        if likelihood is None:
            if prior_type == "mng":
                likelihood_cls = LinearMatrixNormalGamma
            else:
                raise NotImplementedError()

            if linear_hyperparams is None:
                linear_hyperparams = {
                    "use_bias": True,
                    "scale": 1.0,
                    "dof_offset": 1.0,
                    "inv_v_scale": 1e-1,
                }

            likelihood = likelihood_cls(
                batch_shape=(1, num_slots),
                event_shape=(input_dim, slot_dim + linear_hyperparams["use_bias"]),
                **linear_hyperparams,
            )

        self.likelihood = likelihood
        self.pi = CatDirichlet(
            params=ArrayDict(alpha=jnp.ones(num_slots)),
            batch_shape=(),
            event_shape=(num_slots,),
        )
        self.px = MultivariateNormal(
            batch_shape=(num_slots,), event_shape=(slot_dim, 1), event_dim=2
        )

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.multi_modality = multi_modality

        batch_shape = (1, num_slots)
        event_shape = (input_dim, slot_dim)
        super().__init__(2, batch_shape, event_shape)

    def initialize_slot_latents(
        self,
        key: PRNGKeyArray,
        sample_shape: Union[Tuple, int] = 1,
        init_mu_scale: float = 1.0,
        init_inv_sigma_scale: float = 100.0,
    ):
        """Initialize variational distribution over latents."""
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        initial_mu = init_mu_scale * jr.normal(key, shape=len(sample_shape) * (1,) + (self.num_slots, self.slot_dim, 1))
        initial_inv_sigma_mu = jnp.broadcast_to(init_inv_sigma_scale * initial_mu, sample_shape + (self.num_slots, self.slot_dim, 1))
        initial_inv_sigma = jnp.tile(init_inv_sigma_scale * jnp.eye(self.slot_dim), sample_shape + (self.num_slots, 1, 1))

        nat_params = ArrayDict(inv_sigma_mu=initial_inv_sigma_mu, inv_sigma=initial_inv_sigma)
        return MultivariateNormal(
            nat_params=nat_params,
            batch_shape=sample_shape + (self.num_slots,),
            event_shape=(self.slot_dim, 1),
            event_dim=2,
            scale=0.01,
        )

    def __call__(
        self,
        key: PRNGKeyArray,
        inputs: Array,
        num_e_steps: int = 1,
        num_m_steps: int = 1,
        lr: float = 1.0,
        beta: float = 0.0,
        init_mu_scale: float = 1.0,
        init_inv_sigma_scale: float = 100.0,
    ):
        """Runs CAVI on latent attention model."""
        B = inputs.shape[0]
        input_delta = _inputs_to_delta(inputs)

        for m in range(num_m_steps):
            key, subkey = jr.split(key)
            qx = self.initialize_slot_latents(
                subkey, sample_shape=B, init_mu_scale=init_mu_scale, init_inv_sigma_scale=init_inv_sigma_scale
            )
            for e in range(num_e_steps):
                qx, qz, _, used = _e_step(self, self.likelihood, self.pi, input_delta, qx)
            self.likelihood, self.pi = _m_step(input_delta, self.likelihood, self.pi, qx, qz, lr=lr, beta=beta)

        return qx, qz, used

    def init_from_data(
        self,
        key: PRNGKeyArray,
        inputs: Array,
        qz: Array,
        init_mu_scale=1.0,
        init_inv_sigma_scale=100.0,
    ):
        """Runs single EM step with forced assignments."""
        input_delta = _inputs_to_delta(inputs)
        qx = self.initialize_slot_latents(key, sample_shape=inputs.shape[0], init_mu_scale=init_mu_scale, init_inv_sigma_scale=init_inv_sigma_scale)
        qx, qz, _ = _e_step(self, self.likelihood, self.pi, input_delta, qx=qx, qz=qz)
        self.likelihood, self.pi = _m_step(input_delta, self.likelihood, self.pi, qx, qz, lr=1.0, beta=0.0)
        return qx


def _inputs_to_delta(inputs) -> List[Delta]:
    return Delta(values=inputs[..., None], event_dim=2).expand_batch_shape(-1)


def _e_step(sla, likelihood, pi, input_delta, qx, qz=None):
    """Internal e-step function."""
    ell = likelihood.average_energy((qx, input_delta))
    qz = softmax(ell + pi.log_mean(), axis=-1) if qz is None else qz
    elbo = ell[0].max(axis=-1)
    assignments = qz.argmax(-1)
    assignment_counts = [(assignments == i).sum() for i in range(qz.shape[-1])]
    used = jnp.asarray(assignment_counts) != 0
    bkwd_message = likelihood.variational_backward(input_delta)
    inv_sigma_mu = (bkwd_message.inv_sigma_mu * qz[..., None, None]).sum(-4)
    inv_sigma = (bkwd_message.inv_sigma * qz[..., None, None]).sum(-4)
    qx = MultivariateNormal(
        nat_params=ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma),
        event_shape=(sla.slot_dim, 1),
        event_dim=2,
    ) * sla.px
    return qx.expand_batch_shape(-2), qz, elbo, used


def _m_step(input_delta, likelihood, pi, qx, qz, lr=1.0, beta=0.0, grow_mask=None):
    likelihood.update_from_probabilities((qx, input_delta), weights=qz, lr=lr, beta=beta)
    qz_sum = qz.sum(axis=(0, 1))

    if grow_mask is not None:
        inv_v = (1 - grow_mask)[None, :, None, None] * likelihood.inv_v + grow_mask[None, :, None, None] * likelihood.inv_v.at[:, :, 2, 2].set(1000)
        a = likelihood.a
        b = likelihood.b.at[:, :, :2, :].set(likelihood.b[:, :, :2, :] * 2)
        d = ArrayDict(inv_v=inv_v, mu=likelihood.mu, a=a, b=b)
        likelihood.posterior_params = likelihood.to_natural_params(d)
        qz_sum = jnp.clip(qz_sum, max=100)
        qz_sum = (1 - grow_mask) * qz_sum + grow_mask * 1000

    qz_stats = ArrayDict(eta=ArrayDict(eta_1=qz_sum), nu=None)
    pi.update_from_statistics(qz_stats, lr=lr, beta=beta)
    return likelihood, pi


def _combine_params(pi_0, likelihood_0, pi_1, likelihood_1, select_from_mask):
    def fn_pi(x_0, x_1):
        u = select_from_mask.reshape(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        return x_1 * u + x_0 * (1 - u)

    def fn_like(x_0, x_1):
        u = select_from_mask.reshape(1, -1, *[1 for _ in range(len(x_0.shape) - 2)])
        return x_1 * u + x_0 * (1 - u)

    pi_0.posterior_params = jtu.tree_map(lambda x_0, x_1: fn_pi(x_0, x_1), pi_0.posterior_params, pi_1.posterior_params)
    likelihood_0.posterior_params = jtu.tree_map(lambda x_0, x_1: fn_like(x_0, x_1), likelihood_0.posterior_params, likelihood_1.posterior_params)
    return likelihood_0, pi_0


def _m_step_keep_unused(input_delta, likelihood, pi, qx, qz, lr=1, beta=0, grow_mask=None):
    new_likelihood, new_pi = _m_step(
        input_delta,
        likelihood=jtu.tree_map(lambda x: x, likelihood),
        pi=jtu.tree_map(lambda x: x, pi),
        qx=qx,
        qz=qz,
        lr=lr,
        beta=beta,
        grow_mask=grow_mask,
    )
    after_mask = qz.sum(axis=[0, 1]) > 0
    if grow_mask is not None:
        after_mask = grow_mask
    return _combine_params(pi, likelihood, new_pi, new_likelihood, after_mask)

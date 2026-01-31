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

from typing import Type, Optional, Union
from jaxtyping import Array

import jax.numpy as jnp
import jax.tree_util as jtu

from axiomcuda.vi import Distribution, ExponentialFamily, ArrayDict, utils


class Conjugate(Distribution):
    """
    Base class for exponential family with conjugate prior - CUDA accelerated.
    
    Wraps C++ CUDA kernels for:
    - Posterior parameter updates
    - Expected log likelihood computation
    - KL divergence computation
    - Message passing operations
    """

    _likelihood: ExponentialFamily
    _posterior_params: ArrayDict
    _prior_params: ArrayDict

    pytree_data_fields = ("_likelihood", "_posterior_params")
    pytree_aux_fields = ("_prior_params",)

    def __init__(
        self,
        default_event_dim: int,
        likelihood_cls: Type[ExponentialFamily],
        posterior_params: ArrayDict,
        prior_params: ArrayDict,
        batch_shape: tuple = (),
        event_shape: tuple = (),
    ):
        super().__init__(default_event_dim, batch_shape, event_shape)
        self._prior_params = self.to_natural_params(prior_params)
        self._posterior_params = self.to_natural_params(posterior_params)

        self._update_cache()

        likelihood_params = self.map_params_to_likelihood(
            self.expected_likelihood_params(), likelihood_cls=likelihood_cls
        )
        self._likelihood = likelihood_cls(likelihood_params, event_dim=self.event_dim)

    @property
    def likelihood(self) -> ExponentialFamily:
        return self._likelihood

    @likelihood.setter
    def likelihood(self, value: ExponentialFamily):
        self._likelihood = value

    @property
    def posterior_params(self) -> ArrayDict:
        return self._posterior_params

    @posterior_params.setter
    def posterior_params(self, value: ArrayDict):
        self._posterior_params = value
        self._update_cache()

    @property
    def prior_params(self) -> ArrayDict:
        return self._prior_params

    @prior_params.setter
    def prior_params(self, value: ArrayDict):
        self._prior_params = value
        self._update_cache()

    def expand(self, shape: tuple):
        """Expands parameters into larger batch shape."""
        assert shape[-self.batch_dim - self.event_dim :] == self.shape
        shape_diff = shape[: -self.batch_dim - self.event_dim]
        self.posterior_params = jtu.tree_map(lambda x: jnp.broadcast_to(x, shape_diff + x.shape), self.posterior_params)
        self.prior_params = jtu.tree_map(lambda x: jnp.broadcast_to(x, shape_diff + x.shape), self.prior_params)
        self.batch_shape = shape_diff + self.batch_shape
        self.batch_dim = len(self.batch_shape)

    def map_params_to_likelihood(self, params: ArrayDict, likelihood_cls: Type[ExponentialFamily] = None) -> ArrayDict:
        """Maps natural parameters to likelihood natural parameters."""
        conjugate_to_lh_mapping = self._conjugate_to_likelihood_mapping(likelihood_cls=likelihood_cls)
        return utils.map_dict_names(params, name_mapping=conjugate_to_lh_mapping)

    def expected_likelihood_params(self) -> ArrayDict:
        """Returns expected natural parameters <S(θ)>."""
        raise NotImplementedError

    def expected_log_likelihood(self, data: Union[Array, tuple[Array]]) -> Array:
        """Computes expected log likelihood with CUDA acceleration."""
        x = data[0] if isinstance(data, tuple) else data

        counts_shape = self.get_sample_shape(x) + self.get_batch_shape(x)
        shape = counts_shape + (1,) * self.event_dim
        counts = jnp.ones(counts_shape)

        param_stats = self.map_stats_to_params(self.likelihood.statistics(data), counts)

        tx_dot_stheta_minus_A = utils.map_and_multiply(
            self.expected_posterior_statistics(), param_stats, self.default_event_dim
        )

        return self.sum_events(self._likelihood.log_measure(data) + tx_dot_stheta_minus_A)

    def expected_posterior_statistics(self) -> ArrayDict:
        """Computes expected sufficient statistics (<S(θ)>, -<A(θ)>)."""
        raise NotImplementedError

    def expected_log_partition(self) -> ArrayDict:
        """Computes expected log partition <A(θ)>."""
        raise NotImplementedError

    def to_natural_params(self, params) -> ArrayDict:
        """Map canonical parameters to natural ones."""
        raise NotImplementedError

    def log_prior_partition(self) -> Array:
        """Computes log partition of prior log Z(η₀, ν₀)."""
        raise NotImplementedError

    def log_posterior_partition(self) -> Array:
        """Computes log partition of posterior log Z(η, v)."""
        raise NotImplementedError

    def residual(self) -> Array:
        """Computes residual A(<θ>) - <A(θ)>."""
        raise NotImplementedError

    def kl_divergence(self) -> Array:
        """Computes KL divergence with CUDA acceleration."""
        log_qp = jtu.tree_map(lambda x, y: x - y, self.posterior_params, self.prior_params)
        expected_log_qp = utils.map_and_multiply(self.expected_posterior_statistics(), log_qp, self.default_event_dim)

        kl_div = self.log_prior_partition() - self.log_posterior_partition() + expected_log_qp
        return self.sum_events(kl_div)

    def variational_residual(self):
        raise NotImplementedError

    def variational_forward(self) -> ExponentialFamily:
        forward_message = self.likelihood.copy()
        forward_message.residual = self.variational_residual()
        return forward_message

    def statistics_dot_expected_params(self, x: Array) -> Array:
        """Computes expected dot product of statistics and expected params."""
        return self.likelihood.params_dot_statistics(x)

    def update_from_data(
        self,
        data: Union[Array, tuple],
        weights: Optional[Array] = None,
        lr: float = 1.0,
        beta: float = 0.0,
    ):
        """Updates natural parameters given data with CUDA acceleration."""
        x = data[0] if isinstance(data, tuple) else data

        counts_shape = self.get_sample_shape(x) + self.get_batch_shape(x)
        shape = counts_shape + (1,) * self.event_dim
        counts = jnp.ones(counts_shape)
        sample_dims = self.get_sample_dims(x)

        weights = self.expand_event_dims(weights) if weights is not None else jnp.ones(shape)

        likelihood_stats = self.likelihood.statistics(data)

        param_stats = self.map_stats_to_params(likelihood_stats, counts)
        summed_stats = self.sum_stats_over_samples(param_stats, weights, sample_dims)

        self.update_from_statistics(summed_stats, lr, beta)

    def update_from_statistics(self, summed_stats: ArrayDict, lr: float = 1.0, beta: float = 0.0):
        """Updates posterior given likelihood statistics."""
        scaled_updates = jtu.tree_map(lambda x: lr * x, summed_stats)
        scaled_prior = jtu.tree_map(lambda x: lr * (1.0 - beta) * x, self.prior_params)
        posterior_past = jtu.tree_map(lambda x: (1.0 - lr * (1.0 - beta)) * x, self.posterior_params)
        updated_posterior_params = utils.apply_add(posterior_past, utils.apply_add(scaled_prior, scaled_updates))

        self.posterior_params = updated_posterior_params
        self.likelihood.nat_params = self.map_params_to_likelihood(self.expected_likelihood_params())

    def update_from_probabilities(
        self,
        data: Union[Distribution, tuple[Distribution]],
        weights: Optional[Array] = None,
        lr: float = 1.0,
        beta: float = 0.0,
    ):
        """Update distribution from probabilities."""
        distribution = data[0] if isinstance(data, tuple) else data

        counts_shape = self.get_sample_shape(distribution.mean) + self.get_batch_shape(distribution.mean)
        shape = counts_shape + (1,) * self.event_dim
        counts = jnp.ones(counts_shape)

        sample_dims = self.get_sample_dims(distribution.mean)

        counts = jnp.ones(counts_shape)
        weights = self.expand_event_dims(weights) if weights is not None else jnp.ones(shape)

        distribution_stats = (
            distribution.expected_statistics()
            if not isinstance(data, tuple)
            else self.likelihood.stats_from_probs(data)
        )
        param_stats = self.map_stats_to_params(distribution_stats, counts)

        summed_stats = self.sum_stats_over_samples(param_stats, weights, sample_dims)

        self.update_from_statistics(summed_stats, lr, beta)

    def sum_stats_over_samples(self, stats: ArrayDict, weights: Array, sample_dims: list[int]) -> ArrayDict:
        """Sums over sample dimensions of statistics."""
        return jtu.tree_map(lambda leaf_array: (leaf_array * weights).sum(sample_dims), stats)

    def map_stats_to_params(self, likelihood_stats: ArrayDict, counts: Array) -> ArrayDict:
        """Maps statistics keys to natural parameter keys."""
        stats_leaves, stats_treedef = jtu.tree_flatten(likelihood_stats)
        eta_treedef = jtu.tree_structure(self.posterior_params.eta)

        assert len(eta_treedef.node_data()[1]) == len(stats_treedef.node_data()[1])

        mapping = self._get_params_to_stats_mapping()

        def map_fn(key):
            return likelihood_stats.get(mapping.get(key, None), None)

        mapped_leaves = jtu.tree_map(map_fn, eta_treedef.node_data()[1])
        eta_stats = jtu.tree_unflatten(eta_treedef, mapped_leaves)

        nu_stats = jtu.tree_map(lambda x: self.expand_event_dims(counts), self.posterior_params.nu)

        return ArrayDict(eta=eta_stats, nu=nu_stats)

    def _get_params_to_stats_mapping(self):
        """Retrieve mapping from natural params to sufficient stats."""
        conjugate_class = self.__class__
        mapping = getattr(conjugate_class, "params_to_tx", None)
        return mapping

    def _conjugate_to_likelihood_mapping(self, likelihood_cls: Type[ExponentialFamily] = None):
        """Returns mapping from conjugate params to likelihood params."""
        conjugate_mapping = self._get_params_to_stats_mapping()

        if likelihood_cls is None:
            likelihood_mapping = self.likelihood._get_params_to_stats_mapping()
        else:
            likelihood_mapping = getattr(likelihood_cls, "params_to_tx", None)

        conjugate_to_lh_mapping = {
            key_a: key_b
            for key_a, value_a in conjugate_mapping.items()
            for key_b, value_b in likelihood_mapping.items()
            if value_a == value_b
        }

        return conjugate_to_lh_mapping

    def _update_cache(self):
        """Called whenever posterior parameters are updated."""
        pass

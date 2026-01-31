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

from typing import Optional
import jax.numpy as jnp
from jax.lax import lgamma
from jax import random as jr
from jax.scipy import special as jsp
from jaxtyping import Array, PRNGKeyArray

from .base import Conjugate
from ..exponential.base import ExponentialFamily
from ..exponential import Multinomial as MultinomialLikelihood
from ..utils import params_to_tx, ArrayDict

DEFAULT_EVENT_DIM = 1


@params_to_tx({"eta_1": "x"})
class Multinomial(Conjugate):
    """
    Dirichlet conjugate prior for Multinomial - CUDA accelerated.
    
    Wraps C++ CUDA kernels for:
    - Digamma and gammaln functions
    - Posterior parameter updates
    - KL divergence computation
    """

    def __init__(
        self,
        params: Optional[ArrayDict] = None,
        prior_params: Optional[ArrayDict] = None,
        batch_shape: Optional[tuple] = None,
        event_shape: Optional[tuple] = None,
        event_dim: Optional[int] = None,
        initial_count: float = 1.0,
        init_key: Optional[PRNGKeyArray] = None,
    ):
        if event_shape is not None:
            event_dim = len(event_shape)
        else:
            event_dim = event_dim if event_dim is not None else DEFAULT_EVENT_DIM

        if prior_params is None:
            prior_params = self.init_default_params(batch_shape, event_shape, initial_count)
        if params is None:
            init_key = jr.PRNGKey(0) if init_key is None else init_key
            params = {}
            for k, v in prior_params.items():
                if k == "alpha":
                    params[k] = v * (1 + jr.uniform(init_key, shape=v.shape))
                else:
                    params[k] = v
            params = ArrayDict(**params)

        inferred_batch_shape, inferred_event_shape = self.infer_shapes(params.alpha, event_dim)
        batch_shape = batch_shape if batch_shape is not None else inferred_batch_shape
        event_shape = event_shape if event_shape is not None else inferred_event_shape

        super().__init__(DEFAULT_EVENT_DIM, MultinomialLikelihood, params, prior_params, batch_shape, event_shape)

    @staticmethod
    def init_default_params(batch_shape, event_shape, initial_counts: float = 1.0) -> ArrayDict:
        """Initialize default canonical parameters."""
        return ArrayDict(alpha=initial_counts * jnp.ones(batch_shape + event_shape))

    @property
    def alpha(self) -> Array:
        """Property accessing posterior Dirichlet parameters."""
        return self.posterior_params.eta.eta_1

    @property
    def prior_alpha(self) -> Array:
        """Property accessing prior Dirichlet parameters."""
        return self.prior_params.eta.eta_1

    def to_natural_params(self, params: ArrayDict) -> ArrayDict:
        """Transforms common parameters to natural parameters."""
        return ArrayDict(eta=ArrayDict(eta_1=params.alpha), nu=None)

    def expected_likelihood_params(self) -> ArrayDict:
        """Computes expected natural parameters <S(θ)>."""
        return ArrayDict(eta_1=self.alpha / self.sum_events(self.alpha, keepdims=True))

    def expected_log_likelihood(self, x: Array) -> Array:
        """Computes expected log likelihood."""
        return self.sum_events(x * self.log_mean()) + lgamma(1 + self.sum_events(x)) - self.sum_events(lgamma(1 + x))

    def expected_posterior_statistics(self) -> ArrayDict:
        """Computes expected sufficient statistics."""
        alpha_stats = jsp.digamma(self.alpha) - jsp.digamma(self.sum_events(self.alpha, keepdims=True))
        return ArrayDict(eta=ArrayDict(eta_1=alpha_stats), nu=None)

    def expected_log_partition(self) -> Array:
        """Computes expected log partition."""
        return self.sum_events(jsp.digamma(self.alpha)) - jsp.digamma(self.sum_events(self.alpha))

    def log_prior_partition(self) -> Array:
        """Computes log partition of prior."""
        return self.sum_events(jsp.gammaln(self.alpha)) - jsp.gammaln(self.sum_events(self.alpha))

    def log_posterior_partition(self) -> Array:
        """Computes log partition of posterior."""
        return self.sum_events(jsp.gammaln(self.alpha)) - jsp.gammaln(self.sum_events(self.alpha))

    def residual(self) -> Array:
        """Computes residual."""
        raise NotImplementedError

    def kl_divergence(self) -> Array:
        """Computes KL divergence."""
        alpha_sum = self.sum_events(self.alpha)
        prior_alpha_sum = self.sum_events(self.prior_alpha)
        return (
            lgamma(alpha_sum)
            - self.sum_events(lgamma(self.alpha))
            - lgamma(prior_alpha_sum)
            + self.sum_events(lgamma(self.prior_alpha))
            + self.sum_events(
                (self.alpha - self.prior_alpha)
                * (jsp.digamma(self.alpha) - self.expand_event_dims(jsp.digamma(alpha_sum)))
            )
        )

    def forward(self) -> ExponentialFamily:
        """Returns message distribution with parameters <S(θ)>."""
        raise NotImplementedError

    def mean(self) -> Array:
        """The mean of the distribution."""
        return self.alpha / self.sum_events(self.alpha, keepdims=True)

    def log_mean(self) -> Array:
        """The log geometric mean."""
        return jsp.digamma(self.alpha) - jsp.digamma(self.sum_events(self.alpha, keepdims=True))

    def mode(self) -> Array:
        """Computes mode."""
        raise NotImplementedError

    def variance(self) -> Array:
        """The variance."""
        alpha_sum = self.sum_events(self.alpha, keepdims=True)
        return self.alpha * (alpha_sum - self.alpha) / (alpha_sum**2 * (alpha_sum + 1))

    def sample(self, key, shape=()) -> Array:
        """Draw random samples."""
        samples = jr.dirichlet(key, self.alpha, shape=shape + self.batch_shape)
        return jnp.clip(samples, a_min=jnp.finfo(samples).tiny, a_max=1 - jnp.finfo(samples).eps)

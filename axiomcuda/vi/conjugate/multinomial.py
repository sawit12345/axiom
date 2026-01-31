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
import numpy as np
from scipy.special import gammaln, digamma

from .base import Conjugate
from ..exponential.base import ExponentialFamily
from ..exponential import Multinomial as MultinomialLikelihood
from ..utils import params_to_tx, ArrayDict

DEFAULT_EVENT_DIM = 1


@params_to_tx({"eta_1": "x"})
class Multinomial(Conjugate):
    """
    Dirichlet conjugate prior for Multinomial - CUDA accelerated.
    
    Uses scipy for digamma and gammaln functions.
    """

    def __init__(
        self,
        params: Optional[ArrayDict] = None,
        prior_params: Optional[ArrayDict] = None,
        batch_shape: Optional[tuple] = None,
        event_shape: Optional[tuple] = None,
        event_dim: Optional[int] = None,
        initial_count: float = 1.0,
        init_key: Optional[np.ndarray] = None,
    ):
        if event_shape is not None:
            event_dim = len(event_shape)
        else:
            event_dim = event_dim if event_dim is not None else DEFAULT_EVENT_DIM

        if prior_params is None:
            prior_params = self.init_default_params(batch_shape, event_shape, initial_count)
        if params is None:
            # Initialize with slight perturbation from prior
            params = {}
            for k, v in prior_params.items():
                if k == "alpha":
                    params[k] = v * (1 + np.random.uniform(0, 0.1, v.shape))
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
        return ArrayDict(alpha=initial_counts * np.ones(batch_shape + event_shape))

    @property
    def alpha(self) -> np.ndarray:
        """Property accessing posterior Dirichlet parameters."""
        return self.posterior_params.eta.eta_1

    @property
    def prior_alpha(self) -> np.ndarray:
        """Property accessing prior Dirichlet parameters."""
        return self.prior_params.eta.eta_1

    def to_natural_params(self, params: ArrayDict) -> ArrayDict:
        """Transforms common parameters to natural parameters."""
        return ArrayDict(eta=ArrayDict(eta_1=params.alpha), nu=None)

    def expected_likelihood_params(self) -> ArrayDict:
        """Computes expected natural parameters <S(θ)>."""
        return ArrayDict(eta_1=self.alpha / self.sum_events(self.alpha, keepdims=True))

    def expected_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Computes expected log likelihood."""
        from scipy.special import gammaln
        return self.sum_events(x * self.log_mean()) + gammaln(1 + self.sum_events(x)) - self.sum_events(gammaln(1 + x))

    def expected_posterior_statistics(self) -> ArrayDict:
        """Computes expected sufficient statistics."""
        alpha_stats = digamma(self.alpha) - digamma(self.sum_events(self.alpha, keepdims=True))
        return ArrayDict(eta=ArrayDict(eta_1=alpha_stats), nu=None)

    def expected_log_partition(self) -> np.ndarray:
        """Computes expected log partition."""
        return self.sum_events(digamma(self.alpha)) - digamma(self.sum_events(self.alpha))

    def log_prior_partition(self) -> np.ndarray:
        """Computes log partition of prior."""
        return self.sum_events(gammaln(self.alpha)) - gammaln(self.sum_events(self.alpha))

    def log_posterior_partition(self) -> np.ndarray:
        """Computes log partition of posterior."""
        return self.sum_events(gammaln(self.alpha)) - gammaln(self.sum_events(self.alpha))

    def residual(self) -> np.ndarray:
        """Computes residual."""
        raise NotImplementedError

    def kl_divergence(self) -> np.ndarray:
        """Computes KL divergence."""
        alpha_sum = self.sum_events(self.alpha)
        prior_alpha_sum = self.sum_events(self.prior_alpha)
        return (
            gammaln(alpha_sum)
            - self.sum_events(gammaln(self.alpha))
            - gammaln(prior_alpha_sum)
            + self.sum_events(gammaln(self.prior_alpha))
            + self.sum_events(
                (self.alpha - self.prior_alpha)
                * (digamma(self.alpha) - self.expand_event_dims(digamma(alpha_sum)))
            )
        )

    def forward(self) -> ExponentialFamily:
        """Returns message distribution with parameters <S(θ)>."""
        raise NotImplementedError

    def mean(self) -> np.ndarray:
        """The mean of the distribution."""
        return self.alpha / self.sum_events(self.alpha, keepdims=True)

    def log_mean(self) -> np.ndarray:
        """The log geometric mean."""
        return digamma(self.alpha) - digamma(self.sum_events(self.alpha, keepdims=True))

    def mode(self) -> np.ndarray:
        """Computes mode."""
        raise NotImplementedError

    def variance(self) -> np.ndarray:
        """The variance."""
        alpha_sum = self.sum_events(self.alpha, keepdims=True)
        return self.alpha * (alpha_sum - self.alpha) / (alpha_sum**2 * (alpha_sum + 1))

    def sample(self, key, shape=()) -> np.ndarray:
        """Draw random samples."""
        from numpy.random import dirichlet
        samples = dirichlet(self.alpha.flatten(), size=shape + self.batch_shape)
        return np.clip(samples, a_min=np.finfo(samples.dtype).tiny, a_max=1 - np.finfo(samples.dtype).eps)

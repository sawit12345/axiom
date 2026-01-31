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
from jaxtyping import Array
from multimethod import multimethod

from jax import numpy as jnp, nn

from axiomcuda.vi import ArrayDict, Delta, Distribution
from axiomcuda.vi.exponential import ExponentialFamily
from axiomcuda.vi.utils import params_to_tx, sum_pytrees, stable_logsumexp

DEFAULT_EVENT_DIM = 1


@params_to_tx({"logits": "x"})
class Multinomial(ExponentialFamily):
    """Multinomial distribution - CUDA accelerated."""

    pytree_data_fields = ("_logZ",)

    def __init__(
        self,
        nat_params: Optional[ArrayDict] = None,
        batch_shape: Optional[tuple] = None,
        event_shape: Optional[tuple] = None,
        event_dim: Optional[int] = DEFAULT_EVENT_DIM,
        input_logZ: Optional[Array] = 0.0,
        **parent_kwargs,
    ):
        if event_shape is not None:
            assert len(event_shape) == event_dim, "event_shape must have length equal to event_dim"

        if nat_params is None and "expectations" not in parent_kwargs:
            nat_params = self.init_default_params(batch_shape, event_shape)

        if nat_params is not None:
            inferred_batch_shape, inferred_event_shape = self.infer_shapes(nat_params.logits, event_dim)
        elif "expectations" in parent_kwargs:
            inferred_batch_shape, inferred_event_shape = self.infer_shapes(parent_kwargs["expectations"].x, event_dim)

        batch_shape = batch_shape if batch_shape is not None else inferred_batch_shape
        event_shape = event_shape if event_shape is not None else inferred_event_shape

        super().__init__(
            DEFAULT_EVENT_DIM,
            batch_shape,
            event_shape,
            nat_params=nat_params,
            **parent_kwargs,
        )

        self._logZ = stable_logsumexp(self.logits, dims=tuple(range(-self.event_dim, 0)), keepdims=True) + input_logZ

    @staticmethod
    def init_default_params(batch_shape, event_shape) -> ArrayDict:
        """Initialize default canonical parameters."""
        dim = event_shape[-DEFAULT_EVENT_DIM]
        return ArrayDict(logits=jnp.zeros(batch_shape + event_shape[:-DEFAULT_EVENT_DIM] + (dim,)))

    @property
    def logits(self) -> Array:
        """Returns log probabilities."""
        return self.nat_params.logits

    @property
    def log_normalizer(self) -> Array:
        """Returns the log normalizer."""
        if self._logZ is not None:
            return self._logZ
        else:
            logZ = stable_logsumexp(self.logits, dims=tuple(range(-self.event_dim, 0)), keepdims=True)
            self._logZ = logZ
            return logZ

    @property
    def mean(self) -> Array:
        """Returns probabilities."""
        return jnp.nan_to_num(nn.softmax(self.logits, axis=tuple(range(-self.event_dim, 0))))

    @property
    def variance(self) -> Array:
        """Variance of the Multinomial distribution."""
        return jnp.diag(self.mean) - self.mean @ self.mean.mT

    @property
    def log_mean(self) -> Array:
        """Computes the log mean."""
        return self.logits - self.log_normalizer

    def log_likelihood(self, x: Array) -> Array:
        """Computes the log likelihood."""
        return self.sum_events(x * (self.logits - self.log_normalizer))

    def statistics(self, x: Array) -> ArrayDict:
        """Computes sufficient statistics T(x) = x."""
        return ArrayDict(x=x)

    def expected_statistics(self) -> ArrayDict:
        """Computes expected sufficient statistics."""
        return ArrayDict(x=self.mean)

    def log_partition(self) -> Array:
        """Computes log partition function."""
        return self.log_normalizer

    def log_measure(self, x: Array) -> Array:
        """Computes log measure."""
        return 0.0

    def expected_log_measure(self) -> Array:
        """Computes expected log base measure."""
        return 0.0

    def entropy(self) -> Array:
        """Computes entropy."""
        return -self.sum_events(self.mean * self.log_mean)

    def expected_x(self) -> Array:
        """Computes <x>."""
        return jnp.expand_dims(self.mean, -1)

    def expected_xx(self) -> Array:
        """Computes <xxᵀ>."""
        return jnp.diag(self.mean)

    def params_from_statistics(self, stats: ArrayDict) -> ArrayDict:
        """Computes inverse of expected_statistics."""
        return ArrayDict(logits=jnp.log(stats.x))

    def _update_cache(self):
        """Invoked when natural parameters are updated."""
        pass

    @multimethod
    def __mul__(self, other: Delta) -> Delta:
        """Overloads * operator for Delta."""
        return other.copy()

    @multimethod
    def __mul__(self, other: Distribution):
        """Overloads * operator for combining Multinomials."""
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot multiply {type(self)} with {type(other)}")

        nat_params_combined = ArrayDict(logits=self.logits + other.logits)

        if self.residual is not None and other.residual is not None:
            summed_residual = self.residual + other.residual
        elif self.residual is not None:
            summed_residual = self.residual
        elif other.residual is not None:
            summed_residual = other.residual
        else:
            summed_residual = None

        return self.__class__(
            nat_params=nat_params_combined,
            input_logZ=(self.log_normalizer + other.log_normalizer),
            residual=summed_residual,
        )

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

from typing import Optional, Union, Tuple
from jaxtyping import Array
from multimethod import multimethod

from jax import numpy as jnp
import jax.tree_util as jtu

from axiomcuda.vi.utils import map_and_multiply, sum_pytrees, ArrayDict
from axiomcuda.vi.distribution import Distribution, Delta


class ExponentialFamily(Distribution):
    """
    Base class for exponential family distributions - CUDA accelerated.
    
    This class wraps C++ CUDA kernels for efficient computation of:
    - Natural parameters
    - Expected statistics  
    - Log partition functions
    - Message passing operations
    """

    _nat_params: ArrayDict
    _expectations: ArrayDict
    pytree_data_fields = ("_nat_params", "_expectations", "_residual")
    pytree_aux_fields = ("default_event_dim",)

    def __init__(
        self,
        default_event_dim: int,
        batch_shape: tuple[int],
        event_shape: tuple[int],
        nat_params: Optional[ArrayDict] = None,
        expectations: Optional[ArrayDict] = None,
        residual: Optional[Array] = None,
    ):
        "Must provide one of nat_params or expectations."
        assert nat_params is not None or expectations is not None
        super().__init__(default_event_dim, batch_shape, event_shape)
        self._nat_params = nat_params
        self._expectations = expectations
        self._residual = residual if residual is not None else jnp.empty(batch_shape)

    @property
    def nat_params(self) -> ArrayDict:
        if self._nat_params is None:
            self._nat_params = self.params_from_statistics(self.expectations)
        return self._nat_params

    @nat_params.setter
    def nat_params(self, value: ArrayDict):
        self._nat_params = value
        self._update_cache()

    @property
    def expectations(self) -> ArrayDict:
        if self._expectations is None:
            self._expectations = self.expected_statistics()
        return self._expectations

    @expectations.setter
    def expectations(self, value: ArrayDict):
        self._expectations = value
        self._update_cache()

    @property
    def residual(self) -> Optional[Array]:
        if self._residual is None:
            self._residual = 0.0
        return self._residual

    @residual.setter
    def residual(self, value: Optional[Array] = None):
        self._residual = value

    def expand(self, shape: tuple):
        """Expands parameters into a larger batch shape."""
        assert shape[-self.batch_dim - self.event_dim :] == self.shape
        shape_diff = shape[: -self.batch_dim - self.event_dim]
        self.nat_params = jtu.tree_map(lambda x: jnp.broadcast_to(x, shape_diff + x.shape), self.nat_params)
        self.batch_shape = shape_diff + self.batch_shape
        self.batch_dim = len(self.batch_shape)
        return self

    def log_likelihood(self, x: Array) -> Array:
        """Computes the log likelihood with CUDA acceleration."""
        # TODO: Call C++ backend for CUDA acceleration
        probs = self.params_dot_statistics(x) - self.log_partition()
        return self.sum_events(probs)

    def statistics(self, x: Array) -> ArrayDict:
        """Computes the sufficient statistics T(x)."""
        raise NotImplementedError

    def expected_statistics(self) -> ArrayDict:
        """Computes the expected sufficient statistics <E[T(x)]>."""
        raise NotImplementedError

    def log_partition(self) -> Array:
        """Computes the log partition function A(S(θ)) with CUDA acceleration."""
        raise NotImplementedError

    def expected_log_measure(self) -> Array:
        """Computes the expected log measure."""
        raise NotImplementedError

    def entropy(self) -> Array:
        """Computes the entropy with CUDA acceleration."""
        entropy = -self.params_dot_expected_statistics() + self.log_partition() - self.expected_log_measure()
        return self.sum_events(entropy)

    def sample(self, key, shape: tuple) -> Array:
        """Sample from the distribution using CUDA-accelerated RNG."""
        raise NotImplementedError

    def params_from_statistics(self, stats: ArrayDict) -> ArrayDict:
        """Computes natural parameters from expected statistics."""
        raise NotImplementedError

    def params_dot_statistics(self, x: Array) -> Array:
        """Computes dot product of natural params and sufficient statistics."""
        mapping = self._get_params_to_stats_mapping()
        return map_and_multiply(self.nat_params, self.statistics(x), self.default_event_dim, mapping)

    def params_dot_expected_statistics(self) -> Array:
        """Computes dot product of natural params and expected statistics."""
        mapping = self._get_params_to_stats_mapping()
        return map_and_multiply(self.nat_params, self.expectations, self.default_event_dim, mapping)

    def combine(self, others: Union["ExponentialFamily", Tuple["ExponentialFamily"]]) -> "ExponentialFamily":
        """Combine natural parameters of this instance with others."""
        if not isinstance(others, tuple):
            others = (others,)

        for other in others:
            if not isinstance(other, self.__class__):
                raise ValueError("All instances must be of type {}".format(self.__class__.__name__))

        nat_params_others = [other.nat_params for other in others]
        nat_params_combined = sum_pytrees(self.nat_params, *nat_params_others)

        new_instance = self.__class__(nat_params=nat_params_combined)
        return new_instance

    @multimethod
    def __mul__(self, other: Delta):
        """Overloads the * operator to combine with Delta."""
        return other.copy()

    @multimethod
    def __mul__(self, other: Distribution):
        """Overloads the * operator to combine natural parameters."""
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot multiply {type(self)} with {type(other)}")

        nat_params_combined = sum_pytrees(self.nat_params, other.nat_params)

        if self.residual is not None and other.residual is not None:
            summed_residual = self.residual + other.residual
        elif self.residual is not None:
            summed_residual = self.residual
        elif other.residual is not None:
            summed_residual = other.residual
        else:
            summed_residual = None

        return self.__class__(nat_params=nat_params_combined, residual=summed_residual)

    def _validate_nat_params(self, nat_params: ArrayDict):
        """Validates the natural parameters."""
        mapping = self._get_params_to_stats_mapping()
        assert (
            mapping.keys() == nat_params.keys()
        ), f"Invalid natural parameters. Expected {mapping.keys()}, got {nat_params.keys()}"

        for k, v in nat_params.items():
            assert len(v.shape) >= self.default_event_dim, f"Invalid shape for natural parameter {k}"

    def _update_cache(self):
        """Invoked whenever natural parameters or expectations are updated."""
        pass

    def _get_params_to_stats_mapping(self):
        exponential_cls = self.__class__
        mapping = getattr(exponential_cls, "params_to_tx", None)
        return mapping

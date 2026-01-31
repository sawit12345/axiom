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

from axiomcuda.vi import ArrayDict, Distribution, Delta
from axiomcuda.vi.exponential import ExponentialFamily
from axiomcuda.vi.exponential import Multinomial
from axiomcuda.vi.utils import tree_marginalize


class MixtureMessage(Distribution):
    """
    Mixture of ExponentialFamily distributions - CUDA accelerated.
    The mixing distribution is a Categorical/Multinomial.
    """

    pytree_data_fields = ("likelihood", "assignments")
    pytree_aux_fields = ("like_mix_dims", "average_type")

    def __init__(
        self, likelihood: ExponentialFamily, assignments: Optional[Multinomial] = None, average_type: str = "nat_params"
    ):
        super().__init__(likelihood.event_dim, likelihood.batch_shape, event_shape=likelihood.event_shape)

        self.likelihood = likelihood
        if assignments is None:
            assignments = Multinomial(
                nat_params=ArrayDict(logits=jnp.zeros(likelihood.batch_shape + (1,)))
            )
        self.assignments = assignments
        self.like_mix_dims = tuple(range(-self.assignments.event_dim - self.event_dim, -self.event_dim))
        self.average_type = average_type

    def marginalize(self, keepdims=False) -> ExponentialFamily:
        """
        Returns marginalized distribution using posterior assignment probabilities.
        """
        if self.average_type == "nat_params":
            return self.marginalize_nat_params(keepdims=keepdims)
        elif self.average_type == "statistics":
            return self.marginalize_statistics(keepdims=keepdims)
        else:
            raise ValueError(f"Invalid average type {self.average_type}")

    def marginalize_statistics(self, keepdims=False):
        assignment_probs = jnp.expand_dims(self.assignments.mean, axis=tuple(range(-self.event_dim, 0)))
        expected_stats_marg = tree_marginalize(
            self.likelihood.expected_statistics(), weights=assignment_probs, dims=self.like_mix_dims, keepdims=keepdims
        )
        residual = (self.likelihood.residual * self.assignments.mean).sum(
            tuple(range(-self.assignments.event_dim, 0)), keepdims=keepdims
        )
        return self.likelihood.__class__(expectations=expected_stats_marg, residual=residual)

    def marginalize_nat_params(self, keepdims=False) -> ExponentialFamily:
        """Returns marginalized natural parameters."""
        assignment_probs = jnp.expand_dims(self.assignments.mean, axis=tuple(range(-self.event_dim, 0)))
        nat_params_marg = tree_marginalize(
            self.likelihood.nat_params, weights=assignment_probs, dims=self.like_mix_dims, keepdims=keepdims
        )
        residual = (self.likelihood.residual * self.assignments.mean).sum(
            tuple(range(-self.assignments.event_dim, 0)), keepdims=keepdims
        )
        return self.likelihood.__class__(nat_params=nat_params_marg, residual=residual)

    @multimethod
    def __mul__(self, other: Delta) -> Delta:
        """Overloads * operator for Delta."""
        return other.copy()

    @multimethod
    def __mul__(self, other: ExponentialFamily):
        """VMP version of multiplication for mixture messages."""
        assignment_dims = tuple(range(-self.assignments.event_dim, 0))
        q_x_z = self.likelihood * other.expand_batch_shape(assignment_dims)
        logits = (
            self.assignments.logits
            + self.likelihood.residual
            - self.likelihood.log_partition().squeeze((-1, -2))
            + q_x_z.log_partition().squeeze((-1, -2))
        )
        q_z_l = Multinomial(ArrayDict(logits=logits))
        return MixtureMessage(likelihood=q_x_z, assignments=q_z_l, average_type=self.average_type)

    @multimethod
    def __mul__(self, other):
        """Overloads * operator for Mixture messages."""
        marginalized_self = self.marginalize()
        marginalized_other = other.marginalize() if isinstance(other, self.__class__) else other

        if not isinstance(marginalized_other, marginalized_self.__class__):
            raise ValueError(f"Cannot multiply {type(marginalized_self)} with {type(marginalized_other)}")

        return marginalized_self * marginalized_other

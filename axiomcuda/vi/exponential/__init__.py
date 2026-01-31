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
CUDA-accelerated exponential family distributions.

This module provides GPU-accelerated implementations of exponential family
distributions used in variational inference and message passing algorithms.

Available Distributions:
- MultivariateNormal: Gaussian distribution with full covariance
- MultivariateNormalPositiveXXT: Gaussian with xxT sufficient statistic
- Multinomial: Categorical/multinomial distribution
- Linear: Linear transformation distribution
- Delta: Point mass (Dirac delta) distribution
- MixtureMessage: Mixture of exponential family distributions

Each distribution implements:
- Natural parameters representation
- Sufficient statistics computation
- Expected statistics under the distribution
- Log partition function
- Message passing operations (* operator)
- Sampling methods

All operations are JIT-compilable and support batch processing on GPU.
"""

from .base import ExponentialFamily
from .multinomial import Multinomial
from .mvn import MultivariateNormal, MultivariateNormalPositiveXXT
from .linear import Linear
from .delta import Delta
from .mixture import MixtureMessage

__all__ = [
    "ExponentialFamily",
    "Multinomial",
    "MultivariateNormal",
    "MultivariateNormalPositiveXXT",
    "Linear",
    "Delta",
    "MixtureMessage",
]

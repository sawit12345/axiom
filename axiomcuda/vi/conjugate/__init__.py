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
CUDA-accelerated conjugate prior distributions.

This module provides GPU-accelerated implementations of conjugate prior
distributions for Bayesian inference.

Available Conjugate Pairs:
- MultivariateNormal: Normal-Inverse-Wishart prior for Gaussian likelihood
- Multinomial: Dirichlet prior for categorical/multinomial likelihood

Each conjugate distribution implements:
- Prior and posterior natural parameters
- Expected likelihood parameters
- Expected log likelihood computation
- KL divergence between posterior and prior
- Variational message passing (forward/backward)
- Parameter updates from data
- Parameter updates from probability distributions

All operations support batch processing and GPU acceleration.
"""

from .base import Conjugate
from .mvn import MultivariateNormal
from .multinomial import Multinomial

__all__ = [
    "Conjugate",
    "MultivariateNormal",
    "Multinomial",
]

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
CUDA-accelerated transform distributions.

This module provides GPU-accelerated implementations of conditional distributions
that model p(y|x,θ), useful for building complex probabilistic models.

Available Transforms:
- LinearMatrixNormalGamma: Linear transformation with Matrix-Normal-Gamma prior
  Models: y = Ax + ε with Bayesian treatment of A and noise covariance

Each transform implements:
- Forward message passing (predictive distribution)
- Backward message passing (input inference)
- Variational forward/backward approximations
- Parameter updates from data or distributions
- ELBO computation
- KL divergence computation

All transforms support GPU acceleration and batch processing.
"""

from .base import Transform
from .linear_mng import LinearMatrixNormalGamma

__all__ = [
    "Transform",
    "LinearMatrixNormalGamma",
]

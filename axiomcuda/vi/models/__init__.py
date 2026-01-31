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
CUDA-accelerated probabilistic models.

This module provides GPU-accelerated implementations of probabilistic models
for variational inference.

Available Models:
- Mixture: Standard mixture model with EM algorithm
- HybridMixture: Mixture with both discrete and continuous likelihoods
- SlotMixtureModel: Latent attention model with slots

Each model implements:
- update_from_data: Fit model to observed data
- update_from_probabilities: Fit model to distributions
- ELBO computation
- Assignment inference
- Prediction methods

All models support GPU batch processing and JIT compilation.
"""

from .base import Model
from .mixture import Mixture
from .hybrid_mixture_model import HybridMixture
from .slot_mixture_model import SlotMixtureModel

__all__ = [
    "Model",
    "Mixture",
    "HybridMixture",
    "SlotMixtureModel",
]

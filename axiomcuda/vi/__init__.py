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
CUDA-accelerated Variational Inference module.

This module provides GPU-accelerated implementations of variational inference
algorithms, wrapping C++ CUDA kernels for high-performance computation.

The module structure mirrors the original axiom/vi/ module but with CUDA-accelerated
backends for computationally intensive operations:

- Distribution: Base class for all probability distributions
- ExponentialFamily: Base class for exponential family distributions
- Conjugate: Base class for conjugate prior distributions
- Transform: Base class for conditional distributions p(y|x,θ)
- Model: Base class for probabilistic models

Key Features:
- GPU-accelerated matrix operations (inverse, logdet, etc.)
- CUDA kernels for sampling from distributions
- Accelerated message passing operations
- Batch processing for large-scale inference
- Compatible with JAX's JIT compilation and automatic differentiation

Example Usage:
    >>> from axiomcuda.vi import MultivariateNormal, Multinomial
    >>> from axiomcuda.vi import Mixture, LinearMatrixNormalGamma
    >>> 
    >>> # Create distributions
    >>> mvn = MultivariateNormal(...)
    >>> cat = Multinomial(...)
    >>> 
    >>> # Use in mixture models
    >>> mixture = Mixture(likelihood=mvn, prior=cat)
    >>> 
    >>> # Fit to data
    >>> posterior, elbo = mixture.update_from_data(data, iters=10)
"""

from .utils import ArrayDict, utils
from .distribution import Distribution, Delta
from .exponential import (
    ExponentialFamily,
    MultivariateNormal,
    MultivariateNormalPositiveXXT,
    Multinomial,
    Linear,
    MixtureMessage,
)
from .conjugate import (
    Conjugate,
    MultivariateNormal as MultivariateNormalConjugate,
    Multinomial as MultinomialConjugate,
)
from .transforms import Transform, LinearMatrixNormalGamma
from .models import Model, Mixture, HybridMixture, SlotMixtureModel

__version__ = "0.1.0"
__all__ = [
    # Base classes
    "Distribution",
    "Delta",
    "ExponentialFamily",
    "Conjugate",
    "Transform",
    "Model",
    # Exponential family
    "MultivariateNormal",
    "MultivariateNormalPositiveXXT",
    "Multinomial",
    "Linear",
    "MixtureMessage",
    # Conjugate priors
    "MultivariateNormalConjugate",
    "MultinomialConjugate",
    # Transforms
    "LinearMatrixNormalGamma",
    # Models
    "Mixture",
    "HybridMixture",
    "SlotMixtureModel",
    # Utilities
    "ArrayDict",
    "utils",
]

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

from typing import Optional, List, Tuple, Union
import numpy as np

from axiomcuda.vi.utils import params_to_tx, ArrayDict, inv_and_logdet, bdot
from .base import ExponentialFamily

# Import the C++ backend - this is the ONLY backend we use
try:
    import axiomcuda_backend as backend
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False
    raise RuntimeError(
        "AXIOMCUDA C++ backend not found. Please build the C++ extensions:\n"
        "  pip install -e .\n"
        "The C++ backend is required - there is no JAX fallback."
    )

DEFAULT_EVENT_DIM = 2


@params_to_tx({"inv_sigma_mu": "x", "inv_sigma": "minus_half_xxT"})
class MultivariateNormal(ExponentialFamily):
    """
    Multivariate Normal (Gaussian) distribution - CUDA accelerated.
    
    Uses C++ backend for:
    - Matrix inversions and log-determinants
    - Batch matrix operations
    - Sampling from multivariate normal
    """

    _mu: np.ndarray
    _sigma: np.ndarray
    _logdet_inv_sigma: np.ndarray
    _cache_update_functions: List[Tuple]

    pytree_data_fields = ("_sigma", "_mu", "_logdet_inv_sigma")
    pytree_aux_fields = ("_cache_update_functions",)

    def __init__(
        self,
        nat_params: Optional[ArrayDict] = None,
        batch_shape: Optional[tuple] = None,
        event_shape: Optional[tuple] = None,
        event_dim: Optional[int] = DEFAULT_EVENT_DIM,
        init_key: Optional[np.ndarray] = None,
        scale: float = 1.0,
        cache_to_compute: Union[str, Optional[List[str]]] = "all",
        **parent_kwargs,
    ):
        if event_shape is not None:
            assert len(event_shape) == event_dim, "event_shape must have length equal to event_dim"

        if nat_params is None and "expectations" not in parent_kwargs:
            # Initialize with default params using numpy
            nat_params = self.init_default_params(batch_shape, event_shape, scale, DEFAULT_EVENT_DIM)

        if nat_params is not None:
            inferred_batch_shape, inferred_event_shape = self.infer_shapes(nat_params.inv_sigma_mu, event_dim)
        elif "expectations" in parent_kwargs:
            inferred_batch_shape, inferred_event_shape = self.infer_shapes(parent_kwargs["expectations"].x, event_dim)

        batch_shape = batch_shape if batch_shape is not None else inferred_batch_shape
        event_shape = event_shape if event_shape is not None else inferred_event_shape

        super().__init__(DEFAULT_EVENT_DIM, batch_shape, event_shape, nat_params=nat_params, **parent_kwargs)

        if cache_to_compute == "all" or isinstance(cache_to_compute, list) and len(cache_to_compute) == 0:
            cache_attrs = ["sigma", "mu", "logdet_inv_sigma"]
        else:
            if isinstance(cache_to_compute, str):
                cache_attrs = [cache_to_compute]
            else:
                cache_attrs = cache_to_compute.copy()

        self._cache_update_functions = self._get_cache_update_functions(cache_attrs)
        self._reset_cache()

        if nat_params is not None:
            self._validate_nat_params(nat_params)

    @staticmethod
    def init_default_params(batch_shape, event_shape, scale: float = 1.0, default_event_dim: int = 2) -> ArrayDict:
        """Initialize default canonical parameters using numpy."""
        dim = event_shape[-default_event_dim]
        # Use numpy for random initialization
        inv_sigma_mu = np.zeros(batch_shape + event_shape)
        inv_sigma = np.broadcast_to(scale * np.eye(dim), batch_shape + event_shape[:-default_event_dim] + (dim, dim))
        return ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma)

    @property
    def mu(self) -> np.ndarray:
        """Returns the mean parameter."""
        if self._mu is None:
            self._mu = self.compute_mu()
        return self._mu

    @property
    def inv_sigma(self) -> np.ndarray:
        """Returns the inverse covariance."""
        return self.nat_params.inv_sigma

    @property
    def inv_sigma_mu(self) -> np.ndarray:
        """Returns the precision-weighted mean."""
        return self.nat_params.inv_sigma_mu

    @property
    def logdet_inv_sigma(self) -> np.ndarray:
        """Returns the log determinant of inverse covariance."""
        if self._logdet_inv_sigma is None:
            self._logdet_inv_sigma = self.compute_logdet_inv_sigma()
        return self._logdet_inv_sigma

    @property
    def mu_inv_sigma_mu(self) -> np.ndarray:
        return (self.inv_sigma_mu * self.mu).sum((-2, -1), keepdims=True)

    @property
    def sigma(self) -> np.ndarray:
        """Returns the covariance."""
        if self._sigma is None:
            self._sigma, self._logdet_inv_sigma = self.compute_sigma_and_logdet_inv_sigma()
        return self._sigma

    @property
    def mean(self) -> np.ndarray:
        """Returns the mean (alias for mu)."""
        return self.mu

    def statistics(self, x: np.ndarray) -> ArrayDict:
        """Returns sufficient statistics T(x): [x, -0.5 * xxT]"""
        return ArrayDict(x=x, minus_half_xxT=-0.5 * x @ x.T)

    def log_measure(self, x: np.ndarray) -> np.ndarray:
        return -0.5 * self.dim * np.log(2 * np.pi)

    def expected_log_measure(self) -> np.ndarray:
        return self.log_measure(np.array(0.0))

    def expected_statistics(self) -> ArrayDict:
        """Computes expected sufficient statistics <T(x)>."""
        minus_half_xxT = -0.5 * self.expected_xx()
        return ArrayDict(x=self.mu, minus_half_xxT=minus_half_xxT)

    def log_partition(self) -> np.ndarray:
        """Computes log partition function with CUDA acceleration."""
        term1 = 0.5 * bdot(self.mu.T, self.inv_sigma_mu)
        term2 = -0.5 * self.logdet_inv_sigma
        return self.sum_default_events(term1 + term2, keepdims=True)

    def expected_x(self) -> np.ndarray:
        """Computes <x>."""
        return self.mu

    def expected_xx(self) -> np.ndarray:
        """Computes <xxT>."""
        return self.sigma + bdot(self.mu, self.mu.T)

    def sample(self, key, shape=()) -> np.ndarray:
        """Draw samples using CUDA-accelerated RNG via backend."""
        custom_event_shape = self.event_shape[: -self.default_event_dim]
        shape = shape + self.batch_shape + custom_event_shape
        return backend.sample_multivariate_normal(key, mean=self.mu.squeeze(), cov=self.sigma, shape=shape)

    @staticmethod
    def params_from_statistics(stats: ArrayDict) -> ArrayDict:
        """Computes natural parameters from expectations."""
        exp_xx = -2 * stats.minus_half_xxT
        mu = stats.x
        covariance = exp_xx - bdot(mu, mu.T)
        inv_sigma, _ = inv_and_logdet(covariance)
        inv_sigma_mu = bdot(inv_sigma, mu)
        return ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma)

    def _reset_cache(self):
        self._sigma, self._logdet_inv_sigma = self.compute_sigma_and_logdet_inv_sigma()
        self._mu = self.compute_mu()

    def compute_sigma_and_logdet_inv_sigma(self):
        """Compute inverse of precision matrix and its logdet - uses CUDA."""
        return inv_and_logdet(self.inv_sigma)

    def compute_logdet_inv_sigma(self):
        """Compute logdet of precision matrix - uses CUDA."""
        return inv_and_logdet(self.inv_sigma, return_inverse=False)

    def compute_mu(self):
        """Compute mean from natural parameters."""
        return bdot(self.sigma, self.inv_sigma_mu)

    def compute_sigma(self):
        """Compute covariance from precision matrix."""
        return inv_and_logdet(self.inv_sigma, return_logdet=False)

    def _entropy(self):
        """Computes entropy."""
        return 0.5 * (self.dim * (1 + np.log(2 * np.pi)) - self.logdet_inv_sigma)

    def _order_cache_computations(self, cache_attrs):
        """Orders cache computations based on dependencies."""
        ordered_cache_attrs = []
        if "sigma" in cache_attrs:
            ordered_cache_attrs.append("sigma")
            cache_attrs.remove("sigma")
        if "mu" in cache_attrs:
            if "sigma" not in ordered_cache_attrs:
                ordered_cache_attrs.append("sigma")
            ordered_cache_attrs.append("mu")
            cache_attrs.remove("mu")
        ordered_cache_attrs.extend(cache_attrs)
        return ordered_cache_attrs

    def _get_cache_update_functions(self, cache_attrs):
        """Returns method names for updating each cache attribute."""
        ordered_cache_attrs = self._order_cache_computations(cache_attrs)
        method_names = []
        for attr in ordered_cache_attrs:
            if hasattr(self, f"compute_{attr}"):
                method_names.append((attr, f"compute_{attr}"))
        return method_names

    def _update_cache(self):
        """Dynamically calls cache update methods."""
        for attr_name, method_name in self._cache_update_functions:
            method = getattr(self, method_name)
            setattr(self, f"_{attr_name}", method())

    def shift(self, deltax):
        inv_sigma_mu = self.inv_sigma_mu + self.inv_sigma @ deltax
        inv_sigma = self.inv_sigma
        residual = self.residual
        return MultivariateNormal(ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma), residual=residual)


@params_to_tx({"inv_sigma_mu": "x", "inv_sigma": "xxT"})
class MultivariateNormalPositiveXXT(MultivariateNormal):
    """
    Multivariate Normal with xxT sufficient statistic - CUDA accelerated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def statistics(self, x: np.ndarray) -> ArrayDict:
        """Returns sufficient statistics T(x): [x, xxT]"""
        return ArrayDict(x=x, xxT=bdot(x, x.T))

    def expected_statistics(self) -> ArrayDict:
        """Computes expected sufficient statistics <T(x)>."""
        xxT = self.expected_xx()
        return ArrayDict(x=self.mu, xxT=xxT)

    def expected_x(self) -> np.ndarray:
        return self.mu

    def expected_xx(self) -> np.ndarray:
        return self.sigma + self.mu @ self.mu.T

    def sample(self, key, shape=()) -> np.ndarray:
        custom_event_shape = self.event_shape[: -self.default_event_dim]
        shape = shape + self.batch_shape + custom_event_shape
        return backend.sample_multivariate_normal(key, mean=self.mu.squeeze(), cov=self.sigma, shape=shape)

    @staticmethod
    def params_from_statistics(stats: ArrayDict) -> ArrayDict:
        """Computes natural parameters from expectations of [x, xxT]."""
        expected_outer_xx = stats.xxT
        outer_product = stats.x @ stats.x.T
        covariance = expected_outer_xx - outer_product
        inv_sigma = np.linalg.inv(covariance)
        inv_sigma_mu = inv_sigma @ stats.x
        return ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma)

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
from scipy.special import digamma

from ..distribution import Distribution
from ..exponential import MultivariateNormal as MultivariateNormalLikelihood
from .base import Conjugate
from ..utils import mvgammaln, mvdigamma, params_to_tx, ArrayDict
from ..utils import inv_and_logdet, bdot

DEFAULT_EVENT_DIM = 2


@params_to_tx({"eta_1": "x", "eta_2": "minus_half_xxT"})
class MultivariateNormal(Conjugate):
    """
    Normal-Inverse-Wishart conjugate prior - CUDA accelerated.
    
    Uses C++ backend for:
    - Matrix operations (inverse, logdet)
    - Multivariate gamma and digamma functions
    - Posterior parameter updates
    """

    _u: np.ndarray
    _logdet_inv_u: np.ndarray
    _prior_logdet_inv_u: np.ndarray

    pytree_data_fields = ("_u", "_logdet_inv_u", "_prior_logdet_inv_u")
    pytree_aux_fields = ("fixed_precision",)

    def __init__(
        self,
        params: Optional[ArrayDict] = None,
        prior_params: Optional[ArrayDict] = None,
        batch_shape: Optional[tuple] = None,
        event_shape: Optional[tuple] = None,
        event_dim: Optional[int] = DEFAULT_EVENT_DIM,
        fixed_precision: bool = False,
        scale: float = 1.0,
        dof_offset: float = 0.0,
        init_key: Optional[np.ndarray] = None,
    ):
        if event_shape is not None:
            assert len(event_shape) == event_dim, "event_shape must have length equal to event_dim"

        if prior_params is None:
            assert dof_offset >= 0.0, "dof_offset must be non-negative"
            prior_params = self.init_default_params(batch_shape, event_shape, scale, dof_offset, DEFAULT_EVENT_DIM)
        if params is None:
            # Initialize params with slight random perturbation from prior
            params = {}
            for k, v in prior_params.items():
                if k == "mean":
                    params[k] = v + np.random.normal(0, 0.1, v.shape)
                else:
                    params[k] = v
            params = ArrayDict(**params)

        inferred_batch_shape, inferred_event_shape = self.infer_shapes(params.mean, event_dim)
        batch_shape = batch_shape if batch_shape is not None else inferred_batch_shape
        event_shape = event_shape if event_shape is not None else inferred_event_shape

        self.fixed_precision = fixed_precision

        super().__init__(
            DEFAULT_EVENT_DIM,
            MultivariateNormalLikelihood,
            params,
            prior_params,
            batch_shape,
            event_shape,
        )

        _prior_logdet_inv_u = np.linalg.slogdet(self.prior_inv_u)[1]
        self._prior_logdet_inv_u = self.expand_default_event_dims(_prior_logdet_inv_u)

    @staticmethod
    def init_default_params(batch_shape, event_shape, scale: float = 1.0, dof_offset: float = 0.0, default_event_dim=2):
        """Initialize default canonical parameters."""
        dim = event_shape[-default_event_dim]
        mean = np.zeros(batch_shape + event_shape)
        kappa = np.full(batch_shape + event_shape[:-default_event_dim] + (1, 1), 1.0)
        u = (scale**2) * np.broadcast_to(
            np.eye(dim),
            batch_shape + event_shape[:-default_event_dim] + (dim, dim),
        )
        n = np.full(batch_shape + event_shape[:-default_event_dim] + (1, 1), 1.0 + dim + dof_offset)
        return ArrayDict(mean=mean, kappa=kappa, u=u, n=n)

    @property
    def sqrt_diag_norm(self) -> np.ndarray:
        diag = np.diagonal(np.abs(self.posterior_params.eta.eta_2), axis1=-2, axis2=-1)
        diag = np.maximum(diag, 1.0)
        return np.sqrt(diag)

    @property
    def norm(self) -> np.ndarray:
        sqrt_diag = self.sqrt_diag_norm
        norm = bdot(sqrt_diag[..., :, None], sqrt_diag[..., None, :])
        return norm

    @property
    def mean(self):
        return self.posterior_params.eta.eta_1 / self.posterior_params.nu.nu_1

    @property
    def prior_mean(self):
        return self.prior_params.eta.eta_1 / self.prior_params.nu.nu_1

    @property
    def kappa(self):
        return self.posterior_params.nu.nu_1

    @property
    def prior_kappa(self):
        return self.prior_params.nu.nu_1

    @property
    def n(self):
        if self.fixed_precision:
            return self.prior_n
        else:
            return self.posterior_params.nu.nu_2 + self.dim

    @property
    def prior_n(self):
        return self.prior_params.nu.nu_2 + self.dim

    @property
    def _scaled_inv_u(self):
        norm = self.norm
        sqrt_diag_norm = self.sqrt_diag_norm
        scaled_eta_2 = self.posterior_params.eta.eta_2 / norm
        kappa = self.posterior_params.nu.nu_1
        scaled_eta_1 = self.posterior_params.eta.eta_1 / sqrt_diag_norm[..., None]
        scaled_inv_u = -2 * scaled_eta_2 - (1.0 / kappa) * bdot(scaled_eta_1, scaled_eta_1.T)
        clipped_diag = np.diagonal(scaled_inv_u, axis1=-2, axis2=-1).clip(min=1e-3)
        # Clip diagonal
        for i in range(scaled_inv_u.shape[-1]):
            scaled_inv_u[..., i, i] = clipped_diag[..., i]
        return scaled_inv_u

    @property
    def inv_u(self):
        if self.fixed_precision:
            return self.prior_inv_u
        else:
            return self._scaled_inv_u * self.norm

    @property
    def prior_inv_u(self):
        return -2 * self.prior_params.eta.eta_2 - (1.0 / self.prior_params.nu.nu_1) * bdot(
            self.prior_params.eta.eta_1, self.prior_params.eta.eta_1.T
        )

    @property
    def u(self):
        if self._u is None:
            if self.fixed_precision:
                self._u, self._logdet_inv_u = inv_and_logdet(self.prior_inv_u)
            else:
                _scaled_u, _logdet_inv_u_scaled = inv_and_logdet(self._scaled_inv_u)
                self._u = _scaled_u / self.norm
                log_diag = np.log(np.diagonal(self.norm, axis1=-1, axis2=-2)).sum(axis=-1)
                self._logdet_inv_u = _logdet_inv_u_scaled + np.expand_dims(log_diag, (-1, -2))
        return self._u

    @property
    def logdet_inv_u(self):
        if self._logdet_inv_u is None:
            _logdet_inv_u_scaled = inv_and_logdet(self._scaled_inv_u, return_inverse=False)
            log_diag = np.log(np.diagonal(self.norm, axis1=-1, axis2=-2)).sum(axis=-1)
            self._logdet_inv_u = _logdet_inv_u_scaled + np.expand_dims(log_diag, (-1, -2))
        return self._logdet_inv_u

    @property
    def prior_logdet_inv_u(self):
        if self._prior_logdet_inv_u is None:
            self._prior_logdet_inv_u = inv_and_logdet(self.prior_inv_u, return_inverse=False)
        return self._prior_logdet_inv_u

    def to_natural_params(self, params: ArrayDict) -> ArrayDict:
        """Converts canonical parameters to natural parameters."""
        eta_1 = params.mean * params.kappa
        eta_2 = -0.5 * (inv_and_logdet(params.u, return_logdet=False) + bdot(eta_1, params.mean.T))
        nu_1 = params.kappa
        nu_2 = params.n - self.dim
        eta = ArrayDict(eta_1=eta_1, eta_2=eta_2)
        nu = ArrayDict(nu_1=nu_1, nu_2=nu_2)
        return ArrayDict(eta=eta, nu=nu)

    def expected_likelihood_params(self) -> ArrayDict:
        """Returns expected natural parameters <S(θ)>."""
        inv_sigma_mu = self.expected_inv_sigma_mu()
        inv_sigma = self.expected_inv_sigma()
        return ArrayDict(eta_1=inv_sigma_mu, eta_2=inv_sigma)

    def expected_posterior_statistics(self) -> ArrayDict:
        """Computes expected sufficient statistics."""
        eta_stats = self.expected_likelihood_params()
        nu_stats = ArrayDict(nu_1=-self.expected_log_partition().nu_1, nu_2=-self.expected_log_partition().nu_2)
        return ArrayDict(eta=eta_stats, nu=nu_stats)

    def expected_log_partition(self) -> ArrayDict:
        """Computes expected log partition <A(θ)>."""
        nu1_term = 0.5 * self.expected_mu_inv_sigma_mu()
        nu2_term = -0.5 * self.expected_logdet_inv_sigma()
        return ArrayDict(nu_1=nu1_term, nu_2=nu2_term)

    def log_prior_partition(self) -> np.ndarray:
        """Computes log partition of prior."""
        return self._log_partition(self.prior_kappa, self.prior_n, self.prior_logdet_inv_u)

    def log_posterior_partition(self) -> np.ndarray:
        """Computes log partition of posterior."""
        return self._log_partition(self.kappa, self.n, self.logdet_inv_u)

    def _log_partition(self, kappa: np.ndarray, n: np.ndarray, logdet_inv_u: np.ndarray) -> np.ndarray:
        half_dim = 0.5 * self.dim
        term_1 = -half_dim * np.log(kappa)
        term_2 = -0.5 * n * logdet_inv_u
        term_3 = half_dim * (np.log(2 * np.pi) + n * np.log(2))
        term_4 = mvgammaln(n / 2.0, self.dim)
        return term_1 + term_2 + term_3 + term_4

    def expected_logdet_inv_sigma(self) -> np.ndarray:
        return self.dim * np.log(2) - self.logdet_inv_u + mvdigamma(self.n / 2.0, self.dim)

    def logdet_expected_inv_sigma(self):
        return -self.logdet_inv_u + self.dim * np.log(self.n)

    def variational_residual(self):
        return 0.5 * (
            self.dim * (np.log(2) - np.log(self.n) - 1.0 / self.kappa)
            + mvdigamma(self.n / 2.0, self.dim)
        ).squeeze((-2, -1))

    def collapsed_residual(self):
        return self.variational_residual()

    def update_from_probabilities(self, pX: Distribution, weights: Optional[np.ndarray] = None, **kwargs):
        """Update parameters given expected sufficient statistics."""
        sample_shape = pX.shape[: -self.event_dim - self.batch_dim]
        sample_dims = tuple(range(len(sample_shape)))

        if weights is None:
            SEx = pX.expected_x().sum(sample_dims)
            SExx = pX.expected_xx().sum(sample_dims)
            N = np.broadcast_to(np.prod(np.array(sample_shape)), SEx.shape[:-2] + (1, 1))
        else:
            weights = self.expand_event_dims(weights)
            SEx = (weights * pX.expected_x()).sum(sample_dims)
            SExx = (weights * pX.expected_xx()).sum(sample_dims)
            N = weights.sum(sample_dims)

        summed_stats = ArrayDict(
            eta=ArrayDict(eta_1=SEx, eta_2=-0.5 * SExx),
            nu=ArrayDict(nu_1=N, nu_2=N),
        )
        lr = kwargs.get("lr", 1.0)
        beta = kwargs.get("beta", 0.0)
        self.update_from_statistics(summed_stats, lr=lr, beta=beta)

    def expected_inv_sigma(self) -> np.ndarray:
        """Compute expected inverse covariance E[Σ⁻¹] = nU."""
        return self.u * self.n

    def expected_inv_sigma_mu(self) -> np.ndarray:
        """Compute E[Σ⁻¹μ] = κUM."""
        return bdot(self.expected_inv_sigma(), self.mean)

    def expected_sigma(self) -> np.ndarray:
        """Compute expected covariance E[Σ] = U⁻¹/(n-d-1)."""
        return self.inv_u / (self.n - self.dim - 1)

    def inv_expected_inv_sigma(self) -> np.ndarray:
        return self.inv_u / self.n

    def expected_mu_inv_sigma_mu(self) -> np.ndarray:
        """Compute E[μᵀΣ⁻¹μ]."""
        return bdot(self.mean.T, bdot(self.expected_inv_sigma(), self.mean)) + self.dim / self.kappa

    def expected_xx(self) -> np.ndarray:
        """Compute E[xxT] = Σ + μμT."""
        return self.expected_sigma() + bdot(self.mean, self.mean.T)

    def _update_cache(self):
        """Update scale matrix and logdet."""
        if self.fixed_precision:
            self._u, self._logdet_inv_u = inv_and_logdet(self.prior_inv_u)
        else:
            norm = self.norm
            _scaled_u, _logdet_inv_u_scaled = inv_and_logdet(self._scaled_inv_u)
            self._u = _scaled_u / norm
            log_diag = np.log(np.diagonal(self.norm, axis1=-1, axis2=-2)).sum(axis=-1)
            self._logdet_inv_u = _logdet_inv_u_scaled + np.expand_dims(log_diag, (-1, -2))

    def _kl_divergence(self) -> np.ndarray:
        """Computes KL divergence between prior and posterior."""
        kl = 0.5 * (self.prior_kappa / self.kappa - 1 + np.log(self.kappa / self.prior_kappa)) * self.dim
        pred_error = self.mean - self.prior_mean
        kl = kl + 0.5 * self.prior_kappa * bdot(pred_error.T, bdot(self.expected_inv_sigma(), pred_error))
        kl = kl + self.kl_divergence_wishart()
        return self.sum_events(kl)

    def kl_divergence_wishart(self) -> np.ndarray:
        """Compute KL divergence between posterior and prior Wishart."""
        kl = self.prior_n / 2.0 * (self.logdet_inv_u - self.prior_logdet_inv_u)
        kl = kl + self.n / 2.0 * (self.prior_inv_u * self.u).sum((-2, -1), keepdims=True)
        kl = kl - self.n * self.dim / 2.0
        kl = kl + mvgammaln(self.prior_n / 2.0, self.dim) - mvgammaln(self.n / 2.0, self.dim)
        kl = kl + (self.n - self.prior_n) / 2.0 * mvdigamma(self.n / 2.0, self.dim)
        return kl

    def _expected_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Computes expected log likelihood."""
        diff = x - self.mean
        tx_dot_stheta = -0.5 * bdot(diff.T, bdot(self.expected_inv_sigma(), diff))
        atheta_1 = -0.5 * self.dim / self.kappa
        atheta_2 = 0.5 * self.expected_logdet_inv_sigma()
        log_base_measure = -0.5 * self.dim * np.log(2 * np.pi)
        negative_expected_atheta = atheta_1 + atheta_2
        return self.sum_events(log_base_measure + tx_dot_stheta + negative_expected_atheta)

    def expected_log_likelihood(self, data: np.ndarray) -> np.ndarray:
        return self._expected_log_likelihood(data)

    def average_energy(self, x: Distribution):
        """Computes average energy term."""
        expected_x = x.expected_x()
        expected_xx = x.expected_xx()
        sigma = expected_xx - bdot(expected_x, expected_x.T)
        diff = expected_x - self.mean
        exp_inv_sigma = self.expected_inv_sigma()
        energy = -0.5 * np.sum(exp_inv_sigma * sigma, (-2, -1), keepdims=True)
        energy -= 0.5 * bdot(diff.T, bdot(exp_inv_sigma, diff))
        energy -= 0.5 * self.dim / self.kappa
        energy += 0.5 * self.expected_logdet_inv_sigma()
        energy -= 0.5 * self.dim * np.log(2 * np.pi)
        return self.sum_events(energy)

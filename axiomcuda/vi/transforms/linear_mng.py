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

from typing import Optional, Tuple
from jaxtyping import Array

from jax.lax import lgamma
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util
from jax.scipy import special as jsp
from jax.numpy import expand_dims as expand

from axiomcuda.vi.distribution import Distribution
from axiomcuda.vi.exponential import Linear as LinearLikelihood
from axiomcuda.vi.transforms.base import Transform
from axiomcuda.vi.exponential import MultivariateNormal, ExponentialFamily, MixtureMessage
from axiomcuda.vi.utils import params_to_tx, ArrayDict, bdot, tree_at, inv_and_logdet

DEFAULT_EVENT_DIM = 2


@params_to_tx({"eta_1": "xx", "eta_2": "yx", "eta_3": "yy", "eta_4": "ones"})
class LinearMatrixNormalGamma(Transform):
    """
    Linear Matrix-Normal-Gamma distribution - CUDA accelerated.
    
    Models y = Ax + ε with Matrix-Normal-Gamma prior on A and Σ.
    Wraps C++ CUDA kernels for efficient message passing.
    """

    _v: Array
    _logdet_inv_v: Array
    _b: Array
    _prior_logdet_inv_v: Array
    _prior_v: Array

    x_dim: int
    y_dim: int

    pytree_data_fields = ("_v", "_logdet_inv_v", "_b", "_prior_logdet_inv_v", "_prior_v", "_prior_b")
    pytree_aux_fields = ("x_dim", "y_dim", "use_bias", "fixed_precision", "trivial_batch_axes")

    def __init__(
        self,
        params: ArrayDict = None,
        prior_params: ArrayDict = None,
        event_dim: int = DEFAULT_EVENT_DIM,
        use_bias: bool = True,
        fixed_precision: bool = False,
        scale: float = 1.0,
        dof_offset: float = 1.0,
        inv_v_scale: float = 1.0,
        batch_shape: Optional[Tuple[int]] = None,
        event_shape: Optional[Tuple[int]] = None,
        init_key=None,
    ):
        if params is not None:
            self.x_dim, self.y_dim = params.mu.shape[-(DEFAULT_EVENT_DIM - 1)], params.mu.shape[-DEFAULT_EVENT_DIM]
        elif prior_params is not None:
            self.x_dim, self.y_dim = prior_params.mu.shape[-(DEFAULT_EVENT_DIM - 1)], prior_params.mu.shape[-DEFAULT_EVENT_DIM]
        elif event_shape is not None:
            self.x_dim, self.y_dim = event_shape[-(DEFAULT_EVENT_DIM - 1)], event_shape[-DEFAULT_EVENT_DIM]

        self.use_bias = use_bias
        self.fixed_precision = fixed_precision

        if prior_params is None:
            prior_params = self.init_default_params(batch_shape, event_shape, self.x_dim, self.y_dim, scale, dof_offset, inv_v_scale, DEFAULT_EVENT_DIM)

        if fixed_precision:
            self._prior_v = jnp.linalg.inv(prior_params.inv_v)
        self._prior_b = prior_params.b

        if params is None:
            key = init_key if init_key is not None else jr.PRNGKey(0)
            s = scale / jnp.sqrt(self.x_dim)
            mu = prior_params.mu + jr.uniform(key, batch_shape + event_shape[:-event_dim] + (self.y_dim, self.x_dim), minval=-3 * s, maxval=3 * s)
            inv_v = jnp.where(prior_params.inv_v > 0, 1.0, 0.0)
            a = jnp.ones_like(prior_params.a) * 2.0
            b = jnp.ones_like(prior_params.b)
            params = tree_at(lambda x: (x.mu, x.inv_v, x.a, x.b), prior_params, (mu, inv_v, a, b))

        inferred_batch_shape, inferred_event_shape = self.infer_shapes(params.mu, event_dim)
        if batch_shape is None:
            batch_shape = inferred_batch_shape
        if event_shape is None:
            event_shape = inferred_event_shape

        self._prior_b = prior_params.b
        super().__init__(DEFAULT_EVENT_DIM, LinearLikelihood, params, prior_params, batch_shape, event_shape)
        self._prior_v, self._prior_logdet_inv_v = inv_and_logdet(self.prior_inv_v)
        self.trivial_batch_axes = tuple([i for i, d in enumerate(batch_shape) if d == 1])

    @staticmethod
    def init_default_params(batch_shape, event_shape, x_dim, y_dim, scale=1.0, dof_offset=1.0, inv_v_scale=1.0, default_event_dim=2):
        """Initialize default parameters."""
        prior_mu = jnp.full(batch_shape + event_shape[:-default_event_dim] + (y_dim, x_dim), 0.0)
        prior_inv_v = inv_v_scale * jnp.broadcast_to(jnp.eye(x_dim), batch_shape + event_shape[:-default_event_dim] + (x_dim, x_dim))
        prior_a = jnp.full(batch_shape + event_shape[:-default_event_dim] + (y_dim, 1), 1.0 + dof_offset)
        prior_b = scale**2 * jnp.broadcast_to(jnp.ones((y_dim, 1)), batch_shape + event_shape[:-default_event_dim] + (y_dim, 1))
        return ArrayDict(mu=prior_mu, inv_v=prior_inv_v, a=prior_a, b=prior_b)

    @property
    def norm(self):
        return jnp.expand_dims(self.inv_v.reshape(self.batch_shape + (-1,)).max(-1), (-1, -2))

    @property
    def mu(self):
        return bdot(self.posterior_params.eta.eta_2, self.v)

    @property
    def inv_v(self):
        return self.posterior_params.eta.eta_1

    @property
    def b(self):
        if self.fixed_precision:
            return self.prior_b
        if self._b is None:
            norm = self.norm
            rho = jnp.diagonal(bdot(self.mu, self.posterior_params.eta.eta_2.mT / norm), axis1=-1, axis2=-2)
            self._b = (self.posterior_params.eta.eta_3 / norm - expand(rho, -1)) * norm / 2
        return jnp.clip(self._b, min=1e-6)

    @property
    def a(self):
        if self.fixed_precision:
            return self.prior_a
        else:
            return (self.posterior_params.eta.eta_4 - self.x_dim) / 2 + 1

    @property
    def prior_mu(self):
        return bdot(self.prior_params.eta.eta_2, self.prior_v)

    @property
    def prior_inv_v(self):
        return self.prior_params.eta.eta_1

    @property
    def prior_b(self):
        return self._prior_b

    @property
    def prior_a(self):
        return (self.prior_params.eta.eta_4 - self.x_dim) / 2 + 1

    @property
    def v(self):
        if self._v is None:
            norm = self.norm
            scaled_v, scaled_logdet_inv_v = inv_and_logdet(self.inv_v / norm)
            self._v = scaled_v / norm
            self._logdet_inv_v = scaled_logdet_inv_v + self.x_dim * jnp.log(norm)
        return self._v

    @property
    def prior_v(self):
        if self._prior_v is None:
            self._prior_v, self._prior_logdet_inv_v = inv_and_logdet(self.prior_inv_v)
        return self._prior_v

    @property
    def logdet_inv_v(self):
        if self._logdet_inv_v is None:
            norm = self.norm
            self._logdet_inv_v = inv_and_logdet(self.inv_v / norm, return_inverse=False) + self.x_dim * jnp.log(norm)
        return self._logdet_inv_v

    @property
    def prior_logdet_inv_v(self):
        if self._prior_logdet_inv_v is None:
            self._prior_logdet_inv_v = inv_and_logdet(self.prior_inv_v, return_inverse=False)
        return self._prior_logdet_inv_v

    @property
    def weights(self):
        return self.mu[..., :-1] if self.use_bias else self.mu

    @property
    def bias(self):
        return self.mu[..., -1:] if self.use_bias else jnp.broadcast_to(jnp.zeros(1), self.mu.shape[:-1] + (1,))

    def update_from_probabilities(self, pXY, weights=None, lr=1.0, beta=0.0, apply_updates=True):
        """Custom update_from_probs with bias handling."""
        pX, pY = pXY
        if isinstance(pX, MixtureMessage):
            pX = pX.marginalize(keepdims=False)

        sample_shape = self.get_sample_shape(pX.mean)
        sample_dims = self.get_sample_dims(pX.mean)

        px_exp_xx = pX.expected_xx()
        px_exp_x = pX.expected_x()
        if self.use_bias:
            sigma = px_exp_xx - px_exp_x * px_exp_x.mT
            pad_width = [(0, 0)] * sigma.ndim
            pad_width[-1] = (0, 1)
            pad_width[-2] = (0, 1)
            sigma = jnp.pad(sigma, pad_width=pad_width)
            pad_width = [(0, 0)] * px_exp_x.ndim
            pad_width[-2] = (0, 1)
            px_exp_x = jnp.pad(px_exp_x, pad_width, constant_values=1.0)
            px_exp_xx = sigma + px_exp_x * px_exp_x.mT

        py_exp_x = pY.expected_x()
        py_exp_xx = pY.expected_xx()

        pX_batch_shape = self.get_batch_shape(pX.mean)
        pY_batch_shape = self.get_batch_shape(pY.mean)
        common_batch_shape = jnp.broadcast_shapes(pX_batch_shape, pY_batch_shape)

        weights = jnp.ones((1,) * len(sample_dims) + common_batch_shape) if weights is None else weights
        weights = self.expand_event_dims(weights)
        weights_batch_shape = self.get_batch_shape(weights)
        common_batch_shape = jnp.broadcast_shapes(common_batch_shape, weights_batch_shape)
        py_exp_x = py_exp_x * weights
        ones = jnp.broadcast_to(weights, sample_shape + weights.shape[len(sample_shape):])
        summed_stats = ArrayDict(
            xx=(px_exp_xx * weights).sum(sample_dims),
            yx=(py_exp_x * px_exp_x.mT).sum(sample_dims),
            yy=expand(jnp.diagonal(py_exp_xx * weights, axis1=-1, axis2=-2), -1).sum(sample_dims),
            ones=ones.sum(sample_dims),
        )

        summed_stats = tree_util.tree_map(lambda se: se.sum(self.trivial_batch_axes, keepdims=True), summed_stats)
        if apply_updates:
            self.update_from_statistics(self.map_stats_to_params(summed_stats, None), lr, beta)
        else:
            return self.map_stats_to_params(summed_stats, None)

    def update_from_data(self, data, weights=None, lr=1.0, beta=0.0):
        """Custom update_from_data with bias handling."""
        likelihood_stats = self.likelihood.statistics(data, has_bias=self.use_bias)
        yy = expand(jnp.diagonal(likelihood_stats.yy, axis1=-1, axis2=-2), -1)
        likelihood_stats = tree_at(lambda lls: lls.yy, likelihood_stats, yy)

        X, Y = data
        sample_shape = self.get_sample_shape(X)
        sample_dims = self.get_sample_dims(X)

        if weights is None:
            summed_stats = tree_util.tree_map(lambda x: x.sum(sample_dims), likelihood_stats, is_leaf=lambda x: isinstance(x, Array))
        else:
            weights = weights.reshape(weights.shape + self.event_dim * (1,))
            summed_stats = tree_util.tree_map(lambda x: (x * weights).sum(sample_dims), likelihood_stats, is_leaf=lambda x: isinstance(x, Array))

        self.update_from_statistics(self.map_stats_to_params(summed_stats, None), lr, beta)

    def to_natural_params(self, params: ArrayDict) -> ArrayDict:
        """Convert parameters to natural parameters."""
        norm = jnp.expand_dims(params.inv_v.reshape(self.batch_shape + (-1,)).max(-1), (-1, -2))
        eta_1 = params.inv_v
        _eta_2 = bdot(params.mu, params.inv_v / norm)
        rho = jnp.diagonal(bdot(params.mu, _eta_2.mT), axis1=-1, axis2=-2)
        eta_3 = (2 * params.b / norm + expand(rho, -1)) * norm
        eta_4 = 2 * (params.a - 1) + self.x_dim
        return ArrayDict(eta=ArrayDict(eta_1=eta_1, eta_2=_eta_2 * norm, eta_3=eta_3, eta_4=eta_4), nu=None)

    def expected_posterior_statistics(self) -> ArrayDict:
        """Computes expected sufficient statistics."""
        eta_stats = self.expected_likelihood_params()
        return ArrayDict(eta=eta_stats, nu=None)

    def expected_likelihood_params(self) -> ArrayDict:
        """Compute expected likelihood parameters."""
        inv_sigma = self.expected_inv_sigma()
        inv_sigma_x = self.expected_inv_sigma_x(inv_sigma)
        x_inv_sigma_x = self.expected_x_inv_sigma_x(inv_sigma_x)
        return ArrayDict(
            eta_1=-0.5 * x_inv_sigma_x,
            eta_2=inv_sigma_x,
            eta_3=-0.5 * jnp.diagonal(inv_sigma, axis1=-1, axis2=-2)[..., None],
            eta_4=0.5 * self.expected_logdet_inv_sigma(),
        )

    def expected_log_partition(self):
        raise NotImplementedError

    def log_prior_partition(self) -> Array:
        return self._log_partition(self.prior_mu, self.prior_logdet_inv_v, self.prior_a, self.prior_b)

    def log_posterior_partition(self) -> Array:
        return self._log_partition(self.mu, self.logdet_inv_v, self.a, self.b)

    def _log_partition(self, mean: Array, logdet_inv_v: Array, a: Array, b: Array) -> Array:
        d = self.y_dim
        p = self.x_dim
        term_1 = ((d * p) / 2) * jnp.log(2 * jnp.pi)
        term_2 = -(d / 2) * logdet_inv_v
        term_3 = (lgamma(a) - a * jnp.log(b)).sum((-2, -1), keepdims=True)
        return term_1 + term_2 + term_3

    def predict(self, x: Array) -> ExponentialFamily:
        """Computes variational prediction distribution."""
        if self.use_bias:
            inv_sigma_mu = bdot(self.expected_inv_sigma_x()[..., :-1], x) + self.expected_inv_sigma_x()[..., -1:]
            nat_params = ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=self.expected_inv_sigma())
            res = -0.5 * bdot(x.mT, bdot(self.expected_x_inv_sigma_x()[..., :-1, :-1], x))
            res = res - bdot(self.expected_x_inv_sigma_x()[..., -1:, :-1], x) - 0.5 * self.expected_x_inv_sigma_x()[..., -1:, -1:]
        else:
            nat_params = ArrayDict(inv_sigma_mu=bdot(self.expected_inv_sigma_x(), x), inv_sigma=self.expected_inv_sigma())
            res = -0.5 * bdot(x.mT, bdot(self.expected_x_inv_sigma_x(), x))

        res += 0.5 * self.expected_logdet_inv_sigma().sum((-2, -1), keepdims=True)
        res = res.squeeze((-2, -1)) + self._likelihood.log_measure(x)
        return MultivariateNormal(nat_params, residual=res)

    def expected_inv_sigma_x(self, inv_sigma=None) -> Array:
        """Compute E[Σ⁻¹A]."""
        if inv_sigma is None:
            return bdot(self.expected_inv_sigma(), self.mu)
        else:
            return bdot(inv_sigma, self.mu)

    def expected_inv_sigma(self) -> Array:
        """Compute E[Σ⁻¹]."""
        return (self.a / self.b) * jnp.eye(self.y_dim)

    def expected_sigma(self) -> Array:
        """Compute E[Σ]."""
        return (self.b / (self.a - 1)) * jnp.eye(self.y_dim)

    def expected_inv_sigma_diag(self) -> Array:
        return self.a / self.b

    def inv_expected_inv_sigma(self) -> Array:
        """Compute E[Σ⁻¹]⁻¹."""
        return (self.b / self.a) * jnp.eye(self.y_dim)

    def expected_logdet_inv_sigma(self) -> Array:
        """Compute E[log|Σ⁻¹|]."""
        return jnp.sum(jsp.digamma(self.a) - jnp.log(self.b), -2, keepdims=True)

    def logdet_expected_inv_sigma(self) -> Array:
        """Compute log|E[Σ⁻¹]|."""
        return jnp.sum(jnp.log(self.a) - jnp.log(self.b), -2, keepdims=True)

    def expected_log_det_inv_sigma_minus_log_det_expected_inv_sigma(self) -> Array:
        return jnp.sum(jsp.digamma(self.a) - jnp.log(self.a), -2, keepdims=True)

    def expected_x_inv_sigma_x(self, inv_sigma_mu=None) -> Array:
        """Compute E[AᵀΣ⁻¹A]."""
        if inv_sigma_mu is None:
            return self.y_dim * self.v + bdot(self.mu.mT, bdot(self.expected_inv_sigma(), self.mu))
        else:
            return self.y_dim * self.v + bdot(self.mu.mT, inv_sigma_mu)

    def _update_cache(self):
        """Update parameters for computing expectations."""
        norm = self.norm
        _v, _logdet_inv_v = inv_and_logdet(self.inv_v / norm)
        self._v, self._logdet_inv_v = _v / norm, _logdet_inv_v + self.x_dim * jnp.log(norm)
        rho = jnp.diagonal(bdot(self.posterior_params.eta.eta_2 / norm, self.mu.mT), axis1=-1, axis2=-2)
        self._b = (self.posterior_params.eta.eta_3 / norm - expand(rho, -1)) * norm / 2

    def expected_log_likelihood(self, data) -> Array:
        """Compute expected log likelihood."""
        x, y = data
        ex_inv_sigma_x = self.expected_x_inv_sigma_x()
        e_inv_sigma_x = self.expected_inv_sigma_x()
        probs = -0.5 * bdot(y.mT, bdot(self.expected_inv_sigma(), y)).squeeze((-1, -2))
        if self.use_bias:
            probs = probs + (bdot(y.mT, bdot(e_inv_sigma_x[..., :, :-1], x) + e_inv_sigma_x[..., :, -1:])).squeeze((-1, -2))
            probs -= 0.5 * bdot(x.mT, bdot(ex_inv_sigma_x[..., :-1, :-1], x)).squeeze((-1, -2))
            probs -= bdot(ex_inv_sigma_x[..., -1:, :-1], x).squeeze((-1, -2))
            probs -= 0.5 * ex_inv_sigma_x[..., -1, -1]
        else:
            probs += bdot(y.mT, bdot(e_inv_sigma_x, x)).squeeze((-1, -2))
            probs -= 0.5 * bdot(x.mT, bdot(ex_inv_sigma_x, x)).squeeze((-1, -2))
        probs += 0.5 * self.expected_logdet_inv_sigma().sum((-1, -2))
        probs -= 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        return probs

    def average_energy(self, inputs) -> Array:
        """Compute average energy term."""
        pX, pY = inputs
        py_exp_xx = pY.expected_xx()
        px_exp_xx = pX.expected_xx()
        U = -0.5 * (py_exp_xx * self.expected_inv_sigma()).sum((-2, -1))
        EXTinvUX = self.expected_x_inv_sigma_x()
        EinvUX = self.expected_inv_sigma_x()

        if self.use_bias:
            U += bdot(pY.mean.mT, bdot(EinvUX[..., :, :-1], pX.mean) + EinvUX[..., :, -1:]).squeeze((-1, -2))
            U -= 0.5 * (px_exp_xx * EXTinvUX[..., :-1, :-1]).sum((-2, -1))
            U -= bdot(EXTinvUX[..., -1:, :-1], pX.mean).squeeze((-1, -2))
            U -= 0.5 * EXTinvUX[..., -1, -1]
        else:
            U += bdot(pY.mean.mT, bdot(EinvUX, pX.mean)).squeeze((-1, -2))
            U -= 0.5 * (px_exp_xx * EXTinvUX).sum((-1, -2))

        U += 0.5 * self.expected_logdet_inv_sigma().sum((-1, -2)) - 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        return U

    def kl_divergence(self) -> Array:
        """Compute KL divergence."""
        kl = (self.y_dim / 2.0 * self.logdet_inv_v.squeeze((-2, -1)) - self.y_dim / 2.0 * self.prior_logdet_inv_v.squeeze((-2, -1)) - self.y_dim * self.x_dim / 2.0)
        traceV = (self.prior_inv_v * self.v).sum((-2, -1))
        kl = kl + 0.5 * self.y_dim * traceV
        traceMuUMu = (self.prior_inv_v * bdot((self.mu - self.prior_mu).mT, bdot(self.expected_inv_sigma(), self.mu - self.prior_mu))).sum((-2, -1))
        kl = kl + 0.5 * traceMuUMu
        kl_gamma = self._kl_divergence_gamma()
        return kl + kl_gamma

    def _kl_divergence_gamma(self) -> Array:
        """Compute KL divergence for gamma distributions."""
        kl = self.prior_a * (jnp.log(self.b) - jnp.log(self.prior_b))
        kl = kl + lgamma(self.prior_a) - lgamma(self.a)
        kl = kl + (self.a - self.prior_a) * jsp.digamma(self.a)
        kl = kl - (self.b - self.prior_b) * (self.a / self.b)
        return kl.sum((-2, -1))

    def forward_from_normal(self, pX: MultivariateNormal, pass_residual=False) -> MultivariateNormal:
        """Forward message when input is multivariate normal."""
        res = -pX.log_partition().squeeze((-1, -2))
        res += 0.5 * self.expected_logdet_inv_sigma().squeeze((-2, -1))
        if self.use_bias is False:
            A = self.expected_inv_sigma()
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x() + pX.inv_sigma
            C_y = 0.0
            C_x = pX.inv_sigma_mu
            invD, logdetD = inv_and_logdet(D)
            res += 0.5 * bdot(C_x.mT, bdot(invD, C_x)).squeeze((-2, -1)) - 0.5 * logdetD.squeeze((-2, -1))
        else:
            A = self.expected_inv_sigma()
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x()
            C_y = B[..., :, -1:]
            B = B[..., :, :-1]
            C_x = -D[..., :-1, -1:] + pX.inv_sigma_mu
            res += -0.5 * D[..., -1, -1]
            D = D[..., :-1, :-1] + pX.inv_sigma
            invD, logdetD = inv_and_logdet(D)
            res += 0.5 * bdot(C_x.mT, bdot(invD, C_x)).squeeze((-2, -1)) - 0.5 * logdetD.squeeze((-2, -1))

        inv_sigma_yy = A - bdot(B, bdot(invD, B.mT))
        inv_sigma_mu_y = C_y + bdot(B, bdot(invD, C_x))

        ndim_diff = inv_sigma_mu_y.ndim - inv_sigma_yy.ndim
        if ndim_diff > 0:
            inv_sigma_yy = jnp.expand_dims(inv_sigma_yy, tuple(range(ndim_diff)))

        if pass_residual:
            res += pX.residual
        pY = MultivariateNormal(ArrayDict(inv_sigma_mu=inv_sigma_mu_y, inv_sigma=inv_sigma_yy), residual=res)
        pY.residual += pY.log_partition().squeeze((-2, -1))
        return pY

    def backward_from_normal(self, pY: MultivariateNormal, pass_residual=False) -> MultivariateNormal:
        """Backward message when output is multivariate normal."""
        res = -pY.log_partition().squeeze((-1, -2))
        res += 0.5 * self.expected_logdet_inv_sigma().squeeze((-1, -2))
        A = self.expected_inv_sigma() + pY.inv_sigma
        invA, logdetA = inv_and_logdet(A)

        if self.use_bias is False:
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x()
            C_y = pY.inv_sigma_mu
            C_x = 0.0
            res += 0.5 * bdot(C_y.mT, bdot(invA, C_y)).squeeze((-2, -1)) - 0.5 * logdetA.squeeze((-2, -1))
        else:
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x()
            C_y = pY.inv_sigma_mu + B[..., :, -1:]
            B = B[..., :, :-1]
            C_x = -D[..., :-1, -1:]
            res += -0.5 * D[..., -1, -1]
            D = D[..., :-1, :-1]
            res += 0.5 * bdot(C_y.mT, bdot(invA, C_y)).squeeze((-2, -1)) - 0.5 * logdetA.squeeze((-2, -1))

        inv_sigma_xx = D - bdot(B.mT, bdot(invA, B))
        inv_sigma_mu_x = C_x + bdot(B.mT, bdot(invA, C_y))

        ndim_diff = inv_sigma_mu_x.ndim - inv_sigma_xx.ndim
        if ndim_diff > 0:
            inv_sigma_xx = jnp.expand_dims(inv_sigma_xx, tuple(range(ndim_diff)))

        if pass_residual:
            res += pY.residual
        pX = MultivariateNormal(ArrayDict(inv_sigma_mu=inv_sigma_mu_x, inv_sigma=inv_sigma_xx), residual=res)
        return pX

    def variational_forward(self, pX: Distribution, pass_residual=False) -> MultivariateNormal:
        """Variational forward (doesn't marginalize over joint)."""
        inv_sigma_y = self.expected_inv_sigma()
        res = 0.5 * self.expected_logdet_inv_sigma().squeeze((-1, -2))
        expected_inv_sigma_x = self.expected_inv_sigma_x(inv_sigma=inv_sigma_y)
        inv_sigma_xx = self.expected_x_inv_sigma_x(inv_sigma_mu=expected_inv_sigma_x)

        if self.use_bias:
            inv_sigma_mu_y = bdot(expected_inv_sigma_x[..., :, :-1], pX.expected_x()) + expected_inv_sigma_x[..., :, -1:]
            res -= 0.5 * jnp.sum(inv_sigma_xx[..., :-1, :-1] * pX.expected_xx(), (-2, -1))
            res -= jnp.sum(inv_sigma_xx[..., :-1, -1:] * pX.expected_x(), (-2, -1))
            res -= 0.5 * inv_sigma_xx[..., -1, -1]
        else:
            inv_sigma_mu_y = bdot(expected_inv_sigma_x, pX.expected_x())
            res -= 0.5 * jnp.sum(inv_sigma_xx * pX.expected_xx(), (-2, -1))

        shape = jnp.broadcast_shapes(inv_sigma_y.shape[:-2], inv_sigma_mu_y.shape[:-2])
        inv_sigma_y = jnp.broadcast_to(inv_sigma_y, shape + inv_sigma_y.shape[-2:])
        inv_sigma_mu_y = jnp.broadcast_to(inv_sigma_mu_y, shape + inv_sigma_mu_y.shape[-2:])

        if pass_residual:
            res += pX.residual
        pY = MultivariateNormal(ArrayDict(inv_sigma_mu=inv_sigma_mu_y, inv_sigma=inv_sigma_y), residual=res)
        pY.residual += pY.log_partition().squeeze((-1, -2))
        return pY

    def variational_backward(self, pY: Distribution, pass_residual=False) -> Distribution:
        """Variational backward."""
        inv_sigma = self.expected_inv_sigma()
        res = 0.5 * self.expected_logdet_inv_sigma().sum((-2, -1))
        res -= 0.5 * jnp.sum(inv_sigma * pY.expected_xx(), (-2, -1))

        if self.use_bias is False:
            inv_sigma_mu = self.expected_inv_sigma_x(inv_sigma=inv_sigma)
            inv_sigma_mu_x = bdot(inv_sigma_mu.mT, pY.expected_x())
            inv_sigma_xx = self.expected_x_inv_sigma_x(inv_sigma_mu=inv_sigma_mu)
        else:
            Einv_sigma_x = self.expected_inv_sigma_x(inv_sigma=inv_sigma)
            Ex_inv_sigma_x = self.expected_x_inv_sigma_x(inv_sigma_mu=Einv_sigma_x)
            inv_sigma_mu_x = bdot(Einv_sigma_x[..., :, :-1].mT, pY.expected_x()) - Ex_inv_sigma_x[..., :-1, -1:]
            res += bdot(Einv_sigma_x[..., :, -1:].mT, pY.expected_x()).squeeze((-2, -1)) - 0.5 * Ex_inv_sigma_x[..., -1, -1]
            inv_sigma_xx = Ex_inv_sigma_x[..., :-1, :-1]

        shape = jnp.broadcast_shapes(inv_sigma_xx.shape[:-2], pY.expected_x().shape[:-2])
        inv_sigma_xx = jnp.broadcast_to(inv_sigma_xx, shape + inv_sigma_xx.shape[-2:])

        pX = MultivariateNormal(ArrayDict(inv_sigma_mu=inv_sigma_mu_x, inv_sigma=inv_sigma_xx), residual=res)
        if pass_residual:
            pX.residual += pY.residual
        return pX

    def joint(self, pX: Distribution, pY: Distribution) -> Distribution:
        """Compute joint distribution."""
        raise NotImplementedError

    def elbo(self, data, weights=None) -> Array:
        """Compute ELBO."""
        X, Y = data
        sample_dims = self.get_sample_dims(X)
        if weights is None:
            ELL = self.expected_log_likelihood((X, Y)).sum(sample_dims)
        else:
            ELL = (self.expected_log_likelihood(data) * weights).sum(sample_dims)
        return ELL - self.kl_divergence()

    def elbo_contrib(self, pXY, weights=None) -> Array:
        """Compute ELBO contribution."""
        pX, pY = pXY
        sample_dims = self.get_sample_dims(pX.mean)
        if weights is None:
            ELL = self.average_energy(pXY).sum(sample_dims)
        else:
            ELL = (self.average_energy(pXY) * weights).sum(sample_dims)
        return ELL - self.kl_divergence()

// Copyright 2025 VERSES AI, Inc.
//
// Licensed under the VERSES Academic Research License (the "License");
// you may not use this file except in compliance with the license.
//
// You may obtain a copy of the License at
//     https://github.com/VersesTech/axiom/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "conjugate.h"
#include "mvn_exponential.h"

namespace axiom {
namespace distributions {

/**
 * @brief Normal-Inverse-Wishart conjugate prior for Multivariate Normal likelihood.
 * 
 * The conjugate prior for N(μ, Σ) is:
 *   p(μ, Σ⁻¹ | m, κ, n, U) = exp(⟨Σ⁻¹μ, -0.5Σ⁻¹⟩ · ⟨κm, U⁻¹ + κmmᵀ⟩
 *                             - ⟨κ, n - D - 1⟩ · ⟨0.5μᵀΣ⁻¹μ, 0.5 log|Σ⁻¹|⟩)
 * 
 * Natural parameters:
 *   - eta_1 = κm
 *   - eta_2 = U⁻¹ + κmmᵀ
 *   - nu_1 = κ
 *   - nu_2 = n - D - 1
 */
template<typename Scalar>
class MultivariateNormalConjugate : public Conjugate<Scalar> {
public:
    // Cached derived quantities
    std::vector<Scalar> _u;                   // Scale matrix U
    std::vector<Scalar> _logdet_inv_u;        // log|U⁻¹|
    std::vector<Scalar> _prior_logdet_inv_u;  // Prior log|U⁻¹|
    
    bool fixed_precision = false;

    MultivariateNormalConjugate(
        const ArrayDict<std::vector<Scalar>>& params = {},
        const ArrayDict<std::vector<Scalar>>& prior_params = {},
        const std::vector<int>& batch_shape = {},
        const std::vector<int>& event_shape = {},
        int event_dim = MVN_DEFAULT_EVENT_DIM,
        bool fixed_precision = false,
        Scalar scale = Scalar(1.0),
        Scalar dof_offset = Scalar(0.0));

    // Factory method
    static ArrayDict<std::vector<Scalar>> init_default_params(
        const std::vector<int>& batch_shape,
        const std::vector<int>& event_shape,
        Scalar scale = Scalar(1.0),
        Scalar dof_offset = Scalar(0.0));

    // Properties
    std::vector<Scalar> mean() const;
    std::vector<Scalar> prior_mean() const;
    std::vector<Scalar> kappa() const;
    std::vector<Scalar> prior_kappa() const;
    std::vector<Scalar> n() const;
    std::vector<Scalar> prior_n() const;
    std::vector<Scalar> inv_u() const;
    std::vector<Scalar> prior_inv_u() const;
    std::vector<Scalar> u();
    std::vector<Scalar> logdet_inv_u();
    std::vector<Scalar> prior_logdet_inv_u();

    // Conjugate implementations
    ArrayDict<std::vector<Scalar>> to_natural_params(
        const ArrayDict<std::vector<Scalar>>& params) const override;
    ArrayDict<std::vector<Scalar>> expected_likelihood_params() const override;
    ArrayDict<std::vector<Scalar>> expected_posterior_statistics() const override;
    ArrayDict<std::vector<Scalar>> expected_log_partition() const override;
    std::vector<Scalar> log_prior_partition() const override;
    std::vector<Scalar> log_posterior_partition() const override;
    std::vector<Scalar> residual() const override;
    std::vector<Scalar> variational_residual() const override;
    std::vector<Scalar> collapsed_residual() const;

    // Expected values
    std::vector<Scalar> expected_inv_sigma() const;
    std::vector<Scalar> expected_inv_sigma_mu() const;
    std::vector<Scalar> expected_sigma() const;
    std::vector<Scalar> inv_expected_inv_sigma() const;
    std::vector<Scalar> expected_mu_inv_sigma_mu() const;
    std::vector<Scalar> expected_xx() const;
    std::vector<Scalar> expected_logdet_inv_sigma() const;
    std::vector<Scalar> logdet_expected_inv_sigma() const;

    // KL divergence
    std::vector<Scalar> kl_divergence() const override;
    std::vector<Scalar> kl_divergence_wishart() const;

    // Expected log likelihood
    std::vector<Scalar> expected_log_likelihood(const std::vector<Scalar>& data) const override;
    std::vector<Scalar> average_energy(std::shared_ptr<Distribution> x) const;

    // Update methods
    void update_from_probabilities(std::shared_ptr<Distribution> pX,
                                   const std::vector<Scalar>& weights = {},
                                   Scalar lr = Scalar(1.0),
                                   Scalar beta = Scalar(0.0)) override;

    // Internal KL
    std::vector<Scalar> _kl_divergence() const;

protected:
    // Internal computation
    std::vector<Scalar> _log_partition(const std::vector<Scalar>& kappa,
                                       const std::vector<Scalar>& n,
                                       const std::vector<Scalar>& logdet_inv_u) const;
    std::vector<Scalar> _scaled_inv_u() const;
    std::vector<Scalar> sqrt_diag_norm() const;
    std::vector<Scalar> norm() const;
    
    void _update_cache() override;
    static std::unordered_map<std::string, std::string> params_to_tx();
};

} // namespace distributions
} // namespace axiom

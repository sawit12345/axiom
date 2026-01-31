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

#include "exponential_family.h"
#include <memory>

namespace axiom {
namespace distributions {

// Default event dimension for MVN
constexpr int MVN_DEFAULT_EVENT_DIM = 2;

/**
 * @brief Multivariate Normal distribution in exponential family form.
 * 
 * The log likelihood is:
 *   log p(x | μ, Σ) = -0.5 * (x - μ)ᵀ Σ⁻¹(x - μ) + 0.5 * log |Σ⁻¹| - 0.5 * D * log(2π)
 * 
 * Natural parameters S(θ):
 *   - inv_sigma_mu = Σ⁻¹μ  (precision-weighted mean)
 *   - inv_sigma = Σ⁻¹      (precision matrix)
 * 
 * Sufficient statistics T(x):
 *   - x = x
 *   - minus_half_xxT = -0.5 * xxᵀ
 * 
 * Log partition:
 *   A(S(θ)) = 0.5 * μᵀΣ⁻¹μ - 0.5 * log|Σ⁻¹| + 0.5 * D * log(2π)
 */
template<typename Scalar>
class MultivariateNormal : public ExponentialFamily<Scalar> {
public:
    // Cached derived quantities
    std::vector<Scalar> _mu;              // Mean μ = Σ inv_sigma_mu
    std::vector<Scalar> _sigma;           // Covariance Σ = inv_sigma⁻¹
    std::vector<Scalar> _logdet_inv_sigma; // log|Σ⁻¹|

    MultivariateNormal(
        const ArrayDict<std::vector<Scalar>>& nat_params = {},
        const ArrayDict<std::vector<Scalar>>& expectations = {},
        const std::vector<int>& batch_shape = {},
        const std::vector<int>& event_shape = {},
        int event_dim = MVN_DEFAULT_EVENT_DIM);

    // Factory method for initialization
    static ArrayDict<std::vector<Scalar>> init_default_params(
        const std::vector<int>& batch_shape,
        const std::vector<int>& event_shape,
        Scalar scale = Scalar(1.0));

    // Property accessors
    std::vector<Scalar>& mu();
    const std::vector<Scalar>& mu() const;
    
    std::vector<Scalar>& inv_sigma();
    const std::vector<Scalar>& inv_sigma() const;
    
    std::vector<Scalar>& inv_sigma_mu();
    const std::vector<Scalar>& inv_sigma_mu() const;
    
    std::vector<Scalar>& logdet_inv_sigma();
    const std::vector<Scalar>& logdet_inv_sigma() const;
    
    std::vector<Scalar> mu_inv_sigma_mu() const;
    
    std::vector<Scalar>& sigma();
    const std::vector<Scalar>& sigma() const;
    
    std::vector<Scalar> mean() const;

    // Exponential family implementations
    ArrayDict<std::vector<Scalar>> statistics(const std::vector<Scalar>& x) const override;
    std::vector<Scalar> log_measure(const std::vector<Scalar>& x) const override;
    std::vector<Scalar> expected_log_measure() const override;
    ArrayDict<std::vector<Scalar>> expected_statistics() const override;
    std::vector<Scalar> log_partition() const override;
    std::vector<Scalar> sample(const std::vector<Scalar>& key, 
                               const std::vector<int>& shape) const override;
    ArrayDict<std::vector<Scalar>> params_from_statistics(
        const ArrayDict<std::vector<Scalar>>& stats) const override;

    // Additional expected value methods
    std::vector<Scalar> expected_x() const;
    std::vector<Scalar> expected_xx() const;

    // Shift operator for message passing
    std::shared_ptr<MultivariateNormal<Scalar>> shift(const std::vector<Scalar>& deltax) const;

    // Internal entropy computation
    std::vector<Scalar> _entropy() const;

protected:
    // Cache management
    void _reset_cache();
    void _update_cache() override;
    
    // Computation methods
    std::pair<std::vector<Scalar>, std::vector<Scalar>> 
        compute_sigma_and_logdet_inv_sigma() const;
    std::vector<Scalar> compute_logdet_inv_sigma() const;
    std::vector<Scalar> compute_mu() const;
    std::vector<Scalar> compute_sigma() const;

    // Parameter to statistics mapping
    static std::unordered_map<std::string, std::string> params_to_tx();

    // Cache update function ordering
    std::vector<std::pair<std::string, std::string>> _cache_update_functions;
    std::vector<std::pair<std::string, std::string>> 
        _get_cache_update_functions(const std::vector<std::string>& cache_attrs) const;
    std::vector<std::string> _order_cache_computations(
        const std::vector<std::string>& cache_attrs) const;
};

/**
 * @brief Multivariate Normal with positive xxᵀ sufficient statistic.
 * 
 * Alternative parameterization where the sufficient statistic for inv_sigma
 * is the outer product xxᵀ instead of -0.5 * xxᵀ.
 */
template<typename Scalar>
class MultivariateNormalPositiveXXT : public MultivariateNormal<Scalar> {
public:
    MultivariateNormalPositiveXXT(
        const ArrayDict<std::vector<Scalar>>& nat_params = {},
        const ArrayDict<std::vector<Scalar>>& expectations = {},
        const std::vector<int>& batch_shape = {},
        const std::vector<int>& event_shape = {},
        int event_dim = MVN_DEFAULT_EVENT_DIM);

    // Override statistics
    ArrayDict<std::vector<Scalar>> statistics(const std::vector<Scalar>& x) const override;
    ArrayDict<std::vector<Scalar>> expected_statistics() const override;
    
    std::vector<Scalar> expected_x() const;
    std::vector<Scalar> expected_xx() const override;
    
    std::vector<Scalar> sample(const std::vector<Scalar>& key, 
                               const std::vector<int>& shape) const override;
    ArrayDict<std::vector<Scalar>> params_from_statistics(
        const ArrayDict<std::vector<Scalar>>& stats) const override;

protected:
    static std::unordered_map<std::string, std::string> params_to_tx();
};

} // namespace distributions
} // namespace axiom

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
#include "multinomial_exponential.h"

namespace axiom {
namespace distributions {

/**
 * @brief Dirichlet conjugate prior for Multinomial likelihood.
 * 
 * The conjugate prior is:
 *   p(p | α) = Dirichlet(p | α) = (1/B(α)) ∏ p_i^(α_i - 1)
 * 
 * where B(α) is the multivariate beta function.
 * 
 * Natural parameters:
 *   - eta_1 = α (concentration parameters)
 * 
 * The likelihood sufficient statistic is x itself.
 */
template<typename Scalar>
class MultinomialConjugate : public Conjugate<Scalar> {
public:
    MultinomialConjugate(
        const ArrayDict<std::vector<Scalar>>& params = {},
        const ArrayDict<std::vector<Scalar>>& prior_params = {},
        const std::vector<int>& batch_shape = {},
        const std::vector<int>& event_shape = {},
        int event_dim = MULTINOMIAL_DEFAULT_EVENT_DIM,
        Scalar initial_count = Scalar(1.0));

    // Factory method
    static ArrayDict<std::vector<Scalar>> init_default_params(
        const std::vector<int>& batch_shape,
        const std::vector<int>& event_shape,
        Scalar initial_count = Scalar(1.0));

    // Properties
    std::vector<Scalar> alpha() const;
    std::vector<Scalar> prior_alpha() const;
    std::vector<Scalar> mean() const;
    std::vector<Scalar> log_mean() const;
    std::vector<Scalar> variance() const;
    std::vector<Scalar> mode() const;

    // Conjugate implementations
    ArrayDict<std::vector<Scalar>> to_natural_params(
        const ArrayDict<std::vector<Scalar>>& params) const override;
    ArrayDict<std::vector<Scalar>> expected_likelihood_params() const override;
    std::vector<Scalar> expected_log_likelihood(const std::vector<Scalar>& x) const override;
    ArrayDict<std::vector<Scalar>> expected_posterior_statistics() const override;
    ArrayDict<std::vector<Scalar>> expected_log_partition() const override;
    std::vector<Scalar> log_prior_partition() const override;
    std::vector<Scalar> log_posterior_partition() const override;
    std::vector<Scalar> residual() const override;
    std::vector<Scalar> kl_divergence() const override;
    std::vector<Scalar> variational_residual() const override;
    std::shared_ptr<typename Conjugate<Scalar>::LikelihoodType> forward() const override;

    // Sampling
    std::vector<Scalar> sample(const std::vector<Scalar>& key,
                               const std::vector<int>& shape = {}) const;

protected:
    static std::unordered_map<std::string, std::string> params_to_tx();
    
    // Gamma and digamma functions (multivariate)
    std::vector<Scalar> _mvgammaln(const std::vector<Scalar>& x, int d) const;
    std::vector<Scalar> _mvdigamma(const std::vector<Scalar>& x, int d) const;
    std::vector<Scalar> _gammaln(const std::vector<Scalar>& x) const;
    std::vector<Scalar> _digamma(const std::vector<Scalar>& x) const;
};

} // namespace distributions
} // namespace axiom

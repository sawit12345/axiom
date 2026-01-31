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

#include "distribution.h"
#include "exponential_family.h"

namespace axiom {
namespace distributions {

/**
 * @brief Base class for conjugate prior distributions.
 * 
 * The likelihood has the form:
 *   p(x|θ) = φ(x) * exp(S(θ) · T(x) - A(S(θ)))
 * 
 * The conjugate prior has the form:
 *   p(θ|η₀, ν₀) = exp(S(θ) · η₀ - ν₀ A(S(θ)) - log Z(η₀, ν₀))
 * 
 * Posterior update:
 *   η = η₀ + Σ T(xₙ)
 *   ν = ν₀ + N
 */
template<typename Scalar>
class Conjugate : public Distribution {
public:
    // Likelihood class type
    using LikelihoodType = ExponentialFamily<Scalar>;

    // Natural parameters
    ArrayDict<std::vector<Scalar>> _posterior_params;  // (η, ν)
    ArrayDict<std::vector<Scalar>> _prior_params;      // (η₀, ν₀)
    
    // Cached likelihood with expected parameters
    std::shared_ptr<LikelihoodType> _likelihood;

    Conjugate(int default_event_dim,
              std::shared_ptr<LikelihoodType> likelihood,
              const ArrayDict<std::vector<Scalar>>& posterior_params,
              const ArrayDict<std::vector<Scalar>>& prior_params,
              const std::vector<int>& batch_shape = {},
              const std::vector<int>& event_shape = {});

    // Property accessors
    std::shared_ptr<LikelihoodType> likelihood() const;
    void set_likelihood(std::shared_ptr<LikelihoodType> value);
    
    ArrayDict<std::vector<Scalar>>& posterior_params();
    const ArrayDict<std::vector<Scalar>>& posterior_params() const;
    void set_posterior_params(const ArrayDict<std::vector<Scalar>>& value);
    
    ArrayDict<std::vector<Scalar>>& prior_params();
    const ArrayDict<std::vector<Scalar>>& prior_params() const;
    void set_prior_params(const ArrayDict<std::vector<Scalar>>& value);

    // Core conjugate methods (to be implemented by subclasses)
    
    /**
     * @brief Expected natural parameters of likelihood: <S(θ)>_{q(θ|η, v)}
     */
    virtual ArrayDict<std::vector<Scalar>> expected_likelihood_params() const = 0;

    /**
     * @brief Expected log likelihood: <log p(x|θ)>_{q(θ|η, v)}
     */
    virtual std::vector<Scalar> expected_log_likelihood(const std::vector<Scalar>& data) const = 0;

    /**
     * @brief Expected posterior statistics: (<S(θ)>, -<A(θ)>)
     */
    virtual ArrayDict<std::vector<Scalar>> expected_posterior_statistics() const = 0;

    /**
     * @brief Expected log partition: <A(θ)>_{q(θ|η, v)}
     */
    virtual ArrayDict<std::vector<Scalar>> expected_log_partition() const = 0;

    /**
     * @brief Convert canonical params to natural params
     */
    virtual ArrayDict<std::vector<Scalar>> to_natural_params(
        const ArrayDict<std::vector<Scalar>>& params) const = 0;

    /**
     * @brief Log partition of prior: log Z(η₀, ν₀)
     */
    virtual std::vector<Scalar> log_prior_partition() const = 0;

    /**
     * @brief Log partition of posterior: log Z(η, v)
     */
    virtual std::vector<Scalar> log_posterior_partition() const = 0;

    /**
     * @brief Residual: A(<S(θ)>) - <A(θ)>
     */
    virtual std::vector<Scalar> residual() const = 0;

    /**
     * @brief KL divergence: KL(q(θ|η, v), p(θ|η₀, ν₀))
     */
    virtual std::vector<Scalar> kl_divergence() const;

    /**
     * @brief Variational residual
     */
    virtual std::vector<Scalar> variational_residual() const = 0;

    /**
     * @brief Variational forward message
     */
    virtual std::shared_ptr<LikelihoodType> variational_forward() const;

    /**
     * @brief Statistics dot expected params: T(x) · <S(θ)>
     */
    virtual std::vector<Scalar> statistics_dot_expected_params(
        const std::vector<Scalar>& x) const;

    // Update methods
    virtual void update_from_data(const std::vector<Scalar>& data,
                                  const std::vector<Scalar>& weights = {},
                                  Scalar lr = Scalar(1.0),
                                  Scalar beta = Scalar(0.0));
    
    virtual void update_from_statistics(const ArrayDict<std::vector<Scalar>>& summed_stats,
                                        Scalar lr = Scalar(1.0),
                                        Scalar beta = Scalar(0.0));
    
    virtual void update_from_probabilities(std::shared_ptr<Distribution> distribution,
                                           const std::vector<Scalar>& weights = {},
                                           Scalar lr = Scalar(1.0),
                                           Scalar beta = Scalar(0.0));

    // Utility methods
    std::shared_ptr<Distribution> expand(const std::vector<int>& shape);
    
    ArrayDict<std::vector<Scalar>> map_params_to_likelihood(
        const ArrayDict<std::vector<Scalar>>& params) const;
    
    ArrayDict<std::vector<Scalar>> map_stats_to_params(
        const ArrayDict<std::vector<Scalar>>& likelihood_stats,
        const std::vector<Scalar>& counts) const;

protected:
    virtual std::unordered_map<std::string, std::string> _get_params_to_stats_mapping() const;
    virtual std::unordered_map<std::string, std::string> 
        _conjugate_to_likelihood_mapping(std::shared_ptr<LikelihoodType> likelihood_cls = nullptr) const;
    
    ArrayDict<std::vector<Scalar>> sum_stats_over_samples(
        const ArrayDict<std::vector<Scalar>>& stats,
        const std::vector<Scalar>& weights,
        const std::vector<int>& sample_dims) const;
    
    void _update_cache();
};

} // namespace distributions
} // namespace axiom

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
#include <memory>
#include <unordered_map>
#include <string>

namespace axiom {
namespace distributions {

/**
 * @brief Base class for exponential family distributions.
 * 
 * The exponential family is defined as:
 *   p(x|θ) = φ(x) * exp(S(θ) · T(x) - A(S(θ)))
 * 
 * where:
 *   - x is data
 *   - S(θ) are the natural parameters
 *   - φ(x) is the measure function
 *   - T(x) are the sufficient statistics  
 *   - A(S(θ)) is the log partition function
 * 
 * This class stores either the natural parameters S(θ) in `nat_params`,
 * or the expected statistics <E[T(x) | S(θ)]> in `expectations`.
 */
template<typename Scalar>
class ExponentialFamily : public Distribution {
public:
    // Natural parameters S(θ)
    ArrayDict<std::vector<Scalar>> _nat_params;
    
    // Expected sufficient statistics <E[T(x) | S(θ)]>
    ArrayDict<std::vector<Scalar>> _expectations;
    
    // Residual array
    std::vector<Scalar> _residual;

    ExponentialFamily(int default_event_dim,
                      const std::vector<int>& batch_shape,
                      const std::vector<int>& event_shape,
                      const ArrayDict<std::vector<Scalar>>& nat_params = {},
                      const ArrayDict<std::vector<Scalar>>& expectations = {},
                      const std::vector<Scalar>& residual = {});

    // Property accessors
    ArrayDict<std::vector<Scalar>>& nat_params();
    const ArrayDict<std::vector<Scalar>>& nat_params() const;
    void set_nat_params(const ArrayDict<std::vector<Scalar>>& value);

    ArrayDict<std::vector<Scalar>>& expectations();
    const ArrayDict<std::vector<Scalar>>& expectations() const;
    void set_expectations(const ArrayDict<std::vector<Scalar>>& value);

    std::vector<Scalar>& residual();
    const std::vector<Scalar>& residual() const;
    void set_residual(const std::vector<Scalar>& value);

    // Core exponential family methods (to be implemented by subclasses)
    
    /**
     * @brief Compute the log likelihood log p(x|θ)
     * 
     * log p(x|θ) = S(θ) · T(x) - A(S(θ))
     */
    virtual std::vector<Scalar> log_likelihood(const std::vector<Scalar>& x) const;

    /**
     * @brief Compute sufficient statistics T(x)
     */
    virtual ArrayDict<std::vector<Scalar>> statistics(const std::vector<Scalar>& x) const = 0;

    /**
     * @brief Compute expected sufficient statistics <E[T(x)|S(θ)]>
     */
    virtual ArrayDict<std::vector<Scalar>> expected_statistics() const = 0;

    /**
     * @brief Compute log partition function A(S(θ))
     */
    virtual std::vector<Scalar> log_partition() const = 0;

    /**
     * @brief Compute log base measure log φ(x)
     */
    virtual std::vector<Scalar> log_measure(const std::vector<Scalar>& x) const = 0;

    /**
     * @brief Compute expected log base measure <log φ(x)|S(θ)>
     */
    virtual std::vector<Scalar> expected_log_measure() const = 0;

    /**
     * @brief Compute entropy of the distribution
     * 
     * H(p) = -<T(x)|S(θ))> · S(θ) + A(S(θ)) - <log φ(x)|S(θ)>
     */
    virtual std::vector<Scalar> entropy() const;

    /**
     * @brief Sample from the distribution
     */
    virtual std::vector<Scalar> sample(const std::vector<Scalar>& key, 
                                       const std::vector<int>& shape) const = 0;

    /**
     * @brief Compute natural parameters from expected statistics
     * 
     * S(θ) = μ_T⁻¹(<E[T(x)]>)
     */
    virtual ArrayDict<std::vector<Scalar>> params_from_statistics(
        const ArrayDict<std::vector<Scalar>>& stats) const = 0;

    // Utility methods
    
    /**
     * @brief Compute dot product of natural params and sufficient statistics
     * 
     * S(θ) · T(x)
     */
    std::vector<Scalar> params_dot_statistics(const std::vector<Scalar>& x) const;

    /**
     * @brief Compute dot product of natural params and expected statistics
     * 
     * S(θ) · <E[T(x)]>
     */
    std::vector<Scalar> params_dot_expected_statistics() const;

    /**
     * @brief Combine natural parameters with another distribution
     */
    virtual std::shared_ptr<ExponentialFamily<Scalar>> combine(
        const std::shared_ptr<ExponentialFamily<Scalar>>& other) const;

    /**
     * @brief Multiply distributions (combine natural parameters)
     */
    virtual std::shared_ptr<ExponentialFamily<Scalar>> operator*(
        const std::shared_ptr<ExponentialFamily<Scalar>>& other) const;

    /**
     * @brief Expand distribution to larger batch shape
     */
    std::shared_ptr<Distribution> expand(const std::vector<int>& shape);

protected:
    /**
     * @brief Validate natural parameters
     */
    virtual void _validate_nat_params(const ArrayDict<std::vector<Scalar>>& nat_params) const;

    /**
     * @brief Update cache when parameters change
     */
    virtual void _update_cache();

    /**
     * @brief Get mapping from natural params to sufficient statistics
     */
    virtual std::unordered_map<std::string, std::string> _get_params_to_stats_mapping() const;

    // Helper for mapping and multiplying
    std::vector<Scalar> _map_and_multiply(
        const ArrayDict<std::vector<Scalar>>& params,
        const ArrayDict<std::vector<Scalar>>& stats,
        int default_event_dim,
        const std::unordered_map<std::string, std::string>& mapping) const;

    bool _nat_params_initialized = false;
    bool _expectations_initialized = false;
};

} // namespace distributions
} // namespace axiom

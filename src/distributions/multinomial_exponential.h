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

namespace axiom {
namespace distributions {

// Default event dimension for multinomial
constexpr int MULTINOMIAL_DEFAULT_EVENT_DIM = 1;

/**
 * @brief Multinomial (Categorical) distribution in exponential family form.
 * 
 * The likelihood is given by:
 *   log p(x|θ) = x · θ - log(sum(exp(θ)))
 * 
 * Natural parameters:
 *   - logits = θ (unnormalized log probabilities)
 * 
 * Sufficient statistics:
 *   - x = x (one-hot encoding for categorical)
 * 
 * Log partition:
 *   A(θ) = log(sum(exp(θ))) = log_normalizer
 */
template<typename Scalar>
class Multinomial : public ExponentialFamily<Scalar> {
public:
    // Cached log normalizer
    std::vector<Scalar> _logZ;

    Multinomial(
        const ArrayDict<std::vector<Scalar>>& nat_params = {},
        const ArrayDict<std::vector<Scalar>>& expectations = {},
        const std::vector<int>& batch_shape = {},
        const std::vector<int>& event_shape = {},
        int event_dim = MULTINOMIAL_DEFAULT_EVENT_DIM,
        const std::vector<Scalar>& input_logZ = {});

    // Factory method
    static ArrayDict<std::vector<Scalar>> init_default_params(
        const std::vector<int>& batch_shape,
        const std::vector<int>& event_shape);

    // Property accessors
    std::vector<Scalar>& logits();
    const std::vector<Scalar>& logits() const;
    
    std::vector<Scalar>& log_normalizer();
    const std::vector<Scalar>& log_normalizer() const;
    
    std::vector<Scalar> mean() const;  // Probabilities (softmax of logits)
    std::vector<Scalar> variance() const;
    std::vector<Scalar> log_mean() const;

    // Exponential family implementations
    std::vector<Scalar> log_likelihood(const std::vector<Scalar>& x) const override;
    ArrayDict<std::vector<Scalar>> statistics(const std::vector<Scalar>& x) const override;
    ArrayDict<std::vector<Scalar>> expected_statistics() const override;
    std::vector<Scalar> log_partition() const override;
    std::vector<Scalar> log_measure(const std::vector<Scalar>& x) const override;
    std::vector<Scalar> expected_log_measure() const override;
    std::vector<Scalar> entropy() const override;
    std::vector<Scalar> sample(const std::vector<Scalar>& key, 
                               const std::vector<int>& shape) const override;
    ArrayDict<std::vector<Scalar>> params_from_statistics(
        const ArrayDict<std::vector<Scalar>>& stats) const override;

    // Additional methods
    std::vector<Scalar> expected_x() const;
    std::vector<Scalar> expected_xx() const;

    // Multiplication with another Multinomial
    std::shared_ptr<Multinomial<Scalar>> operator*(
        const std::shared_ptr<Multinomial<Scalar>>& other) const;

protected:
    void _update_cache() override;
    static std::unordered_map<std::string, std::string> params_to_tx();
    
    // Stable softmax computation
    std::vector<Scalar> _stable_softmax(const std::vector<Scalar>& x) const;
    std::vector<Scalar> _stable_logsumexp(const std::vector<Scalar>& x) const;
};

} // namespace distributions
} // namespace axiom

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

namespace axiom {
namespace distributions {

// Default event dimension for Delta
constexpr int DELTA_DEFAULT_EVENT_DIM = 1;

/**
 * @brief Delta (Dirac) distribution - point mass at a single point.
 * 
 * The delta distribution places all probability mass at a single point p:
 *   p(x) = δ(x - p)
 * 
 * This is useful for:
 *   - Point estimates in variational inference
 *   - Message passing with exact values
 *   - Observed data points
 */
template<typename Scalar>
class Delta : public Distribution {
public:
    // Point mass location
    std::vector<Scalar> p;
    
    // Dummy fields for conjugate compatibility
    ArrayDict<std::vector<Scalar>> posterior_params;
    ArrayDict<std::vector<Scalar>> prior_params;
    std::vector<Scalar> residual;

    Delta(const std::vector<int>& batch_shape,
          const std::vector<int>& event_shape,
          const std::vector<Scalar>& p_values = {});

    // Properties
    std::vector<Scalar> get_p() const;
    std::vector<Scalar> mean() const;
    std::vector<Scalar> expected_x() const;
    std::vector<Scalar> expected_xx() const;

    // Distribution methods
    std::vector<Scalar> log_partition() const;
    std::vector<Scalar> entropy() const;
    std::vector<Scalar> sample() const;
    
    // Conjugate-like methods for compatibility
    std::vector<Scalar> expected_log_likelihood(const std::vector<Scalar>& x) const;
    std::vector<Scalar> expected_statistics() const;
    
    void update_from_data(const std::vector<Scalar>& x,
                          const std::vector<Scalar>& weights = {},
                          Scalar beta = Scalar(0.0),
                          Scalar lr = Scalar(1.0));
    
    void update_from_probabilities(const std::vector<Scalar>& x,
                                   const std::vector<Scalar>& weights = {});

    // Copy
    std::shared_ptr<Delta<Scalar>> copy() const;

    // Multiplication (returns self)
    std::shared_ptr<Delta<Scalar>> operator*(const std::shared_ptr<Delta<Scalar>>& other) const;
    std::shared_ptr<Delta<Scalar>> operator*(const std::shared_ptr<Distribution>& other) const;
};

} // namespace distributions
} // namespace axiom

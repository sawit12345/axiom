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

#include "exponential_family.h"
#include <numeric>
#include <stdexcept>

namespace axiom {
namespace distributions {

template<typename Scalar>
ExponentialFamily<Scalar>::ExponentialFamily(
    int default_event_dim,
    const std::vector<int>& batch_shape,
    const std::vector<int>& event_shape,
    const ArrayDict<std::vector<Scalar>>& nat_params,
    const ArrayDict<std::vector<Scalar>>& expectations,
    const std::vector<Scalar>& residual)
    : Distribution(default_event_dim, batch_shape, event_shape),
      _nat_params(nat_params),
      _expectations(expectations),
      _residual(residual) {
    
    // Must provide one of nat_params or expectations
    if (nat_params.data.empty() && expectations.data.empty()) {
        throw std::invalid_argument("Must provide one of nat_params or expectations");
    }
    
    if (residual.empty()) {
        // Initialize with empty batch shape
        _residual.resize(std::accumulate(batch_shape.begin(), batch_shape.end(), 
                                          1, std::multiplies<int>()), Scalar(0));
    }
}

template<typename Scalar>
ArrayDict<std::vector<Scalar>>& ExponentialFamily<Scalar>::nat_params() {
    if (!_nat_params_initialized && _nat_params.data.empty()) {
        _nat_params = params_from_statistics(_expectations);
        _nat_params_initialized = true;
    }
    return _nat_params;
}

template<typename Scalar>
const ArrayDict<std::vector<Scalar>>& ExponentialFamily<Scalar>::nat_params() const {
    if (!_nat_params_initialized && _nat_params.data.empty()) {
        const_cast<ExponentialFamily*>(this)->_nat_params = 
            const_cast<ExponentialFamily*>(this)->params_from_statistics(_expectations);
        const_cast<ExponentialFamily*>(this)->_nat_params_initialized = true;
    }
    return _nat_params;
}

template<typename Scalar>
void ExponentialFamily<Scalar>::set_nat_params(const ArrayDict<std::vector<Scalar>>& value) {
    _validate_nat_params(value);
    _nat_params = value;
    _nat_params_initialized = true;
    _update_cache();
}

template<typename Scalar>
ArrayDict<std::vector<Scalar>>& ExponentialFamily<Scalar>::expectations() {
    if (!_expectations_initialized && _expectations.data.empty()) {
        _expectations = expected_statistics();
        _expectations_initialized = true;
    }
    return _expectations;
}

template<typename Scalar>
const ArrayDict<std::vector<Scalar>>& ExponentialFamily<Scalar>::expectations() const {
    if (!_expectations_initialized && _expectations.data.empty()) {
        const_cast<ExponentialFamily*>(this)->_expectations = 
            const_cast<ExponentialFamily*>(this)->expected_statistics();
        const_cast<ExponentialFamily*>(this)->_expectations_initialized = true;
    }
    return _expectations;
}

template<typename Scalar>
void ExponentialFamily<Scalar>::set_expectations(const ArrayDict<std::vector<Scalar>>& value) {
    _expectations = value;
    _expectations_initialized = true;
    _update_cache();
}

template<typename Scalar>
std::vector<Scalar>& ExponentialFamily<Scalar>::residual() {
    if (_residual.empty()) {
        _residual.resize(std::accumulate(batch_shape.begin(), batch_shape.end(), 
                                          1, std::multiplies<int>()), Scalar(0));
    }
    return _residual;
}

template<typename Scalar>
const std::vector<Scalar>& ExponentialFamily<Scalar>::residual() const {
    if (_residual.empty()) {
        const_cast<ExponentialFamily*>(this)->_residual.resize(
            std::accumulate(batch_shape.begin(), batch_shape.end(), 
                            1, std::multiplies<int>()), Scalar(0));
    }
    return _residual;
}

template<typename Scalar>
void ExponentialFamily<Scalar>::set_residual(const std::vector<Scalar>& value) {
    _residual = value;
}

template<typename Scalar>
std::vector<Scalar> ExponentialFamily<Scalar>::log_likelihood(const std::vector<Scalar>& x) const {
    // log p(x|θ) = S(θ) · T(x) - A(S(θ))
    auto probs = params_dot_statistics(x);
    auto partition = log_partition();
    
    // Subtract partition: probs - partition
    std::vector<Scalar> result(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        result[i] = probs[i] - partition[i % partition.size()];
    }
    
    return sum_events(result);
}

template<typename Scalar>
std::vector<Scalar> ExponentialFamily<Scalar>::entropy() const {
    // H(p) = -<T(x)|S(θ))> · S(θ) + A(S(θ)) - <log φ(x)|S(θ)>
    auto params_dot_exp_stats = params_dot_expected_statistics();
    auto partition = log_partition();
    auto exp_log_measure = expected_log_measure();
    
    std::vector<Scalar> result(params_dot_exp_stats.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = -params_dot_exp_stats[i] + partition[i % partition.size()] - 
                    exp_log_measure[i % exp_log_measure.size()];
    }
    
    return sum_events(result);
}

template<typename Scalar>
std::vector<Scalar> ExponentialFamily<Scalar>::params_dot_statistics(const std::vector<Scalar>& x) const {
    auto stats = statistics(x);
    auto mapping = _get_params_to_stats_mapping();
    return _map_and_multiply(nat_params(), stats, default_event_dim, mapping);
}

template<typename Scalar>
std::vector<Scalar> ExponentialFamily<Scalar>::params_dot_expected_statistics() const {
    auto mapping = _get_params_to_stats_mapping();
    return _map_and_multiply(nat_params(), expectations(), default_event_dim, mapping);
}

template<typename Scalar>
std::shared_ptr<ExponentialFamily<Scalar>> ExponentialFamily<Scalar>::combine(
    const std::shared_ptr<ExponentialFamily<Scalar>>& other) const {
    
    // Check if same class
    if (typeid(*other) != typeid(*this)) {
        throw std::invalid_argument("Cannot combine distributions of different types");
    }
    
    // Sum natural parameters
    auto combined_params = _nat_params.map_pairwise(other->_nat_params, 
        [](const std::vector<Scalar>& a, const std::vector<Scalar>& b) {
            std::vector<Scalar> result(a.size());
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] + b[i % b.size()];
            }
            return result;
        });
    
    // Create new instance with combined params
    // Note: This requires a virtual create method or factory
    return nullptr;  // Subclasses must override
}

template<typename Scalar>
std::shared_ptr<ExponentialFamily<Scalar>> ExponentialFamily<Scalar>::operator*(
    const std::shared_ptr<ExponentialFamily<Scalar>>& other) const {
    return combine(other);
}

template<typename Scalar>
std::shared_ptr<Distribution> ExponentialFamily<Scalar>::expand(const std::vector<int>& shape) {
    // Expand parameters into larger batch shape
    // TODO: Generalize using tree operations
    return nullptr;
}

template<typename Scalar>
void ExponentialFamily<Scalar>::_validate_nat_params(
    const ArrayDict<std::vector<Scalar>>& nat_params) const {
    
    auto mapping = _get_params_to_stats_mapping();
    
    // Check keys match
    for (const auto& pair : nat_params.data) {
        if (mapping.find(pair.first) == mapping.end()) {
            throw std::invalid_argument("Invalid natural parameter: " + pair.first);
        }
    }
    
    // Validate shapes
    for (const auto& pair : nat_params.data) {
        if (static_cast<int>(pair.second.size()) < default_event_dim) {
            throw std::invalid_argument("Invalid shape for natural parameter: " + pair.first);
        }
    }
}

template<typename Scalar>
void ExponentialFamily<Scalar>::_update_cache() {
    // Called when parameters are updated
    // Subclasses can override to update derived quantities
}

template<typename Scalar>
std::unordered_map<std::string, std::string> 
ExponentialFamily<Scalar>::_get_params_to_stats_mapping() const {
    // Subclasses should define params_to_tx static member
    return {};
}

template<typename Scalar>
std::vector<Scalar> ExponentialFamily<Scalar>::_map_and_multiply(
    const ArrayDict<std::vector<Scalar>>& params,
    const ArrayDict<std::vector<Scalar>>& stats,
    int default_event_dim,
    const std::unordered_map<std::string, std::string>& mapping) const {
    
    std::vector<Scalar> result;
    
    // Map params to stats and compute dot product
    for (const auto& param_pair : params.data) {
        const std::string& param_key = param_pair.first;
        const std::vector<Scalar>& param_val = param_pair.second;
        
        auto it = mapping.find(param_key);
        if (it != mapping.end()) {
            const std::string& stat_key = it->second;
            if (stats.has(stat_key)) {
                const std::vector<Scalar>& stat_val = stats.get(stat_key);
                
                // Element-wise multiplication and sum over event dims
                // This is a simplified version - proper implementation needs shape handling
                for (size_t i = 0; i < param_val.size(); ++i) {
                    if (i < result.size()) {
                        result[i] += param_val[i] * stat_val[i % stat_val.size()];
                    } else {
                        result.push_back(param_val[i] * stat_val[i % stat_val.size()]);
                    }
                }
            }
        }
    }
    
    return result;
}

// Explicit template instantiation
template class ExponentialFamily<float>;
template class ExponentialFamily<double>;

} // namespace distributions
} // namespace axiom

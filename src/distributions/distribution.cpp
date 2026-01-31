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

#include "distribution.h"
#include <numeric>
#include <algorithm>

namespace axiom {
namespace distributions {

Distribution::Distribution(int default_event_dim,
                           const std::vector<int>& batch_shape,
                           const std::vector<int>& event_shape)
    : default_event_dim(default_event_dim),
      batch_shape(batch_shape),
      event_shape(event_shape) {
    this->event_dim = static_cast<int>(event_shape.size());
    this->batch_dim = static_cast<int>(batch_shape.size());
    this->dim = (event_shape.size() >= static_cast<size_t>(default_event_dim)) 
                ? event_shape[event_shape.size() - default_event_dim] 
                : 0;
}

std::vector<int> Distribution::shape() const {
    std::vector<int> result = batch_shape;
    result.insert(result.end(), event_shape.begin(), event_shape.end());
    return result;
}

std::shared_ptr<Distribution> Distribution::to_event(int n) const {
    // This is a base implementation - subclasses should override
    return nullptr;
}

std::vector<int> Distribution::get_sample_shape(const std::vector<int>& data_shape) const {
    int num_sample_dims = static_cast<int>(data_shape.size()) - event_dim - batch_dim;
    if (num_sample_dims <= 0) return {};
    return std::vector<int>(data_shape.begin(), data_shape.begin() + num_sample_dims);
}

std::vector<int> Distribution::get_batch_shape_from_data(const std::vector<int>& data_shape) const {
    int num_sample_dims = static_cast<int>(data_shape.size()) - event_dim - batch_dim;
    int start = num_sample_dims;
    int end = num_sample_dims + batch_dim;
    if (start >= static_cast<int>(data_shape.size()) || end > static_cast<int>(data_shape.size())) {
        return {};
    }
    return std::vector<int>(data_shape.begin() + start, data_shape.begin() + end);
}

std::vector<int> Distribution::get_sample_dims(int data_ndim) const {
    int num_sample_dims = data_ndim - event_dim - batch_dim;
    std::vector<int> result;
    for (int i = 0; i < num_sample_dims; ++i) {
        result.push_back(i);
    }
    return result;
}

std::vector<int> Distribution::get_event_dims() const {
    std::vector<int> result;
    int total_ndim = batch_dim + event_dim;
    for (int i = -event_dim; i < 0; ++i) {
        result.push_back(total_ndim + i);
    }
    return result;
}

std::pair<std::vector<std::shared_ptr<void>>, std::vector<std::shared_ptr<void>>>
Distribution::tree_flatten() const {
    std::vector<std::shared_ptr<void>> data_values;
    std::vector<std::shared_ptr<void>> aux_values;
    
    auto data_fields = gather_pytree_data_fields();
    auto aux_fields = gather_pytree_aux_fields();
    
    // Note: In actual implementation, we'd use reflection or registration
    // to properly serialize the fields
    
    return {data_values, aux_values};
}

void Distribution::tree_unflatten(const std::vector<std::shared_ptr<void>>& aux_data,
                                  const std::vector<std::shared_ptr<void>>& params) {
    // Reconstruct from flattened representation
    // Subclasses should implement field restoration
}

void Distribution::tree_unflatten_and_replace(
    const std::vector<std::shared_ptr<void>>& aux_data,
    const std::vector<std::shared_ptr<void>>& params,
    const std::unordered_map<std::string, std::shared_ptr<void>>& replace) {
    tree_unflatten(aux_data, params);
    
    // Apply replacements
    for (const auto& pair : replace) {
        // Set field if it exists and is in aux_fields
    }
}

std::shared_ptr<Distribution> Distribution::swap_axes(int axis1, int axis2) const {
    // Base implementation - subclasses override
    return nullptr;
}

std::shared_ptr<Distribution> Distribution::moveaxis(int source, int destination) const {
    // Base implementation - subclasses override
    return nullptr;
}

std::shared_ptr<Distribution> Distribution::expand_batch_shape(
    const std::vector<int>& batch_relative_axes) const {
    // Base implementation - subclasses override
    return nullptr;
}

std::pair<std::vector<int>, std::vector<int>> 
Distribution::infer_shapes(const std::vector<int>& tensor_shape, int event_dim) const {
    int batch_size = static_cast<int>(tensor_shape.size()) - event_dim;
    std::vector<int> batch_shape(tensor_shape.begin(), tensor_shape.begin() + batch_size);
    std::vector<int> event_shape(tensor_shape.begin() + batch_size, tensor_shape.end());
    return {batch_shape, event_shape};
}

std::vector<std::string> Distribution::gather_pytree_data_fields() const {
    // Collect from class hierarchy
    return pytree_data_fields;
}

std::vector<std::string> Distribution::gather_pytree_aux_fields() const {
    // Collect from class hierarchy  
    std::vector<std::string> result = pytree_aux_fields;
    result.push_back("dim");
    result.push_back("default_event_dim");
    result.push_back("batch_dim");
    result.push_back("event_dim");
    result.push_back("batch_shape");
    result.push_back("event_shape");
    return result;
}

} // namespace distributions
} // namespace axiom

// Copyright 2025 VERSES AI, Inc.
//
// Licensed under the VERSES Academic Research License (the "License");
// you may not use this file except in compliance with the license.
//
// You may obtain a copy of the License at
//
//     https://github.com/VersesTech/axiom/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transform.h"
#include <stdexcept>

namespace axiom {
namespace transforms {

// Base Transform implementation is header-only as it's an abstract class
// Concrete implementations will provide the actual functionality

// Utility functions that can be shared across transform implementations

namespace utils {

/**
 * @brief Compute the shape of a tensor given event dimensions
 */
inline std::vector<int> compute_shape(const std::vector<int>& shape, int event_dim) {
    if (static_cast<int>(shape.size()) < event_dim) {
        throw std::invalid_argument("Shape must have at least event_dim dimensions");
    }
    return std::vector<int>(shape.begin(), shape.end() - event_dim);
}

/**
 * @brief Compute batch shape from full shape
 */
inline std::vector<int> batch_shape_from_full(const std::vector<int>& shape, int batch_dim) {
    if (static_cast<int>(shape.size()) < batch_dim) {
        throw std::invalid_argument("Shape must have at least batch_dim dimensions");
    }
    return std::vector<int>(shape.begin(), shape.begin() + batch_dim);
}

/**
 * @brief Compute event shape from full shape
 */
inline std::vector<int> event_shape_from_full(const std::vector<int>& shape, int event_dim) {
    if (static_cast<int>(shape.size()) < event_dim) {
        throw std::invalid_argument("Shape must have at least event_dim dimensions");
    }
    return std::vector<int>(shape.end() - event_dim, shape.end());
}

/**
 * @brief Validate that two shapes are compatible for broadcasting
 */
inline bool shapes_compatible(const std::vector<int>& shape1, const std::vector<int>& shape2) {
    size_t max_len = std::max(shape1.size(), shape2.size());
    
    for (size_t i = 0; i < max_len; ++i) {
        int dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        int dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Compute broadcasted shape from two shapes
 */
inline std::vector<int> broadcast_shapes(
    const std::vector<int>& shape1, 
    const std::vector<int>& shape2) {
    size_t max_len = std::max(shape1.size(), shape2.size());
    std::vector<int> result(max_len);
    
    for (size_t i = 0; i < max_len; ++i) {
        int dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        int dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;
        result[max_len - 1 - i] = std::max(dim1, dim2);
    }
    
    return result;
}

} // namespace utils

} // namespace transforms
} // namespace axiom

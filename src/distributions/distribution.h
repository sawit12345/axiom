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

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

namespace axiom {
namespace distributions {

// Forward declarations
template<typename T>
class ArrayView;

/**
 * @brief Base class for probability distributions with PyTree-like functionality.
 * 
 * This class provides the foundation for all distributions in AXIOM, implementing
 * PyTree-style flattening/unflattening for JAX compatibility and supporting
 * batch and event shapes for efficient parallel computation.
 */
class Distribution {
public:
    // Shape information
    int dim;                           // Dimensionality of the event
    int event_dim;                     // Number of event dimensions
    int batch_dim;                     // Number of batch dimensions  
    int default_event_dim;             // Default event dimensions for this distribution type
    std::vector<int> event_shape;      // Shape of event dimensions
    std::vector<int> batch_shape;      // Shape of batch dimensions

    // PyTree fields specification
    std::vector<std::string> pytree_data_fields;
    std::vector<std::string> pytree_aux_fields;

    Distribution(int default_event_dim, 
                 const std::vector<int>& batch_shape,
                 const std::vector<int>& event_shape);
    
    virtual ~Distribution() = default;

    // Shape-related methods
    std::vector<int> shape() const;
    
    // Convert to event distribution
    virtual std::shared_ptr<Distribution> to_event(int n) const;
    
    // Shape inference helpers
    std::vector<int> get_sample_shape(const std::vector<int>& data_shape) const;
    std::vector<int> get_batch_shape_from_data(const std::vector<int>& data_shape) const;
    std::vector<int> get_sample_dims(int data_ndim) const;
    std::vector<int> get_event_dims() const;
    
    // Reduction operations - inline implementations
    template<typename T>
    T sum_events(const T& x, bool keepdims = false) const {
        return x;  // Placeholder
    }
    
    template<typename T>
    T sum_default_events(const T& x, bool keepdims = false) const {
        return x;  // Placeholder
    }
    
    // Expansion operations - inline implementations
    template<typename T>
    T expand_event_dims(const T& x) const {
        return x;  // Placeholder
    }
    
    template<typename T>
    T expand_default_event_dims(const T& x) const {
        return x;  // Placeholder
    }
    
    template<typename T>
    T expand_batch_dims(const T& x) const {
        return x;  // Placeholder
    }
    
    // PyTree operations (JAX-compatible)
    virtual std::pair<std::vector<std::shared_ptr<void>>, std::vector<std::shared_ptr<void>>>
    tree_flatten() const;
    
    virtual void tree_unflatten(const std::vector<std::shared_ptr<void>>& aux_data,
                                const std::vector<std::shared_ptr<void>>& params);
    
    virtual void tree_unflatten_and_replace(const std::vector<std::shared_ptr<void>>& aux_data,
                                            const std::vector<std::shared_ptr<void>>& params,
                                            const std::unordered_map<std::string, std::shared_ptr<void>>& replace);
    
    // Axis manipulation
    virtual std::shared_ptr<Distribution> swap_axes(int axis1, int axis2) const;
    virtual std::shared_ptr<Distribution> moveaxis(int source, int destination) const;
    virtual std::shared_ptr<Distribution> expand_batch_shape(const std::vector<int>& batch_relative_axes) const;
    
    // Copy
    virtual std::shared_ptr<Distribution> copy() const = 0;
    
    // Shape inference
    std::pair<std::vector<int>, std::vector<int>> 
    infer_shapes(const std::vector<int>& tensor_shape, int event_dim) const;

protected:
    // Gather PyTree fields from class hierarchy
    std::vector<std::string> gather_pytree_data_fields() const;
    std::vector<std::string> gather_pytree_aux_fields() const;
};

/**
 * @brief Array dictionary for PyTree-like parameter storage.
 * 
 * Mimics JAX's PyTree behavior with named array storage.
 */
template<typename T>
class ArrayDict {
public:
    std::unordered_map<std::string, T> data;
    
    ArrayDict() = default;
    
    void set(const std::string& key, const T& value) {
        data[key] = value;
    }
    
    T& get(const std::string& key) {
        return data.at(key);
    }
    
    const T& get(const std::string& key) const {
        return data.at(key);
    }
    
    bool has(const std::string& key) const {
        return data.find(key) != data.end();
    }
    
    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        for (const auto& pair : data) {
            result.push_back(pair.first);
        }
        return result;
    }
    
    // Apply operation to all arrays
    template<typename Func>
    ArrayDict<T> map(Func&& func) const {
        ArrayDict<T> result;
        for (const auto& pair : data) {
            result.set(pair.first, func(pair.second));
        }
        return result;
    }
    
    // Apply operation pairwise
    template<typename Func>
    ArrayDict<T> map_pairwise(const ArrayDict<T>& other, Func&& func) const {
        ArrayDict<T> result;
        for (const auto& pair : data) {
            if (other.has(pair.first)) {
                result.set(pair.first, func(pair.second, other.get(pair.first)));
            }
        }
        return result;
    }
    
};

// Type alias for float arrays using column-major Eigen-like storage
template<typename Scalar>
using Array = std::vector<Scalar>;

} // namespace distributions
} // namespace axiom

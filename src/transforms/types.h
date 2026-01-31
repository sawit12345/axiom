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

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>

namespace axiom {
namespace transforms {

/**
 * @brief Multi-dimensional array structure for numerical computations.
 * 
 * This is a simplified tensor/array structure that supports basic
 * operations needed for the Linear Matrix Normal-Gamma transform.
 * For production use, this would wrap a library like Eigen, cuBLAS,
 * or a custom CUDA tensor library.
 */
struct Array {
    std::vector<float> data;
    std::vector<int> shape;
    
    // Constructors
    Array() = default;
    explicit Array(const std::vector<int>& shape, float init_val = 0.0f);
    Array(const std::vector<float>& data, const std::vector<int>& shape);
    
    // Basic properties
    size_t size() const { return data.size(); }
    int ndim() const { return static_cast<int>(shape.size()); }
    int numel() const { return static_cast<int>(data.size()); }
    
    // Access
    float& operator()(const std::vector<int>& indices);
    float operator()(const std::vector<int>& indices) const;
    float& operator[](size_t idx) { return data[idx]; }
    float operator[](size_t idx) const { return data[idx]; }
    
    // Linear index access
    size_t compute_index(const std::vector<int>& indices) const;
    
    // Shape operations
    Array reshape(const std::vector<int>& new_shape) const;
    Array squeeze() const;
    Array squeeze(int dim) const;
    Array expand_dims(int dim) const;
    
    // Element-wise operations
    Array operator+(const Array& other) const;
    Array operator-(const Array& other) const;
    Array operator*(const Array& other) const;
    Array operator/(const Array& other) const;
    Array operator+(float scalar) const;
    Array operator-(float scalar) const;
    Array operator*(float scalar) const;
    Array operator/(float scalar) const;
    
    // In-place operations
    Array& operator+=(const Array& other);
    Array& operator-=(const Array& other);
    Array& operator*=(const Array& other);
    Array& operator/=(const Array& other);
    Array& operator+=(float scalar);
    Array& operator-=(float scalar);
    Array& operator*=(float scalar);
    Array& operator/=(float scalar);
    
    // Reductions
    float sum() const;
    Array sum(int dim, bool keepdims = false) const;
    Array sum(const std::vector<int>& dims, bool keepdims = false) const;
    
    // Utility
    Array transpose() const;  // For 2D arrays
    Array transpose(int dim1, int dim2) const;
    Array diagonal(int axis1 = -2, int axis2 = -1) const;
    Array clip(float min_val, float max_val) const;
    
    // Static constructors
    static Array zeros(const std::vector<int>& shape);
    static Array ones(const std::vector<int>& shape);
    static Array eye(int n);
    static Array full(const std::vector<int>& shape, float value);
    static Array randn(const std::vector<int>& shape, unsigned int seed = 0);
    static Array uniform(const std::vector<int>& shape, float min, float max, unsigned int seed = 0);
};

/**
 * @brief Dictionary-like structure for storing multiple named arrays.
 * 
 * Used to store parameters, statistics, and natural parameters
 * in a flexible, extensible format.
 */
struct ArrayDict {
    std::unordered_map<std::string, Array> data;
    
    // Constructors
    ArrayDict() = default;
    explicit ArrayDict(const std::unordered_map<std::string, Array>& data);
    
    // Access
    Array& operator[](const std::string& key) { return data[key]; }
    const Array& operator[](const std::string& key) const { return data.at(key); }
    Array& at(const std::string& key) { return data.at(key); }
    const Array& at(const std::string& key) const { return data.at(key); }
    
    // Check existence
    bool has(const std::string& key) const { return data.find(key) != data.end(); }
    bool contains(const std::string& key) const { return has(key); }
    
    // Insert
    void insert(const std::string& key, const Array& value) { data[key] = value; }
    void set(const std::string& key, const Array& value) { data[key] = value; }
    
    // Operations
    ArrayDict operator+(const ArrayDict& other) const;
    ArrayDict operator-(const ArrayDict& other) const;
    ArrayDict operator*(float scalar) const;
    ArrayDict operator/(float scalar) const;
    
    // Tree operations (apply function to all arrays)
    ArrayDict map(const std::function<Array(const Array&)>& fn) const;
    ArrayDict map2(const ArrayDict& other, const std::function<Array(const Array&, const Array&)>& fn) const;
    
    // Broadcasting and shape utilities
    ArrayDict broadcast_to(const std::vector<int>& shape) const;
    ArrayDict sum(int dim, bool keepdims = false) const;
    ArrayDict sum(const std::vector<int>& dims, bool keepdims = false) const;
    
    // Static factory
    static ArrayDict empty();
    static ArrayDict from_dict(const std::unordered_map<std::string, Array>& data);
};

/**
 * @brief Compute matrix multiplication for batched 2D arrays.
 * @param a Left matrix (..., m, k)
 * @param b Right matrix (..., k, n)
 * @return Result (..., m, n)
 */
Array matmul(const Array& a, const Array& b);

/**
 * @brief Batched matrix multiplication with broadcasting.
 */
Array batch_matmul(const Array& a, const Array& b);

/**
 * @brief Compute matrix inverse using Cholesky decomposition for symmetric positive definite matrices.
 */
Array cholesky_inverse(const Array& matrix);

/**
 * @brief Compute inverse and log-determinant together.
 * @return Pair of (inverse, log_determinant)
 */
std::pair<Array, float> inv_and_logdet(const Array& matrix);

/**
 * @brief Compute Cholesky decomposition.
 */
Array cholesky(const Array& matrix);

/**
 * @brief Solve linear system using Cholesky factorization.
 */
Array cholesky_solve(const Array& L, const Array& b);

/**
 * @brief Compute digamma function (psi function).
 */
float digamma(float x);
Array digamma(const Array& x);

/**
 * @brief Compute log-gamma function.
 */
float gammaln(float x);
Array gammaln(const Array& x);

/**
 * @brief Compute log-sum-exp for numerical stability.
 */
float logsumexp(const std::vector<float>& x);
Array logsumexp(const Array& x, int dim, bool keepdims = false);

/**
 * @brief Numerically stable log(1 + exp(x)).
 */
float log1pexp(float x);
Array log1pexp(const Array& x);

} // namespace transforms
} // namespace axiom

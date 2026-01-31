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

#include "types.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>

namespace axiom {
namespace transforms {

// ============================================================================
// Array Implementation
// ============================================================================

Array::Array(const std::vector<int>& shape_, float init_val) : shape(shape_) {
    size_t total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }
    data.resize(total_size, init_val);
}

Array::Array(const std::vector<float>& data_, const std::vector<int>& shape_) 
    : data(data_), shape(shape_) {
    size_t expected = 1;
    for (int dim : shape) expected *= dim;
    if (data.size() != expected) {
        throw std::invalid_argument("Data size doesn't match shape");
    }
}

float& Array::operator()(const std::vector<int>& indices) {
    return data[compute_index(indices)];
}

float Array::operator()(const std::vector<int>& indices) const {
    return data[compute_index(indices)];
}

size_t Array::compute_index(const std::vector<int>& indices) const {
    if (static_cast<int>(indices.size()) != ndim()) {
        throw std::invalid_argument("Number of indices doesn't match array dimensions");
    }
    
    size_t index = 0;
    size_t stride = 1;
    for (int i = ndim() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}

Array Array::reshape(const std::vector<int>& new_shape) const {
    size_t new_size = 1;
    for (int dim : new_shape) new_size *= dim;
    
    if (new_size != data.size()) {
        throw std::invalid_argument("Cannot reshape: size mismatch");
    }
    
    Array result;
    result.data = data;
    result.shape = new_shape;
    return result;
}

Array Array::squeeze() const {
    std::vector<int> new_shape;
    for (int dim : shape) {
        if (dim != 1) new_shape.push_back(dim);
    }
    if (new_shape.empty()) new_shape.push_back(1);
    return reshape(new_shape);
}

Array Array::squeeze(int dim) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("Dimension out of range");
    }
    if (shape[dim] != 1) {
        throw std::invalid_argument("Cannot squeeze dimension with size != 1");
    }
    
    std::vector<int> new_shape = shape;
    new_shape.erase(new_shape.begin() + dim);
    return reshape(new_shape);
}

Array Array::expand_dims(int dim) const {
    if (dim < 0) dim += ndim() + 1;
    if (dim < 0 || dim > ndim()) {
        throw std::out_of_range("Dimension out of range");
    }
    
    std::vector<int> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(new_shape);
}

Array Array::transpose() const {
    if (ndim() != 2) {
        throw std::invalid_argument("Transpose only supported for 2D arrays");
    }
    
    int m = shape[0], n = shape[1];
    Array result({n, m});
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result({j, i}) = (*this)({i, j});
        }
    }
    return result;
}

Array Array::transpose(int dim1, int dim2) const {
    if (dim1 < 0) dim1 += ndim();
    if (dim2 < 0) dim2 += ndim();
    
    std::vector<int> new_shape = shape;
    std::swap(new_shape[dim1], new_shape[dim2]);
    
    Array result(new_shape);
    // Simplified for common 2D case
    if (ndim() == 2 && ((dim1 == 0 && dim2 == 1) || (dim1 == 1 && dim2 == 0))) {
        return transpose();
    }
    
    // General case would require more complex indexing
    throw std::runtime_error("General transpose not fully implemented");
}

Array Array::diagonal(int axis1, int axis2) const {
    if (axis1 < 0) axis1 += ndim();
    if (axis2 < 0) axis2 += ndim();
    
    int dim = std::min(shape[axis1], shape[axis2]);
    std::vector<int> new_shape;
    
    // Build new shape excluding axis1 and axis2, then add dim
    for (int i = 0; i < ndim(); ++i) {
        if (i != axis1 && i != axis2) {
            new_shape.push_back(shape[i]);
        }
    }
    new_shape.push_back(dim);
    new_shape.push_back(1);  // Keep 2D shape like Python
    
    Array result(new_shape);
    
    // Extract diagonal elements
    // For simplicity, assuming 2D input
    if (ndim() == 2) {
        for (int i = 0; i < dim; ++i) {
            result[i] = (*this)({i, i});
        }
    }
    
    return result;
}

Array Array::clip(float min_val, float max_val) const {
    Array result = *this;
    for (float& val : result.data) {
        val = std::max(min_val, std::min(max_val, val));
    }
    return result;
}

// Element-wise operations
Array Array::operator+(const Array& other) const {
    Array result = *this;
    result += other;
    return result;
}

Array Array::operator-(const Array& other) const {
    Array result = *this;
    result -= other;
    return result;
}

Array Array::operator*(const Array& other) const {
    Array result = *this;
    result *= other;
    return result;
}

Array Array::operator/(const Array& other) const {
    Array result = *this;
    result /= other;
    return result;
}

Array Array::operator+(float scalar) const {
    Array result = *this;
    result += scalar;
    return result;
}

Array Array::operator-(float scalar) const {
    Array result = *this;
    result -= scalar;
    return result;
}

Array Array::operator*(float scalar) const {
    Array result = *this;
    result *= scalar;
    return result;
}

Array Array::operator/(float scalar) const {
    Array result = *this;
    result /= scalar;
    return result;
}

// In-place operations
Array& Array::operator+=(const Array& other) {
    if (shape != other.shape) {
        // Try broadcasting
        if (numel() == other.numel()) {
            // Same number of elements, treat as compatible
        } else if (other.numel() == 1) {
            *this += other.data[0];
            return *this;
        } else {
            throw std::invalid_argument("Shape mismatch in array addition");
        }
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i % other.data.size()];
    }
    return *this;
}

Array& Array::operator-=(const Array& other) {
    if (shape != other.shape) {
        if (numel() == other.numel()) {
        } else if (other.numel() == 1) {
            *this -= other.data[0];
            return *this;
        } else {
            throw std::invalid_argument("Shape mismatch in array subtraction");
        }
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i % other.data.size()];
    }
    return *this;
}

Array& Array::operator*=(const Array& other) {
    if (shape != other.shape) {
        if (numel() == other.numel()) {
        } else if (other.numel() == 1) {
            *this *= other.data[0];
            return *this;
        } else {
            throw std::invalid_argument("Shape mismatch in array multiplication");
        }
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= other.data[i % other.data.size()];
    }
    return *this;
}

Array& Array::operator/=(const Array& other) {
    if (shape != other.shape) {
        if (numel() == other.numel()) {
        } else if (other.numel() == 1) {
            *this /= other.data[0];
            return *this;
        } else {
            throw std::invalid_argument("Shape mismatch in array division");
        }
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] /= other.data[i % other.data.size()];
    }
    return *this;
}

Array& Array::operator+=(float scalar) {
    for (float& val : data) val += scalar;
    return *this;
}

Array& Array::operator-=(float scalar) {
    for (float& val : data) val -= scalar;
    return *this;
}

Array& Array::operator*=(float scalar) {
    for (float& val : data) val *= scalar;
    return *this;
}

Array& Array::operator/=(float scalar) {
    for (float& val : data) val /= scalar;
    return *this;
}

// Reductions
float Array::sum() const {
    return std::accumulate(data.begin(), data.end(), 0.0f);
}

Array Array::sum(int dim, bool keepdims) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("Dimension out of range");
    }
    
    std::vector<int> new_shape;
    for (int i = 0; i < ndim(); ++i) {
        if (i != dim || keepdims) {
            new_shape.push_back((i == dim) ? 1 : shape[i]);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);
    
    Array result(new_shape, 0.0f);
    
    // Simplified sum for 2D case
    if (ndim() == 2) {
        if (dim == 0) {
            // Sum over rows
            for (int j = 0; j < shape[1]; ++j) {
                float s = 0.0f;
                for (int i = 0; i < shape[0]; ++i) {
                    s += (*this)({i, j});
                }
                result[j] = s;
            }
        } else {
            // Sum over columns
            for (int i = 0; i < shape[0]; ++i) {
                float s = 0.0f;
                for (int j = 0; j < shape[1]; ++j) {
                    s += (*this)({i, j});
                }
                result[i] = s;
            }
        }
    }
    
    return result;
}

Array Array::sum(const std::vector<int>& dims, bool keepdims) const {
    Array result = *this;
    // Sort dims in descending order to avoid index shifting issues
    std::vector<int> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int>());
    
    for (int dim : sorted_dims) {
        result = result.sum(dim, keepdims);
    }
    return result;
}

// Static constructors
Array Array::zeros(const std::vector<int>& shape) {
    return Array(shape, 0.0f);
}

Array Array::ones(const std::vector<int>& shape) {
    return Array(shape, 1.0f);
}

Array Array::eye(int n) {
    Array result({n, n}, 0.0f);
    for (int i = 0; i < n; ++i) {
        result({i, i}) = 1.0f;
    }
    return result;
}

Array Array::full(const std::vector<int>& shape, float value) {
    return Array(shape, value);
}

Array Array::randn(const std::vector<int>& shape, unsigned int seed) {
    Array result(shape);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float& val : result.data) {
        val = dist(gen);
    }
    return result;
}

Array Array::uniform(const std::vector<int>& shape, float min, float max, unsigned int seed) {
    Array result(shape);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min, max);
    for (float& val : result.data) {
        val = dist(gen);
    }
    return result;
}

// ============================================================================
// ArrayDict Implementation
// ============================================================================

ArrayDict::ArrayDict(const std::unordered_map<std::string, Array>& data_) : data(data_) {}

ArrayDict ArrayDict::operator+(const ArrayDict& other) const {
    ArrayDict result;
    for (const auto& [key, val] : data) {
        if (other.has(key)) {
            result[key] = val + other.at(key);
        } else {
            result[key] = val;
        }
    }
    // Add keys from other that aren't in this
    for (const auto& [key, val] : other.data) {
        if (!has(key)) {
            result[key] = val;
        }
    }
    return result;
}

ArrayDict ArrayDict::operator-(const ArrayDict& other) const {
    ArrayDict result;
    for (const auto& [key, val] : data) {
        if (other.has(key)) {
            result[key] = val - other.at(key);
        } else {
            result[key] = val;
        }
    }
    for (const auto& [key, val] : other.data) {
        if (!has(key)) {
            result[key] = val * (-1.0f);
        }
    }
    return result;
}

ArrayDict ArrayDict::operator*(float scalar) const {
    ArrayDict result;
    for (const auto& [key, val] : data) {
        result[key] = val * scalar;
    }
    return result;
}

ArrayDict ArrayDict::operator/(float scalar) const {
    ArrayDict result;
    for (const auto& [key, val] : data) {
        result[key] = val / scalar;
    }
    return result;
}

ArrayDict ArrayDict::map(const std::function<Array(const Array&)>& fn) const {
    ArrayDict result;
    for (const auto& [key, val] : data) {
        result[key] = fn(val);
    }
    return result;
}

ArrayDict ArrayDict::map2(const ArrayDict& other, 
                          const std::function<Array(const Array&, const Array&)>& fn) const {
    ArrayDict result;
    for (const auto& [key, val] : data) {
        if (other.has(key)) {
            result[key] = fn(val, other.at(key));
        }
    }
    return result;
}

ArrayDict ArrayDict::broadcast_to(const std::vector<int>& target_shape) const {
    ArrayDict result;
    for (const auto& [key, val] : data) {
        // Simple broadcast - just return as-is if shapes don't need changing
        // Full implementation would handle complex broadcasting rules
        result[key] = val;
    }
    return result;
}

ArrayDict ArrayDict::sum(int dim, bool keepdims) const {
    ArrayDict result;
    for (const auto& [key, val] : data) {
        result[key] = val.sum(dim, keepdims);
    }
    return result;
}

ArrayDict ArrayDict::sum(const std::vector<int>& dims, bool keepdims) const {
    ArrayDict result;
    for (const auto& [key, val] : data) {
        result[key] = val.sum(dims, keepdims);
    }
    return result;
}

ArrayDict ArrayDict::empty() {
    return ArrayDict();
}

ArrayDict ArrayDict::from_dict(const std::unordered_map<std::string, Array>& data) {
    return ArrayDict(data);
}

// ============================================================================
// Utility Functions
// ============================================================================

Array matmul(const Array& a, const Array& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D arrays");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    int m = a.shape[0];
    int n = b.shape[1];
    int k = a.shape[1];
    
    Array result({m, n}, 0.0f);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a({i, l}) * b({l, j});
            }
            result({i, j}) = sum;
        }
    }
    
    return result;
}

Array batch_matmul(const Array& a, const Array& b) {
    // Simple implementation assuming batch dimensions match
    // For 3D arrays: (batch, m, k) @ (batch, k, n) = (batch, m, n)
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument("Batch matmul requires at least 2D arrays");
    }
    
    // For now, handle the 2D case as simple matmul
    if (a.ndim() == 2 && b.ndim() == 2) {
        return matmul(a, b);
    }
    
    // For batched case, assume same batch dimensions
    int batch_size = 1;
    for (int i = 0; i < a.ndim() - 2; ++i) {
        batch_size *= a.shape[i];
    }
    
    int m = a.shape[a.ndim() - 2];
    int k = a.shape[a.ndim() - 1];
    int n = b.shape[b.ndim() - 1];
    
    std::vector<int> result_shape(a.shape.begin(), a.shape.end() - 2);
    result_shape.push_back(m);
    result_shape.push_back(n);
    
    Array result(result_shape, 0.0f);
    
    for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    int a_idx = b_idx * m * k + i * k + l;
                    int b_idx_local = b_idx * k * n + l * n + j;
                    sum += a.data[a_idx] * b.data[b_idx_local];
                }
                int r_idx = b_idx * m * n + i * n + j;
                result.data[r_idx] = sum;
            }
        }
    }
    
    return result;
}

Array cholesky(const Array& matrix) {
    if (matrix.ndim() != 2 || matrix.shape[0] != matrix.shape[1]) {
        throw std::invalid_argument("Cholesky requires square 2D matrix");
    }
    
    int n = matrix.shape[0];
    Array L = Array::zeros({n, n});
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = matrix({i, j});
            for (int k = 0; k < j; ++k) {
                sum -= L({i, k}) * L({j, k});
            }
            
            if (i == j) {
                if (sum <= 0.0f) {
                    // Matrix not positive definite - add small epsilon
                    sum = 1e-6f;
                }
                L({i, i}) = std::sqrt(sum);
            } else {
                L({i, j}) = sum / L({j, j});
            }
        }
    }
    
    return L;
}

Array cholesky_inverse(const Array& matrix) {
    Array L = cholesky(matrix);
    int n = L.shape[0];
    
    // Solve L @ L^T @ X = I using forward/backward substitution
    Array inv = Array::eye(n);
    
    // Forward substitution: solve L @ Y = I
    Array Y({n, n}, 0.0f);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            float sum = (i == j) ? 1.0f : 0.0f;
            for (int k = 0; k < i; ++k) {
                sum -= L({i, k}) * Y({k, j});
            }
            Y({i, j}) = sum / L({i, i});
        }
    }
    
    // Backward substitution: solve L^T @ X = Y
    for (int j = 0; j < n; ++j) {
        for (int i = n - 1; i >= 0; --i) {
            float sum = Y({i, j});
            for (int k = i + 1; k < n; ++k) {
                sum -= L({k, i}) * inv({k, j});
            }
            inv({i, j}) = sum / L({i, i});
        }
    }
    
    return inv;
}

std::pair<Array, float> inv_and_logdet(const Array& matrix) {
    Array inv = cholesky_inverse(matrix);
    
    // Compute log determinant from Cholesky factor
    Array L = cholesky(matrix);
    float logdet = 0.0f;
    for (int i = 0; i < L.shape[0]; ++i) {
        logdet += 2.0f * std::log(std::abs(L({i, i})) + 1e-8f);
    }
    
    return {inv, logdet};
}

Array cholesky_solve(const Array& L, const Array& b) {
    // Solve L @ L^T @ x = b
    int n = L.shape[0];
    int nrhs = (b.ndim() > 1) ? b.shape[1] : 1;
    
    Array x = b;
    
    // Forward substitution: L @ y = b
    Array y = b;
    for (int r = 0; r < nrhs; ++r) {
        for (int i = 0; i < n; ++i) {
            float sum = b.data[i * nrhs + r];
            for (int j = 0; j < i; ++j) {
                sum -= L({i, j}) * y.data[j * nrhs + r];
            }
            y.data[i * nrhs + r] = sum / L({i, i});
        }
    }
    
    // Backward substitution: L^T @ x = y
    for (int r = 0; r < nrhs; ++r) {
        for (int i = n - 1; i >= 0; --i) {
            float sum = y.data[i * nrhs + r];
            for (int j = i + 1; j < n; ++j) {
                sum -= L({j, i}) * x.data[j * nrhs + r];
            }
            x.data[i * nrhs + r] = sum / L({i, i});
        }
    }
    
    return x;
}

// Mathematical functions
float digamma(float x) {
    // Approximation of digamma function
    if (x <= 0.0f) {
        return 0.0f;
    }
    
    float result = 0.0f;
    while (x < 7.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    
    float inv_x = 1.0f / x;
    float inv_x2 = inv_x * inv_x;
    float inv_x4 = inv_x2 * inv_x2;
    
    result += std::log(x) - 0.5f * inv_x - inv_x2 / 12.0f + inv_x4 / 120.0f - inv_x4 * inv_x2 / 252.0f;
    
    return result;
}

Array digamma(const Array& x) {
    Array result(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        result.data[i] = digamma(x.data[i]);
    }
    return result;
}

float gammaln(float x) {
    // Lanczos approximation for log gamma
    if (x <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    
    static const float coeffs[] = {
        76.18009172947146f,
        -86.50532032941677f,
        24.01409824083091f,
        -1.231739572450155f,
        0.1208650973866179e-2f,
        -0.5395239384953e-5f
    };
    
    float y = x;
    float tmp = x + 5.5f;
    tmp -= (x + 0.5f) * std::log(tmp);
    float ser = 1.000000000190015f;
    
    for (float coeff : coeffs) {
        y += 1.0f;
        ser += coeff / y;
    }
    
    return -tmp + std::log(2.5066282746310005f * ser / x);
}

Array gammaln(const Array& x) {
    Array result(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        result.data[i] = gammaln(x.data[i]);
    }
    return result;
}

float logsumexp(const std::vector<float>& x) {
    if (x.empty()) return 0.0f;
    
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float val : x) {
        sum += std::exp(val - max_val);
    }
    
    return max_val + std::log(sum + 1e-8f);
}

Array logsumexp(const Array& x, int dim, bool keepdims) {
    if (dim < 0) dim += x.ndim();
    
    std::vector<int> new_shape;
    for (int i = 0; i < x.ndim(); ++i) {
        if (i != dim || keepdims) {
            new_shape.push_back((i == dim) ? 1 : x.shape[i]);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);
    
    Array result(new_shape);
    
    // Simplified for 2D case along dim 0
    if (x.ndim() == 2 && dim == 0) {
        for (int j = 0; j < x.shape[1]; ++j) {
            std::vector<float> column;
            for (int i = 0; i < x.shape[0]; ++i) {
                column.push_back(x({i, j}));
            }
            result[j] = logsumexp(column);
        }
    } else if (x.ndim() == 2 && dim == 1) {
        for (int i = 0; i < x.shape[0]; ++i) {
            std::vector<float> row;
            for (int j = 0; j < x.shape[1]; ++j) {
                row.push_back(x({i, j}));
            }
            result[i] = logsumexp(row);
        }
    }
    
    return result;
}

float log1pexp(float x) {
    if (x > 0.0f) {
        return x + std::log1p(std::exp(-x));
    } else {
        return std::log1p(std::exp(x));
    }
}

Array log1pexp(const Array& x) {
    Array result(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        result.data[i] = log1pexp(x.data[i]);
    }
    return result;
}

} // namespace transforms
} // namespace axiom

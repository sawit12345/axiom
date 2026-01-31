/*
 * Copyright 2025 VERSES AI, Inc.
 *
 * Licensed under the VERSES Academic Research License (the "License");
 * you may not use this file except in compliance with the license.
 *
 * You may obtain a copy of the License at
 *
 *     https://github.com/VersesTech/axiom/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstring>
#include "cuda_utils.h"

namespace axiomcuda {

// ============================================================================
// Data Types
// ============================================================================

enum class DataType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    UINT32,
    UINT64,
    BOOL
};

inline size_t data_type_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT64: return 8;
        case DataType::INT32:   return 4;
        case DataType::INT64:   return 8;
        case DataType::UINT32:  return 4;
        case DataType::UINT64:  return 8;
        case DataType::BOOL:    return 1;
        default: return 0;
    }
}

inline std::string data_type_name(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT64: return "float64";
        case DataType::INT32:   return "int32";
        case DataType::INT64:   return "int64";
        case DataType::UINT32:  return "uint32";
        case DataType::UINT64:  return "uint64";
        case DataType::BOOL:    return "bool";
        default: return "unknown";
    }
}

// ============================================================================
// Device Type
// ============================================================================

enum class DeviceType {
    CPU,
    CUDA
};

// ============================================================================
// Tensor Shape
// ============================================================================

class Shape {
public:
    Shape() = default;
    explicit Shape(std::vector<size_t> dims) : dims_(std::move(dims)) {}
    Shape(std::initializer_list<size_t> dims) : dims_(dims) {}
    
    size_t ndim() const { return dims_.size(); }
    size_t operator[](size_t i) const { return dims_[i]; }
    size_t& operator[](size_t i) { return dims_[i]; }
    
    size_t numel() const {
        size_t n = 1;
        for (auto d : dims_) n *= d;
        return n;
    }
    
    size_t stride(size_t dim) const {
        size_t s = 1;
        for (size_t i = dim + 1; i < dims_.size(); ++i) {
            s *= dims_[i];
        }
        return s;
    }
    
    bool operator==(const Shape& other) const {
        return dims_ == other.dims_;
    }
    
    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }
    
    const std::vector<size_t>& dims() const { return dims_; }
    
    std::string to_string() const;
    
private:
    std::vector<size_t> dims_;
};

// ============================================================================
// Memory Buffer
// ============================================================================

class Buffer {
public:
    Buffer(void* data, size_t size, DeviceType device, bool owns_data = false);
    ~Buffer();
    
    // Disable copy, enable move
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;
    
    void* data() const { return data_; }
    size_t size() const { return size_; }
    DeviceType device() const { return device_; }
    
    void copy_to(Buffer& dst) const;
    void copy_to(void* dst, DeviceType dst_device) const;
    
    std::shared_ptr<Buffer> clone() const;
    
private:
    void* data_;
    size_t size_;
    DeviceType device_;
    bool owns_data_;
};

// ============================================================================
// Tensor Class
// ============================================================================

class Tensor {
public:
    // Constructors
    Tensor() = default;
    
    // Create tensor with shape, allocate on device
    Tensor(const Shape& shape, DataType dtype, DeviceType device);
    
    // Create tensor from existing buffer
    Tensor(const Shape& shape, DataType dtype, std::shared_ptr<Buffer> buffer);
    
    // Create from host data (copies data)
    template<typename T>
    static Tensor from_host(const T* data, const Shape& shape, DeviceType device = DeviceType::CPU);
    
    // Create on CPU
    static Tensor zeros(const Shape& shape, DataType dtype = DataType::FLOAT64);
    static Tensor ones(const Shape& shape, DataType dtype = DataType::FLOAT64);
    static Tensor empty(const Shape& shape, DataType dtype = DataType::FLOAT64);
    
    // Create on CUDA
    static Tensor zeros_cuda(const Shape& shape, DataType dtype = DataType::FLOAT64);
    static Tensor ones_cuda(const Shape& shape, DataType dtype = DataType::FLOAT64);
    static Tensor empty_cuda(const Shape& shape, DataType dtype = DataType::FLOAT64);
    
    // Properties
    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    DeviceType device() const { return buffer_ ? buffer_->device() : DeviceType::CPU; }
    size_t numel() const { return shape_.numel(); }
    size_t nbytes() const { return numel() * data_type_size(dtype_); }
    
    void* data() const { return buffer_ ? buffer_->data() : nullptr; }
    
    template<typename T>
    T* data() const { return static_cast<T*>(data()); }
    
    // Type checks
    bool is_cpu() const { return device() == DeviceType::CPU; }
    bool is_cuda() const { return device() == DeviceType::CUDA; }
    bool is_contiguous() const { return true; }  // Currently always contiguous
    
    // Device transfer
    Tensor to(DeviceType device) const;
    Tensor cpu() const { return to(DeviceType::CPU); }
    Tensor cuda() const { return to(DeviceType::CUDA); }
    
    // Data access (copies to/from host)
    template<typename T>
    void copy_to_host(T* dst) const;
    
    template<typename T>
    void copy_from_host(const T* src);
    
    // Clone
    Tensor clone() const;
    
    // Fill operations
    void fill(double value);
    void fill(int value);
    void zeros_();
    void ones_();
    
    // View operations
    Tensor reshape(const Shape& new_shape) const;
    Tensor view(const Shape& new_shape) const { return reshape(new_shape); }
    
    // Slicing (future: implement actual views)
    // Tensor slice(size_t dim, size_t start, size_t end) const;
    
    // Info
    std::string to_string() const;
    void print() const;
    
    // Element access (slow, for debugging only)
    template<typename T>
    T item() const {
        if (numel() != 1) {
            throw std::runtime_error("item() can only be called on scalar tensors");
        }
        T val;
        copy_to_host(&val);
        return val;
    }
    
    // Data validation
    bool is_valid() const { return buffer_ != nullptr && buffer_->data() != nullptr; }
    
private:
    Shape shape_;
    DataType dtype_;
    std::shared_ptr<Buffer> buffer_;
};

// ============================================================================
// Template Implementations
// ============================================================================

template<typename T>
Tensor Tensor::from_host(const T* data, const Shape& shape, DeviceType device) {
    DataType dtype;
    if (std::is_same<T, float>::value) dtype = DataType::FLOAT32;
    else if (std::is_same<T, double>::value) dtype = DataType::FLOAT64;
    else if (std::is_same<T, int32_t>::value) dtype = DataType::INT32;
    else if (std::is_same<T, int64_t>::value) dtype = DataType::INT64;
    else if (std::is_same<T, uint32_t>::value) dtype = DataType::UINT32;
    else if (std::is_same<T, uint64_t>::value) dtype = DataType::UINT64;
    else if (std::is_same<T, bool>::value) dtype = DataType::BOOL;
    else {
        throw std::runtime_error("Unsupported type for Tensor::from_host");
    }
    
    Tensor tensor(shape, dtype, device);
    tensor.copy_from_host(data);
    return tensor;
}

template<typename T>
void Tensor::copy_to_host(T* dst) const {
    if (!is_valid()) {
        throw std::runtime_error("Cannot copy from invalid tensor");
    }
    
    if (is_cpu()) {
        std::memcpy(dst, data(), nbytes());
    } else {
        // Copy from CUDA to host
        d2h_copy(data<T>(), dst, numel());
    }
}

template<typename T>
void Tensor::copy_from_host(const T* src) {
    if (!is_valid()) {
        throw std::runtime_error("Cannot copy to invalid tensor");
    }
    
    if (is_cpu()) {
        std::memcpy(data(), src, nbytes());
    } else {
        // Copy from host to CUDA
        h2d_copy(src, data<T>(), numel());
    }
}

// ============================================================================
// Tensor Operations (declarations - implementations in cuda_kernels.cu)
// ============================================================================

// Element-wise operations
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

// Scalar operations  
Tensor operator+(const Tensor& a, double scalar);
Tensor operator-(const Tensor& a, double scalar);
Tensor operator*(const Tensor& a, double scalar);
Tensor operator/(const Tensor& a, double scalar);

// Matrix operations
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor batch_matmul(const Tensor& a, const Tensor& b);

// Reductions
Tensor sum(const Tensor& input, int dim = -1);
Tensor mean(const Tensor& input, int dim = -1);
Tensor max(const Tensor& input, int dim = -1);
Tensor min(const Tensor& input, int dim = -1);

// Linear algebra
Tensor cholesky(const Tensor& input);
Tensor solve_cholesky(const Tensor& L, const Tensor& b);
Tensor inverse(const Tensor& input);

// Random
Tensor randn(const Shape& shape, DeviceType device = DeviceType::CPU);
Tensor rand(const Shape& shape, DeviceType device = DeviceType::CPU);
Tensor randint(int low, int high, const Shape& shape, DeviceType device = DeviceType::CPU);

} // namespace axiomcuda

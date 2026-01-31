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

#include "tensor.h"
#include <sstream>
#include <iomanip>
#include <cstring>
#include <iostream>

namespace axiomcuda {

// ============================================================================
// Shape Implementation
// ============================================================================

std::string Shape::to_string() const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << dims_[i];
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// Buffer Implementation
// ============================================================================

Buffer::Buffer(void* data, size_t size, DeviceType device, bool owns_data)
    : data_(data), size_(size), device_(device), owns_data_(owns_data) {}

Buffer::~Buffer() {
    if (owns_data_ && data_) {
        if (device_ == DeviceType::CUDA) {
            device_free(data_);
        } else {
            delete[] static_cast<char*>(data_);
        }
    }
}

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_), size_(other.size_), 
      device_(other.device_), owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        if (owns_data_ && data_) {
            if (device_ == DeviceType::CUDA) {
                device_free(data_);
            } else {
                delete[] static_cast<char*>(data_);
            }
        }
        
        data_ = other.data_;
        size_ = other.size_;
        device_ = other.device_;
        owns_data_ = other.owns_data_;
        
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

void Buffer::copy_to(Buffer& dst) const {
    if (size_ != dst.size_) {
        throw std::runtime_error("Buffer sizes don't match for copy");
    }
    
    if (device_ == DeviceType::CPU && dst.device_ == DeviceType::CPU) {
        std::memcpy(dst.data_, data_, size_);
    } else if (device_ == DeviceType::CUDA && dst.device_ == DeviceType::CUDA) {
        d2d_copy(static_cast<char*>(data_), static_cast<char*>(dst.data_), size_);
    } else if (device_ == DeviceType::CPU && dst.device_ == DeviceType::CUDA) {
        h2d_copy(static_cast<char*>(data_), static_cast<char*>(dst.data_), size_ / sizeof(double));
    } else if (device_ == DeviceType::CUDA && dst.device_ == DeviceType::CPU) {
        d2h_copy(static_cast<double*>(data_), static_cast<double*>(dst.data_), size_ / sizeof(double));
    }
}

std::shared_ptr<Buffer> Buffer::clone() const {
    void* new_data = nullptr;
    if (device_ == DeviceType::CPU) {
        new_data = new char[size_];
        std::memcpy(new_data, data_, size_);
    } else {
        new_data = device_malloc<char>(size_);
        d2d_copy(static_cast<char*>(data_), static_cast<char*>(new_data), size_);
    }
    
    return std::make_shared<Buffer>(new_data, size_, device_, true);
}

// ============================================================================
// Tensor Implementation
// ============================================================================

Tensor::Tensor(const Shape& shape, DataType dtype, DeviceType device)
    : shape_(shape), dtype_(dtype) {
    
    size_t size = shape.numel() * data_type_size(dtype);
    void* data = nullptr;
    
    if (device == DeviceType::CPU) {
        data = new char[size];
        std::memset(data, 0, size);
    } else {
        data = device_malloc<char>(size);
        device_memset(static_cast<char*>(data), size, 0);
    }
    
    buffer_ = std::make_shared<Buffer>(data, size, device, true);
}

Tensor::Tensor(const Shape& shape, DataType dtype, std::shared_ptr<Buffer> buffer)
    : shape_(shape), dtype_(dtype), buffer_(buffer) {}

Tensor Tensor::zeros(const Shape& shape, DataType dtype) {
    Tensor t(shape, dtype, DeviceType::CPU);
    t.zeros_();
    return t;
}

Tensor Tensor::ones(const Shape& shape, DataType dtype) {
    Tensor t(shape, dtype, DeviceType::CPU);
    t.ones_();
    return t;
}

Tensor Tensor::empty(const Shape& shape, DataType dtype) {
    return Tensor(shape, dtype, DeviceType::CPU);
}

Tensor Tensor::zeros_cuda(const Shape& shape, DataType dtype) {
    Tensor t(shape, dtype, DeviceType::CUDA);
    // Already zeroed in constructor
    return t;
}

Tensor Tensor::ones_cuda(const Shape& shape, DataType dtype) {
    Tensor t(shape, dtype, DeviceType::CUDA);
    t.ones_();
    return t;
}

Tensor Tensor::empty_cuda(const Shape& shape, DataType dtype) {
    return Tensor(shape, dtype, DeviceType::CUDA);
}

Tensor Tensor::to(DeviceType device) const {
    if (device == this->device()) {
        return clone();
    }
    
    Tensor result(shape_, dtype_, device);
    buffer_->copy_to(*result.buffer_);
    return result;
}

Tensor Tensor::clone() const {
    if (!is_valid()) {
        return Tensor();
    }
    
    return Tensor(shape_, dtype_, buffer_->clone());
}

void Tensor::fill(double value) {
    if (is_cpu()) {
        double* ptr = data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            ptr[i] = value;
        }
    } else {
        // Launch CUDA kernel
        extern void launch_fill(double* d_data, double value, int n, void* stream);
        launch_fill(data<double>(), value, numel(), nullptr);
    }
}

void Tensor::fill(int value) {
    if (is_cpu()) {
        int* ptr = data<int>();
        for (size_t i = 0; i < numel(); ++i) {
            ptr[i] = value;
        }
    }
}

void Tensor::zeros_() {
    if (is_cpu()) {
        std::memset(data(), 0, nbytes());
    } else {
        device_memset(data<char>(), nbytes(), 0);
    }
}

void Tensor::ones_() {
    if (dtype_ == DataType::FLOAT64) {
        fill(1.0);
    } else if (dtype_ == DataType::FLOAT32) {
        float* ptr = data<float>();
        for (size_t i = 0; i < numel(); ++i) ptr[i] = 1.0f;
    } else if (dtype_ == DataType::INT32 || dtype_ == DataType::INT64) {
        fill(1);
    }
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (new_shape.numel() != numel()) {
        throw std::runtime_error("Reshape: number of elements must match");
    }
    
    return Tensor(new_shape, dtype_, buffer_);
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(";
    oss << "shape=" << shape_.to_string() << ", ";
    oss << "dtype=" << data_type_name(dtype_) << ", ";
    oss << "device=" << (is_cpu() ? "cpu" : "cuda");
    oss << ")";
    return oss.str();
}

void Tensor::print() const {
    std::cout << to_string() << std::endl;
    
    if (numel() <= 10 && is_cpu()) {
        std::cout << "[";
        double* ptr = data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << ptr[i];
        }
        std::cout << "]" << std::endl;
    }
}

// ============================================================================
// Tensor Operations
// ============================================================================

Tensor operator+(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for addition");
    }
    
    Tensor result = a.clone();
    
    if (a.is_cpu() && b.is_cpu()) {
        double* r_ptr = result.data<double>();
        const double* b_ptr = b.data<double>();
        for (size_t i = 0; i < a.numel(); ++i) {
            r_ptr[i] += b_ptr[i];
        }
    } else {
        // CUDA path
        Tensor b_cuda = b.is_cuda() ? b : b.cuda();
        Tensor r_cuda = result.cuda();
        extern void launch_add(const double*, const double*, double*, int, void*);
        launch_add(r_cuda.data<double>(), b_cuda.data<double>(), r_cuda.data<double>(), 
                   a.numel(), nullptr);
        result = r_cuda.cpu();
    }
    
    return result;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for multiplication");
    }
    
    Tensor result = a.clone();
    
    if (a.is_cpu() && b.is_cpu()) {
        double* r_ptr = result.data<double>();
        const double* b_ptr = b.data<double>();
        for (size_t i = 0; i < a.numel(); ++i) {
            r_ptr[i] *= b_ptr[i];
        }
    } else {
        // CUDA path
        Tensor b_cuda = b.is_cuda() ? b : b.cuda();
        Tensor r_cuda = result.cuda();
        extern void launch_mul(const double*, const double*, double*, int, void*);
        launch_mul(r_cuda.data<double>(), b_cuda.data<double>(), r_cuda.data<double>(), 
                   a.numel(), nullptr);
        result = r_cuda.cpu();
    }
    
    return result;
}

Tensor operator*(const Tensor& a, double scalar) {
    Tensor result = a.clone();
    
    if (a.is_cpu()) {
        double* r_ptr = result.data<double>();
        for (size_t i = 0; i < a.numel(); ++i) {
            r_ptr[i] *= scalar;
        }
    } else {
        Tensor r_cuda = result.cuda();
        extern void launch_mul_scalar(const double*, double, double*, int, void*);
        launch_mul_scalar(r_cuda.data<double>(), scalar, r_cuda.data<double>(), 
                          a.numel(), nullptr);
        result = r_cuda.cpu();
    }
    
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // 2D matrix multiplication: [M, K] @ [K, N] = [M, N]
    if (a.shape().ndim() != 2 || b.shape().ndim() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }
    
    int m = a.shape()[0];
    int k = a.shape()[1];
    int n = b.shape()[1];
    
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }
    
    Tensor result = Tensor::zeros({m, n}, a.dtype());
    
    if (a.is_cpu() && b.is_cpu()) {
        const double* a_ptr = a.data<double>();
        const double* b_ptr = b.data<double>();
        double* r_ptr = result.data<double>();
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int l = 0; l < k; ++l) {
                    sum += a_ptr[i * k + l] * b_ptr[l * n + j];
                }
                r_ptr[i * n + j] = sum;
            }
        }
    } else {
        Tensor a_cuda = a.is_cuda() ? a : a.cuda();
        Tensor b_cuda = b.is_cuda() ? b : b.cuda();
        Tensor r_cuda = result.cuda();
        extern void launch_matmul(const double*, const double*, double*, int, int, int, void*);
        launch_matmul(a_cuda.data<double>(), b_cuda.data<double>(), r_cuda.data<double>(),
                      m, k, n, nullptr);
        result = r_cuda.cpu();
    }
    
    return result;
}

Tensor sum(const Tensor& input, int dim) {
    if (dim == -1) {
        // Sum all elements
        Tensor result = Tensor::zeros({1}, input.dtype());
        
        if (input.is_cpu()) {
            const double* ptr = input.data<double>();
            double sum = 0.0;
            for (size_t i = 0; i < input.numel(); ++i) {
                sum += ptr[i];
            }
            result.data<double>()[0] = sum;
        } else {
            Tensor partial = Tensor::empty({256}, input.dtype()).cuda();
            extern void launch_sum(const double*, double*, int, void*);
            launch_sum(input.data<double>(), partial.data<double>(), input.numel(), nullptr);
            result = partial.cpu();
            // TODO: Final reduction on CPU
        }
        
        return result;
    }
    
    // TODO: Sum along specific dimension
    return input;
}

Tensor randn(const Shape& shape, DeviceType device) {
    Tensor result = Tensor::empty(shape, DataType::FLOAT64);
    
    if (device == DeviceType::CPU) {
        // Use std::normal_distribution
        // For now, fill with zeros as placeholder
        result.zeros_();
    } else {
        extern void launch_normal_random(uint64_t, uint64_t, double*, int, void*);
        launch_normal_random(0x123456789ABCDEF0ULL, 0x0FEDCBA987654321ULL,
                            result.data<double>(), result.numel(), nullptr);
    }
    
    return result;
}

Tensor rand(const Shape& shape, DeviceType device) {
    Tensor result = Tensor::empty(shape, DataType::FLOAT64);
    
    if (device == DeviceType::CPU) {
        result.zeros_();
    } else {
        extern void launch_uniform_random(uint64_t, uint64_t, double*, int, void*);
        launch_uniform_random(0x123456789ABCDEF0ULL, 0x0FEDCBA987654321ULL,
                             result.data<double>(), result.numel(), nullptr);
    }
    
    return result;
}

} // namespace axiomcuda

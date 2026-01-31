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
#include <cmath>
#include <algorithm>
#include <stdexcept>

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
#ifdef USE_CUDA
            device_free(data_);
#endif
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
#ifdef USE_CUDA
                device_free(data_);
#endif
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
#ifdef USE_CUDA
        d2d_copy(static_cast<char*>(data_), static_cast<char*>(dst.data_), size_);
#endif
    } else if (device_ == DeviceType::CPU && dst.device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
        h2d_copy(static_cast<char*>(data_), static_cast<char*>(dst.data_), size_ / sizeof(double));
#endif
    } else if (device_ == DeviceType::CUDA && dst.device_ == DeviceType::CPU) {
#ifdef USE_CUDA
        d2h_copy(static_cast<double*>(data_), static_cast<double*>(dst.data_), size_ / sizeof(double));
#endif
    }
}

std::shared_ptr<Buffer> Buffer::clone() const {
    void* new_data = nullptr;
    if (device_ == DeviceType::CPU) {
        new_data = new char[size_];
        std::memcpy(new_data, data_, size_);
    } else {
#ifdef USE_CUDA
        new_data = device_malloc<char>(size_);
        d2d_copy(static_cast<char*>(data_), static_cast<char*>(new_data), size_);
#endif
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
#ifdef USE_CUDA
        data = device_malloc<char>(size);
        device_memset(static_cast<char*>(data), size, 0);
#endif
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
#ifdef USE_CUDA
        // Launch CUDA kernel
        extern void launch_fill(double* d_data, double value, int n, void* stream);
        launch_fill(data<double>(), value, numel(), nullptr);
#endif
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
#ifdef USE_CUDA
        device_memset(data<char>(), nbytes(), 0);
#endif
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
#ifdef USE_CUDA
        // CUDA path
        Tensor b_cuda = b.is_cuda() ? b : b.cuda();
        Tensor r_cuda = result.cuda();
        extern void launch_add(const double*, const double*, double*, int, void*);
        launch_add(r_cuda.data<double>(), b_cuda.data<double>(), r_cuda.data<double>(), 
                   a.numel(), nullptr);
        result = r_cuda.cpu();
#endif
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
#ifdef USE_CUDA
        // CUDA path
        Tensor b_cuda = b.is_cuda() ? b : b.cuda();
        Tensor r_cuda = result.cuda();
        extern void launch_mul(const double*, const double*, double*, int, void*);
        launch_mul(r_cuda.data<double>(), b_cuda.data<double>(), r_cuda.data<double>(), 
                   a.numel(), nullptr);
        result = r_cuda.cpu();
#endif
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
#ifdef USE_CUDA
        Tensor r_cuda = result.cuda();
        extern void launch_mul_scalar(const double*, double, double*, int, void*);
        launch_mul_scalar(r_cuda.data<double>(), scalar, r_cuda.data<double>(), 
                          a.numel(), nullptr);
        result = r_cuda.cpu();
#endif
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
    
    Tensor result = Tensor::zeros({static_cast<size_t>(m), static_cast<size_t>(n)}, a.dtype());
    
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
#ifdef USE_CUDA
        Tensor a_cuda = a.is_cuda() ? a : a.cuda();
        Tensor b_cuda = b.is_cuda() ? b : b.cuda();
        Tensor r_cuda = result.cuda();
        extern void launch_matmul(const double*, const double*, double*, int, int, int, void*);
        launch_matmul(a_cuda.data<double>(), b_cuda.data<double>(), r_cuda.data<double>(),
                      m, k, n, nullptr);
        result = r_cuda.cpu();
#endif
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
#ifdef USE_CUDA
            Tensor partial = Tensor::empty({256}, input.dtype()).cuda();
            extern void launch_sum(const double*, double*, int, void*);
            launch_sum(input.data<double>(), partial.data<double>(), input.numel(), nullptr);
            result = partial.cpu();
            // TODO: Final reduction on CPU
#endif
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
#ifdef USE_CUDA
        extern void launch_normal_random(uint64_t, uint64_t, double*, int, void*);
        launch_normal_random(0x123456789ABCDEF0ULL, 0x0FEDCBA987654321ULL,
                            result.data<double>(), result.numel(), nullptr);
#endif
    }
    
    return result;
}

Tensor rand(const Shape& shape, DeviceType device) {
    Tensor result = Tensor::empty(shape, DataType::FLOAT64);
    
    if (device == DeviceType::CPU) {
        result.zeros_();
    } else {
#ifdef USE_CUDA
        extern void launch_uniform_random(uint64_t, uint64_t, double*, int, void*);
        launch_uniform_random(0x123456789ABCDEF0ULL, 0x0FEDCBA987654321ULL,
                             result.data<double>(), result.numel(), nullptr);
#endif
    }
    
    return result;
}

// ============================================================================
// Tensor View Operations Implementation
// ============================================================================

Tensor Tensor::squeeze() const {
    std::vector<size_t> new_dims;
    for (auto d : shape_.dims()) {
        if (d != 1) new_dims.push_back(d);
    }
    if (new_dims.empty()) new_dims.push_back(1);
    return reshape(Shape(new_dims));
}

Tensor Tensor::squeeze(int dim) const {
    auto dims = shape_.dims();
    if (dim < 0) dim = dims.size() + dim;
    if (dim < 0 || dim >= static_cast<int>(dims.size())) {
        throw std::out_of_range("squeeze dimension out of range");
    }
    if (dims[dim] != 1) {
        return *this;  // No squeeze needed
    }
    std::vector<size_t> new_dims;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (static_cast<int>(i) != dim) new_dims.push_back(dims[i]);
    }
    return reshape(Shape(new_dims));
}

Tensor Tensor::unsqueeze(int dim) const {
    auto dims = shape_.dims();
    if (dim < 0) dim = dims.size() + dim + 1;
    if (dim < 0 || dim > static_cast<int>(dims.size())) {
        throw std::out_of_range("unsqueeze dimension out of range");
    }
    std::vector<size_t> new_dims;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (static_cast<int>(i) == dim) new_dims.push_back(1);
        new_dims.push_back(dims[i]);
    }
    if (dim == static_cast<int>(dims.size())) new_dims.push_back(1);
    return reshape(Shape(new_dims));
}

Tensor Tensor::transpose() const {
    if (shape_.ndim() != 2) {
        throw std::runtime_error("transpose() without arguments only supports 2D tensors");
    }
    return transpose(0, 1);
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    auto dims = shape_.dims();
    if (dim0 < 0) dim0 = dims.size() + dim0;
    if (dim1 < 0) dim1 = dims.size() + dim1;
    std::swap(dims[dim0], dims[dim1]);
    return reshape(Shape(dims));
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    if (dims.size() != shape_.ndim()) {
        throw std::runtime_error("permute dimensions must match tensor rank");
    }
    std::vector<size_t> new_dims;
    for (int d : dims) {
        if (d < 0) d = shape_.ndim() + d;
        new_dims.push_back(shape_.dims()[d]);
    }
    return reshape(Shape(new_dims));
}

// ============================================================================
// Tensor Arithmetic Implementation
// ============================================================================

Tensor Tensor::add(const Tensor& other) const {
    return *this + other;
}

Tensor Tensor::subtract(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::runtime_error("Tensor shapes must match for subtraction");
    }
    
    Tensor result = clone();
    
    if (is_cpu() && other.is_cpu()) {
        double* r_ptr = result.data<double>();
        const double* b_ptr = other.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            r_ptr[i] -= b_ptr[i];
        }
    } else {
        // For non-CPU, use add with negated other
        Tensor neg_other = other.negate();
        return add(neg_other);
    }
    
    return result;
}

Tensor Tensor::multiply(const Tensor& other) const {
    return *this * other;
}

Tensor Tensor::divide(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::runtime_error("Tensor shapes must match for division");
    }
    
    Tensor result = clone();
    
    if (is_cpu() && other.is_cpu()) {
        double* r_ptr = result.data<double>();
        const double* b_ptr = other.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            r_ptr[i] /= b_ptr[i];
        }
    } else {
#ifdef USE_CUDA
        Tensor other_cuda = other.is_cuda() ? other : other.cuda();
        Tensor r_cuda = result.cuda();
        extern void launch_div(const double*, const double*, double*, int, void*);
        launch_div(r_cuda.data<double>(), other_cuda.data<double>(), r_cuda.data<double>(), 
                   numel(), nullptr);
        result = r_cuda.cpu();
#endif
    }
    
    return result;
}

Tensor Tensor::power(double exponent) const {
    Tensor result = clone();
    
    if (is_cpu()) {
        double* r_ptr = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            r_ptr[i] = std::pow(r_ptr[i], exponent);
        }
    } else {
#ifdef USE_CUDA
        Tensor r_cuda = result.cuda();
        extern void launch_pow(const double*, double, double*, int, void*);
        launch_pow(r_cuda.data<double>(), exponent, r_cuda.data<double>(), 
                   numel(), nullptr);
        result = r_cuda.cpu();
#endif
    }
    
    return result;
}

Tensor Tensor::negate() const {
    Tensor result = clone();
    
    if (is_cpu()) {
        double* r_ptr = result.data<double>();
        for (size_t i = 0; i < numel(); ++i) {
            r_ptr[i] = -r_ptr[i];
        }
    } else {
#ifdef USE_CUDA
        Tensor r_cuda = result.cuda();
        extern void launch_neg(const double*, double*, int, void*);
        launch_neg(r_cuda.data<double>(), r_cuda.data<double>(), 
                   numel(), nullptr);
        result = r_cuda.cpu();
#endif
    }
    
    return result;
}

// ============================================================================
// Tensor Indexing Implementation
// ============================================================================

Tensor Tensor::getitem(int index) const {
    if (shape_.ndim() == 0) {
        throw std::runtime_error("Cannot index a scalar tensor");
    }
    
    auto dims = shape_.dims();
    if (index < 0) index = dims[0] + index;
    if (index < 0 || index >= static_cast<int>(dims[0])) {
        throw std::out_of_range("index out of range");
    }
    
    // Create view for the indexed slice
    std::vector<size_t> new_dims(dims.begin() + 1, dims.end());
    if (new_dims.empty()) new_dims.push_back(1);
    
    // Calculate offset
    size_t stride = shape_.stride(0);
    char* offset_ptr = static_cast<char*>(data()) + index * stride * sizeof(double);
    
    // Create new tensor sharing the buffer with offset
    Tensor result(Shape(new_dims), dtype_, buffer_);
    return result;
}

void Tensor::setitem(int index, const Tensor& value) {
    if (shape_.ndim() == 0) {
        throw std::runtime_error("Cannot index a scalar tensor");
    }
    
    auto dims = shape_.dims();
    if (index < 0) index = dims[0] + index;
    if (index < 0 || index >= static_cast<int>(dims[0])) {
        throw std::out_of_range("index out of range");
    }
    
    // Get slice tensor
    Tensor slice = getitem(index);
    
    if (slice.shape() != value.shape()) {
        throw std::runtime_error("value shape doesn't match slice shape");
    }
    
    // Copy data
    if (is_cpu() && value.is_cpu()) {
        size_t stride = shape_.stride(0);
        double* dst = static_cast<double*>(data()) + index * stride;
        const double* src = value.data<double>();
        for (size_t i = 0; i < slice.numel(); ++i) {
            dst[i] = src[i];
        }
    } else {
        // For CUDA, need to handle properly
        Tensor value_copy = value.cpu();
        size_t stride = shape_.stride(0);
        double* dst = static_cast<double*>(data()) + index * stride;
        const double* src = value_copy.data<double>();
        for (size_t i = 0; i < slice.numel(); ++i) {
            dst[i] = src[i];
        }
    }
}

// ============================================================================
// NumPy Interop Implementation
// ============================================================================

Tensor Tensor::from_numpy(const void* data, const std::vector<size_t>& shape) {
    Shape s(shape);
    Tensor result(s, DataType::FLOAT64, DeviceType::CPU);
    result.copy_from_host(static_cast<const double*>(data));
    return result;
}

void Tensor::to_numpy(void* out_data) const {
    copy_to_host(static_cast<double*>(out_data));
}

// ============================================================================
// DeviceManager Implementation
// ============================================================================

int DeviceManager::getDeviceCount() {
#ifdef USE_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
#else
    return 0;
#endif
}

int DeviceManager::getCurrentDevice() {
#ifdef USE_CUDA
    int device = 0;
    cudaGetDevice(&device);
    return device;
#else
    return 0;
#endif
}

void DeviceManager::setDevice(int device_id) {
#ifdef USE_CUDA
    cudaSetDevice(device_id);
#endif
}

void DeviceManager::synchronize(int device_id) {
#ifdef USE_CUDA
    if (device_id >= 0) {
        int current = getCurrentDevice();
        setDevice(device_id);
        cudaDeviceSynchronize();
        setDevice(current);
    } else {
        cudaDeviceSynchronize();
    }
#endif
}

DeviceProperties DeviceManager::getDeviceProperties(int device_id) {
    DeviceProperties props;
    memset(&props, 0, sizeof(props));
    
#ifdef USE_CUDA
    cudaDeviceProp cuda_props;
    cudaError_t err = cudaGetDeviceProperties(&cuda_props, device_id);
    if (err == cudaSuccess) {
        strncpy(props.name, cuda_props.name, sizeof(props.name) - 1);
        props.name[sizeof(props.name) - 1] = '\0';
        props.totalMemory = cuda_props.totalGlobalMem;
        props.major = cuda_props.major;
        props.minor = cuda_props.minor;
        props.multiProcessorCount = cuda_props.multiProcessorCount;
        props.maxThreadsPerBlock = cuda_props.maxThreadsPerBlock;
        props.warpSize = cuda_props.warpSize;
    }
#endif
    
    return props;
}

std::pair<size_t, size_t> DeviceManager::getMemoryInfo(int device_id) {
    size_t free_mem = 0, total_mem = 0;
    
#ifdef USE_CUDA
    if (device_id >= 0) {
        int current = getCurrentDevice();
        setDevice(device_id);
        cudaMemGetInfo(&free_mem, &total_mem);
        setDevice(current);
    } else {
        cudaMemGetInfo(&free_mem, &total_mem);
    }
#endif
    
    return {free_mem, total_mem};
}

} // namespace axiomcuda

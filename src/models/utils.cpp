/**
 * Copyright 2025 VERSES AI, Inc.
 *
 * Licensed under the VERSES Academic Research License (the "License");
 * you may not use this file except in compliance with the license.
 */

#include "utils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace axiom {
namespace models {

// Tensor implementation
void Tensor::allocate(const std::vector<int>& shape_, bool on_device_) {
    shape = shape_;
    on_device = on_device_;
    ndim = shape_.size();
    
    // Compute size
    size = 1;
    for (auto dim : shape_) {
        size *= dim;
    }
    
    // Compute strides
    strides.resize(ndim);
    for (int i = ndim - 1; i >= 0; --i) {
        if (i == ndim - 1) {
            strides[i] = 1;
        } else {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    
    // Allocate memory
    if (on_device) {
#ifdef USE_CUDA
        cudaMalloc(&data, size * sizeof(float));
#else
        data = new float[size];
#endif
    } else {
        data = new float[size];
    }
}

void Tensor::free() {
    if (data) {
        if (on_device) {
#ifdef USE_CUDA
            cudaFree(data);
#else
            delete[] data;
#endif
        } else {
            delete[] data;
        }
        data = nullptr;
    }
}

void Tensor::copyToDevice() {
#ifdef USE_CUDA
    if (!on_device) {
        float* device_data;
        cudaMalloc(&device_data, size * sizeof(float));
        cudaMemcpy(device_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
        delete[] data;
        data = device_data;
        on_device = true;
    }
#endif
}

void Tensor::copyFromDevice() {
#ifdef USE_CUDA
    if (on_device) {
        float* host_data = new float[size];
        cudaMemcpy(host_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(data);
        data = host_data;
        on_device = false;
    }
#endif
}

float& Tensor::operator()(const std::vector<int>& indices) {
    int idx = getIndex(indices);
    return data[idx];
}

const float& Tensor::operator()(const std::vector<int>& indices) const {
    int idx = getIndex(indices);
    return data[idx];
}

int Tensor::getIndex(const std::vector<int>& indices) const {
    int idx = 0;
    for (int i = 0; i < ndim; ++i) {
        idx += indices[i] * strides[i];
    }
    return idx;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    Tensor result;
    result.shape = new_shape;
    result.ndim = new_shape.size();
    result.size = size;
    result.data = data;
    result.on_device = on_device;
    
    // Compute new strides
    result.strides.resize(result.ndim);
    for (int i = result.ndim - 1; i >= 0; --i) {
        if (i == result.ndim - 1) {
            result.strides[i] = 1;
        } else {
            result.strides[i] = result.strides[i + 1] * result.shape[i + 1];
        }
    }
    
    return result;
}

Tensor Tensor::slice(int dim, int start, int end) const {
    Tensor result;
    result.ndim = ndim;
    result.shape = shape;
    result.shape[dim] = end - start;
    result.strides = strides;
    result.on_device = on_device;
    result.size = (end - start) * strides[dim];
    
    // Compute offset
    int offset = start * strides[dim];
    result.data = data + offset;
    
    return result;
}

// ArrayDict implementation
void ArrayDict::add(const std::string& name, const Tensor& tensor) {
    fields.push_back({name, tensor});
}

Tensor& ArrayDict::get(const std::string& name) {
    for (auto& field : fields) {
        if (field.first == name) {
            return field.second;
        }
    }
    throw std::runtime_error("Field not found: " + name);
}

const Tensor& ArrayDict::get(const std::string& name) const {
    for (const auto& field : fields) {
        if (field.first == name) {
            return field.second;
        }
    }
    throw std::runtime_error("Field not found: " + name);
}

// Matrix operations (CPU implementations)
void matmul(const Tensor& A, const Tensor& B, Tensor& out_C,
           bool transpose_A, bool transpose_B) {
    // Simplified CPU matmul
    int m = transpose_A ? A.shape[1] : A.shape[0];
    int n = transpose_B ? B.shape[0] : B.shape[1];
    int k = transpose_A ? A.shape[0] : A.shape[1];
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                float a_val = transpose_A ? A.data[l * A.strides[0] + i] : A.data[i * A.strides[0] + l];
                float b_val = transpose_B ? B.data[j * B.strides[0] + l] : B.data[l * B.strides[0] + j];
                sum += a_val * b_val;
            }
            out_C.data[i * out_C.strides[0] + j] = sum;
        }
    }
}

void matrixInverse(const Tensor& A, Tensor& out_Ainv) {
    // Simplified 2x2 and 3x3 matrix inversion
    if (A.shape[0] == 2 && A.shape[1] == 2) {
        float a = A.data[0];
        float b = A.data[1];
        float c = A.data[2];
        float d = A.data[3];
        
        float det = a * d - b * c;
        float inv_det = 1.0f / det;
        
        out_Ainv.data[0] = d * inv_det;
        out_Ainv.data[1] = -b * inv_det;
        out_Ainv.data[2] = -c * inv_det;
        out_Ainv.data[3] = a * inv_det;
    } else {
        // Identity for simplicity (should use proper LU decomposition)
        for (int i = 0; i < A.shape[0]; ++i) {
            for (int j = 0; j < A.shape[1]; ++j) {
                out_Ainv.data[i * A.shape[1] + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
}

void logDet(const Tensor& A, Tensor& out_logdet) {
    // Simplified for 2x2
    if (A.shape[0] == 2 && A.shape[1] == 2) {
        float a = A.data[0];
        float b = A.data[1];
        float c = A.data[2];
        float d = A.data[3];
        
        float det = a * d - b * c;
        out_logdet.data[0] = std::log(std::abs(det) + 1e-10f);
    } else {
        out_logdet.data[0] = 0.0f;
    }
}

void sum(const Tensor& input, const std::vector<int>& dims, Tensor& out_output) {
    // Simplified sum over all elements
    float total = 0.0f;
    for (size_t i = 0; i < input.size; ++i) {
        total += input.data[i];
    }
    out_output.data[0] = total;
}

void mean(const Tensor& input, const std::vector<int>& dims, Tensor& out_output) {
    sum(input, dims, out_output);
    out_output.data[0] /= static_cast<float>(input.size);
}

void max(const Tensor& input, const std::vector<int>& dims, Tensor& out_output) {
    float max_val = input.data[0];
    for (size_t i = 1; i < input.size; ++i) {
        max_val = std::max(max_val, input.data[i]);
    }
    out_output.data[0] = max_val;
}

void argmax(const Tensor& input, int dim, Tensor& out_indices) {
    // Simplified: find global argmax
    int max_idx = 0;
    float max_val = input.data[0];
    for (size_t i = 1; i < input.size; ++i) {
        if (input.data[i] > max_val) {
            max_val = input.data[i];
            max_idx = i;
        }
    }
    out_indices.data[0] = static_cast<float>(max_idx);
}

void add(const Tensor& A, const Tensor& B, Tensor& out_C) {
    for (size_t i = 0; i < A.size; ++i) {
        out_C.data[i] = A.data[i] + B.data[i];
    }
}

void subtract(const Tensor& A, const Tensor& B, Tensor& out_C) {
    for (size_t i = 0; i < A.size; ++i) {
        out_C.data[i] = A.data[i] - B.data[i];
    }
}

void multiply(const Tensor& A, const Tensor& B, Tensor& out_C) {
    for (size_t i = 0; i < A.size; ++i) {
        out_C.data[i] = A.data[i] * B.data[i];
    }
}

void scale(const Tensor& A, float scalar, Tensor& out_B) {
    for (size_t i = 0; i < A.size; ++i) {
        out_B.data[i] = A.data[i] * scalar;
    }
}

void exp(const Tensor& A, Tensor& out_B) {
    for (size_t i = 0; i < A.size; ++i) {
        out_B.data[i] = std::exp(A.data[i]);
    }
}

void log(const Tensor& A, Tensor& out_B) {
    for (size_t i = 0; i < A.size; ++i) {
        out_B.data[i] = std::log(A.data[i] + 1e-10f);
    }
}

void clip(const Tensor& A, float min_val, float max_val, Tensor& out_B) {
    for (size_t i = 0; i < A.size; ++i) {
        out_B.data[i] = std::max(min_val, std::min(max_val, A.data[i]));
    }
}

void oneHot(const Tensor& indices, int num_classes, Tensor& out_onehot) {
    // Zero out
    for (size_t i = 0; i < out_onehot.size; ++i) {
        out_onehot.data[i] = 0.0f;
    }
    
    // Set one-hot
    for (int b = 0; b < indices.shape[0]; ++b) {
        int idx = static_cast<int>(indices.data[b]);
        out_onehot.data[b * num_classes + idx] = 1.0f;
    }
}

void zeros(Tensor& A) {
    for (size_t i = 0; i < A.size; ++i) {
        A.data[i] = 0.0f;
    }
}

void ones(Tensor& A) {
    for (size_t i = 0; i < A.size; ++i) {
        A.data[i] = 1.0f;
    }
}

void fill(Tensor& A, float value) {
    for (size_t i = 0; i < A.size; ++i) {
        A.data[i] = value;
    }
}

void eye(Tensor& A) {
    int n = A.shape[0];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A.data[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void randomUniform(Tensor& A, float min_val, float max_val, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < A.size; ++i) {
        A.data[i] = min_val + (max_val - min_val) * static_cast<float>(rand()) / RAND_MAX;
    }
}

void randomNormal(Tensor& A, float mean, float std, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < A.size; i += 2) {
        float u1 = static_cast<float>(rand()) / RAND_MAX;
        float u2 = static_cast<float>(rand()) / RAND_MAX;
        
        float mag = std * std::sqrt(-2.0f * std::log(u1));
        float z0 = mag * std::cos(2.0f * M_PI * u2) + mean;
        float z1 = mag * std::sin(2.0f * M_PI * u2) + mean;
        
        A.data[i] = z0;
        if (i + 1 < A.size) {
            A.data[i + 1] = z1;
        }
    }
}

// CUDA utilities
#ifdef USE_CUDA
void setDevice(int device_id) {
    cudaSetDevice(device_id);
}

int getDevice() {
    int device;
    cudaGetDevice(&device);
    return device;
}

void synchronize() {
    cudaDeviceSynchronize();
}

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(1);
    }
}
#endif

// Memory pool
#ifdef USE_CUDA
CudaMemoryPool& CudaMemoryPool::getInstance() {
    static CudaMemoryPool instance;
    return instance;
}

float* CudaMemoryPool::allocate(size_t size) {
    // Check for existing free block
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            return block.ptr;
        }
    }
    
    // Allocate new block
    float* ptr;
    cudaMalloc(&ptr, size * sizeof(float));
    blocks_.push_back({ptr, size, true});
    return ptr;
}

void CudaMemoryPool::deallocate(float* ptr) {
    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            block.in_use = false;
            return;
        }
    }
}

void CudaMemoryPool::clear() {
    for (auto& block : blocks_) {
        cudaFree(block.ptr);
    }
    blocks_.clear();
}

CudaMemoryPool::~CudaMemoryPool() {
    clear();
}
#endif

} // namespace models
} // namespace axiom

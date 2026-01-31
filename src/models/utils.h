/**
 * Copyright 2025 VERSES AI, Inc.
 *
 * Licensed under the VERSES Academic Research License (the "License");
 * you may not use this file except in compliance with the license.
 */

#pragma once

#include <vector>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
// Stub definitions for CUDA types when not available
typedef int cudaError_t;
#define cudaSuccess 0
typedef void* cudaStream_t;
#endif

namespace axiom {
namespace models {

// Tensor descriptor for CUDA operations
struct Tensor {
    float* data;
    std::vector<int> shape;
    std::vector<int> strides;
    int ndim;
    size_t size;
    bool on_device;
    
    Tensor() : data(nullptr), ndim(0), size(0), on_device(false) {}
    
    void allocate(const std::vector<int>& shape_, bool on_device_ = true);
    void free();
    void copyToDevice();
    void copyFromDevice();
    
    // Access element (for host tensors)
    float& operator()(const std::vector<int>& indices);
    const float& operator()(const std::vector<int>& indices) const;
    
    // Get linear index
    int getIndex(const std::vector<int>& indices) const;
    
    // Reshape (no copy)
    Tensor reshape(const std::vector<int>& new_shape) const;
    
    // Slice (no copy)
    Tensor slice(int dim, int start, int end) const;
};

// ArrayDict for structured parameters
struct ArrayDict {
    std::vector<std::pair<std::string, Tensor>> fields;
    
    void add(const std::string& name, const Tensor& tensor);
    Tensor& get(const std::string& name);
    const Tensor& get(const std::string& name) const;
};

// Utility functions

// Matrix operations
void matmul(const Tensor& A, const Tensor& B, Tensor& out_C,
           bool transpose_A = false, bool transpose_B = false);
void matmulBatch(const Tensor& A, const Tensor& B, Tensor& out_C,
                bool transpose_A = false, bool transpose_B = false);

// Matrix inversion (batched)
void matrixInverse(const Tensor& A, Tensor& out_Ainv);
void matrixInverseBatched(const Tensor& A, Tensor& out_Ainv);

// Log determinant
void logDet(const Tensor& A, Tensor& out_logdet);
void logDetBatched(const Tensor& A, Tensor& out_logdet);

// Cholesky decomposition
void cholesky(const Tensor& A, Tensor& out_L);
void choleskyBatched(const Tensor& A, Tensor& out_L);

// Solve linear systems
void solve(const Tensor& A, const Tensor& B, Tensor& out_X);
void solveBatched(const Tensor& A, const Tensor& B, Tensor& out_X);

// Reduction operations
void sum(const Tensor& input, const std::vector<int>& dims, Tensor& out_output);
void mean(const Tensor& input, const std::vector<int>& dims, Tensor& out_output);
void max(const Tensor& input, const std::vector<int>& dims, Tensor& out_output);
void min(const Tensor& input, const std::vector<int>& dims, Tensor& out_output);
void argmax(const Tensor& input, int dim, Tensor& out_indices);
void argmin(const Tensor& input, int dim, Tensor& out_indices);

// Element-wise operations
void add(const Tensor& A, const Tensor& B, Tensor& out_C);
void subtract(const Tensor& A, const Tensor& B, Tensor& out_C);
void multiply(const Tensor& A, const Tensor& B, Tensor& out_C);
void divide(const Tensor& A, const Tensor& B, Tensor& out_C);
void scale(const Tensor& A, float scalar, Tensor& out_B);
void exp(const Tensor& A, Tensor& out_B);
void log(const Tensor& A, Tensor& out_B);
void log1p(const Tensor& A, Tensor& out_B);
void sqrt(const Tensor& A, Tensor& out_B);
void pow(const Tensor& A, float exponent, Tensor& out_B);
void abs(const Tensor& A, Tensor& out_B);
void clip(const Tensor& A, float min_val, float max_val, Tensor& out_B);

// Special functions
void digamma(const Tensor& A, Tensor& out_B);
void lgamma(const Tensor& A, Tensor& out_B);
void softmax(const Tensor& A, int dim, Tensor& out_B);
void logSoftmax(const Tensor& A, int dim, Tensor& out_B);
void logSumExp(const Tensor& A, int dim, Tensor& out_B);

// Linear algebra helpers
void bdot(const Tensor& A, const Tensor& B, Tensor& out_C);  // Batch dot product
void transpose(const Tensor& A, Tensor& out_B);
void transposeBatched(const Tensor& A, Tensor& out_B);
void diagonal(const Tensor& A, Tensor& out_diag);
void setDiagonal(Tensor& A, const Tensor& diag);

// One-hot encoding
void oneHot(const Tensor& indices, int num_classes, Tensor& out_onehot);

// Bincount
void bincount(const Tensor& indices, int minlength, Tensor& out_counts);

// Broadcasting operations
void broadcastTo(const Tensor& A, const std::vector<int>& target_shape, Tensor& out_B);
bool canBroadcast(const std::vector<int>& shape_a, const std::vector<int>& shape_b);
std::vector<int> broadcastShapes(const std::vector<int>& shape_a, 
                                 const std::vector<int>& shape_b);

// Padding
void pad(const Tensor& A, const std::vector<std::pair<int, int>>& pad_width, 
         float constant_values, Tensor& out_B);

// Indexing
void gather(const Tensor& A, const Tensor& indices, int dim, Tensor& out_B);
void scatter(Tensor& A, const Tensor& indices, int dim, const Tensor& values);

// Set operations
void setdiff1d(const Tensor& A, const Tensor& B, Tensor& out_C);

// Block diagonal
void blockDiag(const std::vector<Tensor>& matrices, Tensor& out_block);

// Initialize
void zeros(Tensor& A);
void ones(Tensor& A);
void fill(Tensor& A, float value);
void eye(Tensor& A);
void randomUniform(Tensor& A, float min_val, float max_val, unsigned int seed);
void randomNormal(Tensor& A, float mean, float std, unsigned int seed);

// Statistics
void meanAndVar(const Tensor& A, int dim, Tensor& out_mean, Tensor& out_var);

// Distance metrics
void squaredError(const Tensor& pred, const Tensor& target, Tensor& out_error);
void euclideanDistance(const Tensor& A, const Tensor& B, Tensor& out_dist);

// CUDA utilities
#ifdef USE_CUDA
void setDevice(int device_id);
int getDevice();
void synchronize();
void checkCudaError(cudaError_t error, const char* file, int line);

#define CUDA_CHECK(err) checkCudaError(err, __FILE__, __LINE__)
#else
// No-op versions for non-CUDA builds
inline void setDevice(int) {}
inline int getDevice() { return 0; }
inline void synchronize() {}
inline void checkCudaError(cudaError_t, const char*, int) {}
#define CUDA_CHECK(err) 
#endif

// Memory management
class CudaMemoryPool {
public:
    static CudaMemoryPool& getInstance();
    
    float* allocate(size_t size);
    void deallocate(float* ptr);
    void clear();
    
private:
    CudaMemoryPool() = default;
    ~CudaMemoryPool();
    
    struct Block {
        float* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks_;
};

} // namespace models
} // namespace axiom

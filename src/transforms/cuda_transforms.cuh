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

#include "types.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace axiom {
namespace transforms {
namespace cuda {

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(err)); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t err = call; \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            throw std::runtime_error("cuSOLVER error: " + std::to_string(err)); \
        } \
    } while(0)

// ============================================================================
// Device Array Structure
// ============================================================================

struct DeviceArray {
    float* data;
    int* shape;
    int ndim;
    size_t size;
    
    DeviceArray() : data(nullptr), shape(nullptr), ndim(0), size(0) {}
    DeviceArray(const Array& host_array);
    ~DeviceArray();
    
    void upload(const Array& host_array);
    Array download() const;
};

// ============================================================================
// Matrix Normal-Gamma Parameter Update Kernels
// ============================================================================

/**
 * @brief CUDA kernel for updating Matrix Normal-Gamma parameters from sufficient statistics.
 * 
 * Updates:
 *   inv_v = prior_inv_v + sum(xx)
 *   eta_2 = prior_eta_2 + sum(yx)
 *   eta_3 = prior_eta_3 + sum(yy_diag)
 *   eta_4 = prior_eta_4 + N
 */
__global__ void update_mng_params_kernel(
    float* inv_v,           // [batch, x_dim, x_dim]
    float* eta_2,           // [batch, y_dim, x_dim]
    float* eta_3,           // [batch, y_dim, 1]
    float* eta_4,           // [batch, y_dim, 1]
    const float* prior_inv_v,
    const float* prior_eta_2,
    const float* prior_eta_3,
    const float* prior_eta_4,
    const float* sum_xx,    // [batch, x_dim, x_dim]
    const float* sum_yx,    // [batch, y_dim, x_dim]
    const float* sum_yy,    // [batch, y_dim]
    const float* count,     // [batch, 1]
    int batch_size,
    int x_dim,
    int y_dim,
    float lr,
    float beta);

/**
 * @brief Batch version of parameter update for multiple mixture components.
 */
void launch_update_mng_params(
    const DeviceArray& inv_v,
    const DeviceArray& eta_2,
    const DeviceArray& eta_3,
    const DeviceArray& eta_4,
    const DeviceArray& prior_inv_v,
    const DeviceArray& prior_eta_2,
    const DeviceArray& prior_eta_3,
    const DeviceArray& prior_eta_4,
    const DeviceArray& sum_xx,
    const DeviceArray& sum_yx,
    const DeviceArray& sum_yy,
    const DeviceArray& count,
    int batch_size,
    int x_dim,
    int y_dim,
    float lr,
    float beta,
    cudaStream_t stream = 0);

// ============================================================================
// Forward/Backward Message Passing Kernels
// ============================================================================

/**
 * @brief CUDA kernel for forward message passing (marginalizing over x).
 * 
 * Computes p(y) = int p(y|x) p(x) dx using Schur complement.
 */
__global__ void forward_from_normal_kernel(
    float* out_inv_sigma,      // [batch, y_dim, y_dim]
    float* out_inv_sigma_mu,   // [batch, y_dim, 1]
    float* out_residual,
    const float* A_inv_sigma,  // [batch, y_dim, y_dim]
    const float* A_inv_sigma_x,// [batch, y_dim, x_dim]
    const float* D,            // [batch, x_dim, x_dim] (x_inv_sigma_x + px_inv_sigma)
    const float* C_x,          // [batch, x_dim, 1]
    const float* expected_logdet,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias);

/**
 * @brief CUDA kernel for backward message passing (marginalizing over y).
 */
__global__ void backward_from_normal_kernel(
    float* out_inv_sigma,
    float* out_inv_sigma_mu,
    float* out_residual,
    const float* A,
    const float* B,
    const float* C_y,
    const float* invA,
    float logdetA,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias);

void launch_forward_from_normal(
    const DeviceArray& out_inv_sigma,
    const DeviceArray& out_inv_sigma_mu,
    const DeviceArray& out_residual,
    const DeviceArray& A_inv_sigma,
    const DeviceArray& A_inv_sigma_x,
    const DeviceArray& D,
    const DeviceArray& C_x,
    const DeviceArray& expected_logdet,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias,
    cudaStream_t stream = 0);

void launch_backward_from_normal(
    const DeviceArray& out_inv_sigma,
    const DeviceArray& out_inv_sigma_mu,
    const DeviceArray& out_residual,
    const DeviceArray& A,
    const DeviceArray& B,
    const DeviceArray& C_y,
    const DeviceArray& invA,
    float logdetA,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias,
    cudaStream_t stream = 0);

// ============================================================================
// Schur Computation Kernels
// ============================================================================

/**
 * @brief Compute Schur complement for precision matrix marginalization.
 * 
 * For joint precision [A, B; B^T, D], the Schur complement is:
 *   S = A - B @ inv(D) @ B^T
 */
__global__ void schur_complement_kernel(
    float* S,           // Output: [batch, m, m]
    const float* A,     // [batch, m, m]
    const float* B,     // [batch, m, n]
    const float* invD,  // [batch, n, n]
    int batch_size,
    int m,
    int n);

void launch_schur_complement(
    const DeviceArray& S,
    const DeviceArray& A,
    const DeviceArray& B,
    const DeviceArray& invD,
    int batch_size,
    int m,
    int n,
    cudaStream_t stream = 0);

// ============================================================================
// Precision Matrix Update Kernels
// ============================================================================

/**
 * @brief Update precision matrix from natural parameters.
 */
__global__ void update_precision_kernel(
    float* inv_sigma,       // [batch, dim, dim]
    const float* inv_sigma_x,// [batch, dim, other_dim]
    const float* x_inv_sigma_x, // [batch, other_dim, other_dim]
    int batch_size,
    int dim,
    int other_dim);

void launch_update_precision(
    const DeviceArray& inv_sigma,
    const DeviceArray& inv_sigma_x,
    const DeviceArray& x_inv_sigma_x,
    int batch_size,
    int dim,
    int other_dim,
    cudaStream_t stream = 0);

// ============================================================================
// Batch Linear Prediction Kernels
// ============================================================================

/**
 * @brief Batch linear prediction kernel.
 * 
 * Computes y_pred = W @ x + b for multiple batches and samples.
 */
__global__ void batch_linear_predict_kernel(
    float* y_out,       // [num_samples, batch, y_dim, 1]
    const float* W,     // [batch, y_dim, x_dim]
    const float* x,     // [num_samples, batch, x_dim, 1]
    const float* b,     // [batch, y_dim, 1] or nullptr
    int num_samples,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias);

void launch_batch_linear_predict(
    const DeviceArray& y_out,
    const DeviceArray& W,
    const DeviceArray& x,
    const DeviceArray& b,
    int num_samples,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias,
    cudaStream_t stream = 0);

// ============================================================================
// Expected Value Computation Kernels
// ============================================================================

/**
 * @brief Compute expected inverse covariance from Gamma parameters.
 */
__global__ void expected_inv_sigma_kernel(
    float* inv_sigma,   // [batch, y_dim, y_dim] - diagonal matrix output
    const float* a,     // [batch, y_dim, 1]
    const float* b,     // [batch, y_dim, 1]
    int batch_size,
    int y_dim);

/**
 * @brief Compute expected log determinant from Gamma parameters.
 */
__global__ void expected_logdet_sigma_kernel(
    float* logdet,      // [batch, 1, 1]
    const float* a,     // [batch, y_dim, 1]
    const float* b,     // [batch, y_dim, 1]
    int batch_size,
    int y_dim);

void launch_expected_inv_sigma(
    const DeviceArray& inv_sigma,
    const DeviceArray& a,
    const DeviceArray& b,
    int batch_size,
    int y_dim,
    cudaStream_t stream = 0);

void launch_expected_logdet_sigma(
    const DeviceArray& logdet,
    const DeviceArray& a,
    const DeviceArray& b,
    int batch_size,
    int y_dim,
    cudaStream_t stream = 0);

// ============================================================================
// Sufficient Statistics Computation Kernels
// ============================================================================

/**
 * @brief Compute sufficient statistics from data.
 */
__global__ void compute_statistics_kernel(
    float* sum_xx,      // [batch, x_dim, x_dim]
    float* sum_yx,      // [batch, y_dim, x_dim]
    float* sum_yy,      // [batch, y_dim]
    float* count,       // [batch, 1]
    const float* X,     // [num_samples, batch, x_dim, 1]
    const float* Y,     // [num_samples, batch, y_dim, 1]
    const float* weights, // [num_samples, batch] or nullptr
    int num_samples,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias);

void launch_compute_statistics(
    const DeviceArray& sum_xx,
    const DeviceArray& sum_yx,
    const DeviceArray& sum_yy,
    const DeviceArray& count,
    const DeviceArray& X,
    const DeviceArray& Y,
    const DeviceArray* weights,
    int num_samples,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias,
    cudaStream_t stream = 0);

// ============================================================================
// cuBLAS/cuSOLVER Wrappers
// ============================================================================

/**
 * @brief Batch matrix multiplication using cuBLAS.
 */
void cublas_batch_matmul(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int batch_size,
    int m,
    int n,
    int k,
    float alpha = 1.0f,
    float beta = 0.0f,
    bool transA = false,
    bool transB = false);

/**
 * @brief Batch Cholesky decomposition using cuSOLVER.
 */
void cusolver_batch_cholesky(
    cusolverDnHandle_t handle,
    float* A,           // Input/output matrix [batch, n, n]
    int batch_size,
    int n,
    int* info);         // Array of batch_size integers for error codes

/**
 * @brief Batch Cholesky solve using cuSOLVER.
 */
void cusolver_batch_cholesky_solve(
    cusolverDnHandle_t handle,
    const float* L,     // Cholesky factor [batch, n, n]
    const float* B,     // RHS [batch, n, nrhs]
    float* X,           // Solution [batch, n, nrhs]
    int batch_size,
    int n,
    int nrhs);

/**
 * @brief Batch matrix inverse using Cholesky decomposition.
 */
void cusolver_batch_cholesky_inverse(
    cusolverDnHandle_t handle,
    float* A,           // Input matrix, output inverse [batch, n, n]
    float* work,        // Workspace
    int batch_size,
    int n,
    float* logdet,      // Output: log determinant [batch]
    int* info);

// ============================================================================
// Initialization and Context Management
// ============================================================================

/**
 * @brief Initialize CUDA context for transforms.
 */
void initialize_cuda_context();

/**
 * @brief Cleanup CUDA context.
 */
void cleanup_cuda_context();

/**
 * @brief Get cuBLAS handle.
 */
cublasHandle_t get_cublas_handle();

/**
 * @brief Get cuSOLVER handle.
 */
cusolverDnHandle_t get_cusolver_handle();

} // namespace cuda
} // namespace transforms
} // namespace axiom

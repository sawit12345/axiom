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

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace axiom {
namespace cuda {

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Batch Multivariate Normal Operations
// ============================================================================

/**
 * @brief Compute batch MVN log-likelihood: -0.5 * (x-μ)ᵀΣ⁻¹(x-μ) + 0.5*log|Σ⁻¹| - 0.5*D*log(2π)
 * 
 * @param d_x Input data [batch_size, n_samples, dim]
 * @param d_mu Mean [batch_size, dim, 1]
 * @param d_inv_sigma Inverse covariance [batch_size, dim, dim]
 * @param d_logdet_inv_sigma log|Σ⁻¹| [batch_size, 1, 1]
 * @param d_result Output log-likelihood [batch_size, n_samples]
 * @param batch_size Number of batch elements
 * @param n_samples Number of samples per batch
 * @param dim Dimensionality
 */
template<typename T>
__global__ void batch_mvn_log_likelihood_kernel(
    const T* d_x,
    const T* d_mu,
    const T* d_inv_sigma,
    const T* d_logdet_inv_sigma,
    T* d_result,
    int batch_size,
    int n_samples,
    int dim);

/**
 * @brief Batch MVN log-likelihood computation (host wrapper)
 */
template<typename T>
void batch_mvn_log_likelihood(
    const T* d_x,
    const T* d_mu,
    const T* d_inv_sigma,
    const T* d_logdet_inv_sigma,
    T* d_result,
    int batch_size,
    int n_samples,
    int dim,
    cudaStream_t stream = 0);

/**
 * @brief Compute Mahalanobis distance: (x-μ)ᵀΣ⁻¹(x-μ)
 */
template<typename T>
__global__ void mahalanobis_distance_kernel(
    const T* d_diff,       // [batch, n_samples, dim]
    const T* d_inv_sigma,  // [batch, dim, dim]
    T* d_result,           // [batch, n_samples]
    int batch_size,
    int n_samples,
    int dim);

// ============================================================================
// Batch Multinomial Operations  
// ============================================================================

/**
 * @brief Stable softmax computation: exp(x - max(x)) / sum(exp(x - max(x)))
 */
template<typename T>
__global__ void stable_softmax_kernel(
    const T* d_logits,  // [batch, dim]
    T* d_probs,         // [batch, dim]
    int batch_size,
    int dim);

/**
 * @brief Stable log-sum-exp: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
 */
template<typename T>
__global__ void stable_logsumexp_kernel(
    const T* d_logits,  // [batch, dim]
    T* d_result,        // [batch]
    int batch_size,
    int dim);

/**
 * @brief Multinomial log probability: sum(x * log(p))
 */
template<typename T>
__global__ void multinomial_log_prob_kernel(
    const T* d_x,       // [batch, n_samples, dim]
    const T* d_logits,  // [batch, dim]
    const T* d_logZ,    // [batch, 1]
    T* d_result,        // [batch, n_samples]
    int batch_size,
    int n_samples,
    int dim);

/**
 * @brief Sample from multinomial distribution using Gumbel trick
 */
template<typename T>
__global__ void multinomial_sample_gumbel_kernel(
    const T* d_logits,  // [batch, dim]
    const T* d_uniform, // [batch, n_samples, dim] uniform random samples
    T* d_samples,       // [batch, n_samples, dim] one-hot
    int batch_size,
    int n_samples,
    int dim);

// ============================================================================
// Parameter Updates
// ============================================================================

/**
 * @brief Conjugate posterior update: η = η₀ + lr * Σ T(x)
 */
template<typename T>
__global__ void conjugate_update_eta_kernel(
    T* d_eta,           // [batch, dim]
    const T* d_eta0,    // [batch, dim] prior
    const T* d_sum_tx,  // [batch, dim] summed statistics
    T lr,
    T beta,
    int batch_size,
    int dim);

/**
 * @brief Conjugate posterior update: ν = ν₀ + lr * N
 */
template<typename T>
__global__ void conjugate_update_nu_kernel(
    T* d_nu,           // [batch]
    const T* d_nu0,    // [batch] prior
    const T* d_N,      // [batch] count
    T lr,
    T beta,
    int batch_size);

/**
 * @brief Batch weighted statistics sum: Σ w * T(x)
 */
template<typename T>
__global__ void weighted_stats_sum_kernel(
    const T* d_stats,   // [n_samples, batch, dim]
    const T* d_weights, // [n_samples, batch]
    T* d_result,        // [batch, dim]
    int n_samples,
    int batch_size,
    int dim);

// ============================================================================
// KL Divergence Computation
// ============================================================================

/**
 * @brief KL divergence for Normal-Inverse-Wishart
 * 
 * KL = 0.5 * [(κ₀/κ - 1 + log(κ/κ₀)) * d 
 *       + κ₀ * (μ - μ₀)ᵀ * E[Σ⁻¹] * (μ - μ₀)]
 *       + KL_wishart
 */
template<typename T>
__global__ void mvn_conjugate_kl_kernel(
    const T* d_mean,          // [batch, dim, 1]
    const T* d_prior_mean,    // [batch, dim, 1]
    const T* d_kappa,         // [batch, 1, 1]
    const T* d_prior_kappa,   // [batch, 1, 1]
    const T* d_inv_u,         // [batch, dim, dim]
    const T* d_logdet_inv_u,  // [batch, 1, 1]
    const T* d_prior_inv_u,   // [batch, dim, dim]
    const T* d_prior_logdet_inv_u, // [batch, 1, 1]
    const T* d_n,             // [batch, 1, 1]
    const T* d_prior_n,       // [batch, 1, 1]
    T* d_result,              // [batch]
    int batch_size,
    int dim);

/**
 * @brief Multivariate digamma function for NIW KL computation
 */
template<typename T>
__device__ T mvdigamma_device(T x, int d);

/**
 * @brief Multivariate log-gamma function for NIW KL computation
 */
template<typename T>
__device__ T mvgammaln_device(T x, int d);

/**
 * @brief KL divergence for Dirichlet distribution
 */
template<typename T>
__global__ void dirichlet_kl_kernel(
    const T* d_alpha,       // [batch, dim]
    const T* d_prior_alpha, // [batch, dim]
    T* d_result,            // [batch]
    int batch_size,
    int dim);

/**
 * @brief Digamma function (device implementation)
 */
template<typename T>
__device__ T digamma_device(T x);

/**
 * @brief Log-gamma function (device implementation)
 */
template<typename T>
__device__ T gammaln_device(T x);

// ============================================================================
// Cholesky Decomposition and Matrix Operations
// ============================================================================

/**
 * @brief Batch Cholesky decomposition for sampling and logdet computation
 */
template<typename T>
void batch_cholesky(
    cusolverDnHandle_t handle,
    const T* d_A,      // [batch, dim, dim]
    T* d_L,            // [batch, dim, dim]
    int* d_info,
    int batch_size,
    int dim);

/**
 * @brief Batch matrix inverse using Cholesky
 */
template<typename T>
void batch_cholesky_inverse(
    cusolverDnHandle_t handle,
    cublasHandle_t cublas_handle,
    const T* d_A,      // [batch, dim, dim]
    T* d_inv_A,        // [batch, dim, dim]
    T* d_logdet,       // [batch] (optional, can be null)
    int* d_info,
    int batch_size,
    int dim);

/**
 * @brief Sample from standard normal using Box-Muller transform
 */
template<typename T>
__global__ void box_muller_sample_kernel(
    const T* d_uniform1,  // [batch, n_samples, dim]
    const T* d_uniform2,  // [batch, n_samples, dim]
    T* d_result,          // [batch, n_samples, dim]
    int batch_size,
    int n_samples,
    int dim);

/**
 * @brief Transform standard normal samples to MVN: μ + L * z
 */
template<typename T>
__global__ void mvn_transform_samples_kernel(
    const T* d_z,        // [batch, n_samples, dim]
    const T* d_mu,       // [batch, dim, 1]
    const T* d_L,        // [batch, dim, dim] (Cholesky of covariance)
    T* d_samples,        // [batch, n_samples, dim]
    int batch_size,
    int n_samples,
    int dim);

// ============================================================================
// Matrix-Vector Operations
// ============================================================================

/**
 * @brief Batch matrix-vector multiply: y = A * x
 */
template<typename T>
__global__ void batch_matvec_kernel(
    const T* d_A,  // [batch, m, n]
    const T* d_x,  // [batch, n, 1] or [batch, n]
    T* d_y,        // [batch, m, 1] or [batch, m]
    int batch_size,
    int m,
    int n,
    bool transpose_A = false);

/**
 * @brief Batch quadratic form: xᵀ A x
 */
template<typename T>
__global__ void batch_quadratic_form_kernel(
    const T* d_x,  // [batch, n, 1]
    const T* d_A,  // [batch, n, n]
    T* d_result,   // [batch]
    int batch_size,
    int n);

/**
 * @brief Batch outer product: x xᵀ
 */
template<typename T>
__global__ void batch_outer_product_kernel(
    const T* d_x,  // [batch, n, 1]
    T* d_result,   // [batch, n, n]
    int batch_size,
    int n,
    T scale = T(1.0));

// ============================================================================
// Utility Kernels
// ============================================================================

/**
 * @brief Element-wise addition: c = a + b
 */
template<typename T>
__global__ void elementwise_add_kernel(
    const T* d_a,
    const T* d_b,
    T* d_c,
    int n);

/**
 * @brief Element-wise multiply: c = a * b
 */
template<typename T>
__global__ void elementwise_mul_kernel(
    const T* d_a,
    const T* d_b,
    T* d_c,
    int n);

/**
 * @brief Element-wise scale: b = alpha * a
 */
template<typename T>
__global__ void elementwise_scale_kernel(
    const T* d_a,
    T* d_b,
    T alpha,
    int n);

/**
 * @brief Sum over last dimension
 */
template<typename T>
__global__ void sum_last_dim_kernel(
    const T* d_input,  // [batch, dim]
    T* d_output,       // [batch]
    int batch_size,
    int dim);

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicit instantiations for float and double
extern template void batch_mvn_log_likelihood<float>(
    const float*, const float*, const float*, const float*,
    float*, int, int, int, cudaStream_t);

extern template void batch_mvn_log_likelihood<double>(
    const double*, const double*, const double*, const double*,
    double*, int, int, int, cudaStream_t);

} // namespace cuda
} // namespace axiom

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

#include "cuda_utils.h"
#include "tensor.h"
#include "math.h"
#include <cfloat>
#include <cmath>

// ============================================================================
// Device Math Functions
// ============================================================================

// Implementation of digamma for CUDA device
device double cuda_digamma(double x) {
    if (x <= 0.0) {
        // Reflection formula
        double result = cuda_digamma(1.0 - x) - M_PI / tan(M_PI * x);
        return result;
    }
    
    double result = 0.0;
    while (x < 10.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    double inv_x4 = inv_x2 * inv_x2;
    
    result += log(x) - 0.5 * inv_x - inv_x2 / 12.0 + inv_x4 / 120.0 - inv_x4 * inv_x2 / 252.0;
    return result;
}

device double cuda_gammaln(double x) {
    if (x <= 0.0) return 0.0 / 0.0;  // NaN
    
    if (x < 0.5) {
        return log(M_PI) - log(sin(M_PI * x)) - cuda_gammaln(1.0 - x);
    }
    
    x -= 1.0;
    double a = 0.99999999999980993;
    a += 676.5203681218851 / (x + 1.0);
    a += -1259.1392167224028 / (x + 2.0);
    a += 771.32342877765313 / (x + 3.0);
    a += -176.61502916214059 / (x + 4.0);
    a += 12.507343278686905 / (x + 5.0);
    a += -0.13857109526572012 / (x + 6.0);
    a += 9.9843695780195716e-6 / (x + 7.0);
    a += 1.5056327351493116e-7 / (x + 8.0);
    
    double t = x + 7.0 + 0.5;
    return 0.5 * log(2.0 * M_PI) + log(a) - t + log(t) * (x + 0.5);
}

// ============================================================================
// Gaussian Log-Likelihood Kernels
// ============================================================================

global void gaussian_loglik_kernel(
    const double* x, const double* mean, const double* inv_cov,
    const double* logdet_inv_cov, int dim, int batch_size, double* output) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const double* x_b = x + idx * dim;
    const double* mean_b = mean + idx * dim;
    const double* inv_cov_b = inv_cov + idx * dim * dim;
    
    // Compute (x - mean)
    extern shared double diff[];
    for (int i = threadIdx.y; i < dim; i += blockDim.y) {
        diff[i] = x_b[i] - mean_b[i];
    }
    __syncthreads();
    
    // Compute (x-mean)^T * inv_cov * (x-mean)
    double mahalanobis = 0.0;
    for (int i = threadIdx.y; i < dim; i += blockDim.y) {
        double row_sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            row_sum += inv_cov_b[i * dim + j] * diff[j];
        }
        mahalanobis += diff[i] * row_sum;
    }
    
    // Reduce within block
    mahalanobis = warp_reduce_sum(mahalanobis);
    if (threadIdx.y == 0) {
        // log p(x | mu, Sigma) = -0.5 * mahalanobis + 0.5 * logdet_inv_cov - 0.5 * D * log(2*pi)
        output[idx] = -0.5 * mahalanobis + 0.5 * logdet_inv_cov[idx] - 0.5 * dim * 1.8378770664093453;
    }
}

global void gaussian_loglik_isotropic_kernel(
    const double* x, const double* mean, const double* sigma_sqr,
    int dim, int batch_size, int num_components, double* output) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_components;
    if (idx >= total) return;
    
    int b = idx / num_components;
    int k = idx % num_components;
    
    const double* x_b = x + b * dim;
    const double* mean_bk = mean + (b * num_components + k) * dim;
    double sigma_sqr_bk = sigma_sqr[b * num_components + k];
    
    // Compute squared error
    double sq_error = 0.0;
    for (int i = 0; i < dim; ++i) {
        double diff = x_b[i] - mean_bk[i];
        sq_error += diff * diff;
    }
    
    double logdet = dim * log(sigma_sqr_bk);
    output[idx] = -0.5 * sq_error / sigma_sqr_bk - 0.5 * logdet - 0.5 * dim * 1.8378770664093453;
}

// ============================================================================
// Softmax Kernels
// ============================================================================

global void softmax_kernel(
    const double* input, double* output, int dim_size, int batch_size) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    const double* in_batch = input + batch_idx * dim_size;
    double* out_batch = output + batch_idx * dim_size;
    
    // Find max for numerical stability
    double max_val = in_batch[0];
    for (int i = 1; i < dim_size; ++i) {
        if (in_batch[i] > max_val) max_val = in_batch[i];
    }
    
    // Compute exp(x - max) and sum
    double sum_exp = 0.0;
    for (int i = 0; i < dim_size; ++i) {
        double exp_val = exp(in_batch[i] - max_val);
        out_batch[i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < dim_size; ++i) {
        out_batch[i] /= sum_exp;
    }
}

global void logsumexp_kernel(
    const double* input, double* output, int dim_size, int batch_size) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    const double* in_batch = input + batch_idx * dim_size;
    
    // Find max
    double max_val = in_batch[0];
    for (int i = 1; i < dim_size; ++i) {
        if (in_batch[i] > max_val) max_val = in_batch[i];
    }
    
    // Compute sum of exp
    double sum_exp = 0.0;
    for (int i = 0; i < dim_size; ++i) {
        sum_exp += exp(in_batch[i] - max_val);
    }
    
    output[batch_idx] = max_val + log(sum_exp);
}

// ============================================================================
// Matrix Multiplication Kernels
// ============================================================================

// Simple matrix multiplication kernel
global void matmul_kernel(
    const double* A, const double* B, double* C,
    int m, int k, int n) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        sum += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
}

// Batched matrix multiplication
global void batch_matmul_kernel(
    const double* A, const double* B, double* C,
    int batch_size, int m, int k, int n) {
    
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || row >= m || col >= n) return;
    
    size_t a_offset = batch * m * k;
    size_t b_offset = batch * k * n;
    size_t c_offset = batch * m * n;
    
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        sum += A[a_offset + row * k + i] * B[b_offset + i * n + col];
    }
    C[c_offset + row * n + col] = sum;
}

// Optimized shared memory matrix multiplication
global void matmul_shared_kernel(
    const double* A, const double* B, double* C,
    int m, int k, int n) {
    
    const int BLOCK_SIZE = 16;
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    shared double As[BLOCK_SIZE][BLOCK_SIZE];
    shared double Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    double sum = 0.0;
    
    for (int bk = 0; bk < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; ++bk) {
        // Load tiles into shared memory
        if (row < m && bk * BLOCK_SIZE + threadIdx.x < k) {
            As[threadIdx.y][threadIdx.x] = A[row * k + bk * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        if (col < n && bk * BLOCK_SIZE + threadIdx.y < k) {
            Bs[threadIdx.y][threadIdx.x] = B[(bk * BLOCK_SIZE + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// ============================================================================
// Reduction Kernels
// ============================================================================

template<typename T>
global void sum_kernel(
    const T* input, T* output, int size) {
    
    extern shared T sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    sdata[tid] = (i < size) ? input[i] : T(0);
    __syncthreads();
    
    // Tree-based reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

template<typename T>
global void mean_kernel(
    const T* input, T* output, int size) {
    
    extern shared T sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? input[i] : T(0);
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0] / size;
    }
}

template<typename T>
global void max_kernel(
    const T* input, T* output, int size) {
    
    extern shared T sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? input[i] : T(-1e308);
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Cholesky Decomposition Kernel (simplified - for small matrices)
// ============================================================================

global void cholesky_kernel(
    const double* A, double* L, int n, int batch_size) {
    
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    size_t offset = batch * n * n;
    const double* A_b = A + offset;
    double* L_b = L + offset;
    
    // Copy A to L
    for (int i = threadIdx.x; i < n * n; i += blockDim.x) {
        L_b[i] = A_b[i];
    }
    __syncthreads();
    
    // Cholesky decomposition (Cholesky-Banachiewicz)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (threadIdx.x == 0) {
                double sum = L_b[i * n + j];
                
                for (int k = 0; k < j; ++k) {
                    sum -= L_b[i * n + k] * L_b[j * n + k];
                }
                
                if (i == j) {
                    L_b[i * n + i] = sqrt(max(sum, 1e-10));
                } else {
                    L_b[i * n + j] = sum / L_b[j * n + j];
                }
            }
        }
        __syncthreads();
    }
    
    // Zero out upper triangle
    for (int idx = threadIdx.x; idx < n * n; idx += blockDim.x) {
        int i = idx / n;
        int j = idx % n;
        if (j > i) L_b[idx] = 0.0;
    }
}

// ============================================================================
// Element-wise Operations
// ============================================================================

#define DEFINE_ELEMENTWISE_OP(name, op) \
    global void name##_kernel(const double* a, const double* b, double* c, int n) { \
        int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < n) c[idx] = a[idx] op b[idx]; \
    }

DEFINE_ELEMENTWISE_OP(add, +)
DEFINE_ELEMENTWISE_OP(sub, -)
DEFINE_ELEMENTWISE_OP(mul, *)
DEFINE_ELEMENTWISE_OP(div, /)

global void add_scalar_kernel(const double* a, double scalar, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + scalar;
}

global void mul_scalar_kernel(const double* a, double scalar, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * scalar;
}

global void exp_kernel(const double* input, double* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = exp(input[idx]);
}

global void log_kernel(const double* input, double* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = log(input[idx]);
}

// ============================================================================
// Random Number Generation Kernels (Threefry)
// ============================================================================

// Threefry rotation
static device uint64_t rotl64_dev(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// Threefry 2x64 random generator
device uint64_t threefry_dev(uint64_t key0, uint64_t key1, uint64_t counter) {
    uint64_t X0 = counter;
    uint64_t X1 = counter ^ key0;
    
    uint64_t ks[3];
    ks[0] = key0;
    ks[1] = key1;
    ks[2] = 0x9E3779B97F4A7C15ULL ^ ks[0] ^ ks[1];
    
    // Initial injection
    X0 += ks[0];
    X1 += ks[1];
    
    // Rounds
    X0 += X1; X1 = rotl64_dev(X1, 16) ^ X0;
    X0 += X1; X1 = rotl64_dev(X1, 42) ^ X0;
    X0 += ks[1]; X1 += ks[2] + 1;
    
    X0 += X1; X1 = rotl64_dev(X1, 12) ^ X0;
    X0 += X1; X1 = rotl64_dev(X1, 31) ^ X0;
    X0 += ks[2]; X1 += ks[0] + 2;
    
    X0 += X1; X1 = rotl64_dev(X1, 16) ^ X0;
    X0 += X1; X1 = rotl64_dev(X1, 32) ^ X0;
    X0 += ks[0]; X1 += ks[1] + 3;
    
    X0 += X1; X1 = rotl64_dev(X1, 24) ^ X0;
    X0 += X1; X1 = rotl64_dev(X1, 21) ^ X0;
    X0 += ks[1]; X1 += ks[2] + 4;
    
    return X0;
}

global void uniform_random_kernel(
    uint64_t key0, uint64_t key1, double* output, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint64_t bits = threefry_dev(key0, key1, (uint64_t)idx);
    // Convert to double in [0, 1)
    const uint64_t mantissa = bits & ((1ULL << 53) - 1);
    output[idx] = (double)mantissa / (double)(1ULL << 53);
}

global void normal_random_kernel(
    uint64_t key0, uint64_t key1, double* output, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Box-Muller transform
    uint64_t bits1 = threefry_dev(key0, key1, (uint64_t)idx * 2);
    uint64_t bits2 = threefry_dev(key0, key1, (uint64_t)idx * 2 + 1);
    
    double u1 = (double)(bits1 & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
    double u2 = (double)(bits2 & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
    
    // Avoid log(0)
    if (u1 < 1e-15) u1 = 1e-15;
    
    double radius = sqrt(-2.0 * log(u1));
    double angle = 2.0 * M_PI * u2;
    
    output[idx] = radius * cos(angle);
}

global void fill_kernel(double* data, double value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

// ============================================================================
// Host-Callable Wrapper Functions
// ============================================================================

namespace axiomcuda {

// Gaussian log-likelihood
void launch_gaussian_loglik(
    const double* d_x, const double* d_mean, const double* d_inv_cov,
    const double* d_logdet, int dim, int batch_size, double* d_output,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    gaussian_loglik_kernel<<<blocks, threads, dim * sizeof(double), stream>>>(
        d_x, d_mean, d_inv_cov, d_logdet, dim, batch_size, d_output);
    CUDA_CHECK(cudaGetLastError());
}

void launch_gaussian_loglik_isotropic(
    const double* d_x, const double* d_mean, const double* d_sigma_sqr,
    int dim, int batch_size, int num_components, double* d_output,
    cudaStream_t stream) {
    
    int total = batch_size * num_components;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    gaussian_loglik_isotropic_kernel<<<blocks, threads, 0, stream>>>(
        d_x, d_mean, d_sigma_sqr, dim, batch_size, num_components, d_output);
    CUDA_CHECK(cudaGetLastError());
}

// Softmax
void launch_softmax(
    const double* d_input, double* d_output, int dim_size, int batch_size,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    softmax_kernel<<<blocks, threads, 0, stream>>>(
        d_input, d_output, dim_size, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_logsumexp(
    const double* d_input, double* d_output, int dim_size, int batch_size,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    logsumexp_kernel<<<blocks, threads, 0, stream>>>(
        d_input, d_output, dim_size, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

// Matrix multiplication
void launch_matmul(
    const double* d_A, const double* d_B, double* d_C,
    int m, int k, int n, cudaStream_t stream) {
    
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    
    matmul_shared_kernel<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C, m, k, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_batch_matmul(
    const double* d_A, const double* d_B, double* d_C,
    int batch_size, int m, int k, int n, cudaStream_t stream) {
    
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y, batch_size);
    
    batch_matmul_kernel<<<blocks, threads, 0, stream>>>(
        d_A, d_B, d_C, batch_size, m, k, n);
    CUDA_CHECK(cudaGetLastError());
}

// Reductions
void launch_sum(
    const double* d_input, double* d_output, int size, cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    sum_kernel<<<blocks, threads, threads * sizeof(double), stream>>>(
        d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_max(
    const double* d_input, double* d_output, int size, cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    max_kernel<<<blocks, threads, threads * sizeof(double), stream>>>(
        d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
}

// Element-wise operations
void launch_add(
    const double* d_a, const double* d_b, double* d_c, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mul(
    const double* d_a, const double* d_b, double* d_c, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mul_scalar(
    const double* d_a, double scalar, double* d_c, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_scalar_kernel<<<blocks, threads, 0, stream>>>(d_a, scalar, d_c, n);
    CUDA_CHECK(cudaGetLastError());
}

// Random
void launch_uniform_random(
    uint64_t key0, uint64_t key1, double* d_output, int n, cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    uniform_random_kernel<<<blocks, threads, 0, stream>>>(key0, key1, d_output, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_normal_random(
    uint64_t key0, uint64_t key1, double* d_output, int n, cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    normal_random_kernel<<<blocks, threads, 0, stream>>>(key0, key1, d_output, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fill(double* d_data, double value, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fill_kernel<<<blocks, threads, 0, stream>>>(d_data, value, n);
    CUDA_CHECK(cudaGetLastError());
}

// Cholesky
void launch_cholesky(
    const double* d_A, double* d_L, int n, int batch_size, cudaStream_t stream) {
    
    int threads = 256;
    cholesky_kernel<<<batch_size, threads, 0, stream>>>(d_A, d_L, n, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace axiomcuda

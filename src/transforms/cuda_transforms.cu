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

#include "cuda_transforms.cuh"
#include <cmath>
#include <stdexcept>

namespace axiom {
namespace transforms {
namespace cuda {

// ============================================================================
// Device Array Implementation
// ============================================================================

DeviceArray::DeviceArray(const Array& host_array) {
    size = host_array.size() * sizeof(float);
    ndim = host_array.ndim();
    
    CUDA_CHECK(cudaMalloc(&data, size));
    CUDA_CHECK(cudaMalloc(&shape, ndim * sizeof(int)));
    
    upload(host_array);
}

DeviceArray::~DeviceArray() {
    if (data) cudaFree(data);
    if (shape) cudaFree(shape);
}

void DeviceArray::upload(const Array& host_array) {
    CUDA_CHECK(cudaMemcpy(data, host_array.data.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(shape, host_array.shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice));
}

Array DeviceArray::download() const {
    Array result;
    result.data.resize(size / sizeof(float));
    result.shape.resize(ndim);
    
    CUDA_CHECK(cudaMemcpy(result.data.data(), data, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.shape.data(), shape, ndim * sizeof(int), cudaMemcpyDeviceToHost));
    
    return result;
}

// ============================================================================
// Matrix Normal-Gamma Parameter Update Kernels
// ============================================================================

__global__ void update_mng_params_kernel(
    float* inv_v,
    float* eta_2,
    float* eta_3,
    float* eta_4,
    const float* prior_inv_v,
    const float* prior_eta_2,
    const float* prior_eta_3,
    const float* prior_eta_4,
    const float* sum_xx,
    const float* sum_yx,
    const float* sum_yy,
    const float* count,
    int batch_size,
    int x_dim,
    int y_dim,
    float lr,
    float beta) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int x_sq = x_dim * x_dim;
    int y_x = y_dim * x_dim;
    
    // Update inv_v: inv_v = (1-lr)*inv_v + lr*((1-beta)*prior_inv_v + beta*inv_v + sum_xx)
    for (int i = 0; i < x_sq; ++i) {
        int idx = batch_idx * x_sq + i;
        float prior_term = lr * ((1.0f - beta) * prior_inv_v[idx] + sum_xx[idx]);
        float current_term = (1.0f - lr * (1.0f - beta)) * inv_v[idx];
        inv_v[idx] = current_term + prior_term;
    }
    
    // Update eta_2
    for (int i = 0; i < y_x; ++i) {
        int idx = batch_idx * y_x + i;
        float prior_term = lr * ((1.0f - beta) * prior_eta_2[idx] + sum_yx[idx]);
        float current_term = (1.0f - lr * (1.0f - beta)) * eta_2[idx];
        eta_2[idx] = current_term + prior_term;
    }
    
    // Update eta_3
    for (int i = 0; i < y_dim; ++i) {
        int idx = batch_idx * y_dim + i;
        float prior_term = lr * ((1.0f - beta) * prior_eta_3[idx] + sum_yy[idx]);
        float current_term = (1.0f - lr * (1.0f - beta)) * eta_3[idx];
        eta_3[idx] = current_term + prior_term;
    }
    
    // Update eta_4
    for (int i = 0; i < y_dim; ++i) {
        int idx = batch_idx * y_dim + i;
        float prior_term = lr * ((1.0f - beta) * prior_eta_4[idx] + count[batch_idx]);
        float current_term = (1.0f - lr * (1.0f - beta)) * eta_4[idx];
        eta_4[idx] = current_term + prior_term;
    }
}

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
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    update_mng_params_kernel<<<blocks, threads, 0, stream>>>(
        inv_v.data, eta_2.data, eta_3.data, eta_4.data,
        prior_inv_v.data, prior_eta_2.data, prior_eta_3.data, prior_eta_4.data,
        sum_xx.data, sum_yx.data, sum_yy.data, count.data,
        batch_size, x_dim, y_dim, lr, beta);
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Forward/Backward Message Passing Kernels
// ============================================================================

__global__ void forward_from_normal_kernel(
    float* out_inv_sigma,
    float* out_inv_sigma_mu,
    float* out_residual,
    const float* A_inv_sigma,
    const float* A_inv_sigma_x,
    const float* D,
    const float* C_x,
    const float* expected_logdet,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int y_sq = y_dim * y_dim;
    int y_x = y_dim * x_dim;
    int x_sq = x_dim * x_dim;
    
    // Compute Schur complement: S = A - B @ inv(D) @ B^T
    // For now, simplified version assuming D is already inverted
    
    // Copy A to output as starting point
    for (int i = 0; i < y_sq; ++i) {
        out_inv_sigma[batch_idx * y_sq + i] = A_inv_sigma[batch_idx * y_sq + i];
    }
    
    // Subtract B @ invD @ B^T (simplified - would need proper matmul)
    // This is a placeholder for the full Schur computation
    
    // Compute inv_sigma_mu = B @ invD @ C_x
    for (int i = 0; i < y_dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < x_dim; ++j) {
            sum += A_inv_sigma_x[batch_idx * y_x + i * x_dim + j] * C_x[batch_idx * x_dim + j];
        }
        out_inv_sigma_mu[batch_idx * y_dim + i] = sum;
    }
    
    // Compute residual
    // residual = -log_partition(pX) + 0.5 * expected_logdet + 0.5 * C_x^T @ invD @ C_x - 0.5 * log|D|
    float residual = 0.5f * expected_logdet[batch_idx];
    out_residual[batch_idx] = residual;
}

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
    bool use_bias) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int x_sq = x_dim * x_dim;
    int y_x = y_dim * x_dim;
    
    // Compute Schur complement for backward: D - B^T @ inv(A) @ B
    // Simplified computation
    
    // Copy D to output (assuming D passed in A parameter for this kernel)
    for (int i = 0; i < x_sq; ++i) {
        out_inv_sigma[batch_idx * x_sq + i] = A[batch_idx * x_sq + i];
    }
    
    // Compute inv_sigma_mu = B^T @ invA @ C_y
    for (int i = 0; i < x_dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < y_dim; ++j) {
            // B is y_dim x x_dim, so B^T is x_dim x y_dim
            sum += B[batch_idx * y_x + j * x_dim + i] * C_y[batch_idx * y_dim + j];
        }
        out_inv_sigma_mu[batch_idx * x_dim + i] = sum;
    }
    
    // Compute residual
    float residual = 0.5f * logdetA;
    out_residual[batch_idx] = residual;
}

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
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    forward_from_normal_kernel<<<blocks, threads, 0, stream>>>(
        out_inv_sigma.data, out_inv_sigma_mu.data, out_residual.data,
        A_inv_sigma.data, A_inv_sigma_x.data, D.data, C_x.data, expected_logdet.data,
        batch_size, x_dim, y_dim, use_bias);
    
    CUDA_CHECK(cudaGetLastError());
}

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
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    // Pack logdetA into a device array
    float* d_logdetA;
    CUDA_CHECK(cudaMalloc(&d_logdetA, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_logdetA, &logdetA, sizeof(float), cudaMemcpyHostToDevice));
    
    backward_from_normal_kernel<<<blocks, threads, 0, stream>>>(
        out_inv_sigma.data, out_inv_sigma_mu.data, out_residual.data,
        A.data, B.data, C_y.data, invA.data, logdetA,
        batch_size, x_dim, y_dim, use_bias);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_logdetA));
}

// ============================================================================
// Schur Complement Kernels
// ============================================================================

__global__ void schur_complement_kernel(
    float* S,
    const float* A,
    const float* B,
    const float* invD,
    int batch_size,
    int m,
    int n) {
    
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || row >= m || col >= m) return;
    
    int m_sq = m * m;
    int m_n = m * n;
    int n_sq = n * n;
    
    // S[i,j] = A[i,j] - sum_k sum_l B[i,k] * invD[k,l] * B[j,l]
    // For diagonal invD, this simplifies
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        for (int l = 0; l < n; ++l) {
            float b_ik = B[batch_idx * m_n + row * n + k];
            float invd_kl = invD[batch_idx * n_sq + k * n + l];
            float b_jl = B[batch_idx * m_n + col * n + l];
            sum += b_ik * invd_kl * b_jl;
        }
    }
    
    S[batch_idx * m_sq + row * m + col] = A[batch_idx * m_sq + row * m + col] - sum;
}

void launch_schur_complement(
    const DeviceArray& S,
    const DeviceArray& A,
    const DeviceArray& B,
    const DeviceArray& invD,
    int batch_size,
    int m,
    int n,
    cudaStream_t stream) {
    
    dim3 threads(16, 16);
    dim3 blocks((m + 15) / 16, (m + 15) / 16, batch_size);
    
    schur_complement_kernel<<<blocks, threads, 0, stream>>>(
        S.data, A.data, B.data, invD.data, batch_size, m, n);
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Batch Linear Prediction Kernels
// ============================================================================

__global__ void batch_linear_predict_kernel(
    float* y_out,
    const float* W,
    const float* x,
    const float* b,
    int num_samples,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias) {
    
    int sample_idx = blockIdx.z;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int y_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample_idx >= num_samples || batch_idx >= batch_size || y_idx >= y_dim) return;
    
    int y_x = y_dim * x_dim;
    
    // Compute y[sample, batch, y_idx] = sum_j W[batch, y_idx, j] * x[sample, batch, j]
    float sum = 0.0f;
    for (int j = 0; j < x_dim; ++j) {
        float w = W[batch_idx * y_x + y_idx * x_dim + j];
        float x_val = x[(sample_idx * batch_size + batch_idx) * x_dim + j];
        sum += w * x_val;
    }
    
    if (use_bias && b != nullptr) {
        sum += b[batch_idx * y_dim + y_idx];
    }
    
    y_out[(sample_idx * batch_size + batch_idx) * y_dim + y_idx] = sum;
}

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
    cudaStream_t stream) {
    
    dim3 threads(1, 256);
    dim3 blocks(1, (batch_size + 255) / 256, num_samples);
    
    batch_linear_predict_kernel<<<blocks, threads, 0, stream>>>(
        y_out.data, W.data, x.data, use_bias ? b.data : nullptr,
        num_samples, batch_size, x_dim, y_dim, use_bias);
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Expected Value Computation Kernels
// ============================================================================

__global__ void expected_inv_sigma_kernel(
    float* inv_sigma,
    const float* a,
    const float* b,
    int batch_size,
    int y_dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || y_idx >= y_dim) return;
    
    int y_sq = y_dim * y_dim;
    int idx_3d = batch_idx * y_sq + y_idx * y_dim + y_idx;
    int idx_ab = batch_idx * y_dim + y_idx;
    
    // inv_sigma[i,i] = a[i] / b[i]
    inv_sigma[idx_3d] = a[idx_ab] / (b[idx_ab] + 1e-8f);
}

__global__ void expected_logdet_sigma_kernel(
    float* logdet,
    const float* a,
    const float* b,
    int batch_size,
    int y_dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    float sum = 0.0f;
    for (int i = 0; i < y_dim; ++i) {
        int idx = batch_idx * y_dim + i;
        // digamma(a) - log(b)
        float a_val = a[idx];
        float b_val = b[idx] + 1e-8f;
        sum += lgammaf(a_val) - lgammaf(a_val - 1.0f) - logf(b_val);
    }
    
    logdet[batch_idx] = sum;
}

void launch_expected_inv_sigma(
    const DeviceArray& inv_sigma,
    const DeviceArray& a,
    const DeviceArray& b,
    int batch_size,
    int y_dim,
    cudaStream_t stream) {
    
    dim3 threads(16, 16);
    dim3 blocks((batch_size + 15) / 16, (y_dim + 15) / 16);
    
    expected_inv_sigma_kernel<<<blocks, threads, 0, stream>>>(
        inv_sigma.data, a.data, b.data, batch_size, y_dim);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_expected_logdet_sigma(
    const DeviceArray& logdet,
    const DeviceArray& a,
    const DeviceArray& b,
    int batch_size,
    int y_dim,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    expected_logdet_sigma_kernel<<<blocks, threads, 0, stream>>>(
        logdet.data, a.data, b.data, batch_size, y_dim);
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Sufficient Statistics Computation Kernels
// ============================================================================

__global__ void compute_statistics_kernel(
    float* sum_xx,
    float* sum_yx,
    float* sum_yy,
    float* count,
    const float* X,
    const float* Y,
    const float* weights,
    int num_samples,
    int batch_size,
    int x_dim,
    int y_dim,
    bool use_bias) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int x_sq = x_dim * x_dim;
    int y_x = y_dim * x_dim;
    
    // Initialize accumulators
    for (int i = 0; i < x_sq; ++i) sum_xx[batch_idx * x_sq + i] = 0.0f;
    for (int i = 0; i < y_x; ++i) sum_yx[batch_idx * y_x + i] = 0.0f;
    for (int i = 0; i < y_dim; ++i) sum_yy[batch_idx * y_dim + i] = 0.0f;
    count[batch_idx] = 0.0f;
    
    // Accumulate over samples
    for (int s = 0; s < num_samples; ++s) {
        float w = (weights != nullptr) ? weights[s * batch_size + batch_idx] : 1.0f;
        
        // Update sum_xx
        for (int i = 0; i < x_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                float x_i = X[(s * batch_size + batch_idx) * x_dim + i];
                float x_j = X[(s * batch_size + batch_idx) * x_dim + j];
                sum_xx[batch_idx * x_sq + i * x_dim + j] += w * x_i * x_j;
            }
        }
        
        // Update sum_yx
        for (int i = 0; i < y_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                float y_i = Y[(s * batch_size + batch_idx) * y_dim + i];
                float x_j = X[(s * batch_size + batch_idx) * x_dim + j];
                sum_yx[batch_idx * y_x + i * x_dim + j] += w * y_i * x_j;
            }
        }
        
        // Update sum_yy (diagonal only for Matrix Normal-Gamma)
        for (int i = 0; i < y_dim; ++i) {
            float y_i = Y[(s * batch_size + batch_idx) * y_dim + i];
            sum_yy[batch_idx * y_dim + i] += w * y_i * y_i;
        }
        
        count[batch_idx] += w;
    }
}

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
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    compute_statistics_kernel<<<blocks, threads, 0, stream>>>(
        sum_xx.data, sum_yx.data, sum_yy.data, count.data,
        X.data, Y.data, weights ? weights->data : nullptr,
        num_samples, batch_size, x_dim, y_dim, use_bias);
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// cuBLAS/cuSOLVER Wrappers
// ============================================================================

static cublasHandle_t cublas_handle = nullptr;
static cusolverDnHandle_t cusolver_handle = nullptr;

void initialize_cuda_context() {
    if (!cublas_handle) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
    }
    if (!cusolver_handle) {
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    }
}

void cleanup_cuda_context() {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
    if (cusolver_handle) {
        cusolverDnDestroy(cusolver_handle);
        cusolver_handle = nullptr;
    }
}

cublasHandle_t get_cublas_handle() {
    if (!cublas_handle) {
        initialize_cuda_context();
    }
    return cublas_handle;
}

cusolverDnHandle_t get_cusolver_handle() {
    if (!cusolver_handle) {
        initialize_cuda_context();
    }
    return cusolver_handle;
}

void cublas_batch_matmul(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int batch_size,
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    bool transA,
    bool transB) {
    
    cublasOperation_t trans_a = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t trans_b = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    int lda = transA ? m : k;
    int ldb = transB ? k : n;
    int ldc = n;
    
    // Use batched GEMM for efficiency
    std::vector<const float*> A_array(batch_size);
    std::vector<const float*> B_array(batch_size);
    std::vector<float*> C_array(batch_size);
    
    int strideA = m * k;
    int strideB = k * n;
    int strideC = m * n;
    
    for (int i = 0; i < batch_size; ++i) {
        A_array[i] = A + i * strideA;
        B_array[i] = B + i * strideB;
        C_array[i] = C + i * strideC;
    }
    
    float** d_A_array;
    float** d_B_array;
    float** d_C_array;
    
    CUDA_CHECK(cudaMalloc(&d_A_array, batch_size * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_B_array, batch_size * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_C_array, batch_size * sizeof(float*)));
    
    CUDA_CHECK(cudaMemcpy(d_A_array, A_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_array, B_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_array, C_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    
    CUBLAS_CHECK(cublasSgemmBatched(
        handle,
        trans_b, trans_a,
        n, m, k,
        &alpha,
        d_B_array, ldb,
        d_A_array, lda,
        &beta,
        d_C_array, ldc,
        batch_size));
    
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
}

void cusolver_batch_cholesky(
    cusolverDnHandle_t handle,
    float* A,
    int batch_size,
    int n,
    int* info) {
    
    int lda = n;
    
    // Query workspace size
    int Lwork;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(
        handle, CUBLAS_FILL_MODE_LOWER, n, A, lda, &Lwork));
    
    float* work;
    CUDA_CHECK(cudaMalloc(&work, Lwork * sizeof(float)));
    
    // Perform Cholesky factorization for each batch element
    for (int i = 0; i < batch_size; ++i) {
        CUSOLVER_CHECK(cusolverDnSpotrf(
            handle, CUBLAS_FILL_MODE_LOWER, n,
            A + i * n * n, lda, work, Lwork, info + i));
    }
    
    CUDA_CHECK(cudaFree(work));
}

void cusolver_batch_cholesky_solve(
    cusolverDnHandle_t handle,
    const float* L,
    const float* B,
    float* X,
    int batch_size,
    int n,
    int nrhs) {
    
    // Copy B to X (solution will be stored in X)
    CUDA_CHECK(cudaMemcpy(X, B, batch_size * n * nrhs * sizeof(float), cudaMemcpyDeviceToDevice));
    
    int lda = n;
    int ldb = n;
    
    // Solve L * L^T * X = B using forward and backward substitution
    for (int i = 0; i < batch_size; ++i) {
        CUSOLVER_CHECK(cusolverDnSpotrs(
            handle, CUBLAS_FILL_MODE_LOWER, n, nrhs,
            L + i * n * n, lda,
            X + i * n * nrhs, ldb));
    }
}

void cusolver_batch_cholesky_inverse(
    cusolverDnHandle_t handle,
    float* A,
    float* work,
    int batch_size,
    int n,
    float* logdet,
    int* info) {
    
    // First compute Cholesky factorization
    cusolver_batch_cholesky(handle, A, batch_size, n, info);
    
    int lda = n;
    
    // Compute log determinant from diagonal of L
    for (int b = 0; b < batch_size; ++b) {
        float* L = A + b * n * n;
        float logdet_val = 0.0f;
        for (int i = 0; i < n; ++i) {
            logdet_val += 2.0f * logf(L[i * n + i] + 1e-8f);
        }
        logdet[b] = logdet_val;
    }
    
    // Compute inverse using the Cholesky factor
    for (int i = 0; i < batch_size; ++i) {
        CUSOLVER_CHECK(cusolverDnSpotri(
            handle, CUBLAS_FILL_MODE_LOWER, n,
            A + i * n * n, lda, info + i));
    }
}

} // namespace cuda
} // namespace transforms
} // namespace axiom

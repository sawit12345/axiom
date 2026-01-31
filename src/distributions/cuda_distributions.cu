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

#include "cuda_distributions.cuh"
#include <math.h>
#include <stdio.h>

namespace axiom {
namespace cuda {

// ============================================================================
// Device Helper Functions
// ============================================================================

template<typename T>
__device__ T digamma_device(T x) {
    // Digamma approximation using series expansion
    const T psi_1 = T(-0.5772156649015328606065121);  // -Euler-Mascheroni constant
    
    if (x < T(1.0)) {
        // Reflection formula
        return psi_1 - T(1.0) / x + T(1.0);
    }
    
    T result = T(0.0);
    while (x < T(6.0)) {
        result -= T(1.0) / x;
        x += T(1.0);
    }
    
    // Asymptotic expansion: ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - ...
    T inv_x = T(1.0) / x;
    T inv_x2 = inv_x * inv_x;
    T inv_x4 = inv_x2 * inv_x2;
    
    result += log(x) - T(0.5) * inv_x - T(1.0/12.0) * inv_x2 + 
              T(1.0/120.0) * inv_x4 - T(1.0/252.0) * inv_x4 * inv_x2;
    
    return result;
}

template<typename T>
__device__ T gammaln_device(T x) {
    // Lanczos approximation for log-gamma
    const T g = T(7.0);
    const T coeffs[] = {
        T(0.99999999999980993),
        T(676.5203681218851),
        T(-1259.1392167224028),
        T(771.32342877765313),
        T(-176.61502916214059),
        T(12.507343278686905),
        T(-0.13857109526572012),
        T(9.9843695780195716e-6),
        T(1.5056327351493116e-7)
    };
    
    if (x < T(0.5)) {
        // Reflection formula
        return log(M_PI) - log(sin(M_PI * x)) - gammaln_device(T(1.0) - x);
    }
    
    x -= T(1.0);
    T a = coeffs[0];
    for (int i = 1; i < 9; i++) {
        a += coeffs[i] / (x + T(i));
    }
    
    T t = x + g + T(0.5);
    return log(sqrt(2.0 * M_PI)) + log(a) - t + log(t) * (x + T(0.5));
}

template<typename T>
__device__ T mvdigamma_device(T x, int d) {
    // Multivariate digamma: Σ_{j=1}^d ψ(x + (1-j)/2)
    T result = T(0.0);
    for (int j = 0; j < d; j++) {
        result += digamma_device(x - T(j) / T(2.0));
    }
    return result;
}

template<typename T>
__device__ T mvgammaln_device(T x, int d) {
    // Multivariate log-gamma: π^(d(d-1)/4) Π_{j=1}^d Γ(x + (1-j)/2)
    T result = T(d * (d - 1) / 4.0) * log(M_PI);
    for (int j = 0; j < d; j++) {
        result += gammaln_device(x - T(j) / T(2.0));
    }
    return result;
}

// ============================================================================
// Batch MVN Log-Likelihood
// ============================================================================

template<typename T>
__global__ void mahalanobis_distance_kernel(
    const T* d_diff,
    const T* d_inv_sigma,
    T* d_result,
    int batch_size,
    int n_samples,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || sample_idx >= n_samples) return;
    
    // Compute: diffᵀ * inv_sigma * diff for this sample
    int diff_offset = (batch_idx * n_samples + sample_idx) * dim;
    int inv_sigma_offset = batch_idx * dim * dim;
    
    T result = T(0.0);
    for (int i = 0; i < dim; i++) {
        T temp = T(0.0);
        for (int j = 0; j < dim; j++) {
            temp += d_inv_sigma[inv_sigma_offset + i * dim + j] * 
                    d_diff[diff_offset + j];
        }
        result += temp * d_diff[diff_offset + i];
    }
    
    d_result[batch_idx * n_samples + sample_idx] = result;
}

template<typename T>
__global__ void batch_mvn_log_likelihood_kernel(
    const T* d_x,
    const T* d_mu,
    const T* d_inv_sigma,
    const T* d_logdet_inv_sigma,
    T* d_result,
    int batch_size,
    int n_samples,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || sample_idx >= n_samples) return;
    
    // Compute diff = x - μ
    extern __shared__ T shared_diff[];
    int diff_offset = (batch_idx * n_samples + sample_idx) * dim;
    
    // Load diff into shared memory
    for (int i = threadIdx.z; i < dim; i += blockDim.z) {
        int mu_offset = batch_idx * dim + i;
        shared_diff[threadIdx.y * dim + i] = d_x[diff_offset + i] - d_mu[mu_offset];
    }
    __syncthreads();
    
    // Compute Mahalanobis distance: diffᵀ * inv_sigma * diff
    int inv_sigma_offset = batch_idx * dim * dim;
    T mahal = T(0.0);
    
    for (int i = 0; i < dim; i++) {
        T temp = T(0.0);
        for (int j = 0; j < dim; j++) {
            temp += d_inv_sigma[inv_sigma_offset + i * dim + j] * 
                    shared_diff[threadIdx.y * dim + j];
        }
        mahal += temp * shared_diff[threadIdx.y * dim + i];
    }
    
    // Compute log-likelihood: -0.5 * mahal + 0.5 * logdet_inv_sigma - 0.5 * dim * log(2π)
    T log2pi = T(1.8378770664093453);  // log(2π)
    int logdet_offset = batch_idx;
    
    T log_likelihood = T(-0.5) * mahal + 
                       T(0.5) * d_logdet_inv_sigma[logdet_offset] -
                       T(0.5) * dim * log2pi;
    
    d_result[batch_idx * n_samples + sample_idx] = log_likelihood;
}

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
    cudaStream_t stream) {
    
    dim3 block(8, 8, 4);
    dim3 grid(
        (batch_size + block.x - 1) / block.x,
        (n_samples + block.y - 1) / block.y
    );
    
    size_t shared_mem_size = block.y * dim * sizeof(T);
    
    batch_mvn_log_likelihood_kernel<<<grid, block, shared_mem_size, stream>>>(
        d_x, d_mu, d_inv_sigma, d_logdet_inv_sigma,
        d_result, batch_size, n_samples, dim);
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Multinomial Operations
// ============================================================================

template<typename T>
__global__ void stable_softmax_kernel(
    const T* d_logits,
    T* d_probs,
    int batch_size,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * dim;
    
    // Find max for numerical stability
    T max_val = d_logits[offset];
    for (int i = 1; i < dim; i++) {
        max_val = fmax(max_val, d_logits[offset + i]);
    }
    
    // Compute exp(x - max) and sum
    T sum = T(0.0);
    for (int i = 0; i < dim; i++) {
        T exp_val = exp(d_logits[offset + i] - max_val);
        d_probs[offset + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    T inv_sum = T(1.0) / sum;
    for (int i = 0; i < dim; i++) {
        d_probs[offset + i] *= inv_sum;
    }
}

template<typename T>
__global__ void stable_logsumexp_kernel(
    const T* d_logits,
    T* d_result,
    int batch_size,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * dim;
    
    // Find max
    T max_val = d_logits[offset];
    for (int i = 1; i < dim; i++) {
        max_val = fmax(max_val, d_logits[offset + i]);
    }
    
    // Compute log(sum(exp(x - max))) + max
    T sum = T(0.0);
    for (int i = 0; i < dim; i++) {
        sum += exp(d_logits[offset + i] - max_val);
    }
    
    d_result[batch_idx] = log(sum) + max_val;
}

template<typename T>
__global__ void multinomial_log_prob_kernel(
    const T* d_x,
    const T* d_logits,
    const T* d_logZ,
    T* d_result,
    int batch_size,
    int n_samples,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || sample_idx >= n_samples) return;
    
    int x_offset = (batch_idx * n_samples + sample_idx) * dim;
    int logits_offset = batch_idx * dim;
    
    T log_prob = T(0.0);
    for (int i = 0; i < dim; i++) {
        log_prob += d_x[x_offset + i] * (d_logits[logits_offset + i] - d_logZ[batch_idx]);
    }
    
    d_result[batch_idx * n_samples + sample_idx] = log_prob;
}

template<typename T>
__global__ void multinomial_sample_gumbel_kernel(
    const T* d_logits,
    const T* d_uniform,
    T* d_samples,
    int batch_size,
    int n_samples,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || sample_idx >= n_samples) return;
    
    int sample_offset = (batch_idx * n_samples + sample_idx) * dim;
    int logits_offset = batch_idx * dim;
    
    // Gumbel trick: argmax(logits + log(-log(uniform)))
    T max_val = T(-1e10);
    int max_idx = 0;
    
    for (int i = 0; i < dim; i++) {
        T gumbel = d_logits[logits_offset + i] + log(-log(d_uniform[sample_offset + i]));
        if (gumbel > max_val) {
            max_val = gumbel;
            max_idx = i;
        }
    }
    
    // Set one-hot
    for (int i = 0; i < dim; i++) {
        d_samples[sample_offset + i] = (i == max_idx) ? T(1.0) : T(0.0);
    }
}

// ============================================================================
// Conjugate Updates
// ============================================================================

template<typename T>
__global__ void conjugate_update_eta_kernel(
    T* d_eta,
    const T* d_eta0,
    const T* d_sum_tx,
    T lr,
    T beta,
    int batch_size,
    int dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * dim;
    
    if (idx >= total) return;
    
    int batch_idx = idx / dim;
    
    // η = (1 - lr * (1 - β)) * η + lr * (1 - β) * η₀ + lr * Σ T(x)
    T past = (T(1.0) - lr * (T(1.0) - beta)) * d_eta[idx];
    T prior = lr * (T(1.0) - beta) * d_eta0[idx];
    T update = lr * d_sum_tx[idx];
    
    d_eta[idx] = past + prior + update;
}

template<typename T>
__global__ void conjugate_update_nu_kernel(
    T* d_nu,
    const T* d_nu0,
    const T* d_N,
    T lr,
    T beta,
    int batch_size) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // ν = (1 - lr * (1 - β)) * ν + lr * (1 - β) * ν₀ + lr * N
    T past = (T(1.0) - lr * (T(1.0) - beta)) * d_nu[batch_idx];
    T prior = lr * (T(1.0) - beta) * d_nu0[batch_idx];
    T update = lr * d_N[batch_idx];
    
    d_nu[batch_idx] = past + prior + update;
}

template<typename T>
__global__ void weighted_stats_sum_kernel(
    const T* d_stats,
    const T* d_weights,
    T* d_result,
    int n_samples,
    int batch_size,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || dim_idx >= dim) return;
    
    int result_idx = batch_idx * dim + dim_idx;
    T sum = T(0.0);
    
    for (int s = 0; s < n_samples; s++) {
        int stats_idx = (s * batch_size + batch_idx) * dim + dim_idx;
        int weight_idx = s * batch_size + batch_idx;
        sum += d_stats[stats_idx] * d_weights[weight_idx];
    }
    
    d_result[result_idx] = sum;
}

// ============================================================================
// KL Divergence
// ============================================================================

template<typename T>
__global__ void mvn_conjugate_kl_kernel(
    const T* d_mean,
    const T* d_prior_mean,
    const T* d_kappa,
    const T* d_prior_kappa,
    const T* d_inv_u,
    const T* d_logdet_inv_u,
    const T* d_prior_inv_u,
    const T* d_prior_logdet_inv_u,
    const T* d_n,
    const T* d_prior_n,
    T* d_result,
    int batch_size,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // KL for mean part: 0.5 * [(κ₀/κ - 1 + log(κ/κ₀)) * d + κ₀ * (μ - μ₀)ᵀ E[Σ⁻¹] (μ - μ₀)]
    T kappa_ratio = d_prior_kappa[batch_idx] / d_kappa[batch_idx];
    T kl_mean = T(0.5) * ((kappa_ratio - T(1.0) + log(d_kappa[batch_idx] / d_prior_kappa[batch_idx])) * dim);
    
    // Compute (μ - μ₀)ᵀ E[Σ⁻¹] (μ - μ₀)
    T pred_error_sq = T(0.0);
    for (int i = 0; i < dim; i++) {
        T diff = d_mean[batch_idx * dim + i] - d_prior_mean[batch_idx * dim + i];
        T temp = T(0.0);
        for (int j = 0; j < dim; j++) {
            T diff_j = d_mean[batch_idx * dim + j] - d_prior_mean[batch_idx * dim + j];
            // E[Σ⁻¹] = n * U
            temp += d_n[batch_idx] * d_inv_u[batch_idx * dim * dim + i * dim + j] * diff_j;
        }
        pred_error_sq += diff * temp;
    }
    
    kl_mean += T(0.5) * d_prior_kappa[batch_idx] * pred_error_sq;
    
    // KL for Wishart part
    T half_dim = T(0.5) * dim;
    T kl_wishart = T(0.0);
    
    // Trace term: 0.5 * n * tr(U₀⁻¹ U)
    T trace_term = T(0.0);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            trace_term += d_prior_inv_u[batch_idx * dim * dim + i * dim + j] * 
                          d_inv_u[batch_idx * dim * dim + j * dim + i];
        }
    }
    kl_wishart += half_dim * d_n[batch_idx] * trace_term;
    
    // Log determinant terms
    kl_wishart += half_dim * d_prior_n[batch_idx] * 
                  (d_logdet_inv_u[batch_idx] - d_prior_logdet_inv_u[batch_idx]);
    
    kl_wishart -= half_dim * d_n[batch_idx] * dim;
    
    // Multivariate gamma terms
    kl_wishart += mvgammaln_device(d_prior_n[batch_idx] / T(2.0), dim);
    kl_wishart -= mvgammaln_device(d_n[batch_idx] / T(2.0), dim);
    kl_wishart += (d_n[batch_idx] - d_prior_n[batch_idx]) / T(2.0) * 
                  mvdigamma_device(d_n[batch_idx] / T(2.0), dim);
    
    d_result[batch_idx] = kl_mean + kl_wishart;
}

template<typename T>
__global__ void dirichlet_kl_kernel(
    const T* d_alpha,
    const T* d_prior_alpha,
    T* d_result,
    int batch_size,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * dim;
    
    // Sum of alphas
    T alpha_sum = T(0.0);
    T prior_alpha_sum = T(0.0);
    
    for (int i = 0; i < dim; i++) {
        alpha_sum += d_alpha[offset + i];
        prior_alpha_sum += d_prior_alpha[offset + i];
    }
    
    // KL = log Γ(α_sum) - Σ log Γ(α) - log Γ(α₀_sum) + Σ log Γ(α₀)
    //      + Σ (α - α₀) * (ψ(α) - ψ(α_sum))
    T kl = gammaln_device(alpha_sum) - gammaln_device(prior_alpha_sum);
    
    T digamma_sum = digamma_device(alpha_sum);
    
    for (int i = 0; i < dim; i++) {
        kl -= gammaln_device(d_alpha[offset + i]);
        kl += gammaln_device(d_prior_alpha[offset + i]);
        kl += (d_alpha[offset + i] - d_prior_alpha[offset + i]) * 
              (digamma_device(d_alpha[offset + i]) - digamma_sum);
    }
    
    d_result[batch_idx] = kl;
}

// ============================================================================
// Sampling
// ============================================================================

template<typename T>
__global__ void box_muller_sample_kernel(
    const T* d_uniform1,
    const T* d_uniform2,
    T* d_result,
    int batch_size,
    int n_samples,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (batch_idx >= batch_size || sample_idx >= n_samples || dim_idx >= dim) return;
    
    int idx = ((batch_idx * n_samples + sample_idx) * dim + dim_idx);
    
    // Box-Muller transform
    T u1 = d_uniform1[idx];
    T u2 = d_uniform2[idx];
    
    T r = sqrt(T(-2.0) * log(u1));
    T theta = T(2.0) * T(M_PI) * u2;
    
    d_result[idx] = r * cos(theta);
}

template<typename T>
__global__ void mvn_transform_samples_kernel(
    const T* d_z,
    const T* d_mu,
    const T* d_L,
    T* d_samples,
    int batch_size,
    int n_samples,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || sample_idx >= n_samples) return;
    
    int sample_offset = (batch_idx * n_samples + sample_idx) * dim;
    int mu_offset = batch_idx * dim;
    int L_offset = batch_idx * dim * dim;
    
    // Compute μ + L * z for this sample
    for (int i = 0; i < dim; i++) {
        T temp = T(0.0);
        for (int j = 0; j <= i; j++) {  // L is lower triangular
            temp += d_L[L_offset + i * dim + j] * d_z[sample_offset + j];
        }
        d_samples[sample_offset + i] = d_mu[mu_offset + i] + temp;
    }
}

// ============================================================================
// Utility Kernels
// ============================================================================

template<typename T>
__global__ void elementwise_add_kernel(
    const T* d_a,
    const T* d_b,
    T* d_c,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    d_c[idx] = d_a[idx] + d_b[idx];
}

template<typename T>
__global__ void elementwise_mul_kernel(
    const T* d_a,
    const T* d_b,
    T* d_c,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    d_c[idx] = d_a[idx] * d_b[idx];
}

template<typename T>
__global__ void elementwise_scale_kernel(
    const T* d_a,
    T* d_b,
    T alpha,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    d_b[idx] = alpha * d_a[idx];
}

template<typename T>
__global__ void sum_last_dim_kernel(
    const T* d_input,
    T* d_output,
    int batch_size,
    int dim) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    T sum = T(0.0);
    int offset = batch_idx * dim;
    
    for (int i = 0; i < dim; i++) {
        sum += d_input[offset + i];
    }
    
    d_output[batch_idx] = sum;
}

// ============================================================================
// Template Instantiations
// ============================================================================

template __global__ void batch_mvn_log_likelihood_kernel<float>(
    const float*, const float*, const float*, const float*,
    float*, int, int, int);

template __global__ void batch_mvn_log_likelihood_kernel<double>(
    const double*, const double*, const double*, const double*,
    double*, int, int, int);

template void batch_mvn_log_likelihood<float>(
    const float*, const float*, const float*, const float*,
    float*, int, int, int, cudaStream_t);

template void batch_mvn_log_likelihood<double>(
    const double*, const double*, const double*, const double*,
    double*, int, int, int, cudaStream_t);

template __global__ void stable_softmax_kernel<float>(const float*, float*, int, int);
template __global__ void stable_softmax_kernel<double>(const double*, double*, int, int);

template __global__ void stable_logsumexp_kernel<float>(const float*, float*, int, int);
template __global__ void stable_logsumexp_kernel<double>(const double*, double*, int, int);

template __global__ void multinomial_log_prob_kernel<float>(
    const float*, const float*, const float*, float*, int, int, int);
template __global__ void multinomial_log_prob_kernel<double>(
    const double*, const double*, const double*, double*, int, int, int);

template __global__ void conjugate_update_eta_kernel<float>(
    float*, const float*, const float*, float, float, int, int);
template __global__ void conjugate_update_eta_kernel<double>(
    double*, const double*, const double*, double, double, int, int);

template __global__ void mvn_conjugate_kl_kernel<float>(
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, float*, int, int);
template __global__ void mvn_conjugate_kl_kernel<double>(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, double*, int, int);

template __global__ void dirichlet_kl_kernel<float>(
    const float*, const float*, float*, int, int);
template __global__ void dirichlet_kl_kernel<double>(
    const double*, const double*, double*, int, int);

} // namespace cuda
} // namespace axiom

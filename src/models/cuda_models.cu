/**
 * Copyright 2025 VERSES AI, Inc.
 *
 * Licensed under the VERSES Academic Research License (the "License");
 * you may not use this file except in compliance with the license.
 */

#include "smm.h"
#include "rmm.h"
#include "tmm.h"
#include "imm.h"
#include "mixture.h"
#include "hybrid_mixture.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <math.h>

namespace axiom {
namespace models {

// ============================================================================
// SMM CUDA Kernels
// ============================================================================

__global__ void smmEStepKernel(
    const float* inputs,      // (batch, num_tokens, input_dim)
    const float* qx_mu,       // (batch, num_slots, slot_dim, 1)
    const float* qx_inv_sigma,// (batch, num_slots, slot_dim, slot_dim)
    const float* mu,          // (1, num_slots, input_dim, slot_dim+1)
    const float* a,           // (1, num_slots, input_dim, 1)
    const float* b,           // (1, num_slots, input_dim, 1)
    float* out_ell,           // (batch, num_tokens, num_slots)
    int batch_size,
    int num_tokens,
    int num_slots,
    int input_dim,
    int slot_dim,
    bool use_bias) {
    
    int x_dim = slot_dim + (use_bias ? 1 : 0);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * num_tokens * num_slots;
    
    if (idx >= total_threads) return;
    
    int b_idx = idx / (num_tokens * num_slots);
    int rem = idx % (num_tokens * num_slots);
    int n_idx = rem / num_slots;
    int k = rem % num_slots;
    
    // Compute expected log-likelihood
    float ell = 0.0f;
    
    for (int i = 0; i < input_dim; ++i) {
        float a_val = a[(k * input_dim + i)];
        float b_val = b[(k * input_dim + i)];
        float inv_sigma_ii = a_val / (b_val + 1e-10f);
        
        ell += 0.5f * logf(inv_sigma_ii + 1e-10f);
    }
    
    ell -= 0.5f * input_dim * logf(2.0f * M_PI);
    
    out_ell[(b_idx * num_tokens + n_idx) * num_slots + k] = ell;
}

__global__ void smmSoftmaxKernel(
    float* data,
    int batch_size,
    int num_tokens,
    int num_slots) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_tokens;
    
    if (idx >= total) return;
    
    int b = idx / num_tokens;
    int n = idx % num_tokens;
    
    // Find max
    float max_val = data[(b * num_tokens + n) * num_slots];
    for (int k = 1; k < num_slots; ++k) {
        max_val = fmaxf(max_val, data[(b * num_tokens + n) * num_slots + k]);
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int k = 0; k < num_slots; ++k) {
        sum_exp += expf(data[(b * num_tokens + n) * num_slots + k] - max_val);
    }
    
    // Normalize
    for (int k = 0; k < num_slots; ++k) {
        data[(b * num_tokens + n) * num_slots + k] = 
            expf(data[(b * num_tokens + n) * num_slots + k] - max_val) / (sum_exp + 1e-10f);
    }
}

__global__ void smmUpdateQxKernel(
    const float* inputs,
    const float* qz,
    float* out_qx_mu,
    float* out_qx_inv_sigma,
    int batch_size,
    int num_tokens,
    int num_slots,
    int slot_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_slots * slot_dim * slot_dim;
    
    if (idx >= total) return;
    
    // Simplified: compute weighted inverse sigma
    int b = idx / (num_slots * slot_dim * slot_dim);
    int rem = idx % (num_slots * slot_dim * slot_dim);
    int k = rem / (slot_dim * slot_dim);
    int i = (rem % (slot_dim * slot_dim)) / slot_dim;
    int j = rem % slot_dim;
    
    float inv_sigma = 0.0f;
    for (int n = 0; n < num_tokens; ++n) {
        inv_sigma += qz[(b * num_tokens + n) * num_slots + k] * ((i == j) ? 1e-6f : 0.0f);
    }
    
    out_qx_inv_sigma[(b * num_slots + k) * slot_dim * slot_dim + i * slot_dim + j] = inv_sigma;
}

// ============================================================================
// RMM CUDA Kernels
// ============================================================================

__global__ void rmmContinuousELLKernel(
    const float* c_data,
    const float* mean,
    const float* u,
    const float* n,
    float* out_ell,
    int batch_size,
    int num_components,
    int cont_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_components;
    
    if (idx >= total) return;
    
    int b = idx / num_components;
    int k = idx % num_components;
    
    float ell = 0.0f;
    
    for (int i = 0; i < cont_dim; ++i) {
        float diff = c_data[b * cont_dim + i] - mean[k * cont_dim + i];
        float u_ii = u[(k * cont_dim + i) * cont_dim + i];
        float n_val = n[k];
        float precision = n_val / (u_ii + 1e-10f);
        
        ell += -0.5f * precision * diff * diff;
        ell += 0.5f * logf(precision + 1e-10f);
    }
    
    ell -= 0.5f * cont_dim * logf(2.0f * M_PI);
    
    out_ell[b * num_components + k] = ell;
}

__global__ void rmmInteractionDetectionKernel(
    const float* data,
    const int* tracked_obj_mask,
    int object_idx,
    float r_interacting,
    int n_objects,
    int n_features,
    int* out_other_idx,
    float* out_dist_x,
    float* out_dist_y,
    int* out_is_interacting) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;
    
    float cx_i = data[object_idx * n_features + 0];
    float cy_i = data[object_idx * n_features + 1];
    float w_i = data[object_idx * n_features + 6] * r_interacting;
    float h_i = data[object_idx * n_features + 7] * r_interacting;
    
    int best_idx = -1;
    float best_overlap = -1.0f;
    float best_dist_x = 0.0f;
    float best_dist_y = 0.0f;
    
    for (int j = 1; j < n_objects; ++j) {
        if (j == object_idx) continue;
        
        float cx_j = data[j * n_features + 0];
        float cy_j = data[j * n_features + 1];
        float w_j = data[j * n_features + 6] * r_interacting;
        float h_j = data[j * n_features + 7] * r_interacting;
        
        // Simplified overlap check
        float dx = fabsf(cx_i - cx_j);
        float dy = fabsf(cy_i - cy_j);
        
        if (dx < (w_i + w_j) && dy < (h_i + h_j)) {
            float overlap = (w_i + w_j - dx) * (h_i + h_j - dy);
            
            if (overlap > best_overlap) {
                best_overlap = overlap;
                best_idx = j;
                best_dist_x = cx_i - cx_j;
                best_dist_y = cy_i - cy_j;
            }
        }
    }
    
    *out_other_idx = best_idx;
    *out_dist_x = best_dist_x;
    *out_dist_y = best_dist_y;
    *out_is_interacting = (best_idx >= 0) ? 1 : 0;
}

// ============================================================================
// TMM CUDA Kernels
// ============================================================================

__global__ void tmmForwardKernel(
    const float* transitions,
    const float* x,
    float* out_next_state,
    int K_max,
    int state_dim,
    bool use_bias) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K_max * state_dim;
    
    if (idx >= total) return;
    
    int k = idx / state_dim;
    int i = idx % state_dim;
    
    float sum = 0.0f;
    int x_dim = state_dim + (use_bias ? 1 : 0);
    
    for (int j = 0; j < state_dim; ++j) {
        sum += transitions[(k * state_dim + i) * x_dim + j] * x[j];
    }
    
    if (use_bias) {
        sum += transitions[(k * state_dim + i) * x_dim + state_dim];
    }
    
    out_next_state[k * state_dim + i] = sum;
}

__global__ void tmmGaussianLogLikelihoodKernel(
    const float* x_curr,
    const float* mu,
    float sigma_sqr,
    float* out_logprobs,
    int K_max,
    int state_dim,
    const int* used_mask) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K_max) return;
    
    if (used_mask && used_mask[idx] == 0) {
        out_logprobs[idx] = -1e20f;
        return;
    }
    
    float squared_error = 0.0f;
    for (int i = 0; i < state_dim; ++i) {
        float diff = x_curr[i] - mu[idx * state_dim + i];
        squared_error += diff * diff;
    }
    
    out_logprobs[idx] = -0.5f * squared_error / sigma_sqr 
                       - 0.5f * state_dim * logf(2.0f * M_PI * sigma_sqr);
}

// ============================================================================
// Mixture Model CUDA Kernels
// ============================================================================

__global__ void mixtureLogProbsKernel(
    const float* data,
    const float* likelihood_params,
    float* out_log_probs,
    int batch_size,
    int num_components,
    int data_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_components;
    
    if (idx >= total) return;
    
    int b = idx / num_components;
    int k = idx % num_components;
    
    float log_prob = 0.0f;
    
    for (int i = 0; i < data_dim; ++i) {
        float diff = data[b * data_dim + i] - likelihood_params[k * data_dim + i];
        log_prob += -0.5f * diff * diff;
    }
    
    out_log_probs[b * num_components + k] = log_prob;
}

__global__ void mixtureSoftmaxKernel(
    float* log_probs,
    const float* pi_alpha,
    int batch_size,
    int num_components) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Add prior log mean
    for (int k = 0; k < num_components; ++k) {
        float log_mean = logf(pi_alpha[k] + 1e-10f) - logf(num_components * 0.1f);
        log_probs[idx * num_components + k] += log_mean;
    }
    
    // Find max
    float max_val = log_probs[idx * num_components];
    for (int k = 1; k < num_components; ++k) {
        max_val = fmaxf(max_val, log_probs[idx * num_components + k]);
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int k = 0; k < num_components; ++k) {
        sum_exp += expf(log_probs[idx * num_components + k] - max_val);
    }
    
    // Normalize (in-place)
    for (int k = 0; k < num_components; ++k) {
        log_probs[idx * num_components + k] = 
            expf(log_probs[idx * num_components + k] - max_val) / (sum_exp + 1e-10f);
    }
}

// ============================================================================
// Gaussian Mixture Operations CUDA Kernels
// ============================================================================

__global__ void gaussianMixtureUpdateKernel(
    const float* data,
    const float* qz,
    float* out_mean,
    float* out_precision,
    int batch_size,
    int num_components,
    int data_dim,
    float lr) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_components * data_dim;
    
    if (idx >= total) return;
    
    int k = idx / data_dim;
    int i = idx % data_dim;
    
    float qz_sum = 0.0f;
    float data_sum = 0.0f;
    
    for (int b = 0; b < batch_size; ++b) {
        float qz_val = qz[b * num_components + k];
        qz_sum += qz_val;
        data_sum += qz_val * data[b * data_dim + i];
    }
    
    if (qz_sum > 1e-10f) {
        float new_mean = data_sum / qz_sum;
        out_mean[k * data_dim + i] = (1.0f - lr) * out_mean[k * data_dim + i] + lr * new_mean;
    }
}

// ============================================================================
// Utility Kernels
// ============================================================================

__global__ void addKernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void multiplyKernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void scaleKernel(const float* A, float scalar, float* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = A[idx] * scalar;
    }
}

__global__ void expKernel(const float* A, float* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = expf(A[idx]);
    }
}

__global__ void logKernel(const float* A, float* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = logf(A[idx] + 1e-10f);
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

extern "C" {

void launchSmmEStep(
    const float* inputs,
    const float* qx_mu,
    const float* qx_inv_sigma,
    const float* mu,
    const float* a,
    const float* b,
    float* out_ell,
    int batch_size,
    int num_tokens,
    int num_slots,
    int input_dim,
    int slot_dim,
    bool use_bias,
    cudaStream_t stream) {
    
    int total = batch_size * num_tokens * num_slots;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    smmEStepKernel<<<blocks, threads, 0, stream>>>(
        inputs, qx_mu, qx_inv_sigma, mu, a, b, out_ell,
        batch_size, num_tokens, num_slots, input_dim, slot_dim, use_bias);
}

void launchSmmSoftmax(
    float* data,
    int batch_size,
    int num_tokens,
    int num_slots,
    cudaStream_t stream) {
    
    int total = batch_size * num_tokens;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    smmSoftmaxKernel<<<blocks, threads, 0, stream>>>(
        data, batch_size, num_tokens, num_slots);
}

void launchRmmContinuousELL(
    const float* c_data,
    const float* mean,
    const float* u,
    const float* n,
    float* out_ell,
    int batch_size,
    int num_components,
    int cont_dim,
    cudaStream_t stream) {
    
    int total = batch_size * num_components;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    rmmContinuousELLKernel<<<blocks, threads, 0, stream>>>(
        c_data, mean, u, n, out_ell, batch_size, num_components, cont_dim);
}

void launchTmmForward(
    const float* transitions,
    const float* x,
    float* out_next_state,
    int K_max,
    int state_dim,
    bool use_bias,
    cudaStream_t stream) {
    
    int total = K_max * state_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    tmmForwardKernel<<<blocks, threads, 0, stream>>>(
        transitions, x, out_next_state, K_max, state_dim, use_bias);
}

void launchTmmGaussianLogLikelihood(
    const float* x_curr,
    const float* mu,
    float sigma_sqr,
    float* out_logprobs,
    int K_max,
    int state_dim,
    const int* used_mask,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (K_max + threads - 1) / threads;
    
    tmmGaussianLogLikelihoodKernel<<<blocks, threads, 0, stream>>>(
        x_curr, mu, sigma_sqr, out_logprobs, K_max, state_dim, used_mask);
}

void launchMixtureLogProbs(
    const float* data,
    const float* likelihood_params,
    float* out_log_probs,
    int batch_size,
    int num_components,
    int data_dim,
    cudaStream_t stream) {
    
    int total = batch_size * num_components;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    mixtureLogProbsKernel<<<blocks, threads, 0, stream>>>(
        data, likelihood_params, out_log_probs, batch_size, num_components, data_dim);
}

void launchMixtureSoftmax(
    float* log_probs,
    const float* pi_alpha,
    int batch_size,
    int num_components,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    mixtureSoftmaxKernel<<<blocks, threads, 0, stream>>>(
        log_probs, pi_alpha, batch_size, num_components);
}

void launchElementwiseAdd(
    const float* A,
    const float* B,
    float* C,
    int n,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    addKernel<<<blocks, threads, 0, stream>>>(A, B, C, n);
}

void launchElementwiseMultiply(
    const float* A,
    const float* B,
    float* C,
    int n,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    multiplyKernel<<<blocks, threads, 0, stream>>>(A, B, C, n);
}

void launchScale(
    const float* A,
    float scalar,
    float* B,
    int n,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    scaleKernel<<<blocks, threads, 0, stream>>>(A, scalar, B, n);
}

void launchExp(const float* A, float* B, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    expKernel<<<blocks, threads, 0, stream>>>(A, B, n);
}

void launchLog(const float* A, float* B, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    logKernel<<<blocks, threads, 0, stream>>>(A, B, n);
}

} // extern "C"

} // namespace models
} // namespace axiom

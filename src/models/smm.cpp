/**
 * Copyright 2025 VERSES AI, Inc.
 *
 * Licensed under the VERSES Academic Research License (the "License");
 * you may not use this file except in compliance with the license.
 */

#include "smm.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cmath>
#include <algorithm>

namespace axiom {
namespace models {

// LinearMNGParams implementation
void LinearMNGParams::allocate(int batch_size, int num_slots, int y_dim, int x_dim, bool use_bias) {
    int x_full_dim = x_dim + (use_bias ? 1 : 0);
    
    mu.shape = {batch_size, num_slots, y_dim, x_full_dim};
    mu.allocate(mu.shape, true);
    
    inv_v.shape = {batch_size, num_slots, x_full_dim, x_full_dim};
    inv_v.allocate(inv_v.shape, true);
    
    a.shape = {batch_size, num_slots, y_dim, 1};
    a.allocate(a.shape, true);
    
    b.shape = {batch_size, num_slots, y_dim, 1};
    b.allocate(b.shape, true);
}

void LinearMNGParams::copyToDevice() {
    mu.copyToDevice();
    inv_v.copyToDevice();
    a.copyToDevice();
    b.copyToDevice();
}

void LinearMNGParams::copyFromDevice() {
    mu.copyFromDevice();
    inv_v.copyFromDevice();
    a.copyFromDevice();
    b.copyFromDevice();
}

// SMMState implementation
void SMMState::allocate(int batch_size, int width_, int height_, int input_dim_, 
                        int slot_dim_, int num_slots_, bool use_bias_) {
    width = width_;
    height = height_;
    input_dim = input_dim_;
    slot_dim = slot_dim_;
    num_slots = num_slots_;
    use_bias = use_bias_;
    
    int num_tokens = width * height;
    int x_dim = slot_dim + (use_bias ? 1 : 0);
    
    prior_params.allocate(batch_size, num_slots, input_dim, slot_dim, use_bias);
    posterior_params.allocate(batch_size, num_slots, input_dim, slot_dim, use_bias);
    
    qx_mu.shape = {batch_size, num_slots, slot_dim, 1};
    qx_mu.allocate(qx_mu.shape, true);
    
    qx_inv_sigma.shape = {batch_size, num_slots, slot_dim, slot_dim};
    qx_inv_sigma.allocate(qx_inv_sigma.shape, true);
    
    qz.shape = {batch_size, num_tokens, num_slots};
    qz.allocate(qz.shape, true);
    
    pi_alpha.shape = {num_slots};
    pi_alpha.allocate(pi_alpha.shape, true);
    
    slot_mask.shape = {num_slots, input_dim, x_dim};
    slot_mask.allocate(slot_mask.shape, true);
    
    used_mask.shape = {num_slots};
    used_mask.allocate(used_mask.shape, true);
    
    dirty_mask.shape = {num_slots};
    dirty_mask.allocate(dirty_mask.shape, true);
    
    position_grid.shape = {height, width, 2};
    position_grid.allocate(position_grid.shape, true);
}

void SMMState::copyToDevice() {
    prior_params.copyToDevice();
    posterior_params.copyToDevice();
    qx_mu.copyToDevice();
    qx_inv_sigma.copyToDevice();
    qz.copyToDevice();
    pi_alpha.copyToDevice();
    slot_mask.copyToDevice();
    used_mask.copyToDevice();
    dirty_mask.copyToDevice();
    position_grid.copyToDevice();
}

void SMMState::copyFromDevice() {
    prior_params.copyFromDevice();
    posterior_params.copyFromDevice();
    qx_mu.copyFromDevice();
    qx_inv_sigma.copyFromDevice();
    qz.copyFromDevice();
    pi_alpha.copyFromDevice();
    slot_mask.copyFromDevice();
    used_mask.copyFromDevice();
    dirty_mask.copyFromDevice();
    position_grid.copyFromDevice();
}

// SlotMixtureModel implementation
SlotMixtureModel::SlotMixtureModel(const SMMState& state) : state_(state) {
    cudaStreamCreate(&stream_);
    
    // Allocate internal buffers
    int num_tokens = state_.width * state_.height;
    ell_buffer_.shape = {1, num_tokens, state_.num_slots};
    ell_buffer_.allocate(ell_buffer_.shape, true);
    
    temp_buffer_.shape = {1, num_tokens, state_.num_slots, state_.input_dim, state_.slot_dim + (state_.use_bias ? 1 : 0)};
    temp_buffer_.allocate(temp_buffer_.shape, true);
}

SlotMixtureModel::~SlotMixtureModel() {
    cudaStreamDestroy(stream_);
}

void SlotMixtureModel::eStep(const Tensor& inputs, int num_iterations) {
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Compute expected log-likelihood
        computeExpectedLogLikelihood(inputs, state_.qx_mu, state_.qx_inv_sigma, ell_buffer_);
        
        // Update qz
        updateQz(ell_buffer_, state_.qz);
        
        // Update qx given qz
        updateQx(inputs, state_.qz, state_.qx_mu, state_.qx_inv_sigma);
    }
}

void SlotMixtureModel::mStep(const Tensor& inputs,
                             const Tensor& qx_mu,
                             const Tensor& qx_inv_sigma,
                             const Tensor& qz,
                             float lr,
                             float beta,
                             const Tensor* grow_mask) {
    updateParameters(inputs, qx_mu, qx_inv_sigma, qz, lr, beta);
    
    if (grow_mask != nullptr) {
        // Apply grow mask adjustments to posterior parameters
        // This increases confidence for growing clusters
        for (int i = 0; i < state_.num_slots; ++i) {
            if (grow_mask->data[i] > 0.5f) {
                // Increase position precision for growing slot
                int idx = i * state_.posterior_params.inv_v.shape[2] * state_.posterior_params.inv_v.shape[3] + 2 * state_.posterior_params.inv_v.shape[3] + 2;
                state_.posterior_params.inv_v.data[idx] = 1000.0f;
            }
        }
    }
}

void SlotMixtureModel::emStep(const Tensor& inputs, float lr, float beta) {
    // E-step
    eStep(inputs, state_.num_e_steps);
    
    // M-step
    mStep(inputs, state_.qx_mu, state_.qx_inv_sigma, state_.qz, lr, beta);
}

void SlotMixtureModel::initializeFromData(const Tensor& inputs) {
    // Force all data to first slot
    int num_tokens = inputs.shape[1];
    
    // Set qz to one-hot for first slot
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < state_.num_slots; ++j) {
            state_.qz.data[i * state_.num_slots + j] = (j == 0) ? 1.0f : 0.0f;
        }
    }
    state_.qz.copyToDevice();
    
    // Run EM step
    emStep(inputs, 1.0f, 0.0f);
}

void SlotMixtureModel::variationalForward(const Tensor& qx_mu,
                                         const Tensor& qx_inv_sigma,
                                         Tensor& out_py_mu,
                                         Tensor& out_py_inv_sigma) {
    // Compute forward prediction: p(y|x) with variational approximation
    // E[inv_sigma] = a / b (diagonal)
    // E[inv_sigma_x] = E[inv_sigma] @ mu
    
    int batch_size = qx_mu.shape[0];
    int num_slots = state_.num_slots;
    int y_dim = state_.input_dim;
    int x_dim = state_.slot_dim + (state_.use_bias ? 1 : 0);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < num_slots; ++k) {
            // Compute expected inverse sigma (diagonal Gamma)
            for (int i = 0; i < y_dim; ++i) {
                float a_val = state_.posterior_params.a.data[b * num_slots * y_dim + k * y_dim + i];
                float b_val = state_.posterior_params.b.data[b * num_slots * y_dim + k * y_dim + i];
                float inv_sigma_ii = a_val / b_val;
                
                // Set output inverse sigma (diagonal)
                out_py_inv_sigma.data[(b * num_slots + k) * y_dim * y_dim + i * y_dim + i] = inv_sigma_ii;
            }
            
            // Compute mean prediction: mu @ E[x]
            for (int i = 0; i < y_dim; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < state_.slot_dim; ++j) {
                    float mu_ij = state_.posterior_params.mu.data[(b * num_slots + k) * y_dim * x_dim + i * x_dim + j];
                    float x_j = qx_mu.data[(b * num_slots + k) * state_.slot_dim + j];
                    sum += mu_ij * x_j;
                }
                if (state_.use_bias) {
                    float bias = state_.posterior_params.mu.data[(b * num_slots + k) * y_dim * x_dim + i * x_dim + state_.slot_dim];
                    sum += bias;
                }
                out_py_mu.data[(b * num_slots + k) * y_dim + i] = sum;
            }
        }
    }
}

void SlotMixtureModel::variationalBackward(const Tensor& inputs,
                                          const Tensor& qz,
                                          Tensor& out_qx_mu,
                                          Tensor& out_qx_inv_sigma) {
    // Compute backward message for updating q(x)
    int batch_size = inputs.shape[0];
    int num_tokens = inputs.shape[1];
    
    // Compute weighted sum of backward messages
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_slots; ++k) {
            float qz_sum = 0.0f;
            
            // Sum qz over tokens for this slot
            for (int n = 0; n < num_tokens; ++n) {
                qz_sum += qz.data[(b * num_tokens + n) * state_.num_slots + k];
            }
            
            // Update qx inverse sigma
            for (int i = 0; i < state_.slot_dim; ++i) {
                for (int j = 0; j < state_.slot_dim; ++j) {
                    float inv_sigma_kij = 0.0f;
                    for (int n = 0; n < num_tokens; ++n) {
                        float qz_nk = qz.data[(b * num_tokens + n) * state_.num_slots + k];
                        // Simplified: use expected sufficient statistics
                        inv_sigma_kij += qz_nk * (i == j ? 1.0f : 0.0f);
                    }
                    out_qx_inv_sigma.data[(b * num_slots + k) * state_.slot_dim * state_.slot_dim + i * state_.slot_dim + j] = inv_sigma_kij;
                }
            }
        }
    }
}

bool SlotMixtureModel::growModel(const Tensor& inputs,
                                const Tensor& ell_max,
                                float threshold) {
    // Find if any data point has ELL below threshold
    int num_tokens = ell_max.shape[1];
    bool needs_growth = false;
    
    for (int i = 0; i < num_tokens; ++i) {
        if (ell_max.data[i] < threshold) {
            needs_growth = true;
            break;
        }
    }
    
    if (!needs_growth) {
        return false;
    }
    
    // Find first unused slot
    int unused_slot = -1;
    for (int k = 0; k < state_.num_slots; ++k) {
        if (state_.used_mask.data[k] < 0.5f) {
            unused_slot = k;
            break;
        }
    }
    
    if (unused_slot < 0) {
        return false; // No slots available
    }
    
    // Find data point with minimum ELL
    int min_idx = 0;
    float min_ell = ell_max.data[0];
    for (int i = 1; i < num_tokens; ++i) {
        if (ell_max.data[i] < min_ell) {
            min_ell = ell_max.data[i];
            min_idx = i;
        }
    }
    
    // Force assign this data point to unused slot
    for (int k = 0; k < state_.num_slots; ++k) {
        state_.qz.data[min_idx * state_.num_slots + k] = (k == unused_slot) ? 1.0f : 0.0f;
    }
    
    // Update parameters with grow mask
    Tensor grow_mask;
    grow_mask.shape = {state_.num_slots};
    grow_mask.allocate(grow_mask.shape, true);
    for (int k = 0; k < state_.num_slots; ++k) {
        grow_mask.data[k] = (k == unused_slot) ? 1.0f : 0.0f;
    }
    grow_mask.copyToDevice();
    
    mStep(inputs, state_.qx_mu, state_.qx_inv_sigma, state_.qz, state_.learning_rate, state_.beta, &grow_mask);
    
    // Mark slot as used
    state_.used_mask.data[unused_slot] = 1.0f;
    state_.used_mask.copyToDevice();
    
    return true;
}

float SlotMixtureModel::computeELBO(const Tensor& inputs,
                                   const Tensor& qx_mu,
                                   const Tensor& qx_inv_sigma,
                                   const Tensor& qz) {
    // Compute ELBO = E_q[log p(y|x,z)] - KL(q||p)
    float elbo = 0.0f;
    
    int batch_size = inputs.shape[0];
    int num_tokens = inputs.shape[1];
    
    // Expected log-likelihood term
    for (int b = 0; b < batch_size; ++b) {
        for (int n = 0; n < num_tokens; ++n) {
            for (int k = 0; k < state_.num_slots; ++k) {
                float qz_nk = qz.data[(b * num_tokens + n) * state_.num_slots + k];
                if (qz_nk > 1e-10f) {
                    // Simplified ELL computation
                    elbo += qz_nk * (-0.5f); // Placeholder for actual likelihood
                }
            }
        }
    }
    
    // KL divergence terms (simplified)
    // KL(q(x)||p(x)) + sum_k E[q(z_k)] KL(q(theta_k)||p(theta_k))
    
    return elbo;
}

Tensor SlotMixtureModel::getUsedMask() const {
    return state_.used_mask;
}

Tensor SlotMixtureModel::getAssignments() const {
    return state_.qz;
}

void SlotMixtureModel::setSlotMask(const Tensor& mask, const std::vector<float>& probs) {
    // Set slot mask templates and their probabilities
    state_.slot_mask = mask;
    state_.slot_mask.copyToDevice();
}

void SlotMixtureModel::computeExpectedLogLikelihood(const Tensor& inputs,
                                                   const Tensor& qx_mu,
                                                   const Tensor& qx_inv_sigma,
                                                   Tensor& out_ell) {
    // E-step: Compute E_q(x)[log p(y|x, theta)]
    int batch_size = inputs.shape[0];
    int num_tokens = inputs.shape[1];
    int num_slots = state_.num_slots;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int n = 0; n < num_tokens; ++n) {
            for (int k = 0; k < num_slots; ++k) {
                float ell = 0.0f;
                
                // Expected log-likelihood under variational distribution
                // -0.5 * (y - mu_x)^T E[inv_sigma] (y - mu_x) + 0.5 * E[log|inv_sigma|]
                
                for (int i = 0; i < state_.input_dim; ++i) {
                    float a_val = state_.posterior_params.a.data[k * state_.input_dim + i];
                    float b_val = state_.posterior_params.b.data[k * state_.input_dim + i];
                    float inv_sigma_ii = a_val / std::max(b_val, 1e-10f);
                    
                    // Simplified: just use inverse variance
                    ell += 0.5f * std::log(std::max(inv_sigma_ii, 1e-10f));
                }
                
                out_ell.data[(b * num_tokens + n) * num_slots + k] = ell;
            }
        }
    }
    out_ell.copyToDevice();
}

void SlotMixtureModel::updateQx(const Tensor& inputs,
                               const Tensor& qz,
                               Tensor& out_qx_mu,
                               Tensor& out_qx_inv_sigma) {
    // Update q(x) given q(z) using backward message
    int batch_size = inputs.shape[0];
    int num_tokens = inputs.shape[1];
    
    // Weighted sum of backward messages
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_slots; ++k) {
            // Compute weighted inverse sigma
            for (int i = 0; i < state_.slot_dim; ++i) {
                for (int j = 0; j < state_.slot_dim; ++j) {
                    float inv_sigma_ij = 0.0f;
                    for (int n = 0; n < num_tokens; ++n) {
                        float qz_nk = qz.data[(b * num_tokens + n) * state_.num_slots + k];
                        inv_sigma_ij += qz_nk * (i == j ? 1e-6f : 0.0f); // Small regularization
                    }
                    out_qx_inv_sigma.data[(b * state_.num_slots + k) * state_.slot_dim * state_.slot_dim + i * state_.slot_dim + j] = inv_sigma_ij;
                }
            }
            
            // Compute weighted mean
            for (int i = 0; i < state_.slot_dim; ++i) {
                float mu_i = 0.0f;
                for (int n = 0; n < num_tokens; ++n) {
                    float qz_nk = qz.data[(b * num_tokens + n) * state_.num_slots + k];
                    // Simplified: average of inputs
                    mu_i += qz_nk * 0.0f; // Placeholder
                }
                out_qx_mu.data[(b * state_.num_slots + k) * state_.slot_dim + i] = mu_i;
            }
        }
    }
    out_qx_mu.copyToDevice();
    out_qx_inv_sigma.copyToDevice();
}

void SlotMixtureModel::updateQz(const Tensor& ell,
                               Tensor& out_qz) {
    // Softmax over slots for each token
    int batch_size = ell.shape[0];
    int num_tokens = ell.shape[1];
    int num_slots = ell.shape[2];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int n = 0; n < num_tokens; ++n) {
            // Find max for numerical stability
            float max_ell = ell.data[(b * num_tokens + n) * num_slots];
            for (int k = 1; k < num_slots; ++k) {
                max_ell = std::max(max_ell, ell.data[(b * num_tokens + n) * num_slots + k]);
            }
            
            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int k = 0; k < num_slots; ++k) {
                float exp_ell = std::exp(ell.data[(b * num_tokens + n) * num_slots + k] - max_ell);
                sum_exp += exp_ell;
            }
            
            // Normalize
            for (int k = 0; k < num_slots; ++k) {
                float exp_ell = std::exp(ell.data[(b * num_tokens + n) * num_slots + k] - max_ell);
                out_qz.data[(b * num_tokens + n) * num_slots + k] = exp_ell / std::max(sum_exp, 1e-10f);
            }
        }
    }
    out_qz.copyToDevice();
}

void SlotMixtureModel::updateParameters(const Tensor& inputs,
                                       const Tensor& qx_mu,
                                       const Tensor& qx_inv_sigma,
                                       const Tensor& qz,
                                       float lr,
                                       float beta) {
    // M-step: Update Linear Matrix Normal Gamma parameters
    int batch_size = inputs.shape[0];
    int num_tokens = inputs.shape[1];
    
    // Compute sufficient statistics weighted by qz
    for (int k = 0; k < state_.num_slots; ++k) {
        float qz_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            for (int n = 0; n < num_tokens; ++n) {
                qz_sum += qz.data[(b * num_tokens + n) * state_.num_slots + k];
            }
        }
        
        // Update pi (prior on z)
        float prior_alpha_k = state_.prior_params.a.data[k];
        float old_alpha_k = state_.posterior_params.a.data[k];
        float new_alpha_k = (1.0f - lr) * old_alpha_k + lr * (prior_alpha_k + qz_sum);
        
        // Apply momentum
        new_alpha_k = (1.0f - beta) * new_alpha_k + beta * old_alpha_k;
        
        state_.posterior_params.a.data[k] = new_alpha_k;
        
        // Update mu (with learning rate)
        for (int i = 0; i < state_.input_dim; ++i) {
            for (int j = 0; j < state_.slot_dim + (state_.use_bias ? 1 : 0); ++j) {
                float prior_mu = state_.prior_params.mu.data[k * state_.input_dim * (state_.slot_dim + 1) + i * (state_.slot_dim + 1) + j];
                float old_mu = state_.posterior_params.mu.data[k * state_.input_dim * (state_.slot_dim + 1) + i * (state_.slot_dim + 1) + j];
                
                // Sufficient statistics update (simplified)
                float new_mu = (1.0f - lr) * old_mu + lr * (prior_mu + 0.1f); // Placeholder
                state_.posterior_params.mu.data[k * state_.input_dim * (state_.slot_dim + 1) + i * (state_.slot_dim + 1) + j] = new_mu;
            }
        }
    }
    
    state_.posterior_params.mu.copyToDevice();
    state_.posterior_params.a.copyToDevice();
}

// Factory function
std::unique_ptr<SlotMixtureModel> createSMM(int width, int height, int input_dim,
                                           int slot_dim, int num_slots,
                                           bool use_bias,
                                           float ns_a,
                                           float ns_b,
                                           float dof_offset,
                                           const std::vector<float>& mask_prob,
                                           const std::vector<float>& scale,
                                           float transform_inv_v_scale,
                                           float bias_inv_v_scale) {
    SMMState state;
    
    // Set configuration
    state.width = width;
    state.height = height;
    state.input_dim = input_dim;
    state.slot_dim = slot_dim;
    state.num_slots = num_slots;
    state.use_bias = use_bias;
    state.learning_rate = 1.0f;
    state.beta = 0.0f;
    state.elbo_threshold = 5.0f;
    state.max_grow_steps = 20;
    state.num_e_steps = 2;
    
    // Allocate state
    state.allocate(1, width, height, input_dim, slot_dim, num_slots, use_bias);
    
    // Initialize prior parameters
    int x_dim = slot_dim + (use_bias ? 1 : 0);
    
    // Initialize mu prior
    for (int k = 0; k < num_slots; ++k) {
        for (int i = 0; i < input_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                state.prior_params.mu.data[k * input_dim * x_dim + i * x_dim + j] = 0.0f;
            }
        }
    }
    
    // Initialize inv_v prior
    for (int k = 0; k < num_slots; ++k) {
        for (int i = 0; i < x_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                float val = (i == j) ? ((i < slot_dim) ? transform_inv_v_scale : bias_inv_v_scale) : 0.0f;
                state.prior_params.inv_v.data[k * x_dim * x_dim + i * x_dim + j] = val;
            }
        }
    }
    
    // Initialize a and b priors
    for (int k = 0; k < num_slots; ++k) {
        for (int i = 0; i < input_dim; ++i) {
            state.prior_params.a.data[k * input_dim + i] = 1.0f + dof_offset;
            
            float scale_val = (i < (int)scale.size()) ? scale[i] : scale.back();
            state.prior_params.b.data[k * input_dim + i] = scale_val * scale_val;
        }
    }
    
    // Initialize posterior with noise
    for (int k = 0; k < num_slots; ++k) {
        for (int i = 0; i < input_dim; ++i) {
            for (int j = 0; j < slot_dim; ++j) {
                float noise = 1.0f + static_cast<float>(rand()) / RAND_MAX;
                state.posterior_params.mu.data[k * input_dim * x_dim + i * x_dim + j] = 
                    ns_a * noise * state.prior_params.mu.data[k * input_dim * x_dim + i * x_dim + j];
            }
            if (use_bias) {
                float noise = 1.0f + static_cast<float>(rand()) / RAND_MAX;
                state.posterior_params.mu.data[k * input_dim * x_dim + i * x_dim + slot_dim] = 
                    ns_b * noise * state.prior_params.mu.data[k * input_dim * x_dim + i * x_dim + slot_dim];
            }
        }
    }
    
    // Copy to device
    state.copyToDevice();
    
    return std::make_unique<SlotMixtureModel>(state);
}

// Position encoding
void addPositionEncoding(const Tensor& image,
                        Tensor& out_with_pos,
                        float width_scale,
                        float height_scale) {
    int height = image.shape[0];
    int width = image.shape[1];
    int channels = image.shape[2];
    
    int idx = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            // Add position
            out_with_pos.data[idx * (channels + 2) + 0] = (w - width / 2.0f) / width_scale;
            out_with_pos.data[idx * (channels + 2) + 1] = (h - height / 2.0f) / height_scale;
            
            // Copy color
            for (int c = 0; c < channels; ++c) {
                out_with_pos.data[idx * (channels + 2) + 2 + c] = image.data[h * width * channels + w * channels + c];
            }
            idx++;
        }
    }
}

// Format observation for SMM
void formatObservation(const Tensor& obs,
                      Tensor& out_formatted,
                      const std::vector<float>& offset,
                      const std::vector<float>& stdevs) {
    int num_pixels = obs.shape[0];
    int num_features = obs.shape[1];
    
    for (int i = 0; i < num_pixels; ++i) {
        for (int j = 0; j < num_features; ++j) {
            float val = obs.data[i * num_features + j];
            float off = (j < (int)offset.size()) ? offset[j] : 0.0f;
            float std = (j < (int)stdevs.size()) ? stdevs[j] : 1.0f;
            out_formatted.data[i * num_features + j] = (val - off) / std;
        }
    }
}

} // namespace models
} // namespace axiom

/**
 * Copyright 2025 VERSES AI, Inc.
 *
 * Licensed under the VERSES Academic Research License (the "License");
 * you may not use this file except in compliance with the license.
 */

#include "rmm.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <algorithm>

namespace axiom {
namespace models {

// HybridMixtureState implementation
void HybridMixtureState::allocate(int num_components_, int cont_dim_, 
                                 const std::vector<int>& discrete_dims_) {
    num_components = num_components_;
    cont_dim = cont_dim_;
    discrete_dims = discrete_dims_;
    num_discrete = discrete_dims_.size();
    
    // Continuous parameters
    cont_mean.shape = {num_components, cont_dim, 1};
    cont_mean.allocate(cont_mean.shape, true);
    
    cont_kappa.shape = {num_components, 1, 1};
    cont_kappa.allocate(cont_kappa.shape, true);
    
    cont_u.shape = {num_components, cont_dim, cont_dim};
    cont_u.allocate(cont_u.shape, true);
    
    cont_n.shape = {num_components, 1, 1};
    cont_n.allocate(cont_n.shape, true);
    
    // Discrete parameters
    disc_alpha.resize(num_discrete);
    for (int i = 0; i < num_discrete; ++i) {
        disc_alpha[i].shape = {num_components, discrete_dims[i], 1};
        disc_alpha[i].allocate(disc_alpha[i].shape, true);
    }
    
    // Prior
    prior_alpha.shape = {num_components};
    prior_alpha.allocate(prior_alpha.shape, true);
    
    posterior_alpha.shape = {num_components};
    posterior_alpha.allocate(posterior_alpha.shape, true);
    
    // Masks
    used_mask.shape = {num_components};
    used_mask.allocate(used_mask.shape, true);
    
    dirty_mask.shape = {num_components};
    dirty_mask.allocate(dirty_mask.shape, true);
}

void HybridMixtureState::copyToDevice() {
    cont_mean.copyToDevice();
    cont_kappa.copyToDevice();
    cont_u.copyToDevice();
    cont_n.copyToDevice();
    
    for (auto& alpha : disc_alpha) {
        alpha.copyToDevice();
    }
    
    prior_alpha.copyToDevice();
    posterior_alpha.copyToDevice();
    used_mask.copyToDevice();
    dirty_mask.copyToDevice();
}

void HybridMixtureState::copyFromDevice() {
    cont_mean.copyFromDevice();
    cont_kappa.copyFromDevice();
    cont_u.copyFromDevice();
    cont_n.copyFromDevice();
    
    for (auto& alpha : disc_alpha) {
        alpha.copyFromDevice();
    }
    
    prior_alpha.copyFromDevice();
    posterior_alpha.copyFromDevice();
    used_mask.copyFromDevice();
    dirty_mask.copyFromDevice();
}

// RewardMixtureModel implementation
RewardMixtureModel::RewardMixtureModel(const HybridMixtureState& state) : state_(state) {
#ifdef USE_CUDA
    cudaStreamCreate(&stream_);
#endif
    
    // Allocate discrete ELL buffers
    for (int i = 0; i < state_.num_discrete; ++i) {
        Tensor buffer;
        buffer.shape = {1, 1, state_.num_components}; // Will be resized dynamically
        buffer.allocate(buffer.shape, true);
        disc_ell_buffers_.push_back(buffer);
    }
}

RewardMixtureModel::~RewardMixtureModel() {
#ifdef USE_CUDA
    cudaStreamDestroy(stream_);
#endif
}

void RewardMixtureModel::eStep(const Tensor& c_data,
                              const std::vector<Tensor>& d_data,
                              const std::vector<float>& w_disc,
                              Tensor& out_qz,
                              Tensor& out_c_ell,
                              Tensor& out_d_ell) {
    int batch_size = c_data.shape[0];
    
    // Compute continuous ELL
    computeContinuousELL(c_data, out_c_ell);
    
    // Compute discrete ELL
    computeDiscreteELL(d_data, w_disc, out_d_ell);
    
    // Total ELL
    Tensor total_ell;
    total_ell.shape = {batch_size, state_.num_components};
    total_ell.allocate(total_ell.shape, true);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            float c_val = out_c_ell.data[b * state_.num_components + k];
            float d_val = out_d_ell.data[b * state_.num_components + k];
            total_ell.data[b * state_.num_components + k] = c_val + d_val;
        }
    }
    
    // Softmax to get qz
    for (int b = 0; b < batch_size; ++b) {
        // Find max
        float max_val = total_ell.data[b * state_.num_components];
        for (int k = 1; k < state_.num_components; ++k) {
            max_val = std::max(max_val, total_ell.data[b * state_.num_components + k]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int k = 0; k < state_.num_components; ++k) {
            float exp_val = std::exp(total_ell.data[b * state_.num_components + k] - max_val);
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int k = 0; k < state_.num_components; ++k) {
            float exp_val = std::exp(total_ell.data[b * state_.num_components + k] - max_val);
            out_qz.data[b * state_.num_components + k] = exp_val / std::max(sum_exp, 1e-10f);
        }
    }
    
    out_qz.copyToDevice();
}

void RewardMixtureModel::mStep(const Tensor& c_data,
                              const std::vector<Tensor>& d_data,
                              const Tensor& qz,
                              float lr,
                              float beta) {
    updatePrior(qz, lr, beta);
    updateContinuousLikelihood(c_data, qz, lr, beta);
    updateDiscreteLikelihood(d_data, qz, lr, beta);
}

void RewardMixtureModel::mStepKeepUnused(const Tensor& c_data,
                                        const std::vector<Tensor>& d_data,
                                        const Tensor& qz) {
    // Store old state
    HybridMixtureState old_state = state_;
    
    // Perform regular M-step
    mStep(c_data, d_data, qz, state_.lr, state_.beta);
    
    // Determine which components are active
    int batch_size = qz.shape[0];
    std::vector<float> active_mask(state_.num_components, 0.0f);
    
    for (int k = 0; k < state_.num_components; ++k) {
        float max_qz = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            max_qz = std::max(max_qz, qz.data[b * state_.num_components + k]);
        }
        if (max_qz > 0.25f) {
            active_mask[k] = 1.0f;
        }
    }
    
    // Combine parameters: use new for active, old for inactive
    for (int k = 0; k < state_.num_components; ++k) {
        float use_new = active_mask[k];
        float use_old = 1.0f - use_new;
        
        // Update continuous mean
        for (int i = 0; i < state_.cont_dim; ++i) {
            state_.cont_mean.data[k * state_.cont_dim + i] = 
                use_new * state_.cont_mean.data[k * state_.cont_dim + i] + 
                use_old * old_state.cont_mean.data[k * state_.cont_dim + i];
        }
        
        // Update alpha
        state_.posterior_alpha.data[k] = 
            use_new * state_.posterior_alpha.data[k] + 
            use_old * old_state.posterior_alpha.data[k];
    }
    
    state_.cont_mean.copyToDevice();
    state_.posterior_alpha.copyToDevice();
}

void RewardMixtureModel::predict(const Tensor& c_sample,
                                const std::vector<Tensor>& d_sample,
                                const std::vector<float>& w_disc,
                                int& out_tmm_slot,
                                float& out_reward,
                                Tensor& out_elogp,
                                Tensor& out_qz,
                                int& out_mix_slot) {
    // E-step to infer mixture cluster
    Tensor c_ell, d_ell;
    c_ell.shape = {1, state_.num_components};
    c_ell.allocate(c_ell.shape, true);
    d_ell.shape = {1, state_.num_components};
    d_ell.allocate(d_ell.shape, true);
    
    eStep(c_sample, d_sample, w_disc, out_qz, c_ell, d_ell);
    
    // Mask out unused components
    for (int k = 0; k < state_.num_components; ++k) {
        float mask = state_.used_mask.data[k];
        out_elogp.data[k] = mask * (c_ell.data[k] + d_ell.data[k]) + (1.0f - mask) * (-1e10f);
    }
    
    // Get most likely component
    out_mix_slot = 0;
    float max_qz = out_qz.data[0];
    for (int k = 1; k < state_.num_components; ++k) {
        if (out_qz.data[k] > max_qz) {
            max_qz = out_qz.data[k];
            out_mix_slot = k;
        }
    }
    
    // Get TMM switch from discrete likelihood
    // Last discrete feature is TMM switch
    int tmm_disc_idx = state_.num_discrete - 1;
    int tmm_dim = state_.discrete_dims[tmm_disc_idx];
    
    out_tmm_slot = 0;
    float max_alpha = state_.disc_alpha[tmm_disc_idx].data[out_mix_slot * tmm_dim];
    for (int i = 1; i < tmm_dim; ++i) {
        float alpha = state_.disc_alpha[tmm_disc_idx].data[out_mix_slot * tmm_dim + i];
        if (alpha > max_alpha) {
            max_alpha = alpha;
            out_tmm_slot = i;
        }
    }
    
    // Get reward (second-to-last discrete feature)
    int reward_disc_idx = state_.num_discrete - 2;
    int reward_dim = state_.discrete_dims[reward_disc_idx];
    
    // Compute weighted reward
    out_reward = 0.0f;
    for (int r = 0; r < reward_dim; ++r) {
        float reward_val = (r - 1); // Map [0, 1, 2] to [-1, 0, 1]
        float prob = 0.0f;
        for (int k = 0; k < state_.num_components; ++k) {
            prob += out_qz.data[k] * state_.disc_alpha[reward_disc_idx].data[k * reward_dim + r];
        }
        out_reward += reward_val * prob;
    }
}

void RewardMixtureModel::sample(cudaStream_t stream,
                               int n_samples,
                               Tensor& out_c_sample,
                               std::vector<Tensor>& out_d_samples) {
    // Sample from mixture
    for (int s = 0; s < n_samples; ++s) {
        // Sample component
        float rand_val = static_cast<float>(rand()) / RAND_MAX;
        int component = 0;
        float cum_prob = 0.0f;
        
        for (int k = 0; k < state_.num_components; ++k) {
            cum_prob += state_.posterior_alpha.data[k] / state_.num_components;
            if (cum_prob >= rand_val) {
                component = k;
                break;
            }
        }
        
        // Sample continuous
        for (int i = 0; i < state_.cont_dim; ++i) {
            float mean = state_.cont_mean.data[component * state_.cont_dim + i];
            float std = 1.0f; // Simplified
            // Box-Muller transform for normal sampling
            float u1 = static_cast<float>(rand()) / RAND_MAX;
            float u2 = static_cast<float>(rand()) / RAND_MAX;
            float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
            out_c_sample.data[s * state_.cont_dim + i] = mean + std * z;
        }
        
        // Sample discrete
        for (int d = 0; d < state_.num_discrete; ++d) {
            int disc_dim = state_.discrete_dims[d];
            rand_val = static_cast<float>(rand()) / RAND_MAX;
            int category = 0;
            cum_prob = 0.0f;
            
            float alpha_sum = 0.0f;
            for (int c = 0; c < disc_dim; ++c) {
                alpha_sum += state_.disc_alpha[d].data[component * disc_dim + c];
            }
            
            for (int c = 0; c < disc_dim; ++c) {
                cum_prob += state_.disc_alpha[d].data[component * disc_dim + c] / alpha_sum;
                if (cum_prob >= rand_val) {
                    category = c;
                    break;
                }
            }
            
            out_d_samples[d].data[s * disc_dim + category] = 1.0f;
        }
    }
}

void RewardMixtureModel::getMeansAsData(Tensor& out_c_means,
                                       std::vector<Tensor>& out_d_means) {
    out_c_means = state_.cont_mean;
    
    out_d_means.clear();
    for (int d = 0; d < state_.num_discrete; ++d) {
        int disc_dim = state_.discrete_dims[d];
        Tensor mean;
        mean.shape = {state_.num_components, disc_dim, 1};
        mean.allocate(mean.shape, true);
        
        for (int k = 0; k < state_.num_components; ++k) {
            float alpha_sum = 0.0f;
            for (int c = 0; c < disc_dim; ++c) {
                alpha_sum += state_.disc_alpha[d].data[k * disc_dim + c];
            }
            
            for (int c = 0; c < disc_dim; ++c) {
                mean.data[k * disc_dim + c] = state_.disc_alpha[d].data[k * disc_dim + c] / alpha_sum;
            }
        }
        
        out_d_means.push_back(mean);
    }
}

void RewardMixtureModel::markDirty(const Tensor& elogp,
                                  const std::vector<Tensor>& d_data,
                                  float threshold) {
    // Find best component
    int best_k = 0;
    float max_elogp = elogp.data[0];
    for (int k = 1; k < state_.num_components; ++k) {
        if (elogp.data[k] > max_elogp) {
            max_elogp = elogp.data[k];
            best_k = k;
        }
    }
    
    // Check if well explained but prediction is wrong
    if (max_elogp < threshold) {
        state_.dirty_mask.data[best_k] += 1.0f;
        state_.dirty_mask.copyToDevice();
    }
}

float RewardMixtureModel::computeELBO(const Tensor& c_data,
                                     const std::vector<Tensor>& d_data,
                                     const std::vector<float>& w_disc) {
    Tensor qz, c_ell, d_ell;
    qz.shape = {c_data.shape[0], state_.num_components};
    qz.allocate(qz.shape, true);
    c_ell.shape = qz.shape;
    c_ell.allocate(c_ell.shape, true);
    d_ell.shape = qz.shape;
    d_ell.allocate(d_ell.shape, true);
    
    eStep(c_data, d_data, w_disc, qz, c_ell, d_ell);
    
    float elbo = 0.0f;
    for (int b = 0; b < c_data.shape[0]; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            elbo += qz.data[b * state_.num_components + k] * (c_ell.data[b * state_.num_components + k] + d_ell.data[b * state_.num_components + k]);
        }
    }
    
    // Subtract KL divergence (simplified)
    for (int k = 0; k < state_.num_components; ++k) {
        float alpha = state_.posterior_alpha.data[k];
        float prior_alpha = state_.prior_alpha.data[k];
        if (alpha > prior_alpha) {
            elbo -= (alpha - prior_alpha); // Simplified KL
        }
    }
    
    return elbo;
}

void RewardMixtureModel::mergeClusters(int idx1, int idx2) {
    // Sum two components together
    for (int i = 0; i < state_.cont_dim; ++i) {
        state_.cont_mean.data[idx1 * state_.cont_dim + i] += state_.cont_mean.data[idx2 * state_.cont_dim + i];
        state_.cont_mean.data[idx1 * state_.cont_dim + i] -= 0.0f; // Subtract prior if needed
    }
    
    state_.posterior_alpha.data[idx1] += state_.posterior_alpha.data[idx2];
    state_.posterior_alpha.data[idx1] -= state_.prior_alpha.data[idx2];
    
    // Reset second component to prior
    state_.posterior_alpha.data[idx2] = state_.prior_alpha.data[idx2];
    state_.used_mask.data[idx2] = 0.0f;
    
    state_.cont_mean.copyToDevice();
    state_.posterior_alpha.copyToDevice();
    state_.used_mask.copyToDevice();
}

void RewardMixtureModel::runBMR(int n_samples, float elbo_threshold) {
    // Find pairs of similar components
    std::vector<std::pair<int, int>> pairs;
    
    for (int i = 0; i < state_.num_components; ++i) {
        if (state_.used_mask.data[i] < 0.5f) continue;
        
        for (int j = i + 1; j < state_.num_components; ++j) {
            if (state_.used_mask.data[j] < 0.5f) continue;
            
            // Check if similar (simplified check)
            float dist = 0.0f;
            for (int d = 0; d < state_.cont_dim; ++d) {
                float diff = state_.cont_mean.data[i * state_.cont_dim + d] - state_.cont_mean.data[j * state_.cont_dim + d];
                dist += diff * diff;
            }
            
            if (dist < elbo_threshold) {
                pairs.push_back({i, j});
            }
        }
    }
    
    // Try merging pairs
    for (const auto& pair : pairs) {
        // Compute ELBO before
        float elbo_before = 0.0f; // Would need actual data
        
        // Try merge
        mergeClusters(pair.first, pair.second);
        
        // Compute ELBO after (would need to check if improved)
    }
}

void RewardMixtureModel::trainStep(const Tensor& c_sample,
                                  const std::vector<Tensor>& d_sample,
                                  float logp_threshold,
                                  Tensor& out_qz,
                                  bool& grew_component) {
    Tensor c_ell, d_ell;
    c_ell.shape = {c_sample.shape[0], state_.num_components};
    c_ell.allocate(c_ell.shape, true);
    d_ell.shape = c_ell.shape;
    d_ell.allocate(d_ell.shape, true);
    
    std::vector<float> w_disc(state_.num_discrete, 1.0f);
    eStep(c_sample, d_sample, w_disc, out_qz, c_ell, d_ell);
    
    // Mask out unused components
    for (int k = 0; k < state_.num_components; ++k) {
        float mask = state_.used_mask.data[k];
        c_ell.data[k] = mask * c_ell.data[k] + (1.0f - mask) * (-1e10f);
        d_ell.data[k] = mask * d_ell.data[k] + (1.0f - mask) * (-1e10f);
    }
    
    // Check if well explained
    float max_ell = c_ell.data[0] + d_ell.data[0];
    for (int k = 1; k < state_.num_components; ++k) {
        max_ell = std::max(max_ell, c_ell.data[k] + d_ell.data[k]);
    }
    
    if (max_ell > logp_threshold) {
        // Well explained, just update
        mStepKeepUnused(c_sample, d_sample, out_qz);
        grew_component = false;
    } else {
        // Find unused component
        int unused_idx = -1;
        for (int k = 0; k < state_.num_components; ++k) {
            if (state_.used_mask.data[k] < 0.5f) {
                unused_idx = k;
                break;
            }
        }
        
        if (unused_idx >= 0) {
            // Assign to unused component
            for (int k = 0; k < state_.num_components; ++k) {
                out_qz.data[k] = (k == unused_idx) ? 1.0f : 0.0f;
            }
            
            mStepKeepUnused(c_sample, d_sample, out_qz);
            state_.used_mask.data[unused_idx] = 1.0f;
            state_.used_mask.copyToDevice();
            grew_component = true;
        } else {
            mStepKeepUnused(c_sample, d_sample, out_qz);
            grew_component = false;
        }
    }
    
    // Update used mask based on alpha
    for (int k = 0; k < state_.num_components; ++k) {
        state_.used_mask.data[k] = (state_.posterior_alpha.data[k] > state_.prior_alpha.data[k]) ? 1.0f : 0.0f;
    }
    state_.used_mask.copyToDevice();
}

Tensor RewardMixtureModel::getUsedMask() const {
    return state_.used_mask;
}

void RewardMixtureModel::setUsedMask(const Tensor& mask) {
    state_.used_mask = mask;
    state_.used_mask.copyToDevice();
}

void RewardMixtureModel::computeContinuousELL(const Tensor& c_data, Tensor& out_ell) {
    int batch_size = c_data.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            float ell = 0.0f;
            
            // Gaussian log-likelihood
            float kappa = state_.cont_kappa.data[k];
            float n = state_.cont_n.data[k];
            
            for (int i = 0; i < state_.cont_dim; ++i) {
                float data_val = c_data.data[b * state_.cont_dim + i];
                float mean = state_.cont_mean.data[k * state_.cont_dim + i];
                
                // Expected precision
                float u_ii = state_.cont_u.data[k * state_.cont_dim * state_.cont_dim + i * state_.cont_dim + i];
                float expected_precision = n / u_ii;
                
                // Log-likelihood contribution
                ell += -0.5f * expected_precision * (data_val - mean) * (data_val - mean);
                ell += 0.5f * std::log(expected_precision);
            }
            
            ell -= 0.5f * state_.cont_dim * std::log(2.0f * M_PI);
            
            out_ell.data[b * state_.num_components + k] = ell;
        }
    }
}

void RewardMixtureModel::computeDiscreteELL(const std::vector<Tensor>& d_data,
                                           const std::vector<float>& w_disc,
                                           Tensor& out_ell) {
    int batch_size = d_data[0].shape[0];
    
    // Initialize to zero
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            out_ell.data[b * state_.num_components + k] = 0.0f;
        }
    }
    
    // Add contribution from each discrete feature
    for (int d = 0; d < state_.num_discrete; ++d) {
        int disc_dim = state_.discrete_dims[d];
        float w = (d < (int)w_disc.size()) ? w_disc[d] : 1.0f;
        
        for (int b = 0; b < batch_size; ++b) {
            // Find which category is active
            int active_cat = 0;
            for (int c = 0; c < disc_dim; ++c) {
                if (d_data[d].data[b * disc_dim + c] > 0.5f) {
                    active_cat = c;
                    break;
                }
            }
            
            for (int k = 0; k < state_.num_components; ++k) {
                float alpha_sum = 0.0f;
                for (int c = 0; c < disc_dim; ++c) {
                    alpha_sum += state_.disc_alpha[d].data[k * disc_dim + c];
                }
                
                float alpha_c = state_.disc_alpha[d].data[k * disc_dim + active_cat];
                float expected_log_prob = std::log(alpha_c / alpha_sum);
                
                out_ell.data[b * state_.num_components + k] += w * expected_log_prob;
            }
        }
    }
}

void RewardMixtureModel::updatePrior(const Tensor& qz, float lr, float beta) {
    int batch_size = qz.shape[0];
    
    for (int k = 0; k < state_.num_components; ++k) {
        float qz_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            qz_sum += qz.data[b * state_.num_components + k];
        }
        
        float prior_alpha = state_.prior_alpha.data[k];
        float old_alpha = state_.posterior_alpha.data[k];
        float new_alpha = (1.0f - lr) * old_alpha + lr * (prior_alpha + qz_sum);
        new_alpha = (1.0f - beta) * new_alpha + beta * old_alpha;
        
        state_.posterior_alpha.data[k] = new_alpha;
    }
    
    state_.posterior_alpha.copyToDevice();
}

void RewardMixtureModel::updateContinuousLikelihood(const Tensor& c_data,
                                                   const Tensor& qz,
                                                   float lr,
                                                   float beta) {
    int batch_size = c_data.shape[0];
    
    for (int k = 0; k < state_.num_components; ++k) {
        float qz_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            qz_sum += qz.data[b * state_.num_components + k];
        }
        
        if (qz_sum < 1e-10f) continue;
        
        // Update mean
        for (int i = 0; i < state_.cont_dim; ++i) {
            float data_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                data_sum += qz.data[b * state_.num_components + k] * c_data.data[b * state_.cont_dim + i];
            }
            
            float old_mean = state_.cont_mean.data[k * state_.cont_dim + i];
            float new_mean = data_sum / qz_sum;
            state_.cont_mean.data[k * state_.cont_dim + i] = (1.0f - lr) * old_mean + lr * new_mean;
        }
        
        // Update kappa
        float old_kappa = state_.cont_kappa.data[k];
        float new_kappa = old_kappa + qz_sum;
        state_.cont_kappa.data[k] = (1.0f - lr) * old_kappa + lr * new_kappa;
        
        // Update n
        float old_n = state_.cont_n.data[k];
        float new_n = old_n + qz_sum;
        state_.cont_n.data[k] = (1.0f - lr) * old_n + lr * new_n;
    }
    
    state_.cont_mean.copyToDevice();
    state_.cont_kappa.copyToDevice();
    state_.cont_n.copyToDevice();
}

void RewardMixtureModel::updateDiscreteLikelihood(const std::vector<Tensor>& d_data,
                                                 const Tensor& qz,
                                                 float lr,
                                                 float beta) {
    int batch_size = qz.shape[0];
    
    for (int d = 0; d < state_.num_discrete; ++d) {
        int disc_dim = state_.discrete_dims[d];
        
        for (int k = 0; k < state_.num_components; ++k) {
            for (int c = 0; c < disc_dim; ++c) {
                float alpha_sum = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    alpha_sum += qz.data[b * state_.num_components + k] * d_data[d].data[b * disc_dim + c];
                }
                
                float old_alpha = state_.disc_alpha[d].data[k * disc_dim + c];
                float new_alpha = old_alpha + alpha_sum;
                state_.disc_alpha[d].data[k * disc_dim + c] = (1.0f - lr) * old_alpha + lr * new_alpha;
            }
        }
        
        state_.disc_alpha[d].copyToDevice();
    }
}

void RewardMixtureModel::sumComponents(const HybridMixtureState& other,
                                      const Tensor& self_mask,
                                      const Tensor& other_mask) {
    // Sum two components weighted by masks
    for (int k = 0; k < state_.num_components; ++k) {
        float s_mask = self_mask.data[k];
        float o_mask = other_mask.data[k];
        
        for (int i = 0; i < state_.cont_dim; ++i) {
            float self_val = state_.cont_mean.data[k * state_.cont_dim + i] * s_mask;
            float other_val = other.cont_mean.data[k * state_.cont_dim + i] * o_mask;
            state_.cont_mean.data[k * state_.cont_dim + i] = self_val + other_val;
        }
        
        float self_alpha = state_.posterior_alpha.data[k] * s_mask;
        float other_alpha = other.posterior_alpha.data[k] * o_mask;
        float prior_alpha = state_.prior_alpha.data[k] * o_mask;
        state_.posterior_alpha.data[k] = self_alpha + other_alpha - prior_alpha;
    }
}

void RewardMixtureModel::combineParams(const HybridMixtureState& other,
                                      const Tensor& select_mask) {
    // Select parameters from two models
    for (int k = 0; k < state_.num_components; ++k) {
        float use_other = select_mask.data[k];
        float use_self = 1.0f - use_other;
        
        for (int i = 0; i < state_.cont_dim; ++i) {
            state_.cont_mean.data[k * state_.cont_dim + i] = 
                use_self * state_.cont_mean.data[k * state_.cont_dim + i] + 
                use_other * other.cont_mean.data[k * state_.cont_dim + i];
        }
        
        state_.posterior_alpha.data[k] = 
            use_self * state_.posterior_alpha.data[k] + 
            use_other * other.posterior_alpha.data[k];
    }
}

// Ellipse interaction detection
void detectEllipseInteractions(const Tensor& data,
                              int object_idx,
                              float r_interacting,
                              bool forward_predict,
                              bool exclude_background,
                              bool interact_with_static,
                              const Tensor& tracked_obj_mask,
                              ObjectInteraction& out_result) {
    int n_objects = data.shape[0];
    
    float cx_i = data.data[object_idx * data.shape[1] + 0];
    float cy_i = data.data[object_idx * data.shape[1] + 1];
    float w_i = data.data[object_idx * data.shape[1] + 6] * r_interacting;
    float h_i = data.data[object_idx * data.shape[1] + 7] * r_interacting;
    
    int best_idx = -1;
    float best_overlap = -1.0f;
    float best_dist_x = 0.0f;
    float best_dist_y = 0.0f;
    
    for (int j = 0; j < n_objects; ++j) {
        if (j == object_idx) continue;
        if (exclude_background && j == 0) continue;
        
        float cx_j = data.data[j * data.shape[1] + 0];
        float cy_j = data.data[j * data.shape[1] + 1];
        float w_j = data.data[j * data.shape[1] + 6] * r_interacting;
        float h_j = data.data[j * data.shape[1] + 7] * r_interacting;
        
        // Check ellipse overlap using grid sampling
        int n_grid = 30;
        int overlap_count = 0;
        
        for (int gi = 0; gi < n_grid; ++gi) {
            for (int gj = 0; gj < n_grid; ++gj) {
                float px = cx_j - w_j + 2.0f * w_j * gi / (n_grid - 1);
                float py = cy_j - h_j + 2.0f * h_j * gj / (n_grid - 1);
                
                // Check if point is in both ellipses
                float in_i = ((px - cx_i) / w_i) * ((px - cx_i) / w_i) + 
                            ((py - cy_i) / h_i) * ((py - cy_i) / h_i) <= 1.0f;
                float in_j = ((px - cx_j) / w_j) * ((px - cx_j) / w_j) + 
                            ((py - cy_j) / h_j) * ((py - cy_j) / h_j) <= 1.0f;
                
                if (in_i && in_j) {
                    overlap_count++;
                }
            }
        }
        
        if (overlap_count > 0) {
            float dist_x = cx_i - cx_j;
            float dist_y = cy_i - cy_j;
            float overlap = static_cast<float>(overlap_count) / (n_grid * n_grid);
            
            // Prefer dynamic objects
            bool is_dynamic = tracked_obj_mask.data[j] > 0.5f;
            float sort_metric = is_dynamic ? overlap : overlap - 100.0f;
            
            if (sort_metric > best_overlap) {
                best_overlap = sort_metric;
                best_idx = j;
                best_dist_x = dist_x;
                best_dist_y = dist_y;
            }
        }
    }
    
    out_result.other_idx = best_idx;
    out_result.distance_x = best_dist_x;
    out_result.distance_y = best_dist_y;
    out_result.is_interacting = (best_idx >= 0);
}

// Closest interaction detection
void detectClosestInteractions(const Tensor& data,
                              int object_idx,
                              float r_interacting,
                              bool exclude_background,
                              bool interact_with_static,
                              bool absolute_distance_scale,
                              const Tensor& tracked_obj_mask,
                              ObjectInteraction& out_result) {
    int n_objects = data.shape[0];
    
    float cx_i = data.data[object_idx * data.shape[1] + 0];
    float cy_i = data.data[object_idx * data.shape[1] + 1];
    float w_i = data.data[object_idx * data.shape[1] + 6];
    float h_i = data.data[object_idx * data.shape[1] + 7];
    
    int best_idx = -1;
    float best_dist_sq = 1e10f;
    float best_dist_x = 0.0f;
    float best_dist_y = 0.0f;
    
    for (int j = 0; j < n_objects; ++j) {
        if (j == object_idx) continue;
        if (exclude_background && j == 0) continue;
        
        float cx_j = data.data[j * data.shape[1] + 0];
        float cy_j = data.data[j * data.shape[1] + 1];
        float w_j = data.data[j * data.shape[1] + 6];
        float h_j = data.data[j * data.shape[1] + 7];
        
        // Check if bounding boxes overlap
        bool x_overlap = std::abs(cx_i - cx_j) < (w_i + w_j) * r_interacting;
        bool y_overlap = std::abs(cy_i - cy_j) < (h_i + h_j) * r_interacting;
        
        if (x_overlap && y_overlap) {
            float dist_x = cx_i - cx_j;
            float dist_y = cy_i - cy_j;
            float dist_sq = dist_x * dist_x + dist_y * dist_y;
            
            if (!absolute_distance_scale) {
                dist_x /= w_i;
                dist_y /= h_i;
            }
            
            bool is_dynamic = tracked_obj_mask.data[j] > 0.5f;
            if (is_dynamic || interact_with_static) {
                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_idx = j;
                    best_dist_x = dist_x;
                    best_dist_y = dist_y;
                }
            }
        }
    }
    
    out_result.other_idx = best_idx;
    out_result.distance_x = best_dist_x;
    out_result.distance_y = best_dist_y;
    out_result.is_interacting = (best_idx >= 0);
}

// Factory function
std::unique_ptr<RewardMixtureModel> createRMM(int action_dim,
                                             int num_components_per_switch,
                                             int num_switches,
                                             int num_object_types,
                                             int num_continuous_dims,
                                             int reward_dim,
                                             float cont_scale_switch,
                                             const std::vector<float>& discrete_alphas) {
    HybridMixtureState state;
    
    int num_components = num_components_per_switch * num_switches;
    int cont_dim = num_continuous_dims;
    std::vector<int> disc_dims = {
        num_object_types + 1,  // own identity
        num_object_types + 1,  // interacting identity
        2,                      // used/unused
        action_dim,            // action
        reward_dim,            // reward
        num_switches           // tmm switches
    };
    
    state.allocate(num_components, cont_dim, disc_dims);
    
    // Initialize continuous likelihood
    for (int k = 0; k < num_components; ++k) {
        state.cont_kappa.data[k] = 1e-4f;
        state.cont_n.data[k] = cont_dim + 2.0f;
        
        for (int i = 0; i < cont_dim; ++i) {
            for (int j = 0; j < cont_dim; ++j) {
                float val = (i == j) ? (cont_scale_switch * cont_scale_switch) : 0.0f;
                state.cont_u.data[k * cont_dim * cont_dim + i * cont_dim + j] = val;
            }
        }
    }
    
    // Initialize discrete likelihoods
    std::vector<float> alphas = discrete_alphas;
    if (alphas.empty()) {
        alphas = std::vector<float>(disc_dims.size(), 1e-4f);
    }
    
    for (int d = 0; d < (int)disc_dims.size(); ++d) {
        float alpha = (d < (int)alphas.size()) ? alphas[d] : 1e-4f;
        
        for (int k = 0; k < num_components; ++k) {
            for (int c = 0; c < disc_dims[d]; ++c) {
                // Initialize with eye prior (permuted)
                int perm_idx = (k + c) % disc_dims[d];
                float val = alpha + (perm_idx == 0 ? 10.0f : 0.0f);
                state.disc_alpha[d].data[k * disc_dims[d] + c] = val;
            }
        }
    }
    
    // Initialize prior
    for (int k = 0; k < num_components; ++k) {
        state.prior_alpha.data[k] = 0.1f;
        state.posterior_alpha.data[k] = 0.1f;
    }
    
    state.lr = 1.0f;
    state.beta = 0.0f;
    
    state.copyToDevice();
    
    return std::make_unique<RewardMixtureModel>(state);
}

} // namespace models
} // namespace axiom

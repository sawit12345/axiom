/**
 * Copyright 2025 VERSES AI, Inc.
 */

#include "hybrid_mixture.h"
#include <cmath>

namespace axiom {
namespace models {

void HybridMixtureStateHM::allocate(int num_components_, int cont_dim_,
                                   const std::vector<int>& discrete_dims_) {
    num_components = num_components_;
    cont_dim = cont_dim_;
    discrete_dims = discrete_dims_;
    num_discrete = discrete_dims_.size();
    
    // Continuous parameters
    cont_mean.shape = {num_components, cont_dim, 1};
    cont_mean.allocate(cont_mean.shape, true);
    
    cont_precision.shape = {num_components, cont_dim, cont_dim};
    cont_precision.allocate(cont_precision.shape, true);
    
    cont_kappa.shape = {num_components, 1, 1};
    cont_kappa.allocate(cont_kappa.shape, true);
    
    cont_dof.shape = {num_components, 1, 1};
    cont_dof.allocate(cont_dof.shape, true);
    
    // Discrete parameters
    disc_alphas.resize(num_discrete);
    for (int i = 0; i < num_discrete; ++i) {
        disc_alphas[i].shape = {num_components, discrete_dims[i], 1};
        disc_alphas[i].allocate(disc_alphas[i].shape, true);
    }
    
    // Prior
    prior_alpha.shape = {num_components};
    prior_alpha.allocate(prior_alpha.shape, true);
    
    posterior_alpha.shape = {num_components};
    posterior_alpha.allocate(posterior_alpha.shape, true);
    
    // Masks
    used_mask.shape = {num_components};
    used_mask.allocate(used_mask.shape, true);
    
    lr = 1.0f;
    beta = 0.0f;
}

void HybridMixtureStateHM::copyToDevice() {
    cont_mean.copyToDevice();
    cont_precision.copyToDevice();
    cont_kappa.copyToDevice();
    cont_dof.copyToDevice();
    
    for (auto& alpha : disc_alphas) {
        alpha.copyToDevice();
    }
    
    prior_alpha.copyToDevice();
    posterior_alpha.copyToDevice();
    used_mask.copyToDevice();
}

void HybridMixtureStateHM::copyFromDevice() {
    cont_mean.copyFromDevice();
    cont_precision.copyFromDevice();
    cont_kappa.copyFromDevice();
    cont_dof.copyFromDevice();
    
    for (auto& alpha : disc_alphas) {
        alpha.copyFromDevice();
    }
    
    prior_alpha.copyFromDevice();
    posterior_alpha.copyFromDevice();
    used_mask.copyFromDevice();
}

HybridMixtureModel::HybridMixtureModel(const HybridMixtureStateHM& state) : state_(state) {
    cudaStreamCreate(&stream_);
    
    // Allocate buffers
    for (int i = 0; i < state_.num_discrete; ++i) {
        Tensor buffer;
        buffer.shape = {1, 1, state_.num_components};
        buffer.allocate(buffer.shape, true);
        disc_buffers_.push_back(buffer);
    }
}

HybridMixtureModel::~HybridMixtureModel() {
    cudaStreamDestroy(stream_);
}

void HybridMixtureModel::trainingStep(const Tensor& c_data,
                                     const std::vector<Tensor>& d_data,
                                     float logp_threshold,
                                     Tensor& out_qz,
                                     bool& grew_component) {
    Tensor c_ell, d_ell;
    c_ell.shape = {c_data.shape[0], state_.num_components};
    c_ell.allocate(c_ell.shape, true);
    d_ell.shape = c_ell.shape;
    d_ell.allocate(d_ell.shape, true);
    
    std::vector<float> w_disc(state_.num_discrete, 1.0f);
    eStep(c_data, d_data, w_disc, out_qz, c_ell, d_ell);
    
    // Mask out unused
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
        mStepKeepUnused(c_data, d_data, out_qz);
        grew_component = false;
    } else {
        // Grow
        int unused_idx = findFirstUnused();
        if (unused_idx >= 0) {
            for (int k = 0; k < state_.num_components; ++k) {
                out_qz.data[k] = (k == unused_idx) ? 1.0f : 0.0f;
            }
            mStepKeepUnused(c_data, d_data, out_qz);
            state_.used_mask.data[unused_idx] = 1.0f;
            state_.used_mask.copyToDevice();
            grew_component = true;
        } else {
            mStepKeepUnused(c_data, d_data, out_qz);
            grew_component = false;
        }
    }
}

void HybridMixtureModel::eStep(const Tensor& c_data,
                              const std::vector<Tensor>& d_data,
                              const std::vector<float>& w_disc,
                              Tensor& out_posterior,
                              Tensor& out_c_ell,
                              Tensor& out_d_ell) {
    computeContinuousLogLikelihood(c_data, out_c_ell);
    computeDiscreteLogLikelihood(d_data, w_disc, out_d_ell);
    
    int batch_size = c_data.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        float max_val = out_c_ell.data[b * state_.num_components] + out_d_ell.data[b * state_.num_components];
        for (int k = 1; k < state_.num_components; ++k) {
            max_val = std::max(max_val, out_c_ell.data[b * state_.num_components + k] + out_d_ell.data[b * state_.num_components + k]);
        }
        
        float sum_exp = 0.0f;
        for (int k = 0; k < state_.num_components; ++k) {
            float total_ell = out_c_ell.data[b * state_.num_components + k] + out_d_ell.data[b * state_.num_components + k];
            sum_exp += std::exp(total_ell - max_val);
        }
        
        for (int k = 0; k < state_.num_components; ++k) {
            float total_ell = out_c_ell.data[b * state_.num_components + k] + out_d_ell.data[b * state_.num_components + k];
            out_posterior.data[b * state_.num_components + k] = std::exp(total_ell - max_val) / sum_exp;
        }
    }
}

void HybridMixtureModel::mStep(const Tensor& c_data,
                              const std::vector<Tensor>& d_data,
                              const Tensor& qz,
                              float lr,
                              float beta) {
    updatePrior(qz, lr, beta);
    updateContinuousLikelihood(c_data, qz, lr, beta);
    updateDiscreteLikelihood(d_data, qz, lr, beta);
}

void HybridMixtureModel::mStepKeepUnused(const Tensor& c_data,
                                        const std::vector<Tensor>& d_data,
                                        const Tensor& qz) {
    HybridMixtureStateHM old_state = state_;
    
    mStep(c_data, d_data, qz, state_.lr, state_.beta);
    
    // Determine active components
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
    
    // Combine
    for (int k = 0; k < state_.num_components; ++k) {
        float use_new = active_mask[k];
        float use_old = 1.0f - use_new;
        
        for (int i = 0; i < state_.cont_dim; ++i) {
            state_.cont_mean.data[k * state_.cont_dim + i] = 
                use_new * state_.cont_mean.data[k * state_.cont_dim + i] + 
                use_old * old_state.cont_mean.data[k * state_.cont_dim + i];
        }
        
        state_.posterior_alpha.data[k] = 
            use_new * state_.posterior_alpha.data[k] + 
            use_old * old_state.posterior_alpha.data[k];
    }
}

void HybridMixtureModel::sumComponents(const HybridMixtureStateHM& other,
                                      const Tensor& self_mask,
                                      const Tensor& other_mask) {
    for (int k = 0; k < state_.num_components; ++k) {
        float s_mask = self_mask.data[k];
        float o_mask = other_mask.data[k];
        
        for (int i = 0; i < state_.cont_dim; ++i) {
            state_.cont_mean.data[k * state_.cont_dim + i] = 
                s_mask * state_.cont_mean.data[k * state_.cont_dim + i] + 
                o_mask * other.cont_mean.data[k * state_.cont_dim + i];
        }
        
        state_.posterior_alpha.data[k] = 
            s_mask * state_.posterior_alpha.data[k] + 
            o_mask * other.posterior_alpha.data[k] - 
            o_mask * state_.prior_alpha.data[k];
    }
}

void HybridMixtureModel::combineParams(const HybridMixtureStateHM& other,
                                      const Tensor& select_mask) {
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

void HybridMixtureModel::mergeClusters(int idx1, int idx2) {
    for (int i = 0; i < state_.cont_dim; ++i) {
        state_.cont_mean.data[idx1 * state_.cont_dim + i] += state_.cont_mean.data[idx2 * state_.cont_dim + i];
        state_.cont_mean.data[idx1 * state_.cont_dim + i] -= 0.0f; // Subtract prior if needed
    }
    
    state_.posterior_alpha.data[idx1] += state_.posterior_alpha.data[idx2];
    state_.posterior_alpha.data[idx1] -= state_.prior_alpha.data[idx2];
    
    state_.posterior_alpha.data[idx2] = state_.prior_alpha.data[idx2];
    state_.used_mask.data[idx2] = 0.0f;
    
    state_.cont_mean.copyToDevice();
    state_.posterior_alpha.copyToDevice();
    state_.used_mask.copyToDevice();
}

void HybridMixtureModel::sample(cudaStream_t stream,
                               int n_samples,
                               Tensor& out_c_sample,
                               std::vector<Tensor>& out_d_samples) {
    for (int s = 0; s < n_samples; ++s) {
        // Sample component
        float rand_val = static_cast<float>(rand()) / RAND_MAX;
        int component = 0;
        float cum_prob = 0.0f;
        
        float alpha_sum = 0.0f;
        for (int k = 0; k < state_.num_components; ++k) {
            alpha_sum += state_.posterior_alpha.data[k];
        }
        
        for (int k = 0; k < state_.num_components; ++k) {
            cum_prob += state_.posterior_alpha.data[k] / alpha_sum;
            if (cum_prob >= rand_val) {
                component = k;
                break;
            }
        }
        
        // Sample continuous
        for (int i = 0; i < state_.cont_dim; ++i) {
            float mean = state_.cont_mean.data[component * state_.cont_dim + i];
            float std = 1.0f;
            
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
            
            float disc_alpha_sum = 0.0f;
            for (int c = 0; c < disc_dim; ++c) {
                disc_alpha_sum += state_.disc_alphas[d].data[component * disc_dim + c];
            }
            
            for (int c = 0; c < disc_dim; ++c) {
                cum_prob += state_.disc_alphas[d].data[component * disc_dim + c] / disc_alpha_sum;
                if (cum_prob >= rand_val) {
                    category = c;
                    break;
                }
            }
            
            out_d_samples[d].data[s * disc_dim + category] = 1.0f;
        }
    }
}

void HybridMixtureModel::getMeansAsData(Tensor& out_c_means,
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
                alpha_sum += state_.disc_alphas[d].data[k * disc_dim + c];
            }
            
            for (int c = 0; c < disc_dim; ++c) {
                mean.data[k * disc_dim + c] = state_.disc_alphas[d].data[k * disc_dim + c] / alpha_sum;
            }
        }
        
        out_d_means.push_back(mean);
    }
}

float HybridMixtureModel::computeELBO(const Tensor& c_data,
                                     const std::vector<Tensor>& d_data) {
    Tensor qz, c_ell, d_ell;
    qz.shape = {c_data.shape[0], state_.num_components};
    qz.allocate(qz.shape, true);
    c_ell.shape = qz.shape;
    c_ell.allocate(c_ell.shape, true);
    d_ell.shape = qz.shape;
    d_ell.allocate(d_ell.shape, true);
    
    std::vector<float> w_disc(state_.num_discrete, 1.0f);
    eStep(c_data, d_data, w_disc, qz, c_ell, d_ell);
    
    float elbo = 0.0f;
    for (int b = 0; b < c_data.shape[0]; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            elbo += qz.data[b * state_.num_components + k] * (c_ell.data[b * state_.num_components + k] + d_ell.data[b * state_.num_components + k]);
        }
    }
    
    return elbo;
}

float HybridMixtureModel::computeContinuousELBO(const Tensor& c_data) {
    Tensor ell;
    ell.shape = {c_data.shape[0], state_.num_components};
    ell.allocate(ell.shape, true);
    
    computeContinuousLogLikelihood(c_data, ell);
    
    float elbo = 0.0f;
    for (int b = 0; b < c_data.shape[0]; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            elbo += ell.data[b * state_.num_components + k];
        }
    }
    
    return elbo;
}

float HybridMixtureModel::computeDiscreteELBO(const std::vector<Tensor>& d_data) {
    Tensor ell;
    ell.shape = {d_data[0].shape[0], state_.num_components};
    ell.allocate(ell.shape, true);
    
    std::vector<float> w_disc(state_.num_discrete, 1.0f);
    computeDiscreteLogLikelihood(d_data, w_disc, ell);
    
    float elbo = 0.0f;
    for (int b = 0; b < d_data[0].shape[0]; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            elbo += ell.data[b * state_.num_components + k];
        }
    }
    
    return elbo;
}

Tensor HybridMixtureModel::getUsedMask() const {
    return state_.used_mask;
}

void HybridMixtureModel::setUsedMask(const Tensor& mask) {
    state_.used_mask = mask;
    state_.used_mask.copyToDevice();
}

int HybridMixtureModel::getNumUsed() const {
    int count = 0;
    for (int k = 0; k < state_.num_components; ++k) {
        if (state_.used_mask.data[k] > 0.5f) {
            count++;
        }
    }
    return count;
}

void HybridMixtureModel::computeContinuousLogLikelihood(const Tensor& c_data, Tensor& out_ell) {
    int batch_size = c_data.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            float ell = 0.0f;
            
            for (int i = 0; i < state_.cont_dim; ++i) {
                float diff = c_data.data[b * state_.cont_dim + i] - state_.cont_mean.data[k * state_.cont_dim + i];
                ell += -0.5f * diff * diff;
            }
            
            ell -= 0.5f * state_.cont_dim * std::log(2.0f * M_PI);
            
            out_ell.data[b * state_.num_components + k] = ell;
        }
    }
}

void HybridMixtureModel::computeDiscreteLogLikelihood(const std::vector<Tensor>& d_data,
                                                     const std::vector<float>& w_disc,
                                                     Tensor& out_ell) {
    int batch_size = d_data[0].shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            out_ell.data[b * state_.num_components + k] = 0.0f;
        }
    }
    
    for (int d = 0; d < state_.num_discrete; ++d) {
        int disc_dim = state_.discrete_dims[d];
        float w = (d < (int)w_disc.size()) ? w_disc[d] : 1.0f;
        
        for (int b = 0; b < batch_size; ++b) {
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
                    alpha_sum += state_.disc_alphas[d].data[k * disc_dim + c];
                }
                
                float log_prob = std::log(state_.disc_alphas[d].data[k * disc_dim + active_cat] / alpha_sum);
                out_ell.data[b * state_.num_components + k] += w * log_prob;
            }
        }
    }
}

void HybridMixtureModel::updatePrior(const Tensor& qz, float lr, float beta) {
    int batch_size = qz.shape[0];
    
    for (int k = 0; k < state_.num_components; ++k) {
        float qz_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            qz_sum += qz.data[b * state_.num_components + k];
        }
        
        float prior_alpha = state_.prior_alpha.data[k];
        float old_alpha = state_.posterior_alpha.data[k];
        float new_alpha = (1.0f - lr) * old_alpha + lr * (prior_alpha + qz_sum);
        state_.posterior_alpha.data[k] = (1.0f - beta) * new_alpha + beta * old_alpha;
    }
    
    state_.posterior_alpha.copyToDevice();
}

void HybridMixtureModel::updateContinuousLikelihood(const Tensor& c_data,
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
        
        for (int i = 0; i < state_.cont_dim; ++i) {
            float data_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                data_sum += qz.data[b * state_.num_components + k] * c_data.data[b * state_.cont_dim + i];
            }
            
            float old_mean = state_.cont_mean.data[k * state_.cont_dim + i];
            float new_mean = data_sum / qz_sum;
            state_.cont_mean.data[k * state_.cont_dim + i] = (1.0f - lr) * old_mean + lr * new_mean;
        }
        
        float old_kappa = state_.cont_kappa.data[k];
        float new_kappa = old_kappa + qz_sum;
        state_.cont_kappa.data[k] = (1.0f - lr) * old_kappa + lr * new_kappa;
        
        float old_n = state_.cont_dof.data[k];
        float new_n = old_n + qz_sum;
        state_.cont_dof.data[k] = (1.0f - lr) * old_n + lr * new_n;
    }
    
    state_.cont_mean.copyToDevice();
    state_.cont_kappa.copyToDevice();
    state_.cont_dof.copyToDevice();
}

void HybridMixtureModel::updateDiscreteLikelihood(const std::vector<Tensor>& d_data,
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
                
                float old_alpha = state_.disc_alphas[d].data[k * disc_dim + c];
                float new_alpha = old_alpha + alpha_sum;
                state_.disc_alphas[d].data[k * disc_dim + c] = (1.0f - lr) * old_alpha + lr * new_alpha;
            }
        }
        
        state_.disc_alphas[d].copyToDevice();
    }
}

int HybridMixtureModel::findFirstUnused() {
    for (int k = 0; k < state_.num_components; ++k) {
        if (state_.used_mask.data[k] < 0.5f) {
            return k;
        }
    }
    return -1;
}

std::unique_ptr<HybridMixtureModel> createHybridMixture(
    int num_components,
    int continuous_dim,
    const std::vector<int>& discrete_dims,
    float cont_scale,
    const std::vector<float>& discrete_alphas,
    float prior_alpha) {
    
    HybridMixtureStateHM state;
    state.allocate(num_components, continuous_dim, discrete_dims);
    
    // Initialize continuous likelihood
    for (int k = 0; k < num_components; ++k) {
        state.cont_kappa.data[k] = 1e-4f;
        state.cont_dof.data[k] = continuous_dim + 2.0f;
        
        for (int i = 0; i < continuous_dim; ++i) {
            for (int j = 0; j < continuous_dim; ++j) {
                float val = (i == j) ? (cont_scale * cont_scale) : 0.0f;
                state.cont_precision.data[(k * continuous_dim + i) * continuous_dim + j] = val;
            }
        }
    }
    
    // Initialize discrete likelihoods
    std::vector<float> alphas = discrete_alphas;
    if (alphas.empty()) {
        alphas = std::vector<float>(discrete_dims.size(), 1e-4f);
    }
    
    for (int d = 0; d < (int)discrete_dims.size(); ++d) {
        float alpha = (d < (int)alphas.size()) ? alphas[d] : 1e-4f;
        int disc_dim = discrete_dims[d];
        
        for (int k = 0; k < num_components; ++k) {
            for (int c = 0; c < disc_dim; ++c) {
                int perm_idx = (k + c) % disc_dim;
                float val = alpha + (perm_idx == 0 ? 10.0f : 0.0f);
                state.disc_alphas[d].data[k * disc_dim + c] = val;
            }
        }
    }
    
    // Initialize prior
    for (int k = 0; k < num_components; ++k) {
        state.prior_alpha.data[k] = prior_alpha;
        state.posterior_alpha.data[k] = prior_alpha;
    }
    
    state.copyToDevice();
    
    return std::make_unique<HybridMixtureModel>(state);
}

} // namespace models
} // namespace axiom

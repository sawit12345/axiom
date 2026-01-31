/**
 * Copyright 2025 VERSES AI, Inc.
 */

#include "mixture.h"
#include <cmath>
#include <algorithm>

namespace axiom {
namespace models {

void MixtureState::allocate(int num_components_, 
                           const std::vector<int>& batch_shape_,
                           const std::vector<int>& event_shape_) {
    num_components = num_components_;
    batch_shape = batch_shape_;
    event_shape = event_shape_;
    event_dim = event_shape_.size();
    
    // Total size for likelihood params
    int total_size = num_components_;
    for (auto dim : event_shape_) {
        total_size *= dim;
    }
    
    likelihood_params.shape = {total_size};
    likelihood_params.allocate(likelihood_params.shape, true);
    
    prior_params.shape = {total_size};
    prior_params.allocate(prior_params.shape, true);
    
    pi_alpha.shape = {num_components_};
    pi_alpha.allocate(pi_alpha.shape, true);
    
    pi_prior_alpha.shape = {num_components_};
    pi_prior_alpha.allocate(pi_prior_alpha.shape, true);
    
    used_mask.shape = {num_components_};
    used_mask.allocate(used_mask.shape, true);
    
    pi_lr = 1.0f;
    pi_beta = 0.0f;
    likelihood_lr = 1.0f;
    likelihood_beta = 0.0f;
}

void MixtureState::copyToDevice() {
    likelihood_params.copyToDevice();
    prior_params.copyToDevice();
    pi_alpha.copyToDevice();
    pi_prior_alpha.copyToDevice();
    used_mask.copyToDevice();
}

void MixtureState::copyFromDevice() {
    likelihood_params.copyFromDevice();
    prior_params.copyFromDevice();
    pi_alpha.copyFromDevice();
    pi_prior_alpha.copyFromDevice();
    used_mask.copyFromDevice();
}

MixtureModel::MixtureModel(const MixtureState& state) : state_(state) {
    cudaStreamCreate(&stream_);
    
    log_probs_buffer_.shape = {1, state_.num_components};
    log_probs_buffer_.allocate(log_probs_buffer_.shape, true);
}

MixtureModel::~MixtureModel() {
    cudaStreamDestroy(stream_);
}

void MixtureModel::eStep(const Tensor& data, Tensor& out_posterior) {
    computeLogProbs(data, log_probs_buffer_);
    
    int batch_size = data.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        // Add prior log mean
        for (int k = 0; k < state_.num_components; ++k) {
            float log_mean = std::log(state_.pi_alpha.data[k]) - std::log(state_.num_components * 0.1f);
            log_probs_buffer_.data[b * state_.num_components + k] += log_mean;
        }
        
        // Softmax
        float max_logp = log_probs_buffer_.data[b * state_.num_components];
        for (int k = 1; k < state_.num_components; ++k) {
            max_logp = std::max(max_logp, log_probs_buffer_.data[b * state_.num_components + k]);
        }
        
        float sum_exp = 0.0f;
        for (int k = 0; k < state_.num_components; ++k) {
            sum_exp += std::exp(log_probs_buffer_.data[b * state_.num_components + k] - max_logp);
        }
        
        for (int k = 0; k < state_.num_components; ++k) {
            out_posterior.data[b * state_.num_components + k] = 
                std::exp(log_probs_buffer_.data[b * state_.num_components + k] - max_logp) / sum_exp;
        }
    }
    
    out_posterior.copyToDevice();
}

void MixtureModel::eStepFromProbs(const Tensor& inputs, Tensor& out_posterior) {
    computeAverageEnergy(inputs, log_probs_buffer_);
    
    int batch_size = inputs.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        // Add prior
        for (int k = 0; k < state_.num_components; ++k) {
            float log_mean = std::log(state_.pi_alpha.data[k]) - std::log(state_.num_components * 0.1f);
            log_probs_buffer_.data[b * state_.num_components + k] += log_mean;
        }
        
        // Softmax
        float max_logp = log_probs_buffer_.data[b * state_.num_components];
        for (int k = 1; k < state_.num_components; ++k) {
            max_logp = std::max(max_logp, log_probs_buffer_.data[b * state_.num_components + k]);
        }
        
        float sum_exp = 0.0f;
        for (int k = 0; k < state_.num_components; ++k) {
            sum_exp += std::exp(log_probs_buffer_.data[b * state_.num_components + k] - max_logp);
        }
        
        for (int k = 0; k < state_.num_components; ++k) {
            out_posterior.data[b * state_.num_components + k] = 
                std::exp(log_probs_buffer_.data[b * state_.num_components + k] - max_logp) / sum_exp;
        }
    }
    
    out_posterior.copyToDevice();
}

void MixtureModel::mStep(const Tensor& data,
                        const Tensor& posterior,
                        float pi_lr,
                        float pi_beta,
                        float likelihood_lr,
                        float likelihood_beta) {
    updatePi(posterior, pi_lr, pi_beta);
    updateLikelihood(data, posterior, likelihood_lr, likelihood_beta);
}

void MixtureModel::updateFromData(const Tensor& data, int iters, bool assign_unused) {
    for (int iter = 0; iter < iters; ++iter) {
        Tensor posterior;
        posterior.shape = {data.shape[0], state_.num_components};
        posterior.allocate(posterior.shape, true);
        
        eStep(data, posterior);
        
        if (assign_unused) {
            Tensor elbo_contrib;
            elbo_contrib.shape = {data.shape[0]};
            elbo_contrib.allocate(elbo_contrib.shape, true);
            
            assignUnused(elbo_contrib, posterior, 1.0f, 1.0f, posterior);
        }
        
        mStep(data, posterior, state_.pi_lr, state_.pi_beta, state_.likelihood_lr, state_.likelihood_beta);
    }
}

void MixtureModel::updateFromProbabilities(const Tensor& inputs, int iters, bool assign_unused) {
    for (int iter = 0; iter < iters; ++iter) {
        Tensor posterior;
        posterior.shape = {inputs.shape[0], state_.num_components};
        posterior.allocate(posterior.shape, true);
        
        eStepFromProbs(inputs, posterior);
        
        if (assign_unused) {
            Tensor elbo_contrib;
            elbo_contrib.shape = {inputs.shape[0]};
            elbo_contrib.allocate(elbo_contrib.shape, true);
            
            assignUnused(elbo_contrib, posterior, 1.0f, 1.0f, posterior);
        }
        
        mStep(inputs, posterior, state_.pi_lr, state_.pi_beta, state_.likelihood_lr, state_.likelihood_beta);
    }
}

float MixtureModel::computeELBO(const Tensor& data) {
    computeLogProbs(data, log_probs_buffer_);
    
    int batch_size = data.shape[0];
    float elbo = 0.0f;
    
    // Expected log-likelihood
    for (int b = 0; b < batch_size; ++b) {
        // Log-sum-exp
        float max_logp = log_probs_buffer_.data[b * state_.num_components];
        for (int k = 1; k < state_.num_components; ++k) {
            max_logp = std::max(max_logp, log_probs_buffer_.data[b * state_.num_components + k]);
        }
        
        float sum_exp = 0.0f;
        for (int k = 0; k < state_.num_components; ++k) {
            sum_exp += std::exp(log_probs_buffer_.data[b * state_.num_components + k] - max_logp);
        }
        
        elbo += max_logp + std::log(sum_exp);
    }
    
    // Subtract KL divergence (simplified)
    for (int k = 0; k < state_.num_components; ++k) {
        if (state_.pi_alpha.data[k] > state_.pi_prior_alpha.data[k]) {
            elbo -= (state_.pi_alpha.data[k] - state_.pi_prior_alpha.data[k]);
        }
    }
    
    return elbo;
}

void MixtureModel::getAssignments(const Tensor& data, bool hard, Tensor& out_assignments) {
    Tensor posterior;
    posterior.shape = {data.shape[0], state_.num_components};
    posterior.allocate(posterior.shape, true);
    
    eStep(data, posterior);
    
    if (hard) {
        // Argmax
        for (int b = 0; b < data.shape[0]; ++b) {
            int best_k = 0;
            float best_prob = posterior.data[b * state_.num_components];
            
            for (int k = 1; k < state_.num_components; ++k) {
                if (posterior.data[b * state_.num_components + k] > best_prob) {
                    best_prob = posterior.data[b * state_.num_components + k];
                    best_k = k;
                }
            }
            
            for (int k = 0; k < state_.num_components; ++k) {
                out_assignments.data[b * state_.num_components + k] = (k == best_k) ? 1.0f : 0.0f;
            }
        }
    } else {
        // Soft assignments (copy posterior)
        for (int i = 0; i < data.shape[0] * state_.num_components; ++i) {
            out_assignments.data[i] = posterior.data[i];
        }
    }
    
    out_assignments.copyToDevice();
}

void MixtureModel::assignUnused(const Tensor& elbo_contrib,
                               const Tensor& posterior,
                               float d_alpha_thr,
                               float fill_value,
                               Tensor& out_posterior) {
    // Find components with low d_alpha
    for (int k = 0; k < state_.num_components; ++k) {
        float d_alpha = state_.pi_alpha.data[k] - state_.pi_prior_alpha.data[k];
        
        if (d_alpha < d_alpha_thr) {
            // Find data point with worst ELL
            int worst_idx = 0;
            float worst_ell = elbo_contrib.data[0];
            
            for (int b = 1; b < elbo_contrib.shape[0]; ++b) {
                if (elbo_contrib.data[b] < worst_ell) {
                    worst_ell = elbo_contrib.data[b];
                    worst_idx = b;
                }
            }
            
            // Assign to this component
            for (int kk = 0; kk < state_.num_components; ++kk) {
                out_posterior.data[worst_idx * state_.num_components + kk] = (kk == k) ? fill_value : 0.0f;
            }
        }
    }
}

void MixtureModel::mergeClusters(int idx1, int idx2) {
    // Sum components
    state_.pi_alpha.data[idx1] += state_.pi_alpha.data[idx2];
    state_.pi_alpha.data[idx1] -= state_.pi_prior_alpha.data[idx2];
    
    // Reset second component
    state_.pi_alpha.data[idx2] = state_.pi_prior_alpha.data[idx2];
    state_.used_mask.data[idx2] = 0.0f;
    
    state_.pi_alpha.copyToDevice();
    state_.used_mask.copyToDevice();
}

bool MixtureModel::growComponent(const Tensor& data, float logp_threshold) {
    computeLogProbs(data, log_probs_buffer_);
    
    // Check if any data point is poorly explained
    int batch_size = data.shape[0];
    bool needs_growth = false;
    
    for (int b = 0; b < batch_size; ++b) {
        float max_logp = log_probs_buffer_.data[b * state_.num_components];
        for (int k = 1; k < state_.num_components; ++k) {
            max_logp = std::max(max_logp, log_probs_buffer_.data[b * state_.num_components + k]);
        }
        
        if (max_logp < logp_threshold) {
            needs_growth = true;
            break;
        }
    }
    
    if (!needs_growth) {
        return false;
    }
    
    // Find unused component
    for (int k = 0; k < state_.num_components; ++k) {
        if (state_.used_mask.data[k] < 0.5f) {
            // Initialize with first data point
            state_.used_mask.data[k] = 1.0f;
            state_.used_mask.copyToDevice();
            return true;
        }
    }
    
    return false;
}

void MixtureModel::predict(const Tensor& X, Tensor& out_mu, Tensor& out_sigma, Tensor& out_probs) {
    // Weighted prediction
    int batch_size = X.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        // Compute responsibilities
        Tensor posterior;
        posterior.shape = {state_.num_components};
        posterior.allocate(posterior.shape, true);
        
        for (int k = 0; k < state_.num_components; ++k) {
            posterior.data[k] = std::exp(log_probs_buffer_.data[b * state_.num_components + k]);
        }
        
        // Normalize
        float sum = 0.0f;
        for (int k = 0; k < state_.num_components; ++k) {
            sum += posterior.data[k];
        }
        
        for (int k = 0; k < state_.num_components; ++k) {
            out_probs.data[b * state_.num_components + k] = posterior.data[k] / sum;
        }
        
        // Weighted mean (simplified)
        for (int i = 0; i < out_mu.shape[1]; ++i) {
            float weighted_mean = 0.0f;
            for (int k = 0; k < state_.num_components; ++k) {
                weighted_mean += out_probs.data[b * state_.num_components + k] * state_.likelihood_params.data[k * out_mu.shape[1] + i];
            }
            out_mu.data[b * out_mu.shape[1] + i] = weighted_mean;
        }
    }
}

Tensor MixtureModel::getUsedMask() const {
    return state_.used_mask;
}

void MixtureModel::setUsedMask(const Tensor& mask) {
    state_.used_mask = mask;
    state_.used_mask.copyToDevice();
}

int MixtureModel::getNumUsed() const {
    int count = 0;
    for (int k = 0; k < state_.num_components; ++k) {
        if (state_.used_mask.data[k] > 0.5f) {
            count++;
        }
    }
    return count;
}

void MixtureModel::computeLogProbs(const Tensor& data, Tensor& out_log_probs) {
    int batch_size = data.shape[0];
    int data_dim = 1;
    for (int i = 1; i < data.ndim; ++i) {
        data_dim *= data.shape[i];
    }
    
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_components; ++k) {
            float log_prob = 0.0f;
            
            // Gaussian log-likelihood
            for (int i = 0; i < data_dim; ++i) {
                float diff = data.data[b * data_dim + i] - state_.likelihood_params.data[k * data_dim + i];
                log_prob += -0.5f * diff * diff;
            }
            
            out_log_probs.data[b * state_.num_components + k] = log_prob;
        }
    }
}

void MixtureModel::computeAverageEnergy(const Tensor& inputs, Tensor& out_energy) {
    // Similar to computeLogProbs but for distributions
    computeLogProbs(inputs, out_energy);
}

void MixtureModel::updatePi(const Tensor& posterior, float lr, float beta) {
    int batch_size = posterior.shape[0];
    
    for (int k = 0; k < state_.num_components; ++k) {
        float qz_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            qz_sum += posterior.data[b * state_.num_components + k];
        }
        
        float prior_alpha = state_.pi_prior_alpha.data[k];
        float old_alpha = state_.pi_alpha.data[k];
        float new_alpha = (1.0f - lr) * old_alpha + lr * (prior_alpha + qz_sum);
        state_.pi_alpha.data[k] = (1.0f - beta) * new_alpha + beta * old_alpha;
    }
    
    state_.pi_alpha.copyToDevice();
}

void MixtureModel::updateLikelihood(const Tensor& data,
                                   const Tensor& posterior,
                                   float lr,
                                   float beta) {
    int batch_size = data.shape[0];
    int data_dim = 1;
    for (int i = 1; i < data.ndim; ++i) {
        data_dim *= data.shape[i];
    }
    
    for (int k = 0; k < state_.num_components; ++k) {
        float qz_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            qz_sum += posterior.data[b * state_.num_components + k];
        }
        
        if (qz_sum < 1e-10f) continue;
        
        // Update mean
        for (int i = 0; i < data_dim; ++i) {
            float data_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                data_sum += posterior.data[b * state_.num_components + k] * data.data[b * data_dim + i];
            }
            
            float old_param = state_.likelihood_params.data[k * data_dim + i];
            float new_param = data_sum / qz_sum;
            state_.likelihood_params.data[k * data_dim + i] = (1.0f - lr) * old_param + lr * new_param;
        }
    }
    
    state_.likelihood_params.copyToDevice();
}

// Helper functions
void softmax(const Tensor& input, const std::vector<int>& dims, Tensor& out_output) {
    // Simplified softmax over last dimension
    int batch_size = 1;
    for (int i = 0; i < input.ndim - 1; ++i) {
        batch_size *= input.shape[i];
    }
    int dim_size = input.shape[input.ndim - 1];
    
    for (int b = 0; b < batch_size; ++b) {
        float max_val = input.data[b * dim_size];
        for (int i = 1; i < dim_size; ++i) {
            max_val = std::max(max_val, input.data[b * dim_size + i]);
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < dim_size; ++i) {
            sum_exp += std::exp(input.data[b * dim_size + i] - max_val);
        }
        
        for (int i = 0; i < dim_size; ++i) {
            out_output.data[b * dim_size + i] = std::exp(input.data[b * dim_size + i] - max_val) / sum_exp;
        }
    }
}

void logSumExp(const Tensor& input, const std::vector<int>& dims, Tensor& out_output) {
    // Log-sum-exp for numerical stability
    int batch_size = 1;
    for (int i = 0; i < input.ndim - 1; ++i) {
        batch_size *= input.shape[i];
    }
    int dim_size = input.shape[input.ndim - 1];
    
    for (int b = 0; b < batch_size; ++b) {
        float max_val = input.data[b * dim_size];
        for (int i = 1; i < dim_size; ++i) {
            max_val = std::max(max_val, input.data[b * dim_size + i]);
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < dim_size; ++i) {
            sum_exp += std::exp(input.data[b * dim_size + i] - max_val);
        }
        
        out_output.data[b] = max_val + std::log(sum_exp);
    }
}

// Factory functions
std::unique_ptr<MixtureModel> createGaussianMixture(int num_components,
                                                   int data_dim,
                                                   float scale,
                                                   float prior_alpha) {
    MixtureState state;
    state.allocate(num_components, {1}, {data_dim});
    
    // Initialize Gaussian parameters
    for (int k = 0; k < num_components; ++k) {
        state.pi_prior_alpha.data[k] = prior_alpha;
        state.pi_alpha.data[k] = prior_alpha;
        
        // Random initialization for means
        for (int i = 0; i < data_dim; ++i) {
            state.likelihood_params.data[k * data_dim + i] = scale * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        }
    }
    
    state.copyToDevice();
    
    return std::make_unique<MixtureModel>(state);
}

std::unique_ptr<MixtureModel> createMultinomialMixture(int num_components,
                                                      int num_categories,
                                                      float prior_alpha) {
    MixtureState state;
    state.allocate(num_components, {1}, {num_categories});
    
    for (int k = 0; k < num_components; ++k) {
        state.pi_prior_alpha.data[k] = prior_alpha;
        state.pi_alpha.data[k] = prior_alpha;
        
        // Uniform initialization
        for (int c = 0; c < num_categories; ++c) {
            state.likelihood_params.data[k * num_categories + c] = 1.0f / num_categories;
        }
    }
    
    state.copyToDevice();
    
    return std::make_unique<MixtureModel>(state);
}

} // namespace models
} // namespace axiom

/**
 * Copyright 2025 VERSES AI, Inc.
 */

#include "imm.h"
#include <cmath>

namespace axiom {
namespace models {

void IMMState::allocate(int num_object_types_, int num_features_, bool color_only) {
    num_object_types = num_object_types_;
    num_features = color_only ? 3 : num_features_;
    color_only_identity = color_only;
    
    mean.shape = {num_object_types, num_features, 1};
    mean.allocate(mean.shape, true);
    
    kappa.shape = {num_object_types, 1, 1};
    kappa.allocate(kappa.shape, true);
    
    u.shape = {num_object_types, num_features, num_features};
    u.allocate(u.shape, true);
    
    n.shape = {num_object_types, 1, 1};
    n.allocate(n.shape, true);
    
    prior_alpha.shape = {num_object_types};
    prior_alpha.allocate(prior_alpha.shape, true);
    
    posterior_alpha.shape = {num_object_types};
    posterior_alpha.allocate(posterior_alpha.shape, true);
    
    used_mask.shape = {num_object_types};
    used_mask.allocate(used_mask.shape, true);
    
    cont_scale = 0.5f;
    color_precision_scale = 1.0f;
}

void IMMState::copyToDevice() {
    mean.copyToDevice();
    kappa.copyToDevice();
    u.copyToDevice();
    n.copyToDevice();
    prior_alpha.copyToDevice();
    posterior_alpha.copyToDevice();
    used_mask.copyToDevice();
}

void IMMState::copyFromDevice() {
    mean.copyFromDevice();
    kappa.copyFromDevice();
    u.copyFromDevice();
    n.copyFromDevice();
    prior_alpha.copyFromDevice();
    posterior_alpha.copyFromDevice();
    used_mask.copyFromDevice();
}

IdentityMixtureModel::IdentityMixtureModel(const IMMState& state) : state_(state) {
    cudaStreamCreate(&stream_);
    
    ell_buffer_.shape = {1, state_.num_object_types};
    ell_buffer_.allocate(ell_buffer_.shape, true);
}

IdentityMixtureModel::~IdentityMixtureModel() {
    cudaStreamDestroy(stream_);
}

void IdentityMixtureModel::inferIdentity(const Tensor& x,
                                        bool color_only_identity,
                                        Tensor& out_class_labels) {
    // Scale color features
    Tensor x_scaled;
    x_scaled.shape = x.shape;
    x_scaled.allocate(x_scaled.shape, true);
    
    for (int b = 0; b < x.shape[0]; ++b) {
        for (int i = 0; i < state_.num_features; ++i) {
            float val = x.data[b * state_.num_features + i];
            // Scale color features (indices 2+)
            if (i >= 2) {
                val *= 100.0f;
            }
            x_scaled.data[b * state_.num_features + i] = val;
        }
    }
    
    // Compute ELL
    computeExpectedLogLikelihood(x_scaled, ell_buffer_);
    
    int batch_size = x.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        // Mask out unused
        for (int k = 0; k < state_.num_object_types; ++k) {
            if (state_.posterior_alpha.data[k] <= state_.prior_alpha.data[k]) {
                ell_buffer_.data[b * state_.num_object_types + k] = -1e10f;
            }
        }
        
        // Find best
        int best_k = 0;
        float best_ell = ell_buffer_.data[b * state_.num_object_types];
        
        for (int k = 1; k < state_.num_object_types; ++k) {
            if (ell_buffer_.data[b * state_.num_object_types + k] > best_ell) {
                best_ell = ell_buffer_.data[b * state_.num_object_types + k];
                best_k = k;
            }
        }
        
        out_class_labels.data[b] = static_cast<float>(best_k);
    }
}

void IdentityMixtureModel::inferRemappedColorIdentity(const Tensor& x,
                                                     bool color_only_identity,
                                                     float ell_threshold,
                                                     Tensor& out_qz) {
    // Scale color features
    Tensor x_scaled;
    x_scaled.shape = x.shape;
    x_scaled.allocate(x_scaled.shape, true);
    
    for (int b = 0; b < x.shape[0]; ++b) {
        for (int i = 0; i < state_.num_features; ++i) {
            float val = x.data[b * state_.num_features + i];
            if (i >= 2) {
                val *= 100.0f;
            }
            x_scaled.data[b * state_.num_features + i] = val;
        }
    }
    
    // Compute ELL with all features
    computeExpectedLogLikelihood(x_scaled, ell_buffer_);
    
    int batch_size = x.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        // Check if well explained
        float max_ell = -1e20f;
        for (int k = 0; k < state_.num_object_types; ++k) {
            if (state_.posterior_alpha.data[k] > state_.prior_alpha.data[k]) {
                max_ell = std::max(max_ell, ell_buffer_.data[b * state_.num_object_types + k]);
            }
        }
        
        if (max_ell > ell_threshold) {
            // Use full features
            float sum_exp = 0.0f;
            for (int k = 0; k < state_.num_object_types; ++k) {
                sum_exp += std::exp(ell_buffer_.data[b * state_.num_object_types + k]);
            }
            
            for (int k = 0; k < state_.num_object_types; ++k) {
                out_qz.data[b * state_.num_object_types + k] = 
                    std::exp(ell_buffer_.data[b * state_.num_object_types + k]) / sum_exp;
            }
        } else {
            // Use shape only (first 2 features)
            computeExpectedLogLikelihoodShapeOnly(x_scaled, ell_buffer_);
            
            float sum_exp = 0.0f;
            for (int k = 0; k < state_.num_object_types; ++k) {
                sum_exp += std::exp(100.0f * ell_buffer_.data[b * state_.num_object_types + k]);
            }
            
            for (int k = 0; k < state_.num_object_types; ++k) {
                out_qz.data[b * state_.num_object_types + k] = 
                    std::exp(100.0f * ell_buffer_.data[b * state_.num_object_types + k]) / sum_exp;
            }
        }
    }
}

void IdentityMixtureModel::update(const Tensor& x,
                                 float i_ell_threshold,
                                 bool color_only_identity,
                                 bool& grew_component) {
    trainStep(x, i_ell_threshold, temp_buffer_, grew_component);
}

void IdentityMixtureModel::trainStep(const Tensor& x,
                                    float logp_threshold,
                                    Tensor& out_qz,
                                    bool& grew_component) {
    // Scale features
    Tensor x_scaled;
    x_scaled.shape = x.shape;
    x_scaled.allocate(x_scaled.shape, true);
    
    for (int b = 0; b < x.shape[0]; ++b) {
        for (int i = 0; i < state_.num_features; ++i) {
            float val = x.data[b * state_.num_features + i];
            if (i >= 2) {
                val *= 100.0f;
            }
            x_scaled.data[b * state_.num_features + i] = val;
        }
    }
    
    // Compute ELL
    computeExpectedLogLikelihood(x_scaled, ell_buffer_);
    
    int batch_size = x.shape[0];
    
    // Mask out unused
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_object_types; ++k) {
            if (state_.posterior_alpha.data[k] <= state_.prior_alpha.data[k]) {
                ell_buffer_.data[b * state_.num_object_types + k] = -1e10f;
            }
        }
    }
    
    // Softmax
    for (int b = 0; b < batch_size; ++b) {
        float max_ell = ell_buffer_.data[b * state_.num_object_types];
        for (int k = 1; k < state_.num_object_types; ++k) {
            max_ell = std::max(max_ell, ell_buffer_.data[b * state_.num_object_types + k]);
        }
        
        float sum_exp = 0.0f;
        for (int k = 0; k < state_.num_object_types; ++k) {
            sum_exp += std::exp(ell_buffer_.data[b * state_.num_object_types + k] - max_ell);
        }
        
        for (int k = 0; k < state_.num_object_types; ++k) {
            out_qz.data[b * state_.num_object_types + k] = 
                std::exp(ell_buffer_.data[b * state_.num_object_types + k] - max_ell) / sum_exp;
        }
    }
    
    // Check if well explained
    float max_ell = -1e20f;
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_object_types; ++k) {
            max_ell = std::max(max_ell, ell_buffer_.data[b * state_.num_object_types + k]);
        }
    }
    
    if (max_ell > logp_threshold) {
        // Update parameters
        updateParameters(x_scaled, out_qz, 1.0f, 0.0f);
        grew_component = false;
    } else {
        // Find unused component
        int unused_idx = findFirstUnused();
        
        if (unused_idx >= 0) {
            // Assign to unused
            for (int k = 0; k < state_.num_object_types; ++k) {
                out_qz.data[k] = (k == unused_idx) ? 1.0f : 0.0f;
            }
            
            updateParameters(x_scaled, out_qz, 1.0f, 0.0f);
            state_.used_mask.data[unused_idx] = 1.0f;
            state_.used_mask.copyToDevice();
            grew_component = true;
        } else {
            updateParameters(x_scaled, out_qz, 1.0f, 0.0f);
            grew_component = false;
        }
    }
    
    // Update used mask
    for (int k = 0; k < state_.num_object_types; ++k) {
        state_.used_mask.data[k] = (state_.posterior_alpha.data[k] > state_.prior_alpha.data[k]) ? 1.0f : 0.0f;
    }
    state_.used_mask.copyToDevice();
}

Tensor IdentityMixtureModel::getUsedMask() const {
    return state_.used_mask;
}

void IdentityMixtureModel::setUsedMask(const Tensor& mask) {
    state_.used_mask = mask;
    state_.used_mask.copyToDevice();
}

void IdentityMixtureModel::getPosterior(const Tensor& x,
                                       Tensor& out_posterior) {
    computeExpectedLogLikelihood(x, ell_buffer_);
    
    int batch_size = x.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        float max_ell = ell_buffer_.data[b * state_.num_object_types];
        for (int k = 1; k < state_.num_object_types; ++k) {
            max_ell = std::max(max_ell, ell_buffer_.data[b * state_.num_object_types + k]);
        }
        
        float sum_exp = 0.0f;
        for (int k = 0; k < state_.num_object_types; ++k) {
            sum_exp += std::exp(ell_buffer_.data[b * state_.num_object_types + k] - max_ell);
        }
        
        for (int k = 0; k < state_.num_object_types; ++k) {
            out_posterior.data[b * state_.num_object_types + k] = 
                std::exp(ell_buffer_.data[b * state_.num_object_types + k] - max_ell) / sum_exp;
        }
    }
}

void IdentityMixtureModel::computeELL(const Tensor& x,
                                     Tensor& out_ell) {
    computeExpectedLogLikelihood(x, out_ell);
}

void IdentityMixtureModel::computeExpectedLogLikelihood(const Tensor& x, Tensor& out_ell) {
    int batch_size = x.shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_object_types; ++k) {
            float ell = 0.0f;
            
            float kappa = state_.kappa.data[k];
            float n = state_.n.data[k];
            
            // Expected log precision
            float expected_log_det = 0.0f;
            for (int i = 0; i < state_.num_features; ++i) {
                float u_ii = state_.u.data[(k * state_.num_features + i) * state_.num_features + i];
                expected_log_det += std::log(n / u_ii);
            }
            
            // Quadratic term
            float quad_term = 0.0f;
            for (int i = 0; i < state_.num_features; ++i) {
                float diff = x.data[b * state_.num_features + i] - state_.mean.data[(k * state_.num_features + i)];
                float u_ii = state_.u.data[(k * state_.num_features + i) * state_.num_features + i];
                float precision = n / u_ii;
                quad_term += -0.5f * precision * diff * diff;
            }
            
            // Assemble
            ell = quad_term + 0.5f * expected_log_det - 0.5f * state_.num_features / kappa;
            ell -= 0.5f * state_.num_features * std::log(2.0f * M_PI);
            
            out_ell.data[b * state_.num_object_types + k] = ell;
        }
    }
}

void IdentityMixtureModel::computeExpectedLogLikelihoodShapeOnly(const Tensor& x, Tensor& out_ell) {
    int batch_size = x.shape[0];
    int shape_dim = 2; // Only first 2 features
    
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < state_.num_object_types; ++k) {
            float ell = 0.0f;
            
            float kappa = state_.kappa.data[k];
            float n = state_.n.data[k];
            
            // Expected log precision (shape only)
            float expected_log_det = 0.0f;
            for (int i = 0; i < shape_dim; ++i) {
                float u_ii = state_.u.data[(k * state_.num_features + i) * state_.num_features + i];
                expected_log_det += std::log(n / u_ii);
            }
            
            // Quadratic term (shape only)
            float quad_term = 0.0f;
            for (int i = 0; i < shape_dim; ++i) {
                float diff = x.data[b * state_.num_features + i] - state_.mean.data[(k * state_.num_features + i)];
                float u_ii = state_.u.data[(k * state_.num_features + i) * state_.num_features + i];
                float precision = n / u_ii;
                quad_term += -0.5f * precision * diff * diff;
            }
            
            ell = quad_term + 0.5f * expected_log_det - 0.5f * shape_dim / kappa;
            ell -= 0.5f * shape_dim * std::log(2.0f * M_PI);
            
            out_ell.data[b * state_.num_object_types + k] = ell;
        }
    }
}

void IdentityMixtureModel::updateParameters(const Tensor& x, const Tensor& qz, float lr, float beta) {
    int batch_size = x.shape[0];
    
    for (int k = 0; k < state_.num_object_types; ++k) {
        float qz_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            qz_sum += qz.data[b * state_.num_object_types + k];
        }
        
        if (qz_sum < 1e-10f) continue;
        
        // Update alpha
        float prior_alpha = state_.prior_alpha.data[k];
        float old_alpha = state_.posterior_alpha.data[k];
        float new_alpha = (1.0f - lr) * old_alpha + lr * (prior_alpha + qz_sum);
        state_.posterior_alpha.data[k] = (1.0f - beta) * new_alpha + beta * old_alpha;
        
        // Update mean
        for (int i = 0; i < state_.num_features; ++i) {
            float data_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                data_sum += qz.data[b * state_.num_object_types + k] * x.data[b * state_.num_features + i];
            }
            
            float old_mean = state_.mean.data[k * state_.num_features + i];
            float new_mean = data_sum / qz_sum;
            state_.mean.data[k * state_.num_features + i] = (1.0f - lr) * old_mean + lr * new_mean;
        }
        
        // Update kappa
        float old_kappa = state_.kappa.data[k];
        float new_kappa = old_kappa + qz_sum;
        state_.kappa.data[k] = (1.0f - lr) * old_kappa + lr * new_kappa;
        
        // Update n
        float old_n = state_.n.data[k];
        float new_n = old_n + qz_sum;
        state_.n.data[k] = (1.0f - lr) * old_n + lr * new_n;
    }
    
    state_.posterior_alpha.copyToDevice();
    state_.mean.copyToDevice();
    state_.kappa.copyToDevice();
    state_.n.copyToDevice();
}

int IdentityMixtureModel::findFirstUnused() {
    for (int k = 0; k < state_.num_object_types; ++k) {
        if (state_.used_mask.data[k] < 0.5f) {
            return k;
        }
    }
    return -1;
}

void remapColors(const Tensor& source_colors,
                const Tensor& target_colors,
                const Tensor& image,
                Tensor& out_remapped) {
    // Simplified color remapping
    int H = image.shape[0];
    int W = image.shape[1];
    int C = image.shape[2];
    
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                out_remapped.data[(h * W + w) * C + c] = image.data[(h * W + w) * C + c];
            }
        }
    }
}

std::unique_ptr<IdentityMixtureModel> createIMM(int num_object_types,
                                               int num_features,
                                               float cont_scale_identity,
                                               float color_precision_scale,
                                               bool color_only_identity) {
    IMMState state;
    state.allocate(num_object_types, num_features, color_only_identity);
    
    int identity_features = color_only_identity ? 3 : num_features;
    
    // Initialize continuous likelihood
    for (int k = 0; k < num_object_types; ++k) {
        state.kappa.data[k] = 1e-4f;
        state.n.data[k] = identity_features + 2.0f;
        
        for (int i = 0; i < identity_features; ++i) {
            for (int j = 0; j < identity_features; ++j) {
                float val = (i == j) ? (cont_scale_identity * cont_scale_identity) : 0.0f;
                state.u.data[(k * identity_features + i) * identity_features + j] = val;
            }
        }
    }
    
    // Apply color precision scale
    if (color_precision_scale != 1.0f) {
        int spatial_dim = 2;
        for (int k = 0; k < num_object_types; ++k) {
            for (int i = spatial_dim; i < identity_features; ++i) {
                state.u.data[(k * identity_features + i) * identity_features + i] *= color_precision_scale;
            }
        }
    }
    
    // Initialize prior
    for (int k = 0; k < num_object_types; ++k) {
        state.prior_alpha.data[k] = 0.1f;
        state.posterior_alpha.data[k] = 0.1f;
    }
    
    state.copyToDevice();
    
    return std::make_unique<IdentityMixtureModel>(state);
}

} // namespace models
} // namespace axiom

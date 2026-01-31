/**
 * Copyright 2025 VERSES AI, Inc.
 */

#include "tmm.h"
#include <cmath>
#include <algorithm>

namespace axiom {
namespace models {

void TMMState::allocate(int n_total_components_, int state_dim_, bool use_bias_, bool use_velocity_) {
    n_total_components = n_total_components_;
    state_dim = state_dim_;
    full_state_dim = 2 * state_dim_;
    use_bias = use_bias_;
    use_velocity = use_velocity_;
    
    int x_dim = full_state_dim + (use_bias_ ? 1 : 0);
    
    transitions.shape = {n_total_components, full_state_dim, x_dim};
    transitions.allocate(transitions.shape, true);
    
    used_mask.shape = {n_total_components};
    used_mask.allocate(used_mask.shape, true);
    
    dt = 1.0f;
    vu = 0.05f;
    sigma_sqr = 2.0f;
    logp_threshold = -0.00001f;
    position_threshold = 0.15f;
    clip_value = 5e-4f;
}

void TMMState::copyToDevice() {
    transitions.copyToDevice();
    used_mask.copyToDevice();
}

void TMMState::copyFromDevice() {
    transitions.copyFromDevice();
    used_mask.copyFromDevice();
}

TransitionMixtureModel::TransitionMixtureModel(const TMMState& state) : state_(state) {
#ifdef USE_CUDA
    cudaStreamCreate(&stream_);
#endif
    
    logprobs_buffer_.shape = {state_.n_total_components};
    logprobs_buffer_.allocate(logprobs_buffer_.shape, true);
}

TransitionMixtureModel::~TransitionMixtureModel() {
#ifdef USE_CUDA
    cudaStreamDestroy(stream_);
#endif
}

void TransitionMixtureModel::forward(const Tensor& transitions,
                                    const Tensor& x,
                                    Tensor& out_next_state) {
    int K_max = transitions.shape[0];
    int state_dim = x.shape[0];
    
    for (int k = 0; k < K_max; ++k) {
        for (int i = 0; i < state_dim; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < state_dim; ++j) {
                sum += transitions.data[(k * state_dim + i) * (state_dim + 1) + j] * x.data[j];
            }
            if (state_.use_bias) {
                sum += transitions.data[(k * state_dim + i) * (state_dim + 1) + state_dim];
            }
            out_next_state.data[k * state_dim + i] = sum;
        }
    }
}

void TransitionMixtureModel::forwardSingle(const Tensor& transition,
                                          const Tensor& x,
                                          Tensor& out_next_state) {
    int state_dim = x.shape[0];
    
    for (int i = 0; i < state_dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < state_dim; ++j) {
            sum += transition.data[i * (state_dim + 1) + j] * x.data[j];
        }
        if (state_.use_bias) {
            sum += transition.data[i * (state_dim + 1) + state_dim];
        }
        out_next_state.data[i] = sum;
    }
}

void TransitionMixtureModel::computeLogProbs(const Tensor& x_prev,
                                            const Tensor& x_curr,
                                            float sigma_sqr,
                                            bool use_velocity,
                                            Tensor& out_logprobs) {
    int K_max = state_.n_total_components;
    int state_dim = state_.full_state_dim;
    
    // Compute predicted next states
    Tensor mu;
    mu.shape = {K_max, state_dim};
    mu.allocate(mu.shape, true);
    forward(state_.transitions, x_prev, mu);
    
    // Compute Gaussian log-likelihood
    for (int k = 0; k < K_max; ++k) {
        float squared_error = 0.0f;
        int eval_dim = use_velocity ? state_dim : state_dim / 2;
        
        for (int i = 0; i < eval_dim; ++i) {
            float diff = x_curr.data[i] - mu.data[k * state_dim + i];
            squared_error += diff * diff;
        }
        
        out_logprobs.data[k] = -0.5f * squared_error / sigma_sqr 
                              - 0.5f * eval_dim * std::log(2.0f * M_PI * sigma_sqr);
    }
}

void TransitionMixtureModel::computeLogProbsMasked(const Tensor& x_prev,
                                                  const Tensor& x_curr,
                                                  float sigma_sqr,
                                                  bool use_velocity,
                                                  Tensor& out_logprobs) {
    computeLogProbs(x_prev, x_curr, sigma_sqr, use_velocity, out_logprobs);
    
    // Mask out unused components
    for (int k = 0; k < state_.n_total_components; ++k) {
        if (state_.used_mask.data[k] < 0.5f) {
            out_logprobs.data[k] = -1e20f;
        }
    }
}

void TransitionMixtureModel::updateTransitions(const Tensor& x_prev,
                                              const Tensor& x_curr,
                                              float sigma_sqr,
                                              float logp_thr,
                                              float pos_thr,
                                              float dt,
                                              bool use_unused_counter,
                                              bool use_velocity,
                                              float clip_value) {
    // Compute log probabilities
    computeLogProbsMasked(x_prev, x_curr, sigma_sqr, use_velocity, logprobs_buffer_);
    
    // Find maximum
    float max_logp = logprobs_buffer_.data[0];
    for (int k = 1; k < state_.n_total_components; ++k) {
        max_logp = std::max(max_logp, logprobs_buffer_.data[k]);
    }
    
    // If max is below threshold, add new component
    if (max_logp < logp_thr) {
        addVelOrBiasComponent(x_prev, x_curr, pos_thr, dt, use_unused_counter, use_velocity, clip_value);
    }
    
    // Update used mask
    for (int k = 0; k < state_.n_total_components; ++k) {
        float sum = 0.0f;
        for (int i = 0; i < state_.full_state_dim; ++i) {
            for (int j = 0; j < state_.full_state_dim + (state_.use_bias ? 1 : 0); ++j) {
                sum += std::abs(state_.transitions.data[(k * state_.full_state_dim + i) * (state_.full_state_dim + 1) + j]);
            }
        }
        state_.used_mask.data[k] = (sum > 0.0f) ? 1.0f : 0.0f;
    }
    state_.used_mask.copyToDevice();
}

void TransitionMixtureModel::addComponent(const Tensor& new_transition) {
    int first_unused = findFirstUnused();
    
    if (first_unused >= 0) {
        for (int i = 0; i < state_.full_state_dim; ++i) {
            for (int j = 0; j < state_.full_state_dim + (state_.use_bias ? 1 : 0); ++j) {
                state_.transitions.data[(first_unused * state_.full_state_dim + i) * (state_.full_state_dim + 1) + j] = 
                    new_transition.data[i * (state_.full_state_dim + 1) + j];
            }
        }
        state_.used_mask.data[first_unused] = 1.0f;
        state_.transitions.copyToDevice();
        state_.used_mask.copyToDevice();
    }
}

void TransitionMixtureModel::createVelocityComponent(const Tensor& x_current,
                                                    const Tensor& x_next,
                                                    float dt,
                                                    bool use_unused_counter,
                                                    Tensor& out_component) {
    int state_dim = state_.state_dim;
    int x_dim = state_.full_state_dim + (state_.use_bias ? 1 : 0);
    
    // Initialize with default dynamics
    generateDefaultDynamicsComponent(state_dim, dt, state_.use_bias, out_component);
    
    // Compute velocity
    for (int i = 0; i < state_dim; ++i) {
        float vel = x_next.data[i] - x_current.data[i];
        float prev_vel = x_current.data[state_dim + i];
        float vel_bias = vel - prev_vel;
        
        if (state_.use_bias) {
            out_component.data[i * x_dim + state_.full_state_dim] = vel_bias;
            out_component.data[(state_dim + i) * x_dim + state_.full_state_dim] = vel_bias;
        }
    }
    
    // Zero out unused components
    if (use_unused_counter) {
        for (int j = 0; j < x_dim; ++j) {
            out_component.data[(state_dim - 1) * x_dim + j] = 0.0f;
            out_component.data[(state_.full_state_dim - 1) * x_dim + j] = 0.0f;
        }
    }
}

void TransitionMixtureModel::createBiasComponent(const Tensor& x,
                                                bool use_unused_counter,
                                                Tensor& out_component) {
    int x_dim = state_.full_state_dim + (state_.use_bias ? 1 : 0);
    
    // Initialize to zero
    for (int i = 0; i < state_.full_state_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            out_component.data[i * x_dim + j] = 0.0f;
        }
    }
    
    // Set bias to x
    if (state_.use_bias) {
        for (int i = 0; i < state_.full_state_dim; ++i) {
            out_component.data[i * x_dim + state_.full_state_dim] = x.data[i];
        }
    }
    
    // Zero out unused
    if (use_unused_counter) {
        for (int j = 0; j < x_dim; ++j) {
            out_component.data[(state_.state_dim - 1) * x_dim + j] = 0.0f;
            out_component.data[(state_.full_state_dim - 1) * x_dim + j] = 0.0f;
        }
    }
}

void TransitionMixtureModel::createPositionVelocityComponent(const Tensor& x_prev,
                                                            const Tensor& x_curr,
                                                            bool use_unused_counter,
                                                            Tensor& out_component) {
    int num_coords = state_.state_dim - (use_unused_counter ? 1 : 0);
    int x_dim = state_.full_state_dim + (state_.use_bias ? 1 : 0);
    
    // Initialize to zero
    for (int i = 0; i < state_.full_state_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            out_component.data[i * x_dim + j] = 0.0f;
        }
    }
    
    // Compute velocity
    for (int i = 0; i < num_coords; ++i) {
        float vel = x_curr.data[i] - x_prev.data[i];
        
        // Position identity
        out_component.data[i * x_dim + i] = 1.0f;
        
        // Bias for position
        if (state_.use_bias) {
            out_component.data[i * x_dim + state_.full_state_dim] = vel;
        }
    }
}

void TransitionMixtureModel::createPositionBiasComponent(const Tensor& x_prev,
                                                        const Tensor& x_curr,
                                                        Tensor& out_component) {
    int x_dim = state_.full_state_dim + (state_.use_bias ? 1 : 0);
    
    // Initialize to zero
    for (int i = 0; i < state_.full_state_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            out_component.data[i * x_dim + j] = 0.0f;
        }
    }
    
    // Position bias
    if (state_.use_bias) {
        for (int i = 0; i < 2 && i < state_.full_state_dim; ++i) {
            out_component.data[i * x_dim + state_.full_state_dim] = x_curr.data[i];
        }
    }
}

float TransitionMixtureModel::gaussianLogLikelihood(const Tensor& y,
                                                   const Tensor& mu,
                                                   float sigma_sqr) {
    float squared_error = 0.0f;
    int dim = y.shape[0];
    
    for (int i = 0; i < dim; ++i) {
        float diff = y.data[i] - mu.data[i];
        squared_error += diff * diff;
    }
    
    return -0.5f * squared_error / sigma_sqr - 0.5f * dim * std::log(2.0f * M_PI * sigma_sqr);
}

int TransitionMixtureModel::getBestTransition(const Tensor& x_prev,
                                             const Tensor& x_curr) {
    computeLogProbsMasked(x_prev, x_curr, state_.sigma_sqr, state_.use_velocity, logprobs_buffer_);
    
    int best_idx = 0;
    float max_logp = logprobs_buffer_.data[0];
    
    for (int k = 1; k < state_.n_total_components; ++k) {
        if (logprobs_buffer_.data[k] > max_logp) {
            max_logp = logprobs_buffer_.data[k];
            best_idx = k;
        }
    }
    
    return best_idx;
}

void TransitionMixtureModel::getTransition(int idx, Tensor& out_transition) {
    int x_dim = state_.full_state_dim + (state_.use_bias ? 1 : 0);
    
    for (int i = 0; i < state_.full_state_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            out_transition.data[i * x_dim + j] = 
                state_.transitions.data[(idx * state_.full_state_dim + i) * x_dim + j];
        }
    }
}

void TransitionMixtureModel::setTransition(int idx, const Tensor& transition) {
    int x_dim = state_.full_state_dim + (state_.use_bias ? 1 : 0);
    
    for (int i = 0; i < state_.full_state_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            state_.transitions.data[(idx * state_.full_state_dim + i) * x_dim + j] = 
                transition.data[i * x_dim + j];
        }
    }
    
    state_.used_mask.data[idx] = 1.0f;
    state_.transitions.copyToDevice();
    state_.used_mask.copyToDevice();
}

Tensor TransitionMixtureModel::getUsedMask() const {
    return state_.used_mask;
}

int TransitionMixtureModel::getNumUsed() const {
    int count = 0;
    for (int k = 0; k < state_.n_total_components; ++k) {
        if (state_.used_mask.data[k] > 0.5f) {
            count++;
        }
    }
    return count;
}

void TransitionMixtureModel::updateModel(const Tensor& x_prev,
                                        const Tensor& x_curr,
                                        float sigma_sqr,
                                        float logp_threshold,
                                        float position_threshold,
                                        float dt,
                                        bool use_unused_counter,
                                        bool use_velocity,
                                        float clip_value) {
    updateTransitions(x_prev, x_curr, sigma_sqr, logp_threshold, position_threshold, 
                     dt, use_unused_counter, use_velocity, clip_value);
}

int TransitionMixtureModel::findFirstUnused() {
    for (int k = 0; k < state_.n_total_components; ++k) {
        if (state_.used_mask.data[k] < 0.5f) {
            return k;
        }
    }
    return -1;
}

void TransitionMixtureModel::addVelOrBiasComponent(const Tensor& x_prev,
                                                  const Tensor& x_curr,
                                                  float pos_thr,
                                                  float dt,
                                                  bool use_unused_counter,
                                                  bool use_velocity,
                                                  float clip_value) {
    // Check if teleport (position change > threshold)
    float pos_diff = 0.0f;
    for (int i = 0; i < state_.state_dim; ++i) {
        float diff = x_curr.data[i] - x_prev.data[i];
        pos_diff += diff * diff;
    }
    pos_diff = std::sqrt(pos_diff);
    
    Tensor new_component;
    int x_dim = state_.full_state_dim + (state_.use_bias ? 1 : 0);
    new_component.shape = {state_.full_state_dim, x_dim};
    new_component.allocate(new_component.shape, true);
    
    if (pos_diff > pos_thr) {
        // Create bias component
        createBiasComponent(x_curr, use_unused_counter, new_component);
    } else {
        // Create velocity component
        createVelocityComponent(x_prev, x_curr, dt, use_unused_counter, new_component);
    }
    
    // Clip small values
    for (int i = 0; i < state_.full_state_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            if (std::abs(new_component.data[i * x_dim + j]) < clip_value) {
                new_component.data[i * x_dim + j] = 0.0f;
            }
        }
    }
    
    addComponent(new_component);
}

// Default component generators
void generateDefaultDynamicsComponent(int state_dim,
                                     float dt,
                                     bool use_bias,
                                     Tensor& out_component) {
    int full_dim = 2 * state_dim;
    int x_dim = full_dim + (use_bias ? 1 : 0);
    
    // Initialize to zero
    for (int i = 0; i < full_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            out_component.data[i * x_dim + j] = 0.0f;
        }
    }
    
    // Position identity and velocity coupling
    for (int i = 0; i < state_dim; ++i) {
        out_component.data[i * x_dim + i] = 1.0f;
        out_component.data[(state_dim + i) * x_dim + (state_dim + i)] = 1.0f;
        out_component.data[i * x_dim + (state_dim + i)] = dt;
    }
}

void generateDefaultKeepUnusedComponent(int state_dim,
                                       float dt,
                                       float vu,
                                       bool use_bias,
                                       Tensor& out_component) {
    int full_dim = 2 * state_dim;
    int x_dim = full_dim + (use_bias ? 1 : 0);
    
    for (int i = 0; i < full_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            out_component.data[i * x_dim + j] = 0.0f;
        }
    }
    
    // Keep position
    for (int i = 0; i < state_dim; ++i) {
        out_component.data[i * x_dim + i] = 1.0f;
    }
    
    // Keep velocity (except unused)
    for (int i = 0; i < state_dim - 1; ++i) {
        out_component.data[(state_dim + i) * x_dim + (state_dim + i)] = 1.0f;
    }
    
    // Bias for unused counter
    if (use_bias) {
        out_component.data[(state_dim - 1) * x_dim + full_dim] = dt * vu;
    }
}

void generateDefaultBecomeUnusedComponent(int state_dim,
                                         float dt,
                                         float vu,
                                         bool use_bias,
                                         Tensor& out_component) {
    int full_dim = 2 * state_dim;
    int x_dim = full_dim + (use_bias ? 1 : 0);
    
    for (int i = 0; i < full_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            out_component.data[i * x_dim + j] = 0.0f;
        }
    }
    
    // Keep position (except last)
    for (int i = 0; i < state_dim - 1; ++i) {
        out_component.data[i * x_dim + i] = 1.0f;
    }
    
    // Bias for unused counter
    if (use_bias) {
        out_component.data[(state_dim - 1) * x_dim + full_dim] = dt * vu;
    }
}

void generateDefaultStopComponent(int state_dim,
                                 bool use_bias,
                                 Tensor& out_component) {
    int full_dim = 2 * state_dim;
    int x_dim = full_dim + (use_bias ? 1 : 0);
    
    for (int i = 0; i < full_dim; ++i) {
        for (int j = 0; j < x_dim; ++j) {
            out_component.data[i * x_dim + j] = 0.0f;
        }
    }
    
    // Position identity for first 2 coordinates
    for (int i = 0; i < 2 && i < full_dim; ++i) {
        out_component.data[i * x_dim + i] = 1.0f;
    }
}

std::unique_ptr<TransitionMixtureModel> createTMM(int n_total_components,
                                                 int state_dim,
                                                 float dt,
                                                 float vu,
                                                 bool use_bias,
                                                 bool use_velocity) {
    TMMState state;
    state.allocate(n_total_components, state_dim, use_bias, use_velocity);
    
    // Initialize default components
    int x_dim = 2 * state_dim + (use_bias ? 1 : 0);
    
    if (use_velocity) {
        // Component 0: Default dynamics
        Tensor comp;
        comp.shape = {2 * state_dim, x_dim};
        comp.allocate(comp.shape, true);
        generateDefaultDynamicsComponent(state_dim, dt, use_bias, comp);
        
        for (int i = 0; i < 2 * state_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                state.transitions.data[i * x_dim + j] = comp.data[i * x_dim + j];
            }
        }
        state.used_mask.data[0] = 1.0f;
        
        // Component 1: Keep unused
        generateDefaultKeepUnusedComponent(state_dim, dt, vu, use_bias, comp);
        for (int i = 0; i < 2 * state_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                state.transitions.data[(1 * 2 * state_dim + i) * x_dim + j] = comp.data[i * x_dim + j];
            }
        }
        state.used_mask.data[1] = 1.0f;
        
        // Component 2: Become unused
        generateDefaultBecomeUnusedComponent(state_dim, dt, vu, use_bias, comp);
        for (int i = 0; i < 2 * state_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                state.transitions.data[(2 * 2 * state_dim + i) * x_dim + j] = comp.data[i * x_dim + j];
            }
        }
        state.used_mask.data[2] = 1.0f;
        
        // Component 3: Stop
        generateDefaultStopComponent(state_dim, use_bias, comp);
        for (int i = 0; i < 2 * state_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                state.transitions.data[(3 * 2 * state_dim + i) * x_dim + j] = comp.data[i * x_dim + j];
            }
        }
        state.used_mask.data[3] = 1.0f;
    } else {
        // Just keep unused and become unused
        Tensor comp;
        comp.shape = {2 * state_dim, x_dim};
        comp.allocate(comp.shape, true);
        
        generateDefaultKeepUnusedComponent(state_dim, dt, vu, use_bias, comp);
        for (int i = 0; i < 2 * state_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                state.transitions.data[i * x_dim + j] = comp.data[i * x_dim + j];
            }
        }
        state.used_mask.data[0] = 1.0f;
        
        generateDefaultBecomeUnusedComponent(state_dim, dt, vu, use_bias, comp);
        for (int i = 0; i < 2 * state_dim; ++i) {
            for (int j = 0; j < x_dim; ++j) {
                state.transitions.data[(1 * 2 * state_dim + i) * x_dim + j] = comp.data[i * x_dim + j];
            }
        }
        state.used_mask.data[1] = 1.0f;
    }
    
    state.copyToDevice();
    
    return std::make_unique<TransitionMixtureModel>(state);
}

} // namespace models
} // namespace axiom

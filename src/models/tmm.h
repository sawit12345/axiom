/**
 * Copyright 2025 VERSES AI, Inc.
 *
 * Licensed under the VERSES Academic Research License (the "License");
 * you may not use this file except in compliance with the license.
 */

#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "utils.h"

namespace axiom {
namespace models {

// Transition Mixture Model State
struct TMMState {
    // Transition matrices (K_max, 2*state_dim, 2*state_dim + use_bias)
    Tensor transitions;
    
    // Masks
    Tensor used_mask;           // (K_max,)
    
    // Configuration
    int n_total_components;
    int state_dim;
    int full_state_dim;         // 2*state_dim
    float dt;
    float vu;                   // Velocity uncertainty
    bool use_bias;
    bool use_velocity;
    float sigma_sqr;            // Likelihood variance
    float logp_threshold;
    float position_threshold;
    float clip_value;
    
    void allocate(int n_total_components_, int state_dim_, bool use_bias_, bool use_velocity_);
    void copyToDevice();
    void copyFromDevice();
};

// Transition Mixture Model
class TransitionMixtureModel {
public:
    TransitionMixtureModel(const TMMState& state);
    ~TransitionMixtureModel();
    
    // Forward dynamics: compute next state prediction
    void forward(const Tensor& transitions,     // (K_max, 2*D, 2*D+1)
                const Tensor& x,               // (2*D,)
                Tensor& out_next_state);        // (K_max, 2*D)
    
    // Single component forward
    void forwardSingle(const Tensor& transition,    // (2*D, 2*D+1)
                      const Tensor& x,             // (2*D,)
                      Tensor& out_next_state);      // (2*D,)
    
    // Compute log probabilities for all transitions
    void computeLogProbs(const Tensor& x_prev,      // (2*state_dim,)
                        const Tensor& x_curr,      // (2*state_dim,)
                        float sigma_sqr,
                        bool use_velocity,
                        Tensor& out_logprobs);      // (K_max,)
    
    // Compute log probabilities with mask
    void computeLogProbsMasked(const Tensor& x_prev,
                              const Tensor& x_curr,
                              float sigma_sqr,
                              bool use_velocity,
                              Tensor& out_logprobs);  // -inf for unused
    
    // Update transitions based on observation pair
    void updateTransitions(const Tensor& x_prev,
                          const Tensor& x_curr,
                          float sigma_sqr,
                          float logp_thr,
                          float pos_thr,
                          float dt,
                          bool use_unused_counter,
                          bool use_velocity,
                          float clip_value);
    
    // Add new component
    void addComponent(const Tensor& new_transition);
    
    // Create velocity component from observation pair
    void createVelocityComponent(const Tensor& x_current,
                                const Tensor& x_next,
                                float dt,
                                bool use_unused_counter,
                                Tensor& out_component);
    
    // Create bias/teleport component
    void createBiasComponent(const Tensor& x,
                            bool use_unused_counter,
                            Tensor& out_component);
    
    // Create position-velocity component
    void createPositionVelocityComponent(const Tensor& x_prev,
                                        const Tensor& x_curr,
                                        bool use_unused_counter,
                                        Tensor& out_component);
    
    // Create position bias component
    void createPositionBiasComponent(const Tensor& x_prev,
                                    const Tensor& x_curr,
                                    Tensor& out_component);
    
    // Gaussian log-likelihood computation
    float gaussianLogLikelihood(const Tensor& y,
                               const Tensor& mu,
                               float sigma_sqr);
    
    // Get best transition for state pair
    int getBestTransition(const Tensor& x_prev,
                         const Tensor& x_curr);
    
    // Get transition by index
    void getTransition(int idx, Tensor& out_transition);
    
    // Set transition at index
    void setTransition(int idx, const Tensor& transition);
    
    // Get used mask
    Tensor getUsedMask() const;
    
    // Get number of used components
    int getNumUsed() const;
    
    // Update the model with a new observation
    void updateModel(const Tensor& x_prev,
                    const Tensor& x_curr,
                    float sigma_sqr = 2.0f,
                    float logp_threshold = -0.00001f,
                    float position_threshold = 0.15f,
                    float dt = 1.0f,
                    bool use_unused_counter = true,
                    bool use_velocity = true,
                    float clip_value = 5e-4f);

private:
    TMMState state_;
    cudaStream_t stream_;
    
    // Internal buffers
    Tensor temp_buffer_;
    Tensor logprobs_buffer_;
    
    int findFirstUnused();
    void addVelOrBiasComponent(const Tensor& x_prev,
                              const Tensor& x_curr,
                              float pos_thr,
                              float dt,
                              bool use_unused_counter,
                              bool use_velocity,
                              float clip_value);
};

// Default dynamics component generators
void generateDefaultDynamicsComponent(int state_dim,
                                     float dt,
                                     bool use_bias,
                                     Tensor& out_component);

void generateDefaultKeepUnusedComponent(int state_dim,
                                       float dt,
                                       float vu,
                                       bool use_bias,
                                       Tensor& out_component);

void generateDefaultBecomeUnusedComponent(int state_dim,
                                         float dt,
                                         float vu,
                                         bool use_bias,
                                         Tensor& out_component);

void generateDefaultStopComponent(int state_dim,
                                 bool use_bias,
                                 Tensor& out_component);

// Factory function
std::unique_ptr<TransitionMixtureModel> createTMM(int n_total_components,
                                                 int state_dim,
                                                 float dt = 1.0f,
                                                 float vu = 0.1f,
                                                 bool use_bias = true,
                                                 bool use_velocity = true);

} // namespace models
} // namespace axiom

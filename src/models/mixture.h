/**
 * Copyright 2025 VERSES AI, Inc.
 *
 * Licensed under the VERSES Academic Research License (the "License");
 * you may not use this file except in compliance with the license.
 */

#pragma once

#include <vector>
#include <memory>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
// Stub for cudaStream_t when CUDA is not available
typedef void* cudaStream_t;
#endif

#include "utils.h"

namespace axiom {
namespace models {

// Generic Mixture Model State
struct MixtureState {
    // Likelihood parameters
    Tensor likelihood_params;
    Tensor prior_params;
    
    // Mixture prior (Dirichlet)
    Tensor pi_alpha;        // (num_components,)
    Tensor pi_prior_alpha;  // (num_components,)
    
    // Masks
    Tensor used_mask;       // (num_components,)
    
    // Configuration
    int num_components;
    int event_dim;
    std::vector<int> batch_shape;
    std::vector<int> event_shape;
    
    // Optimization parameters
    float pi_lr;
    float pi_beta;
    float likelihood_lr;
    float likelihood_beta;
    
    void allocate(int num_components_, 
                 const std::vector<int>& batch_shape_,
                 const std::vector<int>& event_shape_);
    void copyToDevice();
    void copyFromDevice();
};

// Generic Mixture Model
class MixtureModel {
public:
    MixtureModel(const MixtureState& state);
    ~MixtureModel();
    
    // E-step: Compute posterior assignments
    void eStep(const Tensor& data,           // (..., event_shape)
              Tensor& out_posterior);        // (..., num_components)
    
    // E-step with probability inputs
    void eStepFromProbs(const Tensor& inputs,
                       Tensor& out_posterior);
    
    // M-step: Update parameters
    void mStep(const Tensor& data,
              const Tensor& posterior,
              float pi_lr = 1.0f,
              float pi_beta = 0.0f,
              float likelihood_lr = 1.0f,
              float likelihood_beta = 0.0f);
    
    // Combined EM update
    void updateFromData(const Tensor& data,
                       int iters = 1,
                       bool assign_unused = false);
    
    // Update from probability distributions
    void updateFromProbabilities(const Tensor& inputs,
                                int iters = 1,
                                bool assign_unused = false);
    
    // Compute ELBO
    float computeELBO(const Tensor& data);
    
    // Get assignments (hard or soft)
    void getAssignments(const Tensor& data,
                       bool hard,
                       Tensor& out_assignments);
    
    // Assign unused components
    void assignUnused(const Tensor& elbo_contrib,
                     const Tensor& posterior,
                     float d_alpha_thr,
                     float fill_value,
                     Tensor& out_posterior);
    
    // Merge two components
    void mergeClusters(int idx1, int idx2);
    
    // Grow model by adding new component
    bool growComponent(const Tensor& data,
                      float logp_threshold);
    
    // Predict from data
    void predict(const Tensor& X,
                Tensor& out_mu,
                Tensor& out_sigma,
                Tensor& out_probs);
    
    // Get used mask
    Tensor getUsedMask() const;
    
    // Set used mask
    void setUsedMask(const Tensor& mask);
    
    // Get number of used components
    int getNumUsed() const;

private:
    MixtureState state_;
    cudaStream_t stream_;
    
    // Internal buffers
    Tensor log_probs_buffer_;
    Tensor temp_buffer_;
    
    void computeLogProbs(const Tensor& data, Tensor& out_log_probs);
    void computeAverageEnergy(const Tensor& inputs, Tensor& out_energy);
    void updatePi(const Tensor& posterior, float lr, float beta);
    void updateLikelihood(const Tensor& data, const Tensor& posterior, 
                         float lr, float beta);
    std::vector<int> getSampleDims(const Tensor& data);
    Tensor expandToCategoricalDims(const Tensor& data);
};

// Softmax helper
void softmax(const Tensor& input,
            const std::vector<int>& dims,
            Tensor& out_output);

// Log-sum-exp helper
void logSumExp(const Tensor& input,
              const std::vector<int>& dims,
              Tensor& out_output);

// Factory function for Gaussian Mixture
std::unique_ptr<MixtureModel> createGaussianMixture(int num_components,
                                                   int data_dim,
                                                   float scale = 1.0f,
                                                   float prior_alpha = 0.5f);

// Factory function for Multinomial Mixture
std::unique_ptr<MixtureModel> createMultinomialMixture(int num_components,
                                                      int num_categories,
                                                      float prior_alpha = 0.1f);

} // namespace models
} // namespace axiom

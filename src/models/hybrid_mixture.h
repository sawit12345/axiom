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

// Hybrid Mixture Model State (continuous + discrete)
struct HybridMixtureStateHM {
    // Continuous likelihood parameters (Gaussian)
    Tensor cont_mean;       // (num_components, cont_dim, 1)
    Tensor cont_precision;  // (num_components, cont_dim, cont_dim)
    Tensor cont_kappa;      // (num_components, 1, 1)
    Tensor cont_dof;        // (num_components, 1, 1)
    
    // Discrete likelihood parameters (Dirichlet-Multinomial)
    std::vector<Tensor> disc_alphas;  // Each: (num_components, disc_dim, 1)
    std::vector<int> disc_dims;
    
    // Prior
    Tensor prior_alpha;     // (num_components,)
    Tensor posterior_alpha; // (num_components,)
    
    // Masks
    Tensor used_mask;       // (num_components,)
    
    // Configuration
    int num_components;
    int cont_dim;
    std::vector<int> discrete_dims;
    int num_discrete;
    
    // Optimization
    float lr;
    float beta;
    
    void allocate(int num_components_, int cont_dim_,
                 const std::vector<int>& discrete_dims_);
    void copyToDevice();
    void copyFromDevice();
};

// Hybrid Mixture Model (continuous + discrete observations)
class HybridMixtureModel {
public:
    HybridMixtureModel(const HybridMixtureStateHM& state);
    ~HybridMixtureModel();
    
    // Combined training step (E-step + M-step)
    void trainingStep(const Tensor& c_data,           // (batch, cont_dim, 1)
                     const std::vector<Tensor>& d_data,  // Discrete features
                     float logp_threshold,
                     Tensor& out_qz,                   // Posterior assignments
                     bool& grew_component);
    
    // E-step for hybrid data
    void eStep(const Tensor& c_data,
              const std::vector<Tensor>& d_data,
              const std::vector<float>& w_disc,   // Weights for discrete
              Tensor& out_posterior,
              Tensor& out_c_ell,
              Tensor& out_d_ell);
    
    // M-step for hybrid data
    void mStep(const Tensor& c_data,
              const std::vector<Tensor>& d_data,
              const Tensor& qz,
              float lr = 1.0f,
              float beta = 0.0f);
    
    // M-step with unused component preservation
    void mStepKeepUnused(const Tensor& c_data,
                        const std::vector<Tensor>& d_data,
                        const Tensor& qz);
    
    // Sum components (for temporal consistency)
    void sumComponents(const HybridMixtureStateHM& other,
                      const Tensor& self_mask,
                      const Tensor& other_mask);
    
    // Combine parameters from two models
    void combineParams(const HybridMixtureStateHM& other,
                      const Tensor& select_mask);
    
    // Merge two clusters
    void mergeClusters(int idx1, int idx2);
    
    // Sample from the model
    void sample(cudaStream_t stream,
               int n_samples,
               Tensor& out_c_sample,
               std::vector<Tensor>& out_d_samples);
    
    // Get means as data
    void getMeansAsData(Tensor& out_c_means,
                       std::vector<Tensor>& out_d_means);
    
    // Compute combined ELBO
    float computeELBO(const Tensor& c_data,
                     const std::vector<Tensor>& d_data);
    
    // Compute continuous-only ELBO
    float computeContinuousELBO(const Tensor& c_data);
    
    // Compute discrete-only ELBO
    float computeDiscreteELBO(const std::vector<Tensor>& d_data);
    
    // Get used mask
    Tensor getUsedMask() const;
    
    // Set used mask
    void setUsedMask(const Tensor& mask);
    
    // Get number of used components
    int getNumUsed() const;

private:
    HybridMixtureStateHM state_;
    cudaStream_t stream_;
    
    // Internal buffers
    std::vector<Tensor> disc_buffers_;
    Tensor temp_buffer_;
    
    void computeContinuousLogLikelihood(const Tensor& c_data, Tensor& out_ell);
    void computeDiscreteLogLikelihood(const std::vector<Tensor>& d_data,
                                     const std::vector<float>& w_disc,
                                     Tensor& out_ell);
    void updatePrior(const Tensor& qz, float lr, float beta);
    void updateContinuousLikelihood(const Tensor& c_data, const Tensor& qz,
                                   float lr, float beta);
    void updateDiscreteLikelihood(const std::vector<Tensor>& d_data,
                                 const Tensor& qz,
                                 float lr, float beta);
    int findFirstUnused();
};

// Factory function
std::unique_ptr<HybridMixtureModel> createHybridMixture(
    int num_components,
    int continuous_dim,
    const std::vector<int>& discrete_dims,
    float cont_scale = 1.0f,
    const std::vector<float>& discrete_alphas = {},
    float prior_alpha = 0.1f);

} // namespace models
} // namespace axiom

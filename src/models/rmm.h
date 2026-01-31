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

// Hybrid Mixture Model with continuous and discrete components
struct HybridMixtureState {
    // Continuous likelihood (Gaussian)
    Tensor cont_mean;           // (num_components, cont_dim, 1)
    Tensor cont_kappa;          // (num_components, 1, 1)
    Tensor cont_u;              // (num_components, cont_dim, cont_dim)
    Tensor cont_n;              // (num_components, 1, 1)
    
    // Discrete likelihoods (Dirichlet-Multinomial)
    std::vector<Tensor> disc_alpha;     // Each: (num_components, disc_dim, 1)
    std::vector<int> disc_dims;
    
    // Prior on mixture components
    Tensor prior_alpha;         // (num_components,)
    Tensor posterior_alpha;     // (num_components,)
    
    // Masks
    Tensor used_mask;           // (num_components,)
    Tensor dirty_mask;          // (num_components,)
    
    // Configuration
    int num_components;
    int cont_dim;
    std::vector<int> discrete_dims;
    int num_discrete;
    
    // Optimization parameters
    float lr;
    float beta;
    
    void allocate(int num_components_, int cont_dim_, 
                  const std::vector<int>& discrete_dims_);
    void copyToDevice();
    void copyFromDevice();
};

// Reward Mixture Model
class RewardMixtureModel {
public:
    RewardMixtureModel(const HybridMixtureState& state);
    ~RewardMixtureModel();
    
    // E-step for hybrid observations
    void eStep(const Tensor& c_data,           // (batch, cont_dim, 1)
               const std::vector<Tensor>& d_data,  // Each: (batch, disc_dim, 1)
               const std::vector<float>& w_disc,   // Weights for discrete
               Tensor& out_qz,
               Tensor& out_c_ell,
               Tensor& out_d_ell);
    
    // M-step with unused component handling
    void mStep(const Tensor& c_data,
               const std::vector<Tensor>& d_data,
               const Tensor& qz,
               float lr = 1.0f,
               float beta = 0.0f);
    
    // M-step that preserves unused components
    void mStepKeepUnused(const Tensor& c_data,
                        const std::vector<Tensor>& d_data,
                        const Tensor& qz);
    
    // Prediction and rollout
    void predict(const Tensor& c_sample,
                const std::vector<Tensor>& d_sample,
                const std::vector<float>& w_disc,
                int& out_tmm_slot,
                float& out_reward,
                Tensor& out_elogp,
                Tensor& out_qz,
                int& out_mix_slot);
    
    // Sample from the model
    void sample(cudaStream_t stream,
               int n_samples,
               Tensor& out_c_sample,
               std::vector<Tensor>& out_d_samples);
    
    // Get means as data format
    void getMeansAsData(Tensor& out_c_means,
                       std::vector<Tensor>& out_d_means);
    
    // Mark component as dirty (incorrect prediction)
    void markDirty(const Tensor& elogp,
                  const std::vector<Tensor>& d_data,
                  float threshold);
    
    // Compute ELBO
    float computeELBO(const Tensor& c_data,
                     const std::vector<Tensor>& d_data,
                     const std::vector<float>& w_disc);
    
    // BMR: Bayesian Model Reduction - merge similar components
    void mergeClusters(int idx1, int idx2);
    
    // Run BMR to find and merge similar components
    void runBMR(int n_samples, float elbo_threshold);
    
    // Training step with automatic component growing
    void trainStep(const Tensor& c_sample,
                  const std::vector<Tensor>& d_sample,
                  float logp_threshold,
                  Tensor& out_qz,
                  bool& grew_component);
    
    // Get used mask
    Tensor getUsedMask() const;
    
    // Set used mask
    void setUsedMask(const Tensor& mask);

private:
    HybridMixtureState state_;
    cudaStream_t stream_;
    
    // Internal buffers
    std::vector<Tensor> disc_ell_buffers_;
    Tensor temp_buffer_;
    
    void computeContinuousELL(const Tensor& c_data, Tensor& out_ell);
    void computeDiscreteELL(const std::vector<Tensor>& d_data,
                           const std::vector<float>& w_disc,
                           Tensor& out_ell);
    void updatePrior(const Tensor& qz, float lr, float beta);
    void updateContinuous(const Tensor& c_data, const Tensor& qz, 
                         float lr, float beta);
    void updateDiscrete(const std::vector<Tensor>& d_data,
                       const Tensor& qz,
                       float lr, float beta);
    void updateContinuousLikelihood(const Tensor& c_data,
                                   const Tensor& qz,
                                   float lr,
                                   float beta);
    void updateDiscreteLikelihood(const std::vector<Tensor>& d_data,
                                 const Tensor& qz,
                                 float lr,
                                 float beta);
    void sumComponents(const HybridMixtureState& other,
                      const Tensor& self_mask,
                      const Tensor& other_mask);
    void combineParams(const HybridMixtureState& other,
                      const Tensor& select_mask);
};

// Interaction detection
struct ObjectInteraction {
    int other_idx;
    float distance_x;
    float distance_y;
    bool is_interacting;
};

// Ellipse interaction detection
void detectEllipseInteractions(const Tensor& data,        // (n_objects, features)
                              int object_idx,
                              float r_interacting,
                              bool forward_predict,
                              bool exclude_background,
                              bool interact_with_static,
                              const Tensor& tracked_obj_mask,
                              ObjectInteraction& out_result);

// Grid-based object interaction detection (closest)
void detectClosestInteractions(const Tensor& data,
                              int object_idx,
                              float r_interacting,
                              bool exclude_background,
                              bool interact_with_static,
                              bool absolute_distance_scale,
                              const Tensor& tracked_obj_mask,
                              ObjectInteraction& out_result);

// Convert observations to hybrid format for RMM
void toHybridObservation(void* imm_ptr,              // Identity model pointer
                        const Tensor& data,         // (n_objects, features)
                        int object_idx,
                        int action,
                        int reward,
                        int tmm_switch,
                        const Tensor& tracked_obj_mask,
                        bool interact_with_static,
                        int max_switches,
                        int action_dim,
                        int num_object_classes,
                        int reward_dim,
                        bool forward_predict,
                        bool stable_r,
                        bool relative_distance,
                        bool color_only_identity,
                        bool exclude_background,
                        bool use_ellipses,
                        float velocity_scale,
                        bool absolute_distance_scale,
                        Tensor& out_c_feat,         // (cont_dim,)
                        std::vector<Tensor>& out_d_feat);

// Factory function
std::unique_ptr<RewardMixtureModel> createRMM(int action_dim,
                                             int num_components_per_switch,
                                             int num_switches,
                                             int num_object_types,
                                             int num_continuous_dims = 7,
                                             int reward_dim = 3,
                                             float cont_scale_switch = 25.0f,
                                             const std::vector<float>& discrete_alphas = {});

} // namespace models
} // namespace axiom

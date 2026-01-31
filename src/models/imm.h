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

// Identity Mixture Model State
struct IMMState {
    // Gaussian likelihood for identity features
    Tensor mean;            // (num_object_types, num_features, 1)
    Tensor kappa;           // (num_object_types, 1, 1)
    Tensor u;               // (num_object_types, num_features, num_features)
    Tensor n;               // (num_object_types, 1, 1)
    
    // Prior
    Tensor prior_alpha;     // (num_object_types,)
    Tensor posterior_alpha; // (num_object_types,)
    
    // Masks
    Tensor used_mask;       // (num_object_types,)
    
    // Configuration
    int num_object_types;
    int num_features;
    float cont_scale;
    float color_precision_scale;
    bool color_only_identity;
    
    void allocate(int num_object_types_, int num_features_, 
                 bool color_only = false);
    void copyToDevice();
    void copyFromDevice();
};

// Identity Mixture Model
class IdentityMixtureModel {
public:
    IdentityMixtureModel(const IMMState& state);
    ~IdentityMixtureModel();
    
    // Infer identity from features
    // Returns: class_labels (batch,)
    void inferIdentity(const Tensor& x,           // (batch, num_features, 1)
                      bool color_only_identity,
                      Tensor& out_class_labels);  // (batch,) - int32
    
    // Infer with remapped colors (shape-only if color match fails)
    void inferRemappedColorIdentity(const Tensor& x,
                                   bool color_only_identity,
                                   float ell_threshold,
                                   Tensor& out_qz);  // Soft assignment
    
    // Update model with new observation
    void update(const Tensor& x,
               float i_ell_threshold,
               bool color_only_identity,
               bool& grew_component);
    
    // Training step with automatic growing
    void trainStep(const Tensor& x,
                  float logp_threshold,
                  Tensor& out_qz,
                  bool& grew_component);
    
    // Get used mask
    Tensor getUsedMask() const;
    
    // Set used mask
    void setUsedMask(const Tensor& mask);
    
    // Get posterior probabilities
    void getPosterior(const Tensor& x,
                     Tensor& out_posterior);  // (batch, num_object_types)
    
    // Compute expected log-likelihood
    void computeELL(const Tensor& x,
                   Tensor& out_ell);  // (batch, num_object_types)

private:
    IMMState state_;
    cudaStream_t stream_;
    
    // Internal buffers
    Tensor ell_buffer_;
    Tensor temp_buffer_;
    
    void computeExpectedLogLikelihood(const Tensor& x, Tensor& out_ell);
    void computeExpectedLogLikelihoodShapeOnly(const Tensor& x, Tensor& out_ell);
    void updateParameters(const Tensor& x, const Tensor& qz, float lr, float beta);
    int findFirstUnused();
};

// Color remapping utilities
void remapColors(const Tensor& source_colors,      // (3,) RGB
                const Tensor& target_colors,      // (3,) RGB
                const Tensor& image,              // (H, W, 3)
                Tensor& out_remapped);            // (H, W, 3)

// Factory function
std::unique_ptr<IdentityMixtureModel> createIMM(int num_object_types,
                                               int num_features = 5,
                                               float cont_scale_identity = 0.5f,
                                               float color_precision_scale = 1.0f,
                                               bool color_only_identity = false);

} // namespace models
} // namespace axiom

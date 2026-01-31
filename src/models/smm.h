/**
 * Copyright 2025 VERSES AI, Inc.
 *
 * Licensed under the VERSES Academic Research License (the "License");
 * you may not use this file except in compliance with the license.
 *
 * You may obtain a copy of the License at
 *
 *     https://github.com/VersesTech/axiom/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

// Linear Matrix Normal Gamma parameters
struct LinearMNGParams {
    // Mean parameters (y_dim, x_dim) - includes bias if use_bias=true
    Tensor mu;
    // Inverse covariance (x_dim, x_dim)
    Tensor inv_v;
    // Gamma shape parameters (y_dim, 1)
    Tensor a;
    // Gamma scale parameters (y_dim, 1)
    Tensor b;
    
    void allocate(int batch_size, int num_slots, int y_dim, int x_dim, bool use_bias);
    void copyToDevice();
    void copyFromDevice();
};

// Slot Mixture Model State
struct SMMState {
    // Model parameters
    LinearMNGParams prior_params;
    LinearMNGParams posterior_params;
    
    // Variational distributions
    Tensor qx_mu;           // (batch, num_slots, slot_dim, 1)
    Tensor qx_inv_sigma;    // (batch, num_slots, slot_dim, slot_dim)
    Tensor qz;              // (batch, num_tokens, num_slots)
    
    // Prior on slots
    Tensor pi_alpha;        // (num_slots,)
    
    // Masks and indices
    Tensor slot_mask;       // (num_slots, y_dim, x_dim) - mask templates
    Tensor used_mask;       // (num_slots,) - which slots are active
    Tensor dirty_mask;      // (num_slots,) - slots needing update
    
    // Position encoding
    Tensor position_grid;   // Precomputed (height, width, 2) position grid
    
    // Configuration
    int width;
    int height;
    int input_dim;      // Usually 5: (x, y, r, g, b)
    int slot_dim;       // Usually 2
    int num_slots;
    bool use_bias;
    
    // Training parameters
    float learning_rate;
    float beta;         // Momentum for updates
    float elbo_threshold;
    int max_grow_steps;
    int num_e_steps;
    
    void allocate(int batch_size, int width_, int height_, int input_dim_, 
                  int slot_dim_, int num_slots_, bool use_bias_);
    void copyToDevice();
    void copyFromDevice();
};

// Slot Mixture Model
class SlotMixtureModel {
public:
    SlotMixtureModel(const SMMState& state);
    ~SlotMixtureModel();
    
    // E-step: Compute q(z|x) and update q(x|z)
    void eStep(const Tensor& inputs,         // (batch, num_tokens, input_dim)
               int num_iterations = 1);
    
    // M-step: Update parameters given q(x) and q(z)
    void mStep(const Tensor& inputs,
               const Tensor& qx_mu,
               const Tensor& qx_inv_sigma,
               const Tensor& qz,
               float lr = 1.0f,
               float beta = 0.0f,
               const Tensor* grow_mask = nullptr);
    
    // Combined EM step
    void emStep(const Tensor& inputs,
                float lr = 1.0f,
                float beta = 0.0f);
    
    // Initialize slots from data (force all data to first slot)
    void initializeFromData(const Tensor& inputs);
    
    // Forward message passing (predict y from x)
    void variationalForward(const Tensor& qx_mu,
                           const Tensor& qx_inv_sigma,
                           Tensor& out_py_mu,
                           Tensor& out_py_inv_sigma);
    
    // Backward message passing (update qx from observations)
    void variationalBackward(const Tensor& inputs,
                            const Tensor& qz,
                            Tensor& out_qx_mu,
                            Tensor& out_qx_inv_sigma);
    
    // Grow the model by adding new components
    bool growModel(const Tensor& inputs,
                   const Tensor& ell_max,
                   float threshold);
    
    // Compute ELBO
    float computeELBO(const Tensor& inputs,
                     const Tensor& qx_mu,
                     const Tensor& qx_inv_sigma,
                     const Tensor& qz);
    
    // Get used component mask
    Tensor getUsedMask() const;
    
    // Get current qz assignments
    Tensor getAssignments() const;
    
    // Set slot mask templates
    void setSlotMask(const Tensor& mask, const std::vector<float>& probs);

private:
    SMMState state_;
    
    // CUDA streams for async operations
    cudaStream_t stream_;
    
    // Internal buffers
    Tensor ell_buffer_;     // Expected log-likelihood
    Tensor temp_buffer_;    // Temporary computations
    
    void computeExpectedLogLikelihood(const Tensor& inputs,
                                     const Tensor& qx_mu,
                                     const Tensor& qx_inv_sigma,
                                     Tensor& out_ell);
    
    void updateQx(const Tensor& inputs,
                 const Tensor& qz,
                 Tensor& out_qx_mu,
                 Tensor& out_qx_inv_sigma);
    
    void updateQz(const Tensor& ell,
                 Tensor& out_qz);
    
    void updateParameters(const Tensor& inputs,
                         const Tensor& qx_mu,
                         const Tensor& qx_inv_sigma,
                         const Tensor& qz,
                         float lr,
                         float beta);
};

// Factory function
std::unique_ptr<SlotMixtureModel> createSMM(int width, int height, int input_dim,
                                            int slot_dim, int num_slots,
                                            bool use_bias = true,
                                            float ns_a = 1.0f,
                                            float ns_b = 1.0f,
                                            float dof_offset = 10.0f,
                                            const std::vector<float>& mask_prob = {},
                                            const std::vector<float>& scale = {},
                                            float transform_inv_v_scale = 100.0f,
                                            float bias_inv_v_scale = 0.001f);

// Position encoding
void addPositionEncoding(const Tensor& image,        // (H, W, C)
                        Tensor& out_with_pos,       // (H*W, C+2)
                        float width_scale,
                        float height_scale);

// Format observation for SMM
void formatObservation(const Tensor& obs,          // (H, W, C)
                      Tensor& out_formatted,      // (H*W, C+2)
                      const std::vector<float>& offset,
                      const std::vector<float>& stdevs);

} // namespace models
} // namespace axiom

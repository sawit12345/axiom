// Copyright 2025 VERSES AI, Inc.
//
// Licensed under the VERSES Academic Research License (the "License");
// you may not use this file except in compliance with the license.
//
// You may obtain a copy of the License at
//
//     https://github.com/VersesTech/axiom/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "types.h"
#include <memory>
#include <vector>
#include <tuple>

namespace axiom {
namespace transforms {

// Forward declarations
class Distribution;
class ExponentialFamily;

/**
 * @brief Base class for transforms - distributions conditioned on latents.
 * 
 * Transforms model p(y|x,θ) where y is the event, x is the input_event, 
 * and θ gives the parameters. They support forward and backward message
 * passing routines that emit message distributions.
 * 
 * This is the C++ equivalent of the Python Transform class which inherits
 * from Conjugate. Transforms are used in variational inference for the
 * Slot Mixture Model (SMM) and other latent variable models.
 */
class Transform {
public:
    virtual ~Transform() = default;

    // =========================================================================
    // Core Properties
    // =========================================================================
    
    /**
     * @brief Get the input dimension (x_dim)
     */
    virtual int get_x_dim() const = 0;
    
    /**
     * @brief Get the output dimension (y_dim)
     */
    virtual int get_y_dim() const = 0;
    
    /**
     * @brief Check if bias is used in the linear transform
     */
    virtual bool uses_bias() const = 0;
    
    /**
     * @brief Get the batch shape of the transform
     */
    virtual std::vector<int> get_batch_shape() const = 0;
    
    /**
     * @brief Get the event shape of the transform
     */
    virtual std::vector<int> get_event_shape() const = 0;

    // =========================================================================
    // Parameter Access
    // =========================================================================
    
    /**
     * @brief Get posterior parameters in canonical form
     */
    virtual ArrayDict get_posterior_params() const = 0;
    
    /**
     * @brief Get prior parameters in canonical form
     */
    virtual ArrayDict get_prior_params() const = 0;
    
    /**
     * @brief Get natural parameters of posterior
     */
    virtual ArrayDict get_posterior_natural_params() const = 0;
    
    /**
     * @brief Get natural parameters of prior
     */
    virtual ArrayDict get_prior_natural_params() const = 0;
    
    /**
     * @brief Set posterior parameters
     */
    virtual void set_posterior_params(const ArrayDict& params) = 0;
    
    /**
     * @brief Set prior parameters
     */
    virtual void set_prior_params(const ArrayDict& params) = 0;

    // =========================================================================
    // Update Methods
    // =========================================================================
    
    /**
     * @brief Update from data with optional weights
     * @param data Tuple of (X, Y) data arrays
     * @param weights Optional weights for each sample
     * @param lr Learning rate (default: 1.0)
     * @param beta Batch decay (default: 0.0)
     */
    virtual void update_from_data(
        const std::tuple<ArrayDict, ArrayDict>& data,
        const ArrayDict* weights = nullptr,
        float lr = 1.0f,
        float beta = 0.0f) = 0;
    
    /**
     * @brief Update from probability distributions
     * @param pXY Tuple of (pX, pY) distributions
     * @param weights Optional weights
     * @param lr Learning rate (default: 1.0)
     * @param beta Batch decay (default: 0.0)
     * @param apply_updates If true, apply updates; if false, return statistics
     * @return Updated statistics if apply_updates=false
     */
    virtual ArrayDict update_from_probabilities(
        const std::tuple<std::shared_ptr<Distribution>, std::shared_ptr<Distribution>>& pXY,
        const ArrayDict* weights = nullptr,
        float lr = 1.0f,
        float beta = 0.0f,
        bool apply_updates = true) = 0;
    
    /**
     * @brief Update from sufficient statistics
     * @param stats Summed sufficient statistics
     * @param lr Learning rate (default: 1.0)
     * @param beta Batch decay (default: 0.0)
     */
    virtual void update_from_statistics(
        const ArrayDict& stats,
        float lr = 1.0f,
        float beta = 0.0f) = 0;

    // =========================================================================
    // Expected Values and Likelihood
    // =========================================================================
    
    /**
     * @brief Compute expected likelihood parameters
     * @return ArrayDict with expected natural parameters
     */
    virtual ArrayDict expected_likelihood_params() const = 0;
    
    /**
     * @brief Compute expected posterior statistics
     * @return ArrayDict with expected sufficient statistics
     */
    virtual ArrayDict expected_posterior_statistics() const = 0;
    
    /**
     * @brief Compute expected log partition function
     * @return Expected log partition value
     */
    virtual ArrayDict expected_log_partition() const = 0;
    
    /**
     * @brief Compute log partition of prior
     */
    virtual ArrayDict log_prior_partition() const = 0;
    
    /**
     * @brief Compute log partition of posterior
     */
    virtual ArrayDict log_posterior_partition() const = 0;
    
    /**
     * @brief Compute expected log likelihood of data
     * @param data Tuple of (X, Y) data
     * @return Log likelihood values
     */
    virtual ArrayDict expected_log_likelihood(
        const std::tuple<ArrayDict, ArrayDict>& data) const = 0;
    
    /**
     * @brief Compute average energy term
     * @param inputs Tuple of (pX, pY) distributions
     * @return Average energy values
     */
    virtual ArrayDict average_energy(
        const std::tuple<std::shared_ptr<Distribution>, std::shared_ptr<Distribution>>& inputs) const = 0;

    // =========================================================================
    // KL Divergence and ELBO
    // =========================================================================
    
    /**
     * @brief Compute KL divergence between posterior and prior
     * @return KL divergence value
     */
    virtual ArrayDict kl_divergence() const = 0;
    
    /**
     * @brief Compute ELBO for data
     * @param data Tuple of (X, Y) data
     * @param weights Optional weights
     * @return ELBO value
     */
    virtual ArrayDict elbo(
        const std::tuple<ArrayDict, ArrayDict>& data,
        const ArrayDict* weights = nullptr) const = 0;
    
    /**
     * @brief Compute ELBO contribution from distributions
     * @param pXY Tuple of (pX, pY) distributions
     * @param weights Optional weights
     * @return ELBO contribution
     */
    virtual ArrayDict elbo_contrib(
        const std::tuple<std::shared_ptr<Distribution>, std::shared_ptr<Distribution>>& pXY,
        const ArrayDict* weights = nullptr) const = 0;

    // =========================================================================
    // Message Passing (Forward/Backward)
    // =========================================================================
    
    /**
     * @brief Forward message passing from normal input
     * @param pX Input distribution (multivariate normal)
     * @param pass_residual Whether to pass residual from input
     * @return Output distribution (multivariate normal)
     */
    virtual std::shared_ptr<ExponentialFamily> forward_from_normal(
        const std::shared_ptr<ExponentialFamily>& pX,
        bool pass_residual = false) const = 0;
    
    /**
     * @brief Backward message passing from normal output
     * @param pY Output distribution (multivariate normal)
     * @param pass_residual Whether to pass residual from output
     * @return Input distribution (multivariate normal)
     */
    virtual std::shared_ptr<ExponentialFamily> backward_from_normal(
        const std::shared_ptr<ExponentialFamily>& pY,
        bool pass_residual = false) const = 0;
    
    /**
     * @brief Variational forward message (fast approximation)
     * @param pX Input distribution
     * @param pass_residual Whether to pass residual
     * @return Output distribution
     */
    virtual std::shared_ptr<ExponentialFamily> variational_forward(
        const std::shared_ptr<Distribution>& pX,
        bool pass_residual = false) const = 0;
    
    /**
     * @brief Variational backward message (fast approximation)
     * @param pY Output distribution
     * @param pass_residual Whether to pass residual
     * @return Input distribution
     */
    virtual std::shared_ptr<ExponentialFamily> variational_backward(
        const std::shared_ptr<Distribution>& pY,
        bool pass_residual = false) const = 0;

    // =========================================================================
    // Prediction and Joint Distribution
    // =========================================================================
    
    /**
     * @brief Predict output given input
     * @param x Input array
     * @return Predicted output distribution
     */
    virtual std::shared_ptr<ExponentialFamily> predict(const ArrayDict& x) const = 0;
    
    /**
     * @brief Compute joint distribution of input and output
     * @param pX Input distribution
     * @param pY Output distribution
     * @return Joint distribution
     */
    virtual std::shared_ptr<Distribution> joint(
        const std::shared_ptr<Distribution>& pX,
        const std::shared_ptr<Distribution>& pY) const = 0;

    // =========================================================================
    // Utility Methods
    // =========================================================================
    
    /**
     * @brief Convert canonical parameters to natural parameters
     * @param params Canonical parameters (mu, inv_v, a, b)
     * @return Natural parameters (eta_1, eta_2, eta_3, eta_4)
     */
    virtual ArrayDict to_natural_params(const ArrayDict& params) const = 0;
    
    /**
     * @brief Map sufficient statistics to parameter updates
     * @param likelihood_stats Statistics from likelihood
     * @param counts Counts for each sample
     * @return Mapped statistics
     */
    virtual ArrayDict map_stats_to_params(
        const ArrayDict& likelihood_stats,
        const ArrayDict& counts) const = 0;
    
    /**
     * @brief Copy the transform
     */
    virtual std::shared_ptr<Transform> copy() const = 0;
    
    /**
     * @brief Update internal cache after parameter changes
     */
    virtual void update_cache() = 0;
};

} // namespace transforms
} // namespace axiom

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

#include "transform.h"
#include <memory>
#include <vector>
#include <cmath>

namespace axiom {
namespace transforms {

// Forward declarations
class MultivariateNormal;

/**
 * @brief Linear Matrix Normal-Gamma transform for the Slot Mixture Model (SMM).
 * 
 * This distribution models the linear transformation:
 *   y = Ax + epsilon
 * 
 * where:
 *   - y is the output vector with size p (y_dim)
 *   - x is the input vector with size d (x_dim)
 *   - A are the linear transformation parameters with size p x d
 *   - epsilon is additive Gaussian noise
 * 
 * The conjugate prior to A and Sigma is the Matrix Normal Gamma distribution:
 *   A | Sigma^{-1} ~ MN(A | mu_0, Sigma^{-1}, V_0)
 *   Sigma^{-1} := Diag(gamma) ~ Gamma(gamma | a_0, b_0)
 * 
 * The natural parameterization is:
 *   eta_1 = V^{-1}
 *   eta_2 = mu V^{-1}
 *   eta_{3,k} = 2b + [U^{-1} + mu V^{-1} mu^T]_{kk}
 *   eta_{4,k} = 2(a - 1) + d
 * 
 * Sufficient statistics T(x):
 *   eta_1: sum(x_i x_i^T)
 *   eta_2: sum(y_i x_i^T)
 *   eta_3: sum(y_i y_i^T) - diagonal only
 *   eta_4: N (count)
 */
class LinearMatrixNormalGamma : public Transform {
public:
    // =================================================================
    // Constructors and Destructor
    // =================================================================
    
    /**
     * @brief Constructor with explicit parameters
     * @param params Posterior parameters (mu, inv_v, a, b)
     * @param prior_params Prior parameters (mu_0, inv_v_0, a_0, b_0)
     * @param x_dim Input dimension
     * @param y_dim Output dimension
     * @param use_bias Whether to include bias term
     * @param fixed_precision Whether precision is fixed
     * @param batch_shape Batch shape for parameters
     */
    LinearMatrixNormalGamma(
        const ArrayDict& params,
        const ArrayDict& prior_params,
        int x_dim,
        int y_dim,
        bool use_bias = true,
        bool fixed_precision = false,
        const std::vector<int>& batch_shape = {});
    
    /**
     * @brief Constructor with initialization
     * @param x_dim Input dimension
     * @param y_dim Output dimension
     * @param use_bias Whether to include bias term
     * @param fixed_precision Whether precision is fixed
     * @param scale Scale parameter for initialization
     * @param dof_offset Degrees of freedom offset
     * @param inv_v_scale Scale for precision matrix V^{-1}
     * @param batch_shape Batch shape for parameters
     */
    LinearMatrixNormalGamma(
        int x_dim,
        int y_dim,
        bool use_bias = true,
        bool fixed_precision = false,
        float scale = 1.0f,
        float dof_offset = 1.0f,
        float inv_v_scale = 1.0f,
        const std::vector<int>& batch_shape = {});
    
    virtual ~LinearMatrixNormalGamma() = default;

    // =================================================================
    // Core Properties (Transform Interface)
    // =================================================================
    
    int get_x_dim() const override { return x_dim_; }
    int get_y_dim() const override { return y_dim_; }
    bool uses_bias() const override { return use_bias_; }
    std::vector<int> get_batch_shape() const override { return batch_shape_; }
    std::vector<int> get_event_shape() const override;

    // =================================================================
    // Parameter Access (Transform Interface)
    // =================================================================
    
    ArrayDict get_posterior_params() const override;
    ArrayDict get_prior_params() const override;
    ArrayDict get_posterior_natural_params() const override { return posterior_natural_params_; }
    ArrayDict get_prior_natural_params() const override { return prior_natural_params_; }
    
    void set_posterior_params(const ArrayDict& params) override;
    void set_prior_params(const ArrayDict& params) override;

    // =================================================================
    // Canonical Parameter Accessors
    // =================================================================
    
    /** @brief Posterior mean mu (y_dim, x_dim) */
    ArrayDict get_mu() const;
    
    /** @brief Posterior inverse covariance inv_v (x_dim, x_dim) */
    ArrayDict get_inv_v() const;
    
    /** @brief Posterior covariance v (x_dim, x_dim) */
    ArrayDict get_v() const;
    
    /** @brief Posterior shape a (y_dim, 1) */
    ArrayDict get_a() const;
    
    /** @brief Posterior scale b (y_dim, 1) */
    ArrayDict get_b() const;
    
    /** @brief Prior mean mu_0 */
    ArrayDict get_prior_mu() const;
    
    /** @brief Prior inverse covariance inv_v_0 */
    ArrayDict get_prior_inv_v() const;
    
    /** @brief Prior covariance v_0 */
    ArrayDict get_prior_v() const;
    
    /** @brief Prior shape a_0 */
    ArrayDict get_prior_a() const;
    
    /** @brief Prior scale b_0 */
    ArrayDict get_prior_b() const;
    
    /** @brief Weight matrix (excluding bias if use_bias) */
    ArrayDict get_weights() const;
    
    /** @brief Bias vector (if use_bias) */
    ArrayDict get_bias() const;

    // =================================================================
    // Update Methods (Transform Interface)
    // =================================================================
    
    void update_from_data(
        const std::tuple<ArrayDict, ArrayDict>& data,
        const ArrayDict* weights = nullptr,
        float lr = 1.0f,
        float beta = 0.0f) override;
    
    ArrayDict update_from_probabilities(
        const std::tuple<std::shared_ptr<Distribution>, std::shared_ptr<Distribution>>& pXY,
        const ArrayDict* weights = nullptr,
        float lr = 1.0f,
        float beta = 0.0f,
        bool apply_updates = true) override;
    
    void update_from_statistics(
        const ArrayDict& stats,
        float lr = 1.0f,
        float beta = 0.0f) override;

    // =================================================================
    // Expected Values
    // =================================================================
    
    /** @brief E[inv_sigma] = a/b * I */
    ArrayDict expected_inv_sigma() const;
    
    /** @brief E[inv_sigma @ mu] */
    ArrayDict expected_inv_sigma_x(const ArrayDict* inv_sigma = nullptr) const;
    
    /** @brief E[sigma] = b/(a-1) * I */
    ArrayDict expected_sigma() const;
    
    /** @brief E[x^T @ inv_sigma @ x] = y_dim * V + mu^T @ E[inv_sigma] @ mu */
    ArrayDict expected_x_inv_sigma_x(const ArrayDict* inv_sigma_mu = nullptr) const;
    
    /** @brief E[log |inv_sigma|] = sum(digamma(a) - log(b)) */
    ArrayDict expected_logdet_inv_sigma() const;
    
    /** @brief log |E[inv_sigma]| = sum(log(a) - log(b)) */
    ArrayDict logdet_expected_inv_sigma() const;
    
    /** @brief E[log |inv_sigma|] - log |E[inv_sigma]| */
    ArrayDict expected_log_det_inv_sigma_minus_log_det_expected_inv_sigma() const;
    
    /** @brief Inverse of E[inv_sigma] */
    ArrayDict inv_expected_inv_sigma() const;
    
    /** @brief Diagonal of E[inv_sigma] as vector */
    ArrayDict expected_inv_sigma_diag() const;

    // =================================================================
    // Likelihood and Statistics (Transform Interface)
    // =================================================================
    
    ArrayDict expected_likelihood_params() const override;
    ArrayDict expected_posterior_statistics() const override;
    ArrayDict expected_log_partition() const override;
    ArrayDict log_prior_partition() const override;
    ArrayDict log_posterior_partition() const override;
    
    ArrayDict expected_log_likelihood(
        const std::tuple<ArrayDict, ArrayDict>& data) const override;
    
    ArrayDict average_energy(
        const std::tuple<std::shared_ptr<Distribution>, std::shared_ptr<Distribution>>& inputs) const override;

    // =================================================================
    // KL Divergence and ELBO (Transform Interface)
    // =================================================================
    
    ArrayDict kl_divergence() const override;
    ArrayDict elbo(
        const std::tuple<ArrayDict, ArrayDict>& data,
        const ArrayDict* weights = nullptr) const override;
    ArrayDict elbo_contrib(
        const std::tuple<std::shared_ptr<Distribution>, std::shared_ptr<Distribution>>& pXY,
        const ArrayDict* weights = nullptr) const override;

    // =================================================================
    // Message Passing (Transform Interface)
    // =================================================================
    
    std::shared_ptr<ExponentialFamily> forward_from_normal(
        const std::shared_ptr<ExponentialFamily>& pX,
        bool pass_residual = false) const override;
    
    std::shared_ptr<ExponentialFamily> backward_from_normal(
        const std::shared_ptr<ExponentialFamily>& pY,
        bool pass_residual = false) const override;
    
    std::shared_ptr<ExponentialFamily> variational_forward(
        const std::shared_ptr<Distribution>& pX,
        bool pass_residual = false) const override;
    
    std::shared_ptr<ExponentialFamily> variational_backward(
        const std::shared_ptr<Distribution>& pY,
        bool pass_residual = false) const override;

    // =================================================================
    // Prediction and Joint (Transform Interface)
    // =================================================================
    
    std::shared_ptr<ExponentialFamily> predict(const ArrayDict& x) const override;
    
    std::shared_ptr<Distribution> joint(
        const std::shared_ptr<Distribution>& pX,
        const std::shared_ptr<Distribution>& pY) const override;

    // =================================================================
    // Utility Methods (Transform Interface)
    // =================================================================
    
    ArrayDict to_natural_params(const ArrayDict& params) const override;
    ArrayDict map_stats_to_params(const ArrayDict& likelihood_stats, const ArrayDict& counts) const override;
    std::shared_ptr<Transform> copy() const override;
    void update_cache() override;

    // =================================================================
    // Static Methods
    // =================================================================
    
    /** @brief Initialize default prior parameters */
    static ArrayDict init_default_params(
        const std::vector<int>& batch_shape,
        int x_dim,
        int y_dim,
        float scale = 1.0f,
        float dof_offset = 1.0f,
        float inv_v_scale = 1.0f);

private:
    // =================================================================
    // Private Helper Methods
    // =================================================================
    
    /** @brief Compute log partition given parameters */
    ArrayDict compute_log_partition(
        const ArrayDict& mean,
        const ArrayDict& logdet_inv_v,
        const ArrayDict& a,
        const ArrayDict& b) const;
    
    /** @brief Compute KL divergence for gamma component */
    ArrayDict kl_divergence_gamma() const;
    
    /** @brief Initialize the transform with given parameters */
    void initialize(
        const ArrayDict& params,
        const ArrayDict& prior_params,
        int x_dim,
        int y_dim,
        bool use_bias,
        bool fixed_precision,
        const std::vector<int>& batch_shape);
    
    /** @brief Compute inverse and logdet with numerical stability */
    std::pair<ArrayDict, ArrayDict> inv_and_logdet_stable(const ArrayDict& matrix) const;

    // =================================================================
    // Member Variables
    // =================================================================
    
    // Dimensions
    int x_dim_;
    int y_dim_;
    int effective_x_dim_;  // x_dim + 1 if use_bias
    
    // Configuration
    bool use_bias_;
    bool fixed_precision_;
    std::vector<int> batch_shape_;
    
    // Parameters (canonical form)
    ArrayDict posterior_params_;  // mu, inv_v, a, b
    ArrayDict prior_params_;      // mu_0, inv_v_0, a_0, b_0
    
    // Natural parameters
    ArrayDict posterior_natural_params_;  // eta_1, eta_2, eta_3, eta_4
    ArrayDict prior_natural_params_;
    
    // Cached values (mutable for lazy evaluation)
    mutable ArrayDict cached_v_;
    mutable ArrayDict cached_prior_v_;
    mutable ArrayDict cached_b_;
    mutable ArrayDict cached_logdet_inv_v_;
    mutable ArrayDict cached_prior_logdet_inv_v_;
    mutable bool cache_valid_;
    
    // Constants
    static constexpr float MIN_B = 1e-6f;
    static constexpr float EPSILON = 1e-8f;
};

} // namespace transforms
} // namespace axiom

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

#include "linear_mng.h"
#include "cuda_transforms.cuh"
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace axiom {
namespace transforms {

// ============================================================================
// Constructors and Initialization
// ============================================================================

LinearMatrixNormalGamma::LinearMatrixNormalGamma(
    const ArrayDict& params,
    const ArrayDict& prior_params,
    int x_dim,
    int y_dim,
    bool use_bias,
    bool fixed_precision,
    const std::vector<int>& batch_shape)
    : x_dim_(x_dim)
    , y_dim_(y_dim)
    , effective_x_dim_(x_dim + (use_bias ? 1 : 0))
    , use_bias_(use_bias)
    , fixed_precision_(fixed_precision)
    , batch_shape_(batch_shape)
    , cache_valid_(false) {
    
    initialize(params, prior_params, x_dim, y_dim, use_bias, fixed_precision, batch_shape);
}

LinearMatrixNormalGamma::LinearMatrixNormalGamma(
    int x_dim,
    int y_dim,
    bool use_bias,
    bool fixed_precision,
    float scale,
    float dof_offset,
    float inv_v_scale,
    const std::vector<int>& batch_shape)
    : x_dim_(x_dim)
    , y_dim_(y_dim)
    , effective_x_dim_(x_dim + (use_bias ? 1 : 0))
    , use_bias_(use_bias)
    , fixed_precision_(fixed_precision)
    , batch_shape_(batch_shape)
    , cache_valid_(false) {
    
    // Initialize default prior parameters
    ArrayDict prior_params = init_default_params(batch_shape, x_dim, y_dim, scale, dof_offset, inv_v_scale);
    
    // Initialize posterior parameters from prior with small random perturbation
    posterior_params_ = prior_params;
    
    // Add small random initialization to mu
    std::vector<int> mu_shape = batch_shape;
    mu_shape.push_back(y_dim);
    mu_shape.push_back(effective_x_dim_);
    Array mu_noise = Array::randn(mu_shape, 42) * (scale / std::sqrt(static_cast<float>(effective_x_dim_)));
    posterior_params_["mu"] = prior_params["mu"] + mu_noise;
    
    // Set inv_v to identity mask (1.0 where prior > 0)
    Array inv_v_mask = prior_params["inv_v"];
    for (int i = 0; i < inv_v_mask.numel(); ++i) {
        inv_v_mask[i] = (inv_v_mask[i] > 0.0f) ? 1.0f : 0.0f;
    }
    posterior_params_["inv_v"] = inv_v_mask;
    
    // Set a = 2.0, b = 1.0 initially
    posterior_params_["a"] = Array::full({y_dim, 1}, 2.0f);
    posterior_params_["b"] = Array::ones({y_dim, 1});
    
    prior_params_ = prior_params;
    
    // Convert to natural parameters
    posterior_natural_params_ = to_natural_params(posterior_params_);
    prior_natural_params_ = to_natural_params(prior_params_);
    
    update_cache();
}

void LinearMatrixNormalGamma::initialize(
    const ArrayDict& params,
    const ArrayDict& prior_params,
    int x_dim,
    int y_dim,
    bool use_bias,
    bool fixed_precision,
    const std::vector<int>& batch_shape) {
    
    posterior_params_ = params;
    prior_params_ = prior_params;
    
    // Convert to natural parameters
    posterior_natural_params_ = to_natural_params(params);
    prior_natural_params_ = to_natural_params(prior_params);
    
    update_cache();
}

ArrayDict LinearMatrixNormalGamma::init_default_params(
    const std::vector<int>& batch_shape,
    int x_dim,
    int y_dim,
    float scale,
    float dof_offset,
    float inv_v_scale) {
    
    ArrayDict params;
    
    // mu: zeros (batch..., y_dim, x_dim)
    std::vector<int> mu_shape = batch_shape;
    mu_shape.push_back(y_dim);
    mu_shape.push_back(x_dim);
    params["mu"] = Array::zeros(mu_shape);
    
    // inv_v: scaled identity (batch..., x_dim, x_dim)
    std::vector<int> v_shape = batch_shape;
    v_shape.push_back(x_dim);
    v_shape.push_back(x_dim);
    params["inv_v"] = Array::eye(x_dim) * inv_v_scale;
    // Broadcast to batch shape if needed
    if (!batch_shape.empty()) {
        Array inv_v_broadcast = Array::zeros(v_shape);
        // Fill with identity pattern for each batch element
        int batch_size = 1;
        for (int d : batch_shape) batch_size *= d;
        int x_sq = x_dim * x_dim;
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < x_dim; ++i) {
                inv_v_broadcast.data[b * x_sq + i * x_dim + i] = inv_v_scale;
            }
        }
        params["inv_v"] = inv_v_broadcast;
    }
    
    // a: 1 + dof_offset (batch..., y_dim, 1)
    std::vector<int> a_shape = batch_shape;
    a_shape.push_back(y_dim);
    a_shape.push_back(1);
    params["a"] = Array::full(a_shape, 1.0f + dof_offset);
    
    // b: scale^2 (batch..., y_dim, 1)
    std::vector<int> b_shape = batch_shape;
    b_shape.push_back(y_dim);
    b_shape.push_back(1);
    params["b"] = Array::full(b_shape, scale * scale);
    
    return params;
}

// ============================================================================
// Core Properties
// ============================================================================

std::vector<int> LinearMatrixNormalGamma::get_event_shape() const {
    std::vector<int> shape = batch_shape_;
    shape.push_back(y_dim_);
    shape.push_back(effective_x_dim_);
    return shape;
}

// ============================================================================
// Parameter Access
// ============================================================================

ArrayDict LinearMatrixNormalGamma::get_posterior_params() const {
    return posterior_params_;
}

ArrayDict LinearMatrixNormalGamma::get_prior_params() const {
    return prior_params_;
}

void LinearMatrixNormalGamma::set_posterior_params(const ArrayDict& params) {
    posterior_params_ = params;
    posterior_natural_params_ = to_natural_params(params);
    cache_valid_ = false;
    update_cache();
}

void LinearMatrixNormalGamma::set_prior_params(const ArrayDict& params) {
    prior_params_ = params;
    prior_natural_params_ = to_natural_params(params);
    cache_valid_ = false;
}

// ============================================================================
// Canonical Parameter Accessors
// ============================================================================

ArrayDict LinearMatrixNormalGamma::get_mu() const {
    // mu = eta_2 @ V
    ArrayDict result;
    result["mu"] = matmul(posterior_natural_params_["eta_2"], get_v()["v"]);
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_inv_v() const {
    ArrayDict result;
    result["inv_v"] = posterior_natural_params_["eta_1"];
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_v() const {
    if (!cache_valid_) {
        update_cache();
    }
    ArrayDict result;
    result["v"] = cached_v_;
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_a() const {
    ArrayDict result;
    if (fixed_precision_) {
        result["a"] = get_prior_a()["a"];
    } else {
        // a = (eta_4 - x_dim) / 2 + 1
        Array eta_4 = posterior_natural_params_["eta_4"];
        result["a"] = (eta_4 - static_cast<float>(x_dim_)) * 0.5f + 1.0f;
    }
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_b() const {
    ArrayDict result;
    if (fixed_precision_) {
        result["b"] = prior_params_["b"];
    } else {
        if (!cache_valid_) {
            update_cache();
        }
        result["b"] = cached_b_;
    }
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_prior_mu() const {
    ArrayDict result;
    result["mu"] = matmul(prior_natural_params_["eta_2"], get_prior_v()["v"]);
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_prior_inv_v() const {
    ArrayDict result;
    result["inv_v"] = prior_natural_params_["eta_1"];
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_prior_v() const {
    if (!cache_valid_) {
        update_cache();
    }
    ArrayDict result;
    result["v"] = cached_prior_v_;
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_prior_a() const {
    ArrayDict result;
    Array eta_4 = prior_natural_params_["eta_4"];
    result["a"] = (eta_4 - static_cast<float>(x_dim_)) * 0.5f + 1.0f;
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_prior_b() const {
    ArrayDict result;
    result["b"] = prior_params_["b"];
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_weights() const {
    Array mu = posterior_params_["mu"];
    ArrayDict result;
    if (use_bias_) {
        // Return all but last column
        std::vector<int> w_shape = mu.shape;
        w_shape.back() = x_dim_;  // Remove bias dimension
        Array weights(w_shape);
        int batch_elems = mu.numel() / (y_dim_ * effective_x_dim_);
        for (int b = 0; b < batch_elems; ++b) {
            for (int i = 0; i < y_dim_; ++i) {
                for (int j = 0; j < x_dim_; ++j) {
                    int src_idx = b * y_dim_ * effective_x_dim_ + i * effective_x_dim_ + j;
                    int dst_idx = b * y_dim_ * x_dim_ + i * x_dim_ + j;
                    weights[dst_idx] = mu[src_idx];
                }
            }
        }
        result["weights"] = weights;
    } else {
        result["weights"] = mu;
    }
    return result;
}

ArrayDict LinearMatrixNormalGamma::get_bias() const {
    ArrayDict result;
    if (use_bias_) {
        Array mu = posterior_params_["mu"];
        // Extract last column
        std::vector<int> b_shape = mu.shape;
        b_shape.back() = 1;
        Array bias(b_shape);
        int batch_elems = mu.numel() / (y_dim_ * effective_x_dim_);
        for (int b = 0; b < batch_elems; ++b) {
            for (int i = 0; i < y_dim_; ++i) {
                int src_idx = b * y_dim_ * effective_x_dim_ + i * effective_x_dim_ + (effective_x_dim_ - 1);
                int dst_idx = b * y_dim_ + i;
                bias[dst_idx] = mu[src_idx];
            }
        }
        result["bias"] = bias;
    } else {
        result["bias"] = Array::zeros({y_dim_, 1});
    }
    return result;
}

// ============================================================================
// Cache Management
// ============================================================================

void LinearMatrixNormalGamma::update_cache() {
    if (cache_valid_) return;
    
    // Compute V = inv(inv_v) for posterior
    auto [v, logdet_v] = inv_and_logdet(posterior_natural_params_["eta_1"]);
    cached_v_ = v;
    cached_logdet_inv_v_ = Array::full({1, 1}, logdet_v);
    
    // Compute V = inv(inv_v) for prior
    auto [prior_v, prior_logdet_v] = inv_and_logdet(prior_natural_params_["eta_1"]);
    cached_prior_v_ = prior_v;
    cached_prior_logdet_inv_v_ = Array::full({1, 1}, prior_logdet_v);
    
    // Compute b from natural parameters
    // b = (eta_3 - rho) / 2 where rho = diag(mu @ eta_2^T)
    Array eta_2 = posterior_natural_params_["eta_2"];
    Array mu = matmul(eta_2, cached_v_);  // mu = eta_2 @ V
    Array mu_eta2_T = batch_matmul(mu, eta_2.transpose());  // mu @ eta_2^T
    Array rho = mu_eta2_T.diagonal();  // Diagonal elements
    Array eta_3 = posterior_natural_params_["eta_3"];
    cached_b_ = (eta_3 - rho) * 0.5f;
    // Clip to minimum value
    cached_b_ = cached_b_.clip(MIN_B, 1e10f);
    
    cache_valid_ = true;
}

// ============================================================================
// Natural Parameter Conversion
// ============================================================================

ArrayDict LinearMatrixNormalGamma::to_natural_params(const ArrayDict& params) const {
    ArrayDict nat_params;
    
    Array mu = params["mu"];
    Array inv_v = params["inv_v"];
    Array a = params["a"];
    Array b = params["b"];
    
    // eta_1 = inv_v
    nat_params["eta_1"] = inv_v;
    
    // eta_2 = mu @ inv_v
    nat_params["eta_2"] = matmul(mu, inv_v);
    
    // Compute rho = diag(mu @ eta_2^T)
    Array eta_2 = nat_params["eta_2"];
    Array mu_eta2_T = batch_matmul(mu, eta_2.transpose());
    Array rho = mu_eta2_T.diagonal();
    
    // eta_3 = 2b + rho
    nat_params["eta_3"] = b * 2.0f + rho;
    
    // eta_4 = 2(a - 1) + x_dim
    nat_params["eta_4"] = (a - 1.0f) * 2.0f + static_cast<float>(effective_x_dim_);
    
    return nat_params;
}

// ============================================================================
// Expected Values
// ============================================================================

ArrayDict LinearMatrixNormalGamma::expected_inv_sigma() const {
    // E[inv_sigma] = a/b * I
    Array a = get_a()["a"];
    Array b = get_b()["b"];
    Array ratio = a / b;  // Element-wise division
    
    ArrayDict result;
    // Create diagonal matrix with ratio values
    std::vector<int> shape = ratio.shape;
    shape.push_back(y_dim_);
    Array inv_sigma = Array::zeros(shape);
    
    int batch_size = ratio.numel() / y_dim_;
    for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (int i = 0; i < y_dim_; ++i) {
            inv_sigma.data[b_idx * y_dim_ * y_dim_ + i * y_dim_ + i] = ratio.data[b_idx * y_dim_ + i];
        }
    }
    result["inv_sigma"] = inv_sigma;
    return result;
}

ArrayDict LinearMatrixNormalGamma::expected_inv_sigma_x(const ArrayDict* inv_sigma) const {
    ArrayDict result;
    Array e_inv_sigma;
    if (inv_sigma) {
        e_inv_sigma = (*inv_sigma)["inv_sigma"];
    } else {
        e_inv_sigma = expected_inv_sigma()["inv_sigma"];
    }
    Array mu = get_mu()["mu"];
    result["inv_sigma_x"] = matmul(e_inv_sigma, mu);
    return result;
}

ArrayDict LinearMatrixNormalGamma::expected_sigma() const {
    // E[sigma] = b/(a-1) * I
    Array a = get_a()["a"];
    Array b = get_b()["b"];
    Array ratio = b / (a - 1.0f);
    
    ArrayDict result;
    std::vector<int> shape = ratio.shape;
    shape.push_back(y_dim_);
    Array sigma = Array::zeros(shape);
    
    int batch_size = ratio.numel() / y_dim_;
    for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (int i = 0; i < y_dim_; ++i) {
            sigma.data[b_idx * y_dim_ * y_dim_ + i * y_dim_ + i] = ratio.data[b_idx * y_dim_ + i];
        }
    }
    result["sigma"] = sigma;
    return result;
}

ArrayDict LinearMatrixNormalGamma::expected_x_inv_sigma_x(const ArrayDict* inv_sigma_mu) const {
    // E[x^T @ inv_sigma @ x] = y_dim * V + mu^T @ inv_sigma @ mu
    Array mu = get_mu()["mu"];
    Array v = get_v()["v"];
    
    Array e_inv_sigma_mu;
    if (inv_sigma_mu) {
        e_inv_sigma_mu = (*inv_sigma_mu)["inv_sigma_x"];
    } else {
        e_inv_sigma_mu = expected_inv_sigma_x()["inv_sigma_x"];
    }
    
    ArrayDict result;
    // y_dim * V term
    Array y_v = v * static_cast<float>(y_dim_);
    // mu^T @ inv_sigma_mu term
    Array mu_T_inv_sigma_mu = batch_matmul(mu.transpose(), e_inv_sigma_mu);
    
    result["x_inv_sigma_x"] = y_v + mu_T_inv_sigma_mu;
    return result;
}

ArrayDict LinearMatrixNormalGamma::expected_logdet_inv_sigma() const {
    // E[log |inv_sigma|] = sum(digamma(a) - log(b)) over output dimensions
    Array a = get_a()["a"];
    Array b = get_b()["b"];
    
    Array digamma_a = digamma(a);
    Array log_b;
    log_b.data.resize(b.numel());
    for (int i = 0; i < b.numel(); ++i) {
        log_b.data[i] = std::log(b.data[i] + EPSILON);
    }
    log_b.shape = b.shape;
    
    Array result_arr = digamma_a - log_b;
    // Sum over y_dim (axis -2)
    result_arr = result_arr.sum(-2, true);  // Keep dimensions
    
    ArrayDict result;
    result["expected_logdet_inv_sigma"] = result_arr;
    return result;
}

ArrayDict LinearMatrixNormalGamma::logdet_expected_inv_sigma() const {
    // log |E[inv_sigma]| = sum(log(a) - log(b))
    Array a = get_a()["a"];
    Array b = get_b()["b"];
    
    Array log_a, log_b;
    log_a.data.resize(a.numel());
    log_b.data.resize(b.numel());
    for (int i = 0; i < a.numel(); ++i) log_a.data[i] = std::log(a.data[i] + EPSILON);
    for (int i = 0; i < b.numel(); ++i) log_b.data[i] = std::log(b.data[i] + EPSILON);
    log_a.shape = a.shape;
    log_b.shape = b.shape;
    
    Array result_arr = (log_a - log_b).sum(-2, true);
    
    ArrayDict result;
    result["logdet_expected_inv_sigma"] = result_arr;
    return result;
}

ArrayDict LinearMatrixNormalGamma::expected_log_det_inv_sigma_minus_log_det_expected_inv_sigma() const {
    Array a = get_a()["a"];
    Array digamma_a = digamma(a);
    
    Array log_a;
    log_a.data.resize(a.numel());
    for (int i = 0; i < a.numel(); ++i) log_a.data[i] = std::log(a.data[i] + EPSILON);
    log_a.shape = a.shape;
    
    Array result_arr = (digamma_a - log_a).sum(-2, true);
    
    ArrayDict result;
    result["value"] = result_arr;
    return result;
}

ArrayDict LinearMatrixNormalGamma::inv_expected_inv_sigma() const {
    // (E[inv_sigma])^{-1} = b/a * I
    Array a = get_a()["a"];
    Array b = get_b()["b"];
    Array ratio = b / a;
    
    ArrayDict result;
    std::vector<int> shape = ratio.shape;
    shape.push_back(y_dim_);
    Array inv = Array::zeros(shape);
    
    int batch_size = ratio.numel() / y_dim_;
    for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (int i = 0; i < y_dim_; ++i) {
            inv.data[b_idx * y_dim_ * y_dim_ + i * y_dim_ + i] = ratio.data[b_idx * y_dim_ + i];
        }
    }
    result["inv_expected_inv_sigma"] = inv;
    return result;
}

ArrayDict LinearMatrixNormalGamma::expected_inv_sigma_diag() const {
    // Return a/b as vector
    Array a = get_a()["a"];
    Array b = get_b()["b"];
    ArrayDict result;
    result["expected_inv_sigma_diag"] = a / b;
    return result;
}

// ============================================================================
// Likelihood and Statistics
// ============================================================================

ArrayDict LinearMatrixNormalGamma::expected_likelihood_params() const {
    ArrayDict params;
    
    Array e_inv_sigma = expected_inv_sigma()["inv_sigma"];
    Array e_inv_sigma_x = expected_inv_sigma_x()["inv_sigma_x"];
    Array e_x_inv_sigma_x = expected_x_inv_sigma_x()["x_inv_sigma_x"];
    Array e_logdet = expected_logdet_inv_sigma()["expected_logdet_inv_sigma"];
    
    // eta_1 = -0.5 * x_inv_sigma_x
    params["eta_1"] = e_x_inv_sigma_x * (-0.5f);
    
    // eta_2 = inv_sigma_x
    params["eta_2"] = e_inv_sigma_x;
    
    // eta_3 = -0.5 * diag(inv_sigma)
    Array diag = e_inv_sigma.diagonal();
    params["eta_3"] = diag * (-0.5f);
    
    // eta_4 = 0.5 * expected_logdet_inv_sigma
    params["eta_4"] = e_logdet * 0.5f;
    
    return params;
}

ArrayDict LinearMatrixNormalGamma::expected_posterior_statistics() const {
    // Returns expected likelihood params as the statistics
    return expected_likelihood_params();
}

ArrayDict LinearMatrixNormalGamma::expected_log_partition() const {
    throw std::runtime_error("expected_log_partition not implemented");
}

ArrayDict LinearMatrixNormalGamma::compute_log_partition(
    const ArrayDict& mean,
    const ArrayDict& logdet_inv_v,
    const ArrayDict& a,
    const ArrayDict& b) const {
    
    int d = y_dim_;
    int p = effective_x_dim_;
    
    // term_1 = (d*p/2) * log(2*pi)
    float term_1_val = (d * p / 2.0f) * std::log(2.0f * M_PI);
    
    // term_2 = -(d/2) * logdet_inv_v
    Array term_2 = logdet_inv_v.at("logdet_inv_v") * (-d / 2.0f);
    
    // term_3 = sum(lgamma(a) - a*log(b)) over dimensions
    Array lgamma_a = gammaln(a.at("a"));
    Array log_b;
    log_b.data.resize(b.at("b").numel());
    for (int i = 0; i < b.at("b").numel(); ++i) {
        log_b.data[i] = std::log(b.at("b").data[i] + EPSILON);
    }
    log_b.shape = b.at("b").shape;
    Array term_3 = (lgamma_a - a.at("a") * log_b).sum({-2, -1}, true);
    
    ArrayDict result;
    result["log_partition"] = Array::full({1, 1}, term_1_val) + term_2 + term_3;
    return result;
}

ArrayDict LinearMatrixNormalGamma::log_prior_partition() const {
    return compute_log_partition(
        get_prior_mu(),
        get_prior_v(),  // Contains logdet
        get_prior_a(),
        get_prior_b());
}

ArrayDict LinearMatrixNormalGamma::log_posterior_partition() const {
    return compute_log_partition(
        get_mu(),
        get_v(),
        get_a(),
        get_b());
}

// ============================================================================
// Update Methods
// ============================================================================

void LinearMatrixNormalGamma::update_from_data(
    const std::tuple<ArrayDict, ArrayDict>& data,
    const ArrayDict* weights,
    float lr,
    float beta) {
    
    const ArrayDict& X = std::get<0>(data);
    const ArrayDict& Y = std::get<1>(data);
    
    // Compute sufficient statistics
    Array xx = batch_matmul(X.at("data"), X.at("data").transpose());
    Array yy_diag = Y.at("data").diagonal();
    Array yx = batch_matmul(Y.at("data"), X.at("data").transpose());
    
    // Count
    int n_samples = X.at("data").shape[0];
    Array ones = Array::full({n_samples, 1}, 1.0f);
    
    // Handle bias if needed
    if (use_bias_) {
        // Pad X with ones for bias
        // This is simplified - in practice would need proper padding
    }
    
    // Apply weights if provided
    if (weights) {
        xx = xx * weights->at("weights");
        yy_diag = yy_diag * weights->at("weights");
        yx = yx * weights->at("weights");
        ones = ones * weights->at("weights");
    }
    
    // Sum over samples
    ArrayDict stats;
    stats["xx"] = xx.sum(0, false);
    stats["yy"] = yy_diag.sum(0, false);
    stats["yx"] = yx.sum(0, false);
    stats["ones"] = ones.sum(0, false);
    
    update_from_statistics(stats, lr, beta);
}

ArrayDict LinearMatrixNormalGamma::update_from_probabilities(
    const std::tuple<std::shared_ptr<Distribution>, std::shared_ptr<Distribution>>& pXY,
    const ArrayDict* weights,
    float lr,
    float beta,
    bool apply_updates) {
    
    // This would compute statistics from distributions
    // For now, return empty stats
    ArrayDict stats;
    
    if (apply_updates) {
        update_from_statistics(stats, lr, beta);
    }
    
    return stats;
}

void LinearMatrixNormalGamma::update_from_statistics(
    const ArrayDict& stats,
    float lr,
    float beta) {
    
    // Update natural parameters
    // eta_new = (1 - lr) * eta_old + lr * (eta_prior + stats)
    
    ArrayDict scaled_stats = stats * lr;
    ArrayDict scaled_prior = prior_natural_params_ * (lr * (1.0f - beta));
    ArrayDict posterior_past = posterior_natural_params_ * (1.0f - lr * (1.0f - beta));
    
    posterior_natural_params_ = posterior_past + scaled_prior + scaled_stats;
    
    // Convert back to canonical parameters
    // This requires solving for mu, inv_v, a, b from eta
    // For now, just mark cache invalid
    cache_valid_ = false;
    update_cache();
}

ArrayDict LinearMatrixNormalGamma::map_stats_to_params(
    const ArrayDict& likelihood_stats,
    const ArrayDict& counts) const {
    
    // Map likelihood statistics to natural parameter updates
    // xx -> eta_1
    // yx -> eta_2
    // yy -> eta_3 (diagonal)
    // ones -> eta_4
    
    ArrayDict mapped;
    mapped["eta_1"] = likelihood_stats.at("xx");
    mapped["eta_2"] = likelihood_stats.at("yx");
    mapped["eta_3"] = likelihood_stats.at("yy");
    mapped["eta_4"] = counts;
    
    return mapped;
}

// ============================================================================
// Prediction
// ============================================================================

ArrayDict LinearMatrixNormalGamma::expected_log_likelihood(
    const std::tuple<ArrayDict, ArrayDict>& data) const {
    
    const ArrayDict& X = std::get<0>(data);
    const ArrayDict& Y = std::get<1>(data);
    
    Array x = X.at("data");
    Array y = Y.at("data");
    
    Array e_inv_sigma = expected_inv_sigma()["inv_sigma"];
    Array e_inv_sigma_x = expected_inv_sigma_x()["inv_sigma_x"];
    Array e_x_inv_sigma_x = expected_x_inv_sigma_x()["x_inv_sigma_x"];
    Array e_logdet = expected_logdet_inv_sigma()["expected_logdet_inv_sigma"];
    
    Array probs;
    
    if (use_bias_) {
        // Split into weight and bias components
        Array weights = get_weights()["weights"];
        Array bias = get_bias()["bias"];
        
        // More complex computation with bias
        // probs = -0.5 * y^T @ inv_sigma @ y + y^T @ inv_sigma @ (W @ x + b) - 0.5 * x^T @ W^T @ inv_sigma @ W @ x
    } else {
        // probs = -0.5 * y^T @ inv_sigma @ y + y^T @ inv_sigma @ W @ x - 0.5 * x^T @ W^T @ inv_sigma @ W @ x
        Array y_T_inv_sigma_y = batch_matmul(y.transpose(), batch_matmul(e_inv_sigma, y));
        Array y_T_inv_sigma_W_x = batch_matmul(y.transpose(), batch_matmul(e_inv_sigma_x, x));
        Array x_T_W_T_inv_sigma_W_x = batch_matmul(x.transpose(), batch_matmul(e_x_inv_sigma_x, x));
        
        probs = y_T_inv_sigma_W_x - y_T_inv_sigma_y * 0.5f - x_T_W_T_inv_sigma_W_x * 0.5f;
    }
    
    // Add logdet term
    probs = probs + e_logdet.sum({-2, -1}, false) * 0.5f;
    // Subtract constant
    probs = probs - static_cast<float>(y_dim_) * 0.5f * std::log(2.0f * M_PI);
    
    ArrayDict result;
    result["log_likelihood"] = probs;
    return result;
}

ArrayDict LinearMatrixNormalGamma::average_energy(
    const std::tuple<std::shared_ptr<Distribution>, std::shared_ptr<Distribution>>& inputs) const {
    // Similar to expected_log_likelihood but using distribution expectations
    // This is a placeholder implementation
    ArrayDict result;
    result["average_energy"] = Array::zeros({1});
    return result;
}

std::shared_ptr<ExponentialFamily> LinearMatrixNormalGamma::predict(const ArrayDict& x) const {
    // Compute predictive distribution p(y|x)
    // This would return a MultivariateNormal
    // Placeholder - would need full ExponentialFamily implementation
    return nullptr;
}

// ============================================================================
// KL Divergence
// ============================================================================

ArrayDict LinearMatrixNormalGamma::kl_divergence() const {
    ArrayDict kl;
    
    // KL = KL_MN + KL_Gamma
    
    // KL_MN part
    float d = static_cast<float>(y_dim_);
    float p = static_cast<float>(effective_x_dim_);
    
    Array logdet_inv_v = get_v()["v"];  // Contains logdet
    Array prior_logdet_inv_v = get_prior_v()["v"];
    
    Array kl_mn = logdet_inv_v - prior_logdet_inv_v;
    kl_mn = kl_mn * (d / 2.0f);
    kl_mn = kl_mn - Array::full({1}, d * p / 2.0f);
    
    // Trace terms
    Array v = get_v()["v"];
    Array prior_inv_v = get_prior_inv_v()["inv_v"];
    Array trace_v = (prior_inv_v * v).sum({-2, -1}, false);
    kl_mn = kl_mn + trace_v * (d / 2.0f);
    
    // Mu term
    Array mu = get_mu()["mu"];
    Array prior_mu = get_prior_mu()["mu"];
    Array diff = mu - prior_mu;
    Array e_inv_sigma = expected_inv_sigma()["inv_sigma"];
    Array mu_term = (prior_inv_v * batch_matmul(diff.transpose(), batch_matmul(e_inv_sigma, diff))).sum({-2, -1}, false);
    kl_mn = kl_mn + mu_term * 0.5f;
    
    // KL_Gamma part
    Array kl_gamma = kl_divergence_gamma()["kl_gamma"];
    
    kl["kl_divergence"] = kl_mn + kl_gamma;
    return kl;
}

ArrayDict LinearMatrixNormalGamma::kl_divergence_gamma() const {
    Array a = get_a()["a"];
    Array b = get_b()["b"];
    Array prior_a = get_prior_a()["a"];
    Array prior_b = get_prior_b()["b"];
    
    // KL = a_prior * (log(b) - log(b_prior)) + lgamma(a_prior) - lgamma(a)
    //      + (a - a_prior) * digamma(a) - (b - b_prior) * a / b
    
    Array log_b, log_prior_b;
    log_b.data.resize(b.numel());
    log_prior_b.data.resize(prior_b.numel());
    for (int i = 0; i < b.numel(); ++i) log_b.data[i] = std::log(b.data[i] + EPSILON);
    for (int i = 0; i < prior_b.numel(); ++i) log_prior_b.data[i] = std::log(prior_b.data[i] + EPSILON);
    log_b.shape = b.shape;
    log_prior_b.shape = prior_b.shape;
    
    Array kl = prior_a * (log_b - log_prior_b);
    kl = kl + gammaln(prior_a) - gammaln(a);
    kl = kl + (a - prior_a) * digamma(a);
    kl = kl - (b - prior_b) * (a / b);
    
    ArrayDict result;
    result["kl_gamma"] = kl.sum({-2, -1}, false);
    return result;
}

// ============================================================================
// ELBO
// ============================================================================

ArrayDict LinearMatrixNormalGamma::elbo(
    const std::tuple<ArrayDict, ArrayDict>& data,
    const ArrayDict* weights) const {
    
    ArrayDict ell = expected_log_likelihood(data);
    ArrayDict kl = kl_divergence();
    
    ArrayDict result;
    if (weights) {
        result["elbo"] = (ell["log_likelihood"] * weights->at("weights")).sum() - kl["kl_divergence"];
    } else {
        result["elbo"] = ell["log_likelihood"].sum() - kl["kl_divergence"];
    }
    return result;
}

ArrayDict LinearMatrixNormalGamma::elbo_contrib(
    const std::tuple<std::shared_ptr<Distribution>, std::shared_ptr<Distribution>>& pXY,
    const ArrayDict* weights) const {
    
    ArrayDict ae = average_energy(pXY);
    ArrayDict kl = kl_divergence();
    
    ArrayDict result;
    if (weights) {
        result["elbo"] = (ae["average_energy"] * weights->at("weights")).sum() - kl["kl_divergence"];
    } else {
        result["elbo"] = ae["average_energy"].sum() - kl["kl_divergence"];
    }
    return result;
}

// ============================================================================
// Message Passing
// ============================================================================

std::shared_ptr<ExponentialFamily> LinearMatrixNormalGamma::forward_from_normal(
    const std::shared_ptr<ExponentialFamily>& pX,
    bool pass_residual) const {
    
    // Forward message passing using Schur complement
    // pY = int p(y|x) p(x) dx
    
    // This is a placeholder - full implementation would:
    // 1. Extract inv_sigma, inv_sigma_mu from pX
    // 2. Compute joint precision matrix
    // 3. Marginalize over x to get pY
    
    return nullptr;  // Would return MultivariateNormal
}

std::shared_ptr<ExponentialFamily> LinearMatrixNormalGamma::backward_from_normal(
    const std::shared_ptr<ExponentialFamily>& pY,
    bool pass_residual) const {
    
    // Backward message passing
    // pX = int p(y|x) p(y) dy / normalization
    
    return nullptr;  // Would return MultivariateNormal
}

std::shared_ptr<ExponentialFamily> LinearMatrixNormalGamma::variational_forward(
    const std::shared_ptr<Distribution>& pX,
    bool pass_residual) const {
    
    // Fast variational approximation
    // Doesn't marginalize over joint, just uses E[x]
    
    return nullptr;
}

std::shared_ptr<ExponentialFamily> LinearMatrixNormalGamma::variational_backward(
    const std::shared_ptr<Distribution>& pY,
    bool pass_residual) const {
    
    // Fast variational backward
    
    return nullptr;
}

// ============================================================================
// Joint Distribution
// ============================================================================

std::shared_ptr<Distribution> LinearMatrixNormalGamma::joint(
    const std::shared_ptr<Distribution>& pX,
    const std::shared_ptr<Distribution>& pY) const {
    
    // Compute joint p(x, y) = p(y|x) p(x)
    // Used for exact inference in linear dynamical systems
    
    throw std::runtime_error("joint not implemented");
}

// ============================================================================
// Copy
// ============================================================================

std::shared_ptr<Transform> LinearMatrixNormalGamma::copy() const {
    auto copy = std::make_shared<LinearMatrixNormalGamma>(
        posterior_params_,
        prior_params_,
        x_dim_,
        y_dim_,
        use_bias_,
        fixed_precision_,
        batch_shape_);
    return copy;
}

// ============================================================================
// Numerical Helpers
// ============================================================================

std::pair<ArrayDict, ArrayDict> LinearMatrixNormalGamma::inv_and_logdet_stable(const ArrayDict& matrix) const {
    // Stable matrix inversion with logdet computation
    // Uses Cholesky decomposition for symmetric positive definite matrices
    
    Array mat = matrix.at("matrix");
    
    // Add small epsilon for numerical stability if needed
    // Check if matrix is positive definite
    
    auto [inv, logdet] = inv_and_logdet(mat);
    
    ArrayDict inv_dict, logdet_dict;
    inv_dict["inverse"] = inv;
    logdet_dict["logdet"] = Array::full({1}, logdet);
    
    return {inv_dict, logdet_dict};
}

} // namespace transforms
} // namespace axiom

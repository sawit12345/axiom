/*
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

#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>

namespace axiomcuda {
namespace math {

// Constants
constexpr double PI = 3.14159265358979323846;
constexpr double LOG_2PI = 1.83787706640934548356;  // log(2*pi)
constexpr double LOG_PI = 1.14472988584940017414;     // log(pi)

// ============================================================================
// Gamma and Related Functions
// ============================================================================

/**
 * Compute the log gamma function: log(Gamma(x))
 * Uses Lanczos approximation for high accuracy
 * 
 * @param x Input value (must be positive)
 * @return log(Gamma(x))
 */
double gammaln(double x);

/**
 * Vectorized log gamma function
 */
void gammaln(const double* input, double* output, size_t n);

/**
 * Compute the digamma function: psi(x) = d/dx log(Gamma(x))
 * Uses asymptotic expansion and recurrence relation
 * 
 * @param x Input value (must be positive)
 * @return digamma(x)
 */
double digamma(double x);

/**
 * Vectorized digamma function
 */
void digamma(const double* input, double* output, size_t n);

// ============================================================================
// Multivariate Gamma and Digamma Functions
// ============================================================================

/**
 * Compute the multivariate log gamma function of dimension d
 * 
 * Gamma_d(x) = pi^(d(d-1)/4) * prod_{j=1}^d Gamma(x + (1-j)/2)
 * log Gamma_d(x) = d(d-1)/4 * log(pi) + sum_{j=1}^d log Gamma(x + (1-j)/2)
 * 
 * @param x Input value (must be > (d-1)/2)
 * @param d Dimension
 * @return log of multivariate gamma function
 */
double mvgammaln(double x, int d);

/**
 * Vectorized multivariate log gamma function
 */
void mvgammaln(const double* input, int d, double* output, size_t n);

/**
 * Compute the multivariate digamma function of dimension d
 * 
 * psi_d(x) = sum_{j=1}^d psi(x + (1-j)/2)
 * 
 * @param x Input value
 * @param d Dimension
 * @return multivariate digamma value
 */
double mvdigamma(double x, int d);

/**
 * Vectorized multivariate digamma function
 */
void mvdigamma(const double* input, int d, double* output, size_t n);

// ============================================================================
// Log-Sum-Exp and Softmax
// ============================================================================

/**
 * Compute numerically stable log-sum-exp
 * log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
 * 
 * @param x Input array
 * @param n Length of array
 * @return log(sum(exp(x)))
 */
double logsumexp(const double* x, size_t n);

/**
 * Compute numerically stable log-sum-exp along a dimension
 * 
 * @param input Input array of shape [...]
 * @param output Output array
 * @param stride Stride along the dimension to reduce
 * @param n Number of elements along dimension
 * @param batch_size Total number of batches
 */
void logsumexp_dim(const double* input, double* output, 
                   size_t stride, size_t n, size_t batch_size);

/**
 * Compute softmax with optional masking
 * 
 * @param input Input array
 * @param output Output array
 * @param n Length
 * @param mask Optional mask (nullptr for no mask)
 */
void softmax(const double* input, double* output, size_t n, const bool* mask = nullptr);

/**
 * Batched softmax along a dimension
 * 
 * @param input Input array
 * @param output Output array  
 * @param stride Stride along dimension
 * @param dim_size Size of softmax dimension
 * @param batch_size Total batches
 * @param mask Optional mask array
 */
void softmax_dim(const double* input, double* output,
                 size_t stride, size_t dim_size, size_t batch_size,
                 const bool* mask = nullptr);

// ============================================================================
// Gaussian Log-Likelihood
// ============================================================================

/**
 * Compute Gaussian log-likelihood
 * 
 * log p(x | mu, sigma) = -0.5 * (x-mu)^T * Sigma^{-1} * (x-mu)
 *                      - 0.5 * log|Sigma| - 0.5 * D * log(2*pi)
 * 
 * @param x Data point (D x 1)
 * @param mean Mean (D x 1)
 * @param inv_cov Inverse covariance (D x D)
 * @param logdet_inv_cov Log determinant of inverse covariance
 * @param dim Dimension D
 * @return Log likelihood value
 */
double gaussian_loglik(const double* x, const double* mean, 
                       const double* inv_cov, double logdet_inv_cov, int dim);

/**
 * Batched Gaussian log-likelihood
 * 
 * @param x Data (batch_size x D)
 * @param mean Mean (batch_size x D)
 * @param inv_cov Inverse covariance (batch_size x D x D) 
 * @param logdet_inv_cov Log determinant (batch_size)
 * @param dim Dimension D
 * @param batch_size Number of batches
 * @param output Output log-likelihoods (batch_size)
 */
void gaussian_loglik_batch(const double* x, const double* mean,
                           const double* inv_cov, const double* logdet_inv_cov,
                           int dim, size_t batch_size, double* output);

/**
 * Simplified Gaussian log-likelihood with isotropic covariance
 * 
 * @param x Data (D)
 * @param mean Mean (D)
 * @param sigma_sqr Variance
 * @param dim Dimension D
 * @return Log likelihood
 */
double gaussian_loglik_isotropic(const double* x, const double* mean, 
                                  double sigma_sqr, int dim);

/**
 * Batched isotropic Gaussian log-likelihood
 * 
 * @param x Data (batch x D)
 * @param mean Mean (batch x K x D)
 * @param sigma_sqr Variance (batch x K)
 * @param dim Dimension D
 * @param batch_size Batch size
 * @param num_components K
 * @param output Output (batch x K)
 */
void gaussian_loglik_isotropic_batch(const double* x, const double* mean,
                                     const double* sigma_sqr, int dim,
                                     size_t batch_size, size_t num_components,
                                     double* output);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Force matrix to be symmetric: (A + A^T) / 2
 */
void symmetrize(double* matrix, int dim);

/**
 * Make matrix positive definite by adding epsilon to diagonal if needed
 */
void make_positive_definite(double* matrix, int dim, double epsilon = 1e-6);

/**
 * Check if matrix is symmetric
 */
bool is_symmetric(const double* matrix, int dim, double tol = 1e-10);

/**
 * Element-wise operations
 */
void add(const double* a, const double* b, double* c, size_t n);
void subtract(const double* a, const double* b, double* c, size_t n);
void multiply(const double* a, const double* b, double* c, size_t n);
void divide(const double* a, const double* b, double* c, size_t n);
void scale(const double* a, double s, double* c, size_t n);

} // namespace math
} // namespace axiomcuda

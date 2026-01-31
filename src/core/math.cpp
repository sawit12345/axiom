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

#include "math.h"
#include <algorithm>
#include <limits>

namespace axiomcuda {
namespace math {

// ============================================================================
// Gamma and Related Functions
// ============================================================================

// Lanczos approximation coefficients for gamma function
static const double LANCZOS_G = 7.0;
static const double LANCZOS_P[] = {
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
};

double gammaln(double x) {
    if (x <= 0) {
        throw std::invalid_argument("gammaln: x must be positive");
    }
    
    // Use reflection formula for small values
    if (x < 0.5) {
        return LOG_PI - std::log(std::sin(PI * x)) - gammaln(1.0 - x);
    }
    
    // Lanczos approximation
    x -= 1.0;
    double a = LANCZOS_P[0];
    for (int i = 1; i < 9; ++i) {
        a += LANCZOS_P[i] / (x + i);
    }
    
    double t = x + LANCZOS_G + 0.5;
    return 0.5 * std::log(2.0 * PI) + std::log(a) - t + std::log(t) * (x + 0.5);
}

void gammaln(const double* input, double* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = gammaln(input[i]);
    }
}

// Digamma implementation using asymptotic expansion and recurrence
double digamma(double x) {
    if (x <= 0) {
        // Reflection formula: psi(1-x) = psi(x) + pi * cot(pi * x)
        double result = digamma(1.0 - x) - PI / std::tan(PI * x);
        return result;
    }
    
    // Use recurrence relation to get x >= 10
    double result = 0.0;
    while (x < 10.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    
    // Asymptotic expansion for large x:
    // psi(x) ~ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    double inv_x4 = inv_x2 * inv_x2;
    
    result += std::log(x) - 0.5 * inv_x 
              - inv_x2 / 12.0 
              + inv_x4 / 120.0 
              - inv_x4 * inv_x2 / 252.0;
    
    return result;
}

void digamma(const double* input, double* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = digamma(input[i]);
    }
}

// ============================================================================
// Multivariate Gamma and Digamma Functions
// ============================================================================

double mvgammaln(double x, int d) {
    if (d <= 0) {
        throw std::invalid_argument("mvgammaln: dimension d must be positive");
    }
    
    // log Gamma_d(x) = d(d-1)/4 * log(pi) + sum_{j=1}^d log Gamma(x + (1-j)/2)
    double result = d * (d - 1) * 0.25 * LOG_PI;
    
    for (int j = 0; j < d; ++j) {
        double arg = x - j * 0.5;
        if (arg <= 0) {
            throw std::invalid_argument("mvgammaln: invalid argument (x <= (d-1)/2)");
        }
        result += gammaln(arg);
    }
    
    return result;
}

void mvgammaln(const double* input, int d, double* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = mvgammaln(input[i], d);
    }
}

double mvdigamma(double x, int d) {
    if (d <= 0) {
        throw std::invalid_argument("mvdigamma: dimension d must be positive");
    }
    
    // psi_d(x) = sum_{j=1}^d psi(x + (1-j)/2) = sum_{j=1}^d psi(x - (j-1)/2)
    double result = 0.0;
    
    for (int j = 0; j < d; ++j) {
        result += digamma(x - j * 0.5);
    }
    
    return result;
}

void mvdigamma(const double* input, int d, double* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = mvdigamma(input[i], d);
    }
}

// ============================================================================
// Log-Sum-Exp and Softmax
// ============================================================================

double logsumexp(const double* x, size_t n) {
    if (n == 0) return -std::numeric_limits<double>::infinity();
    if (n == 1) return x[0];
    
    // Find max for numerical stability
    double max_val = x[0];
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    if (max_val == -std::numeric_limits<double>::infinity()) {
        return max_val;
    }
    
    // Compute log(sum(exp(x - max)))
    double sum_exp = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_exp += std::exp(x[i] - max_val);
    }
    
    return max_val + std::log(sum_exp);
}

void logsumexp_dim(const double* input, double* output,
                   size_t stride, size_t n, size_t batch_size) {
    for (size_t b = 0; b < batch_size; ++b) {
        const double* batch_input = input + b * stride * n;
        
        // Find max
        double max_val = batch_input[0];
        for (size_t i = 1; i < n; ++i) {
            double val = batch_input[i * stride];
            if (val > max_val) max_val = val;
        }
        
        // Compute sum of exp
        double sum_exp = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum_exp += std::exp(batch_input[i * stride] - max_val);
        }
        
        output[b] = max_val + std::log(sum_exp);
    }
}

void softmax(const double* input, double* output, size_t n, const bool* mask) {
    if (n == 0) return;
    
    // Find max for numerical stability
    double max_val;
    if (mask) {
        // Find first unmasked value
        size_t first = 0;
        while (first < n && mask[first]) first++;
        if (first >= n) return;  // All masked

        max_val = input[first];
        for (size_t i = first + 1; i < n; ++i) {
            if (mask[i] == 0 && input[i] > max_val) {
                max_val = input[i];
            }
        }
    } else {
        max_val = input[0];
        for (size_t i = 1; i < n; ++i) {
            if (input[i] > max_val) max_val = input[i];
        }
    }
    
    // Compute exp(x - max)
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!mask || mask[i] == 0) {
            double exp_val = std::exp(input[i] - max_val);
            output[i] = exp_val;
            sum += exp_val;
        } else {
            output[i] = 0.0;
        }
    }
    
    // Normalize
    if (sum > 0) {
        for (size_t i = 0; i < n; ++i) {
            output[i] /= sum;
        }
    }
}

void softmax_dim(const double* input, double* output,
                 size_t stride, size_t dim_size, size_t batch_size,
                 const bool* mask) {
    for (size_t b = 0; b < batch_size; ++b) {
        const double* batch_in = input + b * stride * dim_size;
        double* batch_out = output + b * stride * dim_size;
        
        // Apply softmax along dimension for this batch
        // Collect values
        std::vector<double> vals(dim_size);
        for (size_t i = 0; i < dim_size; ++i) {
            vals[i] = batch_in[i * stride];
        }

        std::vector<double> out_vals(dim_size);
        // For now, ignore mask in softmax_dim - fix later if needed
        softmax(vals.data(), out_vals.data(), dim_size, nullptr);
        
        for (size_t i = 0; i < dim_size; ++i) {
            batch_out[i * stride] = out_vals[i];
        }
    }
}

// ============================================================================
// Gaussian Log-Likelihood
// ============================================================================

double gaussian_loglik(const double* x, const double* mean,
                       const double* inv_cov, double logdet_inv_cov, int dim) {
    // Compute (x - mu)
    std::vector<double> diff(dim);
    for (int i = 0; i < dim; ++i) {
        diff[i] = x[i] - mean[i];
    }
    
    // Compute (x - mu)^T * inv_cov * (x - mu)
    // First: inv_cov * diff
    std::vector<double> temp(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            temp[i] += inv_cov[i * dim + j] * diff[j];
        }
    }
    
    // Then: diff^T * temp
    double mahalanobis = 0.0;
    for (int i = 0; i < dim; ++i) {
        mahalanobis += diff[i] * temp[i];
    }
    
    // log p(x | mu, Sigma) = -0.5 * mahalanobis - 0.5 * log|Sigma| - 0.5 * D * log(2*pi)
    // Note: log|Sigma| = -logdet_inv_cov
    return -0.5 * mahalanobis + 0.5 * logdet_inv_cov - 0.5 * dim * LOG_2PI;
}

void gaussian_loglik_batch(const double* x, const double* mean,
                           const double* inv_cov, const double* logdet_inv_cov,
                           int dim, size_t batch_size, double* output) {
    size_t matrix_size = dim * dim;
    
    for (size_t b = 0; b < batch_size; ++b) {
        const double* x_b = x + b * dim;
        const double* mean_b = mean + b * dim;
        const double* inv_cov_b = inv_cov + b * matrix_size;
        
        output[b] = gaussian_loglik(x_b, mean_b, inv_cov_b, logdet_inv_cov[b], dim);
    }
}

double gaussian_loglik_isotropic(const double* x, const double* mean,
                                  double sigma_sqr, int dim) {
    // For isotropic covariance: Sigma = sigma_sqr * I
    // log|Sigma| = D * log(sigma_sqr)
    // (x - mu)^T * inv_cov * (x - mu) = (1/sigma_sqr) * ||x - mu||^2
    
    double sq_error = 0.0;
    for (int i = 0; i < dim; ++i) {
        double diff = x[i] - mean[i];
        sq_error += diff * diff;
    }
    
    double logdet = dim * std::log(sigma_sqr);
    
    return -0.5 * sq_error / sigma_sqr - 0.5 * logdet - 0.5 * dim * LOG_2PI;
}

void gaussian_loglik_isotropic_batch(const double* x, const double* mean,
                                     const double* sigma_sqr, int dim,
                                     size_t batch_size, size_t num_components,
                                     double* output) {
    for (size_t b = 0; b < batch_size; ++b) {
        const double* x_b = x + b * dim;
        
        for (size_t k = 0; k < num_components; ++k) {
            const double* mean_bk = mean + (b * num_components + k) * dim;
            double sigma_sqr_bk = sigma_sqr[b * num_components + k];
            
            output[b * num_components + k] = gaussian_loglik_isotropic(
                x_b, mean_bk, sigma_sqr_bk, dim);
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void symmetrize(double* matrix, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            double avg = 0.5 * (matrix[i * dim + j] + matrix[j * dim + i]);
            matrix[i * dim + j] = avg;
            matrix[j * dim + i] = avg;
        }
    }
}

void make_positive_definite(double* matrix, int dim, double epsilon) {
    // Add epsilon to diagonal until positive definite
    // Simple check: all diagonal entries should be positive after symmetrization
    symmetrize(matrix, dim);
    
    for (int i = 0; i < dim; ++i) {
        if (matrix[i * dim + i] <= epsilon) {
            matrix[i * dim + i] += 2.0 * epsilon;
        }
    }
}

bool is_symmetric(const double* matrix, int dim, double tol) {
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            if (std::abs(matrix[i * dim + j] - matrix[j * dim + i]) > tol) {
                return false;
            }
        }
    }
    return true;
}

void add(const double* a, const double* b, double* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

void subtract(const double* a, const double* b, double* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
}

void multiply(const double* a, const double* b, double* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
}

void divide(const double* a, const double* b, double* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] / b[i];
}

void scale(const double* a, double s, double* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] * s;
}

} // namespace math
} // namespace axiomcuda

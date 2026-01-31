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

#include "random.h"
#include "linalg.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace axiomcuda {
namespace random {

// ============================================================================
// Threefry PRNG Constants
// ============================================================================

// Threefry rotation constants
static const int R_64x2_0_0 = 16;
static const int R_64x2_1_0 = 42;
static const int R_64x2_2_0 = 12;
static const int R_64x2_3_0 = 31;
static const int R_64x2_4_0 = 16;
static const int R_64x2_5_0 = 32;
static const int R_64x2_6_0 = 24;
static const int R_64x2_7_0 = 21;

// S-box constants (from Philox/Threefry paper)
static const uint64_t S0 = 0x9E3779B97F4A7C15ULL;  // Golden ratio * 2^64
static const uint64_t S1 = 0xBB67AE8584CAA73BULL;  // sqrt(3) - 1 * 2^64

// Helper for rotating 64-bit integers
static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// ============================================================================
// PRNG Key Management
// ============================================================================

PRNGKey PRNGKey::from_seed(uint64_t seed) {
    // Use splitmix64 to expand single seed into 128-bit key
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    uint64_t k0 = z ^ (z >> 31);
    
    z = k0 + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    uint64_t k1 = z ^ (z >> 31);
    
    return PRNGKey(k0, k1);
}

PRNGKey PRNGKey::from_ints(uint32_t i0, uint32_t i1) {
    return PRNGKey((static_cast<uint64_t>(i0) << 32) | i0,
                   (static_cast<uint64_t>(i1) << 32) | i1);
}

void split_key(const PRNGKey& key, int num, PRNGKey* output) {
    // Use Threefry to generate new keys
    for (int i = 0; i < num; ++i) {
        uint64_t bits = threefry_random(key, static_cast<uint64_t>(i));
        uint64_t bits2 = threefry_random(key, static_cast<uint64_t>(i) + 0x100000000ULL);
        output[i] = PRNGKey(bits, bits2);
    }
}

void split_key_2(const PRNGKey& key, PRNGKey& key1, PRNGKey& key2) {
    PRNGKey outputs[2];
    split_key(key, 2, outputs);
    key1 = outputs[0];
    key2 = outputs[1];
}

uint64_t fold_key(const PRNGKey& key, uint64_t counter) {
    return threefry_random(key, counter);
}

// ============================================================================
// Threefry PRNG Core
// ============================================================================

uint64_t threefry_random(const PRNGKey& key, uint64_t counter) {
    // Threefry 2x64 algorithm (simplified version compatible with JAX)
    // Uses 12 rounds
    
    uint64_t X[2];
    X[0] = counter;
    X[1] = counter ^ key.key[0];  // Simple counter setup
    
    uint64_t ks[3];
    ks[0] = key.key[0];
    ks[1] = key.key[1];
    ks[2] = S0 ^ ks[0] ^ ks[1];
    
    // Initial injection
    X[0] += ks[0];
    X[1] += ks[1];
    
    // Rounds
    // Round 1
    X[0] += X[1];
    X[1] = rotl64(X[1], R_64x2_0_0) ^ X[0];
    
    X[0] += X[1];
    X[1] = rotl64(X[1], R_64x2_1_0) ^ X[0];
    
    X[0] += ks[1];
    X[1] += ks[2] + 1;
    
    // Round 2
    X[0] += X[1];
    X[1] = rotl64(X[1], R_64x2_2_0) ^ X[0];
    
    X[0] += X[1];
    X[1] = rotl64(X[1], R_64x2_3_0) ^ X[0];
    
    X[0] += ks[2];
    X[1] += ks[0] + 2;
    
    // Round 3
    X[0] += X[1];
    X[1] = rotl64(X[1], R_64x2_4_0) ^ X[0];
    
    X[0] += X[1];
    X[1] = rotl64(X[1], R_64x2_5_0) ^ X[0];
    
    X[0] += ks[0];
    X[1] += ks[1] + 3;
    
    // Round 4
    X[0] += X[1];
    X[1] = rotl64(X[1], R_64x2_6_0) ^ X[0];
    
    X[0] += X[1];
    X[1] = rotl64(X[1], R_64x2_7_0) ^ X[0];
    
    X[0] += ks[1];
    X[1] += ks[2] + 4;
    
    return X[0];
}

void threefry_random_batch(const PRNGKey& key, uint64_t start_counter, 
                           size_t n, uint64_t* output) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = threefry_random(key, start_counter + i);
    }
}

// ============================================================================
// Uniform Distribution
// ============================================================================

static inline double uint64_to_double(uint64_t bits) {
    // Convert 64 random bits to double in [0, 1)
    // Use 53 bits of precision (IEEE 754 double has 53-bit mantissa)
    const uint64_t mantissa = bits & ((1ULL << 53) - 1);
    return static_cast<double>(mantissa) / static_cast<double>(1ULL << 53);
}

double uniform(const PRNGKey& key, uint64_t counter) {
    return uint64_to_double(threefry_random(key, counter));
}

double uniform_range(const PRNGKey& key, uint64_t counter, double min, double max) {
    return min + uniform(key, counter) * (max - min);
}

void uniform_batch(const PRNGKey& key, size_t n, double* output) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = uniform(key, i);
    }
}

void uniform_range_batch(const PRNGKey& key, size_t n, double min, double max, double* output) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = uniform_range(key, i, min, max);
    }
}

// ============================================================================
// Normal (Gaussian) Distribution
// ============================================================================

// Box-Muller transform: generates 2 normal samples from 2 uniform samples
// We use only the first one per counter to simplify interface
// User should increment counter by 2 for independent samples

double normal(const PRNGKey& key, uint64_t counter) {
    // Box-Muller: generate two independent normals
    // We use counter for u1, counter+1 for u2
    double u1 = uniform(key, counter);
    double u2 = uniform(key, counter + 1);
    
    // Avoid log(0)
    if (u1 <= 0.0) u1 = 1e-15;
    
    double radius = std::sqrt(-2.0 * std::log(u1));
    double angle = 2.0 * M_PI * u2;
    
    return radius * std::cos(angle);
}

double normal_scaled(const PRNGKey& key, uint64_t counter, double mean, double std) {
    return mean + std * normal(key, counter);
}

void normal_batch(const PRNGKey& key, size_t n, double* output) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = normal(key, i * 2);  // Use 2 counters per sample
    }
}

void normal_scaled_batch(const PRNGKey& key, size_t n, double mean, double std, double* output) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = normal_scaled(key, i * 2, mean, std);
    }
}

// ============================================================================
// Multivariate Normal Distribution
// ============================================================================

void multivariate_normal(const PRNGKey& key, const double* mean, 
                         const double* cov_chol, int dim, double* output) {
    // Generate standard normal samples
    std::vector<double> z(dim);
    for (int i = 0; i < dim; ++i) {
        z[i] = normal(key, static_cast<uint64_t>(i) * 2);
    }
    
    // Compute output = mean + L * z
    // where L is the Cholesky factor (lower triangular)
    for (int i = 0; i < dim; ++i) {
        double sum = mean[i];
        for (int j = 0; j <= i; ++j) {
            sum += cov_chol[i * dim + j] * z[j];
        }
        output[i] = sum;
    }
}

void multivariate_normal_batch(const PRNGKey& key, const double* mean,
                               const double* cov_chol, int dim,
                               size_t batch_size, double* output) {
    // Split key for each batch
    std::vector<PRNGKey> keys(batch_size);
    split_key(key, static_cast<int>(batch_size), keys.data());
    
    for (size_t b = 0; b < batch_size; ++b) {
        multivariate_normal(keys[b], mean + b * dim, 
                           cov_chol + b * dim * dim, dim, 
                           output + b * dim);
    }
}

void multivariate_normal_multi(const PRNGKey& key, const double* mean,
                               const double* cov_chol, int dim,
                               size_t num_samples, double* output) {
    // Split key for each sample
    std::vector<PRNGKey> keys(num_samples);
    split_key(key, static_cast<int>(num_samples), keys.data());
    
    for (size_t s = 0; s < num_samples; ++s) {
        multivariate_normal(keys[s], mean, cov_chol, dim, output + s * dim);
    }
}

// ============================================================================
// Dirichlet Distribution
// ============================================================================

void dirichlet(const PRNGKey& key, const double* alpha, int dim, double* output) {
    // Generate gamma samples for each dimension
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        output[i] = gamma(key, static_cast<uint64_t>(i) * 4, alpha[i], 1.0);
        sum += output[i];
    }
    
    // Normalize
    if (sum > 0.0) {
        for (int i = 0; i < dim; ++i) {
            output[i] /= sum;
        }
    }
}

void dirichlet_batch(const PRNGKey& key, const double* alpha, int dim,
                     size_t batch_size, double* output) {
    // Split key for each batch
    std::vector<PRNGKey> keys(batch_size);
    split_key(key, static_cast<int>(batch_size), keys.data());
    
    for (size_t b = 0; b < batch_size; ++b) {
        dirichlet(keys[b], alpha + b * dim, dim, output + b * dim);
    }
}

// ============================================================================
// Categorical Distribution
// ============================================================================

int categorical(const PRNGKey& key, const double* probs, int dim) {
    double u = uniform(key, 0);
    double cumsum = 0.0;
    
    for (int i = 0; i < dim; ++i) {
        cumsum += probs[i];
        if (u < cumsum) {
            return i;
        }
    }
    
    // Fallback (shouldn't happen if probs sum to 1)
    return dim - 1;
}

int categorical_log(const PRNGKey& key, const double* log_probs, int dim) {
    // Compute log-sum-exp for normalization
    double max_log_prob = log_probs[0];
    for (int i = 1; i < dim; ++i) {
        if (log_probs[i] > max_log_prob) {
            max_log_prob = log_probs[i];
        }
    }
    
    double sum_exp = 0.0;
    for (int i = 0; i < dim; ++i) {
        sum_exp += std::exp(log_probs[i] - max_log_prob);
    }
    double log_sum_exp = max_log_prob + std::log(sum_exp);
    
    // Convert to probs
    std::vector<double> probs(dim);
    for (int i = 0; i < dim; ++i) {
        probs[i] = std::exp(log_probs[i] - log_sum_exp);
    }
    
    return categorical(key, probs.data(), dim);
}

void categorical_batch(const PRNGKey& key, const double* probs, int dim,
                       size_t batch_size, int* output) {
    // Split key for each batch
    std::vector<PRNGKey> keys(batch_size);
    split_key(key, static_cast<int>(batch_size), keys.data());
    
    for (size_t b = 0; b < batch_size; ++b) {
        output[b] = categorical(keys[b], probs + b * dim, dim);
    }
}

int gumbel_max(const PRNGKey& key, const double* logits, int dim) {
    // Gumbel-max trick: argmax(logits + gumbel_noise)
    // Gumbel(0,1) = -log(-log(uniform(0,1)))
    
    int max_idx = 0;
    double max_val = logits[0];
    
    for (int i = 0; i < dim; ++i) {
        double u = uniform(key, static_cast<uint64_t>(i));
        // Avoid log(0) and log(1)
        if (u <= 0.0) u = 1e-15;
        if (u >= 1.0) u = 1.0 - 1e-15;
        
        double gumbel = -std::log(-std::log(u));
        double val = logits[i] + gumbel;
        
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    return max_idx;
}

// ============================================================================
// Gamma Distribution
// ============================================================================

// Marsaglia and Tsang's method for Gamma(shape >= 1)
// For shape < 1, we use the fact that Gamma(shape) = Gamma(shape+1) * U^(1/shape)

double gamma(const PRNGKey& key, uint64_t counter, double shape, double scale) {
    if (shape <= 0.0) {
        throw std::invalid_argument("gamma: shape must be positive");
    }
    if (scale <= 0.0) {
        throw std::invalid_argument("gamma: scale must be positive");
    }
    
    bool shape_less_than_1 = shape < 1.0;
    double d, c, x, v, u;
    
    if (shape_less_than_1) {
        // Use boost to shape+1, then scale
        shape += 1.0;
    }
    
    d = shape - 1.0 / 3.0;
    c = 1.0 / std::sqrt(9.0 * d);
    
    while (true) {
        do {
            x = normal(key, counter);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        
        v = v * v * v;
        u = uniform(key, counter + 1);
        
        if (u < 1.0 - 0.0331 * x * x * x * x) {
            break;
        }
        
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
            break;
        }
        
        counter += 2;  // Try again with new random numbers
    }
    
    double result = d * v * scale;
    
    if (shape_less_than_1) {
        // Compensate for the shape boost
        double u_boost = uniform(key, counter + 2);
        if (u_boost <= 0.0) u_boost = 1e-15;
        result *= std::pow(u_boost, 1.0 / (shape - 1.0));
    }
    
    return result;
}

void gamma_batch(const PRNGKey& key, size_t n, double shape, double scale, double* output) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = gamma(key, i * 4, shape, scale);  // Use 4 counters per sample
    }
}

// ============================================================================
// Advanced Utilities
// ============================================================================

void permutation(const PRNGKey& key, int n, int* output) {
    // Initialize with 0 to n-1
    for (int i = 0; i < n; ++i) {
        output[i] = i;
    }
    
    // Fisher-Yates shuffle
    for (int i = n - 1; i > 0; --i) {
        uint64_t counter = static_cast<uint64_t>(n - 1 - i);
        int j = static_cast<int>(uniform(key, counter) * (i + 1));
        if (j > i) j = i;
        std::swap(output[i], output[j]);
    }
}

void randint(const PRNGKey& key, size_t n, int min, int max, int* output) {
    if (min >= max) {
        throw std::invalid_argument("randint: min must be less than max");
    }
    
    int range = max - min;
    for (size_t i = 0; i < n; ++i) {
        output[i] = min + static_cast<int>(uniform(key, i) * range);
    }
}

void shuffle(const PRNGKey& key, void* data, size_t n, size_t elem_size) {
    char* bytes = static_cast<char*>(data);
    std::vector<char> temp(elem_size);
    
    for (size_t i = n - 1; i > 0; --i) {
        uint64_t counter = static_cast<uint64_t>(n - 1 - i);
        size_t j = static_cast<size_t>(uniform(key, counter) * (i + 1));
        if (j > i) j = i;
        
        // Swap elements i and j
        std::memcpy(temp.data(), bytes + i * elem_size, elem_size);
        std::memcpy(bytes + i * elem_size, bytes + j * elem_size, elem_size);
        std::memcpy(bytes + j * elem_size, temp.data(), elem_size);
    }
}

void bernoulli(const PRNGKey& key, size_t n, double p, int* output) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("bernoulli: p must be in [0, 1]");
    }
    
    for (size_t i = 0; i < n; ++i) {
        output[i] = (uniform(key, i) < p) ? 1 : 0;
    }
}

} // namespace random
} // namespace axiomcuda

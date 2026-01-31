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

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <array>

namespace axiomcuda {
namespace random {

// ============================================================================
// PRNG Key Management (JAX-style Threefry)
// ============================================================================

/**
 * Random key structure (similar to JAX PRNGKey)
 * Uses Threefry counter-based PRNG for parallel generation
 * 
 * Key structure: two 64-bit integers (4 32-bit words)
 * This is compatible with JAX's PRNGKey design
 */
struct PRNGKey {
    uint64_t key[2];
    
    PRNGKey(uint64_t k0 = 0, uint64_t k1 = 0) : key{k0, k1} {}
    
    // Initialize from a single 64-bit seed
    static PRNGKey from_seed(uint64_t seed);
    
    // Initialize from two integers (like jax.random.PRNGKey)
    static PRNGKey from_ints(uint32_t i0, uint32_t i1);
    
    bool operator==(const PRNGKey& other) const {
        return key[0] == other.key[0] && key[1] == other.key[1];
    }
    
    bool operator!=(const PRNGKey& other) const {
        return !(*this == other);
    }
};

/**
 * Split a PRNG key into multiple keys (JAX-style key splitting)
 * This is the proper way to generate independent random numbers in parallel
 * 
 * @param key Input key
 * @param num Number of keys to split into
 * @param output Output array of keys (must have space for num keys)
 */
void split_key(const PRNGKey& key, int num, PRNGKey* output);

/**
 * Split key into exactly 2 keys (most common case)
 * 
 * @param key Input key
 * @param key1 First output key
 * @param key2 Second output key
 */
void split_key_2(const PRNGKey& key, PRNGKey& key1, PRNGKey& key2);

/**
 * Fold a key to generate random bits (internal use)
 */
uint64_t fold_key(const PRNGKey& key, uint64_t counter);

// ============================================================================
// Threefry PRNG Core
// ============================================================================

/**
 * Threefry 2x64 counter-based PRNG
 * Generates random bits from key and counter
 * 
 * @param key PRNG key (2x64 bits)
 * @param counter Counter value (unique for each random number in parallel)
 * @return 64 random bits
 */
uint64_t threefry_random(const PRNGKey& key, uint64_t counter);

/**
 * Generate multiple random bits in batch
 * 
 * @param key PRNG key
 * @param start_counter Starting counter
 * @param n Number of values to generate
 * @param output Output array (n values)
 */
void threefry_random_batch(const PRNGKey& key, uint64_t start_counter, 
                           size_t n, uint64_t* output);

// ============================================================================
// Uniform Distribution
// ============================================================================

/**
 * Generate uniform random number in [0, 1)
 * 
 * @param key PRNG key
 * @param counter Counter for this specific random number
 * @return Uniform random number in [0, 1)
 */
double uniform(const PRNGKey& key, uint64_t counter);

/**
 * Generate uniform random number in [min, max)
 * 
 * @param key PRNG key
 * @param counter Counter
 * @param min Minimum value
 * @param max Maximum value
 * @return Uniform random number in [min, max)
 */
double uniform_range(const PRNGKey& key, uint64_t counter, double min, double max);

/**
 * Generate batch of uniform random numbers
 * 
 * @param key PRNG key
 * @param n Number of samples
 * @param output Output array (n values)
 */
void uniform_batch(const PRNGKey& key, size_t n, double* output);

/**
 * Generate batch of uniform random numbers in range
 * 
 * @param key PRNG key
 * @param n Number of samples
 * @param min Minimum value
 * @param max Maximum value
 * @param output Output array
 */
void uniform_range_batch(const PRNGKey& key, size_t n, double min, double max, double* output);

// ============================================================================
// Normal (Gaussian) Distribution
// ============================================================================

/**
 * Generate standard normal random number (Box-Muller transform)
 * 
 * @param key PRNG key
 * @param counter Counter (uses counter and counter+1)
 * @return Standard normal random number
 */
double normal(const PRNGKey& key, uint64_t counter);

/**
 * Generate normal random number with given mean and std
 * 
 * @param key PRNG key
 * @param counter Counter
 * @param mean Mean value
 * @param std Standard deviation
 * @return Normal random number
 */
double normal_scaled(const PRNGKey& key, uint64_t counter, double mean, double std);

/**
 * Generate batch of standard normal random numbers
 * 
 * @param key PRNG key
 * @param n Number of samples
 * @param output Output array
 */
void normal_batch(const PRNGKey& key, size_t n, double* output);

/**
 * Generate batch of scaled normal random numbers
 * 
 * @param key PRNG key
 * @param n Number of samples
 * @param mean Mean value
 * @param std Standard deviation
 * @param output Output array
 */
void normal_scaled_batch(const PRNGKey& key, size_t n, double mean, double std, double* output);

// ============================================================================
// Multivariate Normal Distribution
// ============================================================================

/**
 * Generate multivariate normal sample
 * 
 * x ~ N(mean, cov)
 * 
 * Algorithm:
 * 1. Compute Cholesky decomposition: cov = L * L^T
 * 2. Generate z ~ N(0, I)
 * 3. Return mean + L * z
 * 
 * @param key PRNG key
 * @param mean Mean vector (dim)
 * @param cov Cholesky factor L of covariance (dim x dim, lower triangular)
 * @param dim Dimension
 * @param output Output sample (dim)
 */
void multivariate_normal(const PRNGKey& key, const double* mean, 
                         const double* cov_chol, int dim, double* output);

/**
 * Generate batch of multivariate normal samples
 * 
 * @param key PRNG key
 * @param mean Mean vectors (batch_size x dim)
 * @param cov_chol Cholesky factors (batch_size x dim x dim)
 * @param dim Dimension
 * @param batch_size Number of samples
 * @param output Output samples (batch_size x dim)
 */
void multivariate_normal_batch(const PRNGKey& key, const double* mean,
                               const double* cov_chol, int dim,
                               size_t batch_size, double* output);

/**
 * Generate multiple samples from same distribution
 * 
 * @param key PRNG key
 * @param mean Mean vector (dim)
 * @param cov_chol Cholesky factor (dim x dim)
 * @param dim Dimension
 * @param num_samples Number of samples
 * @param output Output (num_samples x dim)
 */
void multivariate_normal_multi(const PRNGKey& key, const double* mean,
                               const double* cov_chol, int dim,
                               size_t num_samples, double* output);

// ============================================================================
// Dirichlet Distribution
// ============================================================================

/**
 * Generate Dirichlet random sample
 * 
 * x ~ Dirichlet(alpha)
 * where alpha is concentration parameters
 * 
 * Algorithm:
 * 1. Generate gamma samples: y_i ~ Gamma(alpha_i, 1)
 * 2. Normalize: x_i = y_i / sum(y)
 * 
 * @param key PRNG key
 * @param alpha Concentration parameters (dim)
 * @param dim Dimension
 * @param output Output sample (dim)
 */
void dirichlet(const PRNGKey& key, const double* alpha, int dim, double* output);

/**
 * Generate batch of Dirichlet samples
 * 
 * @param key PRNG key
 * @param alpha Concentration parameters (batch_size x dim)
 * @param dim Dimension
 * @param batch_size Number of samples
 * @param output Output (batch_size x dim)
 */
void dirichlet_batch(const PRNGKey& key, const double* alpha, int dim,
                     size_t batch_size, double* output);

// ============================================================================
// Categorical Distribution
// ============================================================================

/**
 * Sample from categorical distribution
 * 
 * P(X = i) = probs[i]
 * 
 * @param key PRNG key
 * @param probs Probabilities (dim, must sum to ~1)
 * @param dim Number of categories
 * @return Sampled index
 */
int categorical(const PRNGKey& key, const double* probs, int dim);

/**
 * Sample from categorical with log probabilities
 * 
 * @param key PRNG key
 * @param log_probs Log probabilities (dim)
 * @param dim Number of categories
 * @return Sampled index
 */
int categorical_log(const PRNGKey& key, const double* log_probs, int dim);

/**
 * Batch categorical sampling
 * 
 * @param key PRNG key
 * @param probs Probabilities (batch_size x dim)
 * @param dim Number of categories
 * @param batch_size Number of samples
 * @param output Output indices (batch_size)
 */
void categorical_batch(const PRNGKey& key, const double* probs, int dim,
                       size_t batch_size, int* output);

/**
 * Gumbel-max trick for categorical sampling from logits
 * Useful for differentiable sampling
 * 
 * @param key PRNG key
 * @param logits Unnormalized log probabilities (dim)
 * @param dim Number of categories
 * @return Index of argmax(logits + gumbel_noise)
 */
int gumbel_max(const PRNGKey& key, const double* logits, int dim);

// ============================================================================
// Gamma Distribution (for Dirichlet and other uses)
// ============================================================================

/**
 * Generate Gamma random variable
 * 
 * Uses Marsaglia and Tsang's method
 * 
 * @param key PRNG key
 * @param shape Shape parameter (alpha > 0)
 * @param scale Scale parameter (beta > 0)
 * @return Gamma(shape, scale) random variable
 */
double gamma(const PRNGKey& key, uint64_t counter, double shape, double scale = 1.0);

/**
 * Batch gamma sampling
 * 
 * @param key PRNG key
 * @param n Number of samples
 * @param shape Shape parameter
 * @param scale Scale parameter
 * @param output Output array
 */
void gamma_batch(const PRNGKey& key, size_t n, double shape, double scale, double* output);

// ============================================================================
// Advanced Utilities
// ============================================================================

/**
 * Generate random permutation
 * 
 * @param key PRNG key
 * @param n Size of permutation
 * @param output Output array (n values: 0 to n-1 in random order)
 */
void permutation(const PRNGKey& key, int n, int* output);

/**
 * Generate random integers in range [min, max)
 * 
 * @param key PRNG key
 * @param n Number of integers
 * @param min Minimum (inclusive)
 * @param max Maximum (exclusive)
 * @param output Output array
 */
void randint(const PRNGKey& key, size_t n, int min, int max, int* output);

/**
 * Shuffle array in-place
 * 
 * @param key PRNG key
 * @param data Array to shuffle
 * @param n Length of array
 * @param elem_size Size of each element in bytes
 */
void shuffle(const PRNGKey& key, void* data, size_t n, size_t elem_size);

/**
 * Bernoulli sampling
 * 
 * @param key PRNG key
 * @param n Number of samples
 * @param p Probability of true
 * @param output Output array (0 or 1)
 */
void bernoulli(const PRNGKey& key, size_t n, double p, int* output);

} // namespace random
} // namespace axiomcuda

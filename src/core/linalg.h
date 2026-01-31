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

#include <cstddef>
#include <vector>
#include <stdexcept>

namespace axiomcuda {
namespace linalg {

// ============================================================================
// Forward Declarations
// ============================================================================

// Simple matrix view structure for clean interfaces
template<typename T>
struct MatrixView {
    T* data;
    size_t rows;
    size_t cols;
    size_t ld;  // Leading dimension (stride between columns for column-major)
    
    MatrixView(T* data_, size_t rows_, size_t cols_, size_t ld_ = 0) 
        : data(data_), rows(rows_), cols(cols_), ld(ld_ == 0 ? rows_ : ld_) {}
    
    inline T& operator()(size_t i, size_t j) { return data[i + j * ld]; }
    inline const T& operator()(size_t i, size_t j) const { return data[i + j * ld]; }
};

// ============================================================================
// Basic Matrix Operations
// ============================================================================

/**
 * Matrix-vector multiplication: y = alpha * A * x + beta * y
 * 
 * @param A Matrix (m x n)
 * @param x Vector (n)
 * @param y Result vector (m)
 * @param m Number of rows
 * @param n Number of columns
 * @param alpha Scale factor for A*x
 * @param beta Scale factor for y
 * @param transpose Whether to use A^T instead of A
 */
void gemv(const double* A, const double* x, double* y,
          size_t m, size_t n, double alpha = 1.0, double beta = 0.0,
          bool transpose = false);

/**
 * Matrix-matrix multiplication: C = alpha * A * B + beta * C
 * 
 * @param A First matrix (m x k)
 * @param B Second matrix (k x n)
 * @param C Result matrix (m x n)
 * @param m Rows of A and C
 * @param k Columns of A, rows of B
 * @param n Columns of B and C
 * @param alpha Scale factor
 * @param beta Scale factor for C
 */
void gemm(const double* A, const double* B, double* C,
          size_t m, size_t k, size_t n, double alpha = 1.0, double beta = 0.0);

/**
 * Batched matrix multiplication: C[i] = A[i] * B[i]
 * 
 * @param A Batch of matrices (batch_size x m x k)
 * @param B Batch of matrices (batch_size x k x n)
 * @param C Result batch (batch_size x m x n)
 * @param batch_size Number of matrices in batch
 * @param m Rows
 * @param k Inner dimension
 * @param n Columns
 */
void gemm_batched(const double* A, const double* B, double* C,
                  size_t batch_size, size_t m, size_t k, size_t n);

/**
 * Batched dot product (bdot) - specialized for common case
 * Multiplies last two dimensions with broadcasting
 * 
 * @param A First tensor (..., m, k)
 * @param B Second tensor (..., k, n)
 * @param C Result tensor (..., m, n)
 * @param batch_dims Product of all batch dimensions
 * @param m Rows of A
 * @param k Inner dimension
 * @param n Columns of B
 */
void bdot(const double* A, const double* B, double* C,
          size_t batch_dims, size_t m, size_t k, size_t n);

// ============================================================================
// Cholesky Decomposition
// ============================================================================

/**
 * Cholesky decomposition: A = L * L^T (potrf)
 * Computes lower triangular L such that L * L^T = A
 * 
 * @param A Input matrix (n x n), positive definite
 * @param L Output lower triangular matrix (n x n)
 * @param n Dimension
 * @return 0 on success, non-zero on failure
 */
int cholesky_decompose(const double* A, double* L, size_t n);

/**
 * Batched Cholesky decomposition
 * 
 * @param A Batch of matrices (batch_size x n x n)
 * @param L Output batch (batch_size x n x n)
 * @param batch_size Number of matrices
 * @param n Dimension
 */
void cholesky_decompose_batched(const double* A, double* L,
                                size_t batch_size, size_t n);

/**
 * Cholesky solve: Solve A * X = B using Cholesky factor L (potrs)
 * where A = L * L^T
 * 
 * @param L Lower triangular Cholesky factor (n x n)
 * @param B Right hand side (n x nrhs)
 * @param X Solution (n x nrhs)
 * @param n Dimension
 * @param nrhs Number of right hand sides
 */
void cholesky_solve(const double* L, const double* B, double* X,
                    size_t n, size_t nrhs);

/**
 * Batched Cholesky solve
 * 
 * @param L Batch of Cholesky factors (batch_size x n x n)
 * @param B Batch of right hand sides (batch_size x n x nrhs)
 * @param X Output solutions (batch_size x n x nrhs)
 * @param batch_size Batch size
 * @param n Dimension
 * @param nrhs Number of right hand sides
 */
void cholesky_solve_batched(const double* L, const double* B, double* X,
                            size_t batch_size, size_t n, size_t nrhs);

/**
 * Triangular solve: Solve L * X = B or L^T * X = B
 * where L is lower triangular
 * 
 * @param L Lower triangular matrix (n x n)
 * @param B Right hand side (n x nrhs)
 * @param X Solution (n x nrhs)
 * @param n Dimension
 * @param nrhs Number of right hand sides
 * @param transpose Solve with L^T instead of L
 */
void trsm(const double* L, const double* B, double* X,
          size_t n, size_t nrhs, bool transpose = false);

// ============================================================================
// Matrix Inverse with Log-Determinant
// ============================================================================

/**
 * Compute matrix inverse and log-determinant using Cholesky
 * 
 * For positive definite matrix A:
 *   - Compute L such that A = L * L^T
 *   - log|A| = 2 * sum(log(diag(L)))
 *   - A^{-1} = (L^{-1})^T * L^{-1}
 * 
 * @param A Input matrix (n x n), positive definite
 * @param A_inv Output inverse matrix (n x n)
 * @param logdet Output log-determinant of A
 * @param n Dimension
 * @return 0 on success, non-zero if A is not positive definite
 */
int inv_and_logdet_cholesky(const double* A, double* A_inv, double& logdet, size_t n);

/**
 * Batched matrix inverse with log-determinant
 * 
 * @param A Batch of matrices (batch_size x n x n)
 * @param A_inv Output inverses (batch_size x n x n)
 * @param logdet Output log-determinants (batch_size)
 * @param batch_size Number of matrices
 * @param n Dimension
 */
void inv_and_logdet_cholesky_batched(const double* A, double* A_inv, double* logdet,
                                     size_t batch_size, size_t n);

/**
 * Compute only log-determinant using Cholesky
 * 
 * @param A Input matrix (n x n)
 * @param logdet Output log-determinant
 * @param n Dimension
 * @return 0 on success
 */
int logdet_cholesky(const double* A, double& logdet, size_t n);

/**
 * Batched log-determinant
 * 
 * @param A Batch of matrices (batch_size x n x n)
 * @param logdet Output log-determinants (batch_size)
 * @param batch_size Batch size
 * @param n Dimension
 */
void logdet_cholesky_batched(const double* A, double* logdet,
                             size_t batch_size, size_t n);

// ============================================================================
// Block Diagonal Matrix Operations
// ============================================================================

/**
 * Construct block diagonal matrix from list of blocks
 * 
 * @param blocks Array of pointers to block matrices
 * @param block_sizes Array of block dimensions
 * @param num_blocks Number of blocks
 * @param output Output matrix (sum(block_sizes) x sum(block_sizes))
 */
void block_diag(const double** blocks, const size_t* block_sizes,
                size_t num_blocks, double* output);

/**
 * Extract diagonal blocks from block diagonal matrix
 * 
 * @param matrix Input block diagonal matrix
 * @param block_sizes Sizes of blocks to extract
 * @param num_blocks Number of blocks
 * @param outputs Array of output buffers for each block
 */
void extract_block_diag(const double* matrix, const size_t* block_sizes,
                        size_t num_blocks, double** outputs);

/**
 * Batched block diagonal construction
 * 
 * @param blocks Batch of block arrays (batch_size x num_blocks x ...)
 * @param block_sizes Block dimensions
 * @param num_blocks Number of blocks per batch
 * @param batch_size Number of batches
 * @param output Output (batch_size x total_size x total_size)
 */
void block_diag_batched(const double** blocks, const size_t* block_sizes,
                        size_t num_blocks, size_t batch_size, double* output);

// ============================================================================
// Matrix Decomposition and Eigenvalue Operations
// ============================================================================

/**
 * Compute eigenvalues of symmetric matrix (using QR algorithm)
 * 
 * @param A Symmetric matrix (n x n)
 * @param eigenvalues Output eigenvalues (n)
 * @param n Dimension
 * @return 0 on success
 */
int eigenvalues_symmetric(const double* A, double* eigenvalues, size_t n);

/**
 * Check if matrix is positive definite by checking all eigenvalues > 0
 * 
 * @param A Matrix (n x n)
 * @param n Dimension
 * @param tol Tolerance
 * @return true if positive definite
 */
bool is_positive_definite(const double* A, size_t n, double tol = 1e-10);

/**
 * Force matrix to be positive definite by adding to diagonal
 * 
 * @param A Input/output matrix (n x n)
 * @param n Dimension
 * @param min_eig Minimum eigenvalue threshold
 * @return Amount added to diagonal
 */
double make_positive_definite_inplace(double* A, size_t n, double min_eig = 1e-6);

// ============================================================================
// Advanced Batched Operations
// ============================================================================

/**
 * Batched matrix-vector multiplication
 * 
 * @param A Batch of matrices (batch_size x m x n)
 * @param x Batch of vectors (batch_size x n)
 * @param y Output (batch_size x m)
 * @param batch_size Batch size
 * @param m Rows
 * @param n Columns
 */
void gemv_batched(const double* A, const double* x, double* y,
                  size_t batch_size, size_t m, size_t n);

/**
 * Compute A^T * B for batches (common for outer products)
 * 
 * @param A Batch of vectors (batch_size x m)
 * @param B Batch of vectors (batch_size x n)
 * @param C Output outer products (batch_size x m x n)
 * @param batch_size Batch size
 * @param m Dimension of A
 * @param n Dimension of B
 */
void outer_product_batched(const double* A, const double* B, double* C,
                           size_t batch_size, size_t m, size_t n);

/**
 * Compute trace of matrix product: trace(A * B)
 * 
 * @param A First matrix (m x n)
 * @param B Second matrix (n x m)
 * @param m Rows of A
 * @param n Columns of A / rows of B
 * @return trace(A * B)
 */
double trace_product(const double* A, const double* B, size_t m, size_t n);

/**
 * Batched trace of matrix products
 * 
 * @param A Batch of matrices (batch_size x m x n)
 * @param B Batch of matrices (batch_size x n x m)
 * @param output Output traces (batch_size)
 * @param batch_size Batch size
 * @param m Rows
 * @param n Columns
 */
void trace_product_batched(const double* A, const double* B, double* output,
                           size_t batch_size, size_t m, size_t n);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Set matrix to identity
 * 
 * @param A Output matrix (n x n)
 * @param n Dimension
 */
void set_identity(double* A, size_t n);

/**
 * Set batch of matrices to identity
 * 
 * @param A Output batch (batch_size x n x n)
 * @param batch_size Batch size
 * @param n Dimension
 */
void set_identity_batched(double* A, size_t batch_size, size_t n);

/**
 * Copy matrix
 * 
 * @param src Source (m x n)
 * @param dst Destination (m x n)
 * @param m Rows
 * @param n Columns
 */
void copy_matrix(const double* src, double* dst, size_t m, size_t n);

/**
 * Transpose matrix
 * 
 * @param src Source (m x n)
 * @param dst Destination (n x m)
 * @param m Rows
 * @param n Columns
 */
void transpose(const double* src, double* dst, size_t m, size_t n);

/**
 * Batched transpose
 * 
 * @param src Source batch (batch_size x m x n)
 * @param dst Destination (batch_size x n x m)
 * @param batch_size Batch size
 * @param m Rows
 * @param n Columns
 */
void transpose_batched(const double* src, double* dst, size_t batch_size, size_t m, size_t n);

/**
 * Extract diagonal elements
 * 
 * @param matrix Input matrix (n x n)
 * @param diagonal Output diagonal (n)
 * @param n Dimension
 */
void extract_diagonal(const double* matrix, double* diagonal, size_t n);

/**
 * Set diagonal elements
 * 
 * @param matrix Input/output matrix (n x n)
 * @param diagonal Diagonal values (n)
 * @param n Dimension
 */
void set_diagonal(double* matrix, const double* diagonal, size_t n);

/**
 * Scale diagonal by factor
 * 
 * @param matrix Input/output matrix (n x n)
 * @param scale Scale factor
 * @param n Dimension
 */
void scale_diagonal(double* matrix, double scale, size_t n);

} // namespace linalg
} // namespace axiomcuda

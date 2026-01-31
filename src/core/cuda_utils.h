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

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace axiomcuda {

// ============================================================================
// CUDA Error Checking
// ============================================================================

/**
 * Check CUDA error and throw exception if error occurred
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - " + \
                                     cudaGetErrorString(error)); \
        } \
    } while(0)

/**
 * Check CUDA error and return error code (for non-throwing contexts)
 */
#define CUDA_CHECK_RET(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            return error; \
        } \
    } while(0)

/**
 * Get last CUDA error as string
 */
std::string get_cuda_error_string();

/**
 * Synchronize device and check for errors
 */
void cuda_sync_check();

// ============================================================================
// Device Properties and Query
// ============================================================================

/**
 * Get number of available CUDA devices
 */
int get_device_count();

/**
 * Get current device ID
 */
int get_current_device();

/**
 * Set current device
 */
void set_device(int device_id);

/**
 * Get device properties
 */
cudaDeviceProp get_device_properties(int device_id = -1);

/**
 * Print device info
 */
void print_device_info(int device_id = -1);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Allocate device memory with error checking
 */
template<typename T>
T* device_malloc(size_t n_elements) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n_elements * sizeof(T)));
    return ptr;
}

/**
 * Allocate pinned host memory
 */
template<typename T>
T* pinned_malloc(size_t n_elements) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, n_elements * sizeof(T)));
    return ptr;
}

/**
 * Free device memory
 */
template<typename T>
void device_free(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

/**
 * Free pinned memory
 */
template<typename T>
void pinned_free(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

/**
 * Copy host to device
 */
template<typename T>
void h2d_copy(const T* host_src, T* device_dst, size_t n_elements) {
    CUDA_CHECK(cudaMemcpy(device_dst, host_src, n_elements * sizeof(T), 
                          cudaMemcpyHostToDevice));
}

/**
 * Copy device to host
 */
template<typename T>
void d2h_copy(const T* device_src, T* host_dst, size_t n_elements) {
    CUDA_CHECK(cudaMemcpy(host_dst, device_src, n_elements * sizeof(T), 
                          cudaMemcpyDeviceToHost));
}

/**
 * Copy device to device
 */
template<typename T>
void d2d_copy(const T* device_src, T* device_dst, size_t n_elements) {
    CUDA_CHECK(cudaMemcpy(device_dst, device_src, n_elements * sizeof(T), 
                          cudaMemcpyDeviceToDevice));
}

/**
 * Set device memory to zero
 */
template<typename T>
void device_memset(T* device_ptr, size_t n_elements, int value = 0) {
    CUDA_CHECK(cudaMemset(device_ptr, value, n_elements * sizeof(T)));
}

/**
 * Async copy host to device
 */
template<typename T>
void h2d_copy_async(const T* host_src, T* device_dst, size_t n_elements, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(device_dst, host_src, n_elements * sizeof(T), 
                               cudaMemcpyHostToDevice, stream));
}

/**
 * Async copy device to host  
 */
template<typename T>
void d2h_copy_async(const T* device_src, T* host_dst, size_t n_elements, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(host_dst, device_src, n_elements * sizeof(T), 
                               cudaMemcpyDeviceToHost, stream));
}

// ============================================================================
// Stream Management
// ============================================================================

/**
 * Create CUDA stream
 */
cudaStream_t create_stream();

/**
 * Destroy CUDA stream
 */
void destroy_stream(cudaStream_t stream);

/**
 * Synchronize stream
 */
void stream_sync(cudaStream_t stream);

/**
 * Get default stream
 */
cudaStream_t default_stream();

// ============================================================================
// Kernel Launch Configuration
// ============================================================================

/**
 * Structure for kernel launch configuration
 */
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    cudaStream_t stream;
    
    LaunchConfig() : shared_mem(0), stream(0) {}
    
    LaunchConfig(int grid_x, int block_x, 
                 size_t shared = 0, cudaStream_t s = 0)
        : grid(grid_x, 1, 1), block(block_x, 1, 1), 
          shared_mem(shared), stream(s) {}
    
    LaunchConfig(int grid_x, int grid_y, int block_x, int block_y,
                 size_t shared = 0, cudaStream_t s = 0)
        : grid(grid_x, grid_y, 1), block(block_x, block_y, 1), 
          shared_mem(shared), stream(s) {}
};

/**
 * Calculate 1D launch configuration for given problem size
 * 
 * @param n Total number of elements
 * @param block_size Threads per block (default 256)
 * @return Launch configuration
 */
LaunchConfig calculate_launch_config_1d(size_t n, int block_size = 256);

/**
 * Calculate 2D launch configuration
 * 
 * @param m Number of rows
 * @param n Number of columns  
 * @param block_x Block size in x (default 16)
 * @param block_y Block size in y (default 16)
 * @return Launch configuration
 */
LaunchConfig calculate_launch_config_2d(size_t m, size_t n, 
                                        int block_x = 16, int block_y = 16);

/**
 * Get maximum threads per block for current device
 */
int get_max_threads_per_block();

/**
 * Get number of SMs on current device
 */
int get_sm_count();

// ============================================================================
// Math Constants on Device
// ============================================================================

// Constants available on both host and device
#ifdef __CUDACC__
#define CUDA_CONSTANT __constant__ __device__
#else
#define CUDA_CONSTANT
#endif

// Mathematical constants
CUDA_CONSTANT extern const double CUDA_PI;
CUDA_CONSTANT extern const double CUDA_2PI;
CUDA_CONSTANT extern const double CUDA_LN_2PI;
CUDA_CONSTANT extern const double CUDA_LN_PI;
CUDA_CONSTANT extern const double CUDA_E;
CUDA_CONSTANT extern const double CUDA_LN2;
CUDA_CONSTANT extern const double CUDA_LN10;
CUDA_CONSTANT extern const double CUDA_SQRT2;
CUDA_CONSTANT extern const double CUDA_INV_SQRT_2PI;
CUDA_CONSTANT extern const double CUDA_EPSILON;

// Infinity and NaN helpers
#ifdef __CUDACC__
__device__ __forceinline__ double cuda_infinity() {
    return __longlong_as_double(0x7ff0000000000000ULL);
}

__device__ __forceinline__ double cuda_nan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

__device__ __forceinline__ bool cuda_is_nan(double x) {
    return x != x;
}

__device__ __forceinline__ bool cuda_is_inf(double x) {
    return (x == cuda_infinity() || x == -cuda_infinity());
}
#endif

// ============================================================================
// Warp and Block Operations
// ============================================================================

#ifdef __CUDACC__

/**
 * Warp-level reduction: sum
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * Warp-level reduction: max
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/**
 * Warp-level reduction: min
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_min(T val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/**
 * Block-level reduction: sum using shared memory
 */
template<typename T>
__device__ T block_reduce_sum(T val, T* shared_mem) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // Warp reduction
    val = warp_reduce_sum(val);
    
    // Write to shared memory
    if (lane == 0) shared_mem[wid] = val;
    __syncthreads();
    
    // Final reduction within first warp
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[lane] : T(0);
        val = warp_reduce_sum(val);
    }
    
    return val;
}

/**
 * Block-level reduction: max using shared memory
 */
template<typename T>
__device__ T block_reduce_max(T val, T* shared_mem) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared_mem[wid] = val;
    __syncthreads();
    
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[lane] : T(0);
        val = warp_reduce_max(val);
    }
    
    return val;
}

#endif // __CUDACC__

// ============================================================================
// Grid-Stride Loops Helper
// ============================================================================

#ifdef __CUDACC__

/**
 * Helper for grid-stride loops (efficient processing of large arrays)
 * 
 * Usage:
 *   for (int i = grid_stride_start(); i < n; i += grid_stride_step()) {
 *       // process element i
 *   }
 */
__device__ __forceinline__ int grid_stride_start() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int grid_stride_step() {
    return gridDim.x * blockDim.x;
}

/**
 * 2D grid-stride
 */
__device__ __forceinline__ int2 grid_stride_2d_start() {
    return make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                     blockIdx.y * blockDim.y + threadIdx.y);
}

__device__ __forceinline__ int2 grid_stride_2d_step() {
    return make_int2(gridDim.x * blockDim.x, gridDim.y * blockDim.y);
}

#endif // __CUDACC__

} // namespace axiomcuda

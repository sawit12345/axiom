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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <sstream>

#include "bindings.h"

namespace py = pybind11;
using namespace axiomcuda;

// Tensor binding
void bind_tensor(py::module_& m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(
        m, "Tensor", R"doc(
            Multi-dimensional array with GPU/CPU storage.
            
            Tensor provides a unified interface for array storage that can
            reside on either CPU or GPU. It supports:
            - Seamless numpy array conversion
            - GPU memory management
            - Automatic device placement
            - Basic arithmetic operations
            
            The Tensor class is the fundamental data structure used
            throughout the axiomcuda backend.
            
            Attributes:
                shape: Tuple of dimensions
                ndim: Number of dimensions
                size: Total number of elements
                device: Device where tensor is stored ('cpu' or 'cuda:N')
                dtype: Data type (always float64)
            
            Examples:
                >>> import numpy as np
                >>> from axiomcuda_backend import tensor
                >>> 
                >>> # Create from numpy array (stays on CPU)
                >>> arr = np.random.randn(10, 5)
                >>> t = tensor.Tensor.from_numpy(arr)
                >>> 
                >>> # Move to GPU
                >>> t_gpu = t.cuda()
                >>> 
                >>> # Move back to CPU
                >>> t_cpu = t_gpu.cpu()
                >>> 
                >>> # Convert back to numpy
                >>> arr_back = t_cpu.to_numpy()
                >>> 
                >>> # Create zeros/ones
                >>> zeros = tensor.zeros((3, 4, 5))
                >>> ones = tensor.ones((2, 3))
                >>> 
                >>> # Basic operations
                >>> t2 = t + 1.0
                >>> t3 = t * 2.0
                >>> t4 = t.sum(axis=0)
        )doc")
        .def_static("from_numpy", &Tensor::from_numpy,
            py::arg("array"),
            py::arg("copy") = false,
            "Create Tensor from numpy array (zero-copy if possible)")
        .def("to_numpy", &Tensor::to_numpy,
            "Convert to numpy array (copies from GPU if needed)")
        .def("cpu", &Tensor::cpu,
            "Return CPU copy of tensor")
        .def("cuda", &Tensor::cuda,
            py::arg("device_id") = 0,
            "Return GPU copy of tensor")
        .def("to", &Tensor::to,
            py::arg("device"),
            "Move tensor to specified device")
        .def_property_readonly("shape", &Tensor::shape,
            "Shape tuple")
        .def_property_readonly("ndim", &Tensor::ndim,
            "Number of dimensions")
        .def_property_readonly("size", &Tensor::size,
            "Total number of elements")
        .def_property_readonly("device", &Tensor::device_str,
            "Device location ('cpu' or 'cuda:N')")
        .def("is_cuda", &Tensor::is_cuda,
            "Check if tensor is on GPU")
        .def("copy", &Tensor::copy,
            "Create a copy of the tensor")
        .def("reshape", &Tensor::reshape,
            py::arg("new_shape"),
            "Reshape tensor to new dimensions")
        .def("view", &Tensor::view,
            py::arg("new_shape"),
            "Create view with new shape (shares data)")
        .def("squeeze", &Tensor::squeeze,
            py::arg("axis") = py::none(),
            "Remove dimensions of size 1")
        .def("unsqueeze", &Tensor::unsqueeze,
            py::arg("axis"),
            "Add dimension of size 1")
        .def("transpose", &Tensor::transpose,
            py::arg("dim0"), py::arg("dim1"),
            "Swap two dimensions")
        .def("permute", &Tensor::permute,
            py::arg("dims"),
            "Permute dimensions according to given order")
        .def("contiguous", &Tensor::contiguous,
            "Return contiguous copy of tensor")
        .def("is_contiguous", &Tensor::is_contiguous,
            "Check if tensor is contiguous in memory")
        // Arithmetic operations
        .def("__add__", &Tensor::add,
            py::arg("other"),
            "Element-wise addition")
        .def("__sub__", &Tensor::subtract,
            py::arg("other"),
            "Element-wise subtraction")
        .def("__mul__", &Tensor::multiply,
            py::arg("other"),
            "Element-wise multiplication")
        .def("__truediv__", &Tensor::divide,
            py::arg("other"),
            "Element-wise division")
        .def("__pow__", &Tensor::power,
            py::arg("exponent"),
            "Element-wise power")
        .def("__neg__", &Tensor::negate,
            "Negate all elements")
        .def("__getitem__", [](Tensor& self, py::object index) {
                return self.getitem(index);
            },
            py::arg("index"),
            "Get item or slice")
        .def("__setitem__", [](Tensor& self, py::object index, py::object value) {
                self.setitem(index, value);
            },
            py::arg("index"), py::arg("value"),
            "Set item or slice")
        .def("__repr__", [](const Tensor& self) {
            std::ostringstream oss;
            oss << "Tensor(";
            oss << "shape=" << shape_to_string(self.shape());
            oss << ", device='" << self.device_str() << "'";
            oss << ", dtype=float64)";
            return oss.str();
        })
        .def("__str__", [](const Tensor& self) {
            // For small tensors, show values; otherwise just show shape
            if (self.size() <= 100) {
                return self.to_numpy().attr("__str__")().cast<std::string>();
            }
            std::ostringstream oss;
            oss << "Tensor(" << shape_to_string(self.shape()) << ")";
            return oss.str();
        });
    
    // Reduction operations
    m.def("sum", [](const Tensor& input, py::object axis, bool keepdims) {
            if (axis.is_none()) {
                return input.sum_all();
            } else {
                return input.sum(axis.cast<int>(), keepdims);
            }
        },
        py::arg("input"),
        py::arg("axis") = py::none(),
        py::arg("keepdims") = false,
        "Sum elements along axis or all elements");
    
    m.def("mean", [](const Tensor& input, py::object axis, bool keepdims) {
            if (axis.is_none()) {
                return input.mean_all();
            } else {
                return input.mean(axis.cast<int>(), keepdims);
            }
        },
        py::arg("input"),
        py::arg("axis") = py::none(),
        py::arg("keepdims") = false,
        "Mean of elements along axis or all elements");
    
    m.def("max", [](const Tensor& input, py::object axis, bool keepdims) {
            if (axis.is_none()) {
                return input.max_all();
            } else {
                return input.max(axis.cast<int>(), keepdims);
            }
        },
        py::arg("input"),
        py::arg("axis") = py::none(),
        py::arg("keepdims") = false,
        "Maximum along axis or all elements");
    
    m.def("min", [](const Tensor& input, py::object axis, bool keepdims) {
            if (axis.is_none()) {
                return input.min_all();
            } else {
                return input.min(axis.cast<int>(), keepdims);
            }
        },
        py::arg("input"),
        py::arg("axis") = py::none(),
        py::arg("keepdims") = false,
        "Minimum along axis or all elements");
    
    // Creation functions
    m.def("zeros", &Tensor::zeros,
        py::arg("shape"),
        py::arg("device") = "cpu",
        "Create tensor filled with zeros");
    
    m.def("ones", &Tensor::ones,
        py::arg("shape"),
        py::arg("device") = "cpu",
        "Create tensor filled with ones");
    
    m.def("full", &Tensor::full,
        py::arg("shape"),
        py::arg("fill_value"),
        py::arg("device") = "cpu",
        "Create tensor filled with given value");
    
    m.def("empty", &Tensor::empty,
        py::arg("shape"),
        py::arg("device") = "cpu",
        "Create uninitialized tensor");
    
    m.def("eye", &Tensor::eye,
        py::arg("n"),
        py::arg("m") = py::none(),
        py::arg("device") = "cpu",
        "Create identity matrix");
    
    m.def("arange", &Tensor::arange,
        py::arg("start") = 0,
        py::arg("stop"),
        py::arg("step") = 1,
        py::arg("device") = "cpu",
        "Create tensor with evenly spaced values");
    
    m.def("linspace", &Tensor::linspace,
        py::arg("start"),
        py::arg("stop"),
        py::arg("num") = 50,
        py::arg("device") = "cpu",
        "Create tensor with linearly spaced values");
    
    m.def("randn", &Tensor::randn,
        py::arg("shape"),
        py::arg("device") = "cpu",
        py::arg("seed") = py::none(),
        "Create tensor with standard normal random values");
    
    m.def("rand", &Tensor::rand,
        py::arg("shape"),
        py::arg("device") = "cpu",
        py::arg("seed") = py::none(),
        "Create tensor with uniform random values in [0, 1)");
    
    // Linear algebra operations
    m.def("matmul", &Tensor::matmul,
        py::arg("a"),
        py::arg("b"),
        "Matrix multiplication");
    
    m.def("dot", &Tensor::dot,
        py::arg("a"),
        py::arg("b"),
        "Dot product of two 1-D tensors");
    
    m.def("tensordot", &Tensor::tensordot,
        py::arg("a"),
        py::arg("b"),
        py::arg("axes"),
        "Tensor contraction over specified axes");
    
    m.def("transpose", [](const Tensor& input, py::object dim0, py::object dim1) {
            if (dim0.is_none() && dim1.is_none()) {
                // Reverse all dimensions
                return input.transpose_all();
            }
            return input.transpose(dim0.cast<int>(), dim1.cast<int>());
        },
        py::arg("input"),
        py::arg("dim0") = py::none(),
        py::arg("dim1") = py::none(),
        "Transpose tensor");
    
    m.def("inverse", &Tensor::inverse,
        py::arg("input"),
        "Compute matrix inverse");
    
    m.def("cholesky", &Tensor::cholesky,
        py::arg("input"),
        py::arg("upper") = false,
        "Cholesky decomposition");
    
    m.def("solve", &Tensor::solve,
        py::arg("a"),
        py::arg("b"),
        "Solve linear system ax = b");
    
    m.def("eig", &Tensor::eig,
        py::arg("input"),
        "Compute eigenvalues and eigenvectors");
    
    m.def("svd", &Tensor::svd,
        py::arg("input"),
        py::arg("full_matrices") = false,
        "Singular value decomposition");
    
    m.def("det", &Tensor::det,
        py::arg("input"),
        "Matrix determinant");
    
    m.def("slogdet", &Tensor::slogdet,
        py::arg("input"),
        "Sign and log of determinant");
    
    m.def("trace", &Tensor::trace,
        py::arg("input"),
        "Sum of diagonal elements");
    
    m.def("diag", [](const Tensor& input, int offset) {
            if (input.ndim() == 1) {
                return Tensor::diag_from_vector(input, offset);
            } else {
                return Tensor::diag_from_matrix(input, offset);
            }
        },
        py::arg("input"),
        py::arg("offset") = 0,
        "Extract diagonal or create diagonal matrix");
    
    // Element-wise operations
    m.def("exp", &Tensor::exp,
        py::arg("input"),
        "Element-wise exponential");
    
    m.def("log", &Tensor::log,
        py::arg("input"),
        "Element-wise natural logarithm");
    
    m.def("sqrt", &Tensor::sqrt,
        py::arg("input"),
        "Element-wise square root");
    
    m.def("square", &Tensor::square,
        py::arg("input"),
        "Element-wise square");
    
    m.def("abs", &Tensor::abs,
        py::arg("input"),
        "Element-wise absolute value");
    
    m.def("sign", &Tensor::sign,
        py::arg("input"),
        "Element-wise sign");
    
    m.def("clip", &Tensor::clip,
        py::arg("input"),
        py::arg("min") = py::none(),
        py::arg("max") = py::none(),
        "Clip values to range [min, max]");
    
    // Shape manipulation
    m.def("concatenate", &Tensor::concatenate,
        py::arg("tensors"),
        py::arg("axis") = 0,
        "Concatenate tensors along axis");
    
    m.def("stack", &Tensor::stack,
        py::arg("tensors"),
        py::arg("axis") = 0,
        "Stack tensors along new axis");
    
    m.def("split", &Tensor::split,
        py::arg("input"),
        py::arg("indices_or_sections"),
        py::arg("axis") = 0,
        "Split tensor into multiple tensors");
    
    m.def("broadcast_to", &Tensor::broadcast_to,
        py::arg("input"),
        py::arg("shape"),
        "Broadcast tensor to new shape");
    
    // Memory management utilities
    py::class_<MemoryPool>(m, "MemoryPool", R"doc(
        Memory pool for efficient GPU memory management.
        
        The memory pool reduces allocation overhead by caching
        GPU memory allocations for reuse.
    )doc")
        .def_static("enable", &MemoryPool::enable,
            "Enable memory pooling")
        .def_static("disable", &MemoryPool::disable,
            "Disable memory pooling and free cached memory")
        .def_static("empty_cache", &MemoryPool::emptyCache,
            "Free unused cached memory")
        .def_static("memory_stats", &MemoryPool::memoryStats,
            "Get memory statistics")
        .def_static("set_limit", &MemoryPool::setLimit,
            py::arg("bytes"),
            "Set maximum cached memory in bytes");
    
    // Stream management for async operations
    py::class_<CudaStream>(m, "CudaStream", R"doc(
        CUDA stream for asynchronous operations.
        
        Streams allow overlapping computation and memory transfers.
    )doc")
        .def(py::init<int>(),
            py::arg("device_id") = 0,
            "Create CUDA stream")
        .def("synchronize", &CudaStream::synchronize,
            "Wait for all operations in stream to complete")
        .def("is_done", &CudaStream::isDone,
            "Check if all operations are complete")
        .def("__enter__", &CudaStream::enter)
        .def("__exit__", &CudaStream::exit);
}

// Math module initialization
void init_math_module(py::module_& m) {
    m.doc() = R"doc(
        Mathematical functions optimized with CUDA.
        
        This module provides GPU-accelerated implementations of:
        - Gamma and digamma functions
        - Log-sum-exp and softmax
        - Gaussian log-likelihood
        - Matrix operations
        
        All functions accept both CPU and GPU tensors and automatically
        dispatch to the appropriate implementation.
    )doc";
    
    // Gamma functions
    m.def("gammaln", [](const Tensor& x) {
            return x.gammaln();
        },
        py::arg("x"),
        "Log gamma function log(Γ(x))");
    
    m.def("digamma", [](const Tensor& x) {
            return x.digamma();
        },
        py::arg("x"),
        "Digamma function ψ(x) = d/dx log(Γ(x))");
    
    m.def("mvgammaln", [](double x, int d) {
            return math::mvgammaln(x, d);
        },
        py::arg("x"),
        py::arg("d"),
        "Multivariate log gamma function");
    
    m.def("mvdigamma", [](double x, int d) {
            return math::mvdigamma(x, d);
        },
        py::arg("x"),
        py::arg("d"),
        "Multivariate digamma function");
    
    // Log-sum-exp and softmax
    m.def("logsumexp", [](const Tensor& x, py::object axis, bool keepdims) {
            if (axis.is_none()) {
                return Tensor::scalar(math::logsumexp(x.data(), x.size()));
            }
            return x.logsumexp(axis.cast<int>(), keepdims);
        },
        py::arg("x"),
        py::arg("axis") = py::none(),
        py::arg("keepdims") = false,
        "Numerically stable log-sum-exp");
    
    m.def("softmax", [](const Tensor& x, py::object axis) {
            int ax = axis.is_none() ? -1 : axis.cast<int>();
            return x.softmax(ax);
        },
        py::arg("x"),
        py::arg("axis") = py::none(),
        "Softmax function");
    
    // Gaussian functions
    m.def("gaussian_loglik", [](const Tensor& x, const Tensor& mean, 
                                const Tensor& inv_cov, double logdet_inv_cov) {
            return math::gaussian_loglik_batch(
                x.data(), mean.data(), inv_cov.data(), 
                &logdet_inv_cov, x.shape().back(), x.shape()[0]
            );
        },
        py::arg("x"),
        py::arg("mean"),
        py::arg("inv_cov"),
        py::arg("logdet_inv_cov"),
        "Gaussian log-likelihood");
    
    m.def("gaussian_loglik_isotropic", [](const Tensor& x, const Tensor& mean, 
                                          double sigma_sqr) {
            return math::gaussian_loglik_isotropic_batch(
                x.data(), mean.data(), nullptr, 
                x.shape().back(), x.shape()[0], mean.shape()[1], nullptr
            );
        },
        py::arg("x"),
        py::arg("mean"),
        py::arg("sigma_sqr"),
        "Isotropic Gaussian log-likelihood");
    
    // Matrix utilities
    m.def("symmetrize", [](Tensor& matrix) {
            math::symmetrize(matrix.data(), matrix.shape().back());
        },
        py::arg("matrix"),
        "Make matrix symmetric: (A + Aᵀ) / 2");
    
    m.def("make_positive_definite", [](Tensor& matrix, double epsilon) {
            math::make_positive_definite(matrix.data(), matrix.shape().back(), epsilon);
        },
        py::arg("matrix"),
        py::arg("epsilon") = 1e-6,
        "Make matrix positive definite");
    
    // Trigonometric
    m.def("sin", [](const Tensor& x) { return x.sin(); },
        py::arg("x"), "Sine");
    m.def("cos", [](const Tensor& x) { return x.cos(); },
        py::arg("x"), "Cosine");
    m.def("tan", [](const Tensor& x) { return x.tan(); },
        py::arg("x"), "Tangent");
    m.def("arcsin", [](const Tensor& x) { return x.arcsin(); },
        py::arg("x"), "Inverse sine");
    m.def("arccos", [](const Tensor& x) { return x.arccos(); },
        py::arg("x"), "Inverse cosine");
    m.def("arctan", [](const Tensor& x) { return x.arctan(); },
        py::arg("x"), "Inverse tangent");
    
    // Hyperbolic
    m.def("sinh", [](const Tensor& x) { return x.sinh(); },
        py::arg("x"), "Hyperbolic sine");
    m.def("cosh", [](const Tensor& x) { return x.cosh(); },
        py::arg("x"), "Hyperbolic cosine");
    m.def("tanh", [](const Tensor& x) { return x.tanh(); },
        py::arg("x"), "Hyperbolic tangent");
    
    // Constants
    m.attr("pi") = math::PI;
    m.attr("e") = std::exp(1.0);
    m.attr("inf") = std::numeric_limits<double>::infinity();
    m.attr("nan") = std::numeric_limits<double>::quiet_NaN();
}

// Main tensor module initialization
void init_tensor_module(py::module_& m) {
    m.doc() = R"doc(
        Tensor operations and array utilities.
        
        This module provides the Tensor class for multi-dimensional arrays
        with seamless CPU/GPU interoperability and numpy compatibility.
        
        Key features:
        - Zero-copy numpy array conversion when possible
        - Automatic memory management on GPU
        - Broadcasting and shape manipulation
        - Linear algebra operations
        - Element-wise mathematical functions
    )doc";
    
    // Bind Tensor class
    bind_tensor(m);
}

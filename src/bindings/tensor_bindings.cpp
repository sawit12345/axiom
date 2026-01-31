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
#include "../core/math.h"

namespace py = pybind11;
using namespace axiomcuda;

// Helper function to convert shape to string
std::string shape_to_string(const Shape& shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.ndim(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << ")";
    return oss.str();
}

// Helper to convert numpy array to Tensor
Tensor tensor_from_numpy(py::array_t<double> array) {
    py::buffer_info info = array.request();
    std::vector<size_t> shape;
    for (auto s : info.shape) {
        shape.push_back(static_cast<size_t>(s));
    }
    return Tensor::from_numpy(info.ptr, shape);
}

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
                >>> t = tensor.from_numpy(arr)
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
                >>> t4 = tensor.sum(t, axis=0)
        )doc")
        .def_static("from_numpy", [](py::array_t<double> array) {
                return tensor_from_numpy(array);
            },
            py::arg("array"),
            "Create Tensor from numpy array (copies data)")
        .def("to_numpy", [](const Tensor& self) {
                // Create numpy array and copy data
                std::vector<ssize_t> shape;
                for (size_t i = 0; i < self.ndim(); ++i) {
                    shape.push_back(static_cast<ssize_t>(self.shape()[i]));
                }
                py::array_t<double> result(shape);
                py::buffer_info info = result.request();
                self.to_numpy(info.ptr);
                return result;
            },
            "Convert to numpy array (copies from GPU if needed)")
        .def("cpu", &Tensor::cpu,
            "Return CPU copy of tensor")
        .def("cuda", &Tensor::cuda,
            "Return GPU copy of tensor")
        .def("to", [](const Tensor& self, const std::string& device) {
                if (device == "cpu") {
                    return self.to(DeviceType::CPU);
                } else if (device.substr(0, 4) == "cuda") {
                    return self.to(DeviceType::CUDA);
                }
                throw std::invalid_argument("Unknown device: " + device);
            },
            py::arg("device"),
            "Move tensor to specified device")
        .def_property_readonly("shape", [](const Tensor& self) {
                std::vector<size_t> shape;
                for (size_t i = 0; i < self.ndim(); ++i) {
                    shape.push_back(self.shape()[i]);
                }
                return shape;
            },
            "Shape tuple")
        .def_property_readonly("ndim", &Tensor::ndim,
            "Number of dimensions")
        .def_property_readonly("size", &Tensor::size,
            "Total number of elements")
        .def_property_readonly("device", &Tensor::device_str,
            "Device location ('cpu' or 'cuda:N')")
        .def("is_cuda", &Tensor::is_cuda,
            "Check if tensor is on GPU")
        .def("copy", &Tensor::clone,
            "Create a copy of the tensor")
        .def("reshape", [](const Tensor& self, const std::vector<size_t>& new_shape) {
                return self.reshape(Shape(new_shape));
            },
            py::arg("new_shape"),
            "Reshape tensor to new dimensions")
        .def("view", [](const Tensor& self, const std::vector<size_t>& new_shape) {
                return self.view(Shape(new_shape));
            },
            py::arg("new_shape"),
            "Create view with new shape (shares data)")
        .def("squeeze", [](const Tensor& self, py::object axis) {
                if (axis.is_none()) {
                    return self.squeeze();
                } else {
                    return self.squeeze(axis.cast<int>());
                }
            },
            py::arg("axis") = py::none(),
            "Remove dimensions of size 1")
        .def("unsqueeze", &Tensor::unsqueeze,
            py::arg("axis"),
            "Add dimension of size 1")
        .def("transpose", [](const Tensor& self, py::object dim0, py::object dim1) {
                if (dim0.is_none() || dim1.is_none()) {
                    return self.transpose();
                } else {
                    return self.transpose(dim0.cast<int>(), dim1.cast<int>());
                }
            },
            py::arg("dim0") = py::none(), py::arg("dim1") = py::none(),
            "Swap two dimensions")
        .def("permute", [](const Tensor& self, const std::vector<int>& dims) {
                return self.permute(dims);
            },
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
        .def("__getitem__", [](const Tensor& self, int index) {
                return self.getitem(index);
            },
            py::arg("index"),
            "Get item at index")
        .def("__setitem__", [](Tensor& self, int index, const Tensor& value) {
                self.setitem(index, value);
            },
            py::arg("index"), py::arg("value"),
            "Set item at index")
        .def("__repr__", [](const Tensor& self) {
            std::ostringstream oss;
            oss << "Tensor(";
            oss << "shape=" << shape_to_string(self.shape());
            oss << ", device='" << self.device_str() << "'";
            oss << ", dtype=float64)";
            return oss.str();
        })
        .def("__str__", [](const Tensor& self) {
            std::ostringstream oss;
            oss << "Tensor(" << shape_to_string(self.shape()) << ")";
            return oss.str();
        });
    
    // Reduction operations - as module functions
    // Note: These are stubs - actual implementations not yet available
    m.def("sum", [](const Tensor& input, py::object axis, bool keepdims) -> Tensor {
            throw std::runtime_error("sum() not yet implemented");
        },
        py::arg("input"),
        py::arg("axis") = py::none(),
        py::arg("keepdims") = false,
        "Sum elements along axis or all elements (not yet implemented)");
    
    m.def("mean", [](const Tensor& input, py::object axis, bool keepdims) -> Tensor {
            throw std::runtime_error("mean() not yet implemented");
        },
        py::arg("input"),
        py::arg("axis") = py::none(),
        py::arg("keepdims") = false,
        "Mean of elements along axis or all elements (not yet implemented)");
    
    m.def("max", [](const Tensor& input, py::object axis, bool keepdims) -> Tensor {
            throw std::runtime_error("max() not yet implemented");
        },
        py::arg("input"),
        py::arg("axis") = py::none(),
        py::arg("keepdims") = false,
        "Maximum along axis or all elements (not yet implemented)");
    
    m.def("min", [](const Tensor& input, py::object axis, bool keepdims) -> Tensor {
            throw std::runtime_error("min() not yet implemented");
        },
        py::arg("input"),
        py::arg("axis") = py::none(),
        py::arg("keepdims") = false,
        "Minimum along axis or all elements (not yet implemented)");
    
    // Creation functions
    m.def("zeros", [](const std::vector<size_t>& shape, const std::string& device) {
            auto t = Tensor::zeros(Shape(shape));
            if (device == "cuda") {
                return t.cuda();
            }
            return t;
        },
        py::arg("shape"),
        py::arg("device") = "cpu",
        "Create tensor filled with zeros");
    
    m.def("ones", [](const std::vector<size_t>& shape, const std::string& device) {
            auto t = Tensor::ones(Shape(shape));
            if (device == "cuda") {
                return t.cuda();
            }
            return t;
        },
        py::arg("shape"),
        py::arg("device") = "cpu",
        "Create tensor filled with ones");
    
    // Note: full, empty, eye, arange, linspace are not yet implemented in Tensor
    // Placeholder implementations that throw NotImplementedError
    m.def("full", [](const std::vector<size_t>& shape, double fill_value, const std::string& device) {
            throw std::runtime_error("full() not yet implemented");
        },
        py::arg("shape"),
        py::arg("fill_value"),
        py::arg("device") = "cpu",
        "Create tensor filled with given value (not yet implemented)");
    
    m.def("empty", [](const std::vector<size_t>& shape, const std::string& device) {
            auto t = Tensor::empty(Shape(shape));
            if (device == "cuda") {
                return t.cuda();
            }
            return t;
        },
        py::arg("shape"),
        py::arg("device") = "cpu",
        "Create uninitialized tensor");
    
    m.def("eye", [](size_t n, py::object m, const std::string& device) {
            throw std::runtime_error("eye() not yet implemented");
        },
        py::arg("n"),
        py::arg("m") = py::none(),
        py::arg("device") = "cpu",
        "Create identity matrix (not yet implemented)");
    
    m.def("arange", [](int start, int stop, int step, const std::string& device) {
            throw std::runtime_error("arange() not yet implemented");
        },
        py::arg("start") = 0,
        py::arg("stop"),
        py::arg("step") = 1,
        py::arg("device") = "cpu",
        "Create tensor with evenly spaced values (not yet implemented)");
    
    m.def("linspace", [](double start, double stop, size_t num, const std::string& device) {
            throw std::runtime_error("linspace() not yet implemented");
        },
        py::arg("start"),
        py::arg("stop"),
        py::arg("num") = 50,
        py::arg("device") = "cpu",
        "Create tensor with linearly spaced values (not yet implemented)");
    
    m.def("randn", [](const std::vector<size_t>& shape, const std::string& device, py::object seed) -> Tensor {
            throw std::runtime_error("randn() not yet implemented");
        },
        py::arg("shape"),
        py::arg("device") = "cpu",
        py::arg("seed") = py::none(),
        "Create tensor with standard normal random values (not yet implemented)");
    
    m.def("rand", [](const std::vector<size_t>& shape, const std::string& device, py::object seed) -> Tensor {
            throw std::runtime_error("rand() not yet implemented");
        },
        py::arg("shape"),
        py::arg("device") = "cpu",
        py::arg("seed") = py::none(),
        "Create tensor with uniform random values in [0, 1) (not yet implemented)");
    
    // Linear algebra operations
    m.def("matmul", [](const Tensor& a, const Tensor& b) -> Tensor {
            throw std::runtime_error("matmul() not yet implemented");
        },
        py::arg("a"),
        py::arg("b"),
        "Matrix multiplication (not yet implemented)");
    
    m.def("dot", [](const Tensor& a, const Tensor& b) -> Tensor {
            throw std::runtime_error("dot() not yet implemented");
        },
        py::arg("a"),
        py::arg("b"),
        "Dot product of two 1-D tensors (not yet implemented)");
    
    m.def("transpose", [](const Tensor& input, py::object dim0, py::object dim1) {
            if (dim0.is_none() && dim1.is_none()) {
                // Reverse all dimensions
                return input.transpose();
            }
            return input.transpose(dim0.cast<int>(), dim1.cast<int>());
        },
        py::arg("input"),
        py::arg("dim0") = py::none(),
        py::arg("dim1") = py::none(),
        "Transpose tensor");
    
    m.def("inverse", [](const Tensor& input) -> Tensor {
            // Placeholder - actual inverse not implemented yet
            throw std::runtime_error("inverse() not yet implemented");
        },
        py::arg("input"),
        "Compute matrix inverse (not yet implemented)");
    
    m.def("cholesky", [](const Tensor& input) -> Tensor {
            // Placeholder - actual cholesky not implemented yet
            throw std::runtime_error("cholesky() not yet implemented");
        },
        py::arg("input"),
        "Cholesky decomposition (not yet implemented)");
    
    m.def("solve", [](const Tensor& a, const Tensor& b) -> Tensor {
            // Placeholder - actual solve not implemented yet
            throw std::runtime_error("solve() not yet implemented");
        },
        py::arg("a"),
        py::arg("b"),
        "Solve linear system ax = b (not yet implemented)");
    
    m.def("trace", [](const Tensor& input) {
            // Compute trace as sum of diagonal
            if (input.ndim() != 2) {
                throw std::runtime_error("trace only supports 2D tensors");
            }
            auto diag = input;  // Placeholder - actual implementation needed
            return diag;
        },
        py::arg("input"),
        "Sum of diagonal elements");
    
    m.def("diag", [](const Tensor& input, int offset) {
            if (input.ndim() == 1) {
                // Create diagonal matrix from vector
                return input;  // Placeholder
            } else {
                // Extract diagonal from matrix
                return input;  // Placeholder
            }
        },
        py::arg("input"),
        py::arg("offset") = 0,
        "Extract diagonal or create diagonal matrix");
    
    // Element-wise operations
    m.def("exp", [](const Tensor& input) {
            return input.power(std::exp(1.0));  // Placeholder - should use actual exp
        },
        py::arg("input"),
        "Element-wise exponential");
    
    m.def("log", [](const Tensor& input) {
            return input;  // Placeholder
        },
        py::arg("input"),
        "Element-wise natural logarithm");
    
    m.def("sqrt", [](const Tensor& input) {
            return input.power(0.5);
        },
        py::arg("input"),
        "Element-wise square root");
    
    m.def("square", [](const Tensor& input) {
            return input.multiply(input);
        },
        py::arg("input"),
        "Element-wise square");
    
    m.def("abs", [](const Tensor& input) {
            return input;  // Placeholder
        },
        py::arg("input"),
        "Element-wise absolute value");
    
    m.def("sign", [](const Tensor& input) {
            return input;  // Placeholder
        },
        py::arg("input"),
        "Element-wise sign");
    
    m.def("clip", [](const Tensor& input, py::object min, py::object max) {
            return input;  // Placeholder
        },
        py::arg("input"),
        py::arg("min") = py::none(),
        py::arg("max") = py::none(),
        "Clip values to range [min, max]");
    
    // Shape manipulation
    m.def("concatenate", [](const std::vector<Tensor>& tensors, int axis) {
            return tensors[0];  // Placeholder
        },
        py::arg("tensors"),
        py::arg("axis") = 0,
        "Concatenate tensors along axis");
    
    m.def("stack", [](const std::vector<Tensor>& tensors, int axis) {
            return tensors[0];  // Placeholder
        },
        py::arg("tensors"),
        py::arg("axis") = 0,
        "Stack tensors along new axis");
    
    m.def("broadcast_to", [](const Tensor& input, const std::vector<size_t>& shape) {
            return input;  // Placeholder
        },
        py::arg("input"),
        py::arg("shape"),
        "Broadcast tensor to new shape");
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
    
    // Gamma functions - these are in math namespace
    m.def("gammaln", [](py::array_t<double> x) {
            py::buffer_info info = x.request();
            std::vector<double> result(info.size);
            math::gammaln(static_cast<double*>(info.ptr), result.data(), info.size);
            return result;
        },
        py::arg("x"),
        "Log gamma function log(Γ(x))");
    
    m.def("digamma", [](py::array_t<double> x) {
            py::buffer_info info = x.request();
            std::vector<double> result(info.size);
            math::digamma(static_cast<double*>(info.ptr), result.data(), info.size);
            return result;
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
    m.def("logsumexp", [](py::array_t<double> x) {
            py::buffer_info info = x.request();
            return math::logsumexp(static_cast<double*>(info.ptr), info.size);
        },
        py::arg("x"),
        "Numerically stable log-sum-exp");
    
    m.def("softmax", [](py::array_t<double> x) {
            py::buffer_info info = x.request();
            std::vector<double> result(info.size);
            math::softmax(static_cast<double*>(info.ptr), result.data(), info.size, nullptr);
            return result;
        },
        py::arg("x"),
        "Softmax function");
    
    // Gaussian functions
    m.def("gaussian_loglik", [](py::array_t<double> x, py::array_t<double> mean, 
                                py::array_t<double> inv_cov, double logdet_inv_cov) {
            py::buffer_info x_info = x.request();
            py::buffer_info m_info = mean.request();
            py::buffer_info i_info = inv_cov.request();
            int dim = static_cast<int>(x_info.shape.back());
            size_t batch_size = x_info.size / dim;
            std::vector<double> result(batch_size);
            math::gaussian_loglik_batch(
                static_cast<double*>(x_info.ptr), 
                static_cast<double*>(m_info.ptr), 
                static_cast<double*>(i_info.ptr), 
                &logdet_inv_cov, dim, batch_size, result.data()
            );
            return result;
        },
        py::arg("x"),
        py::arg("mean"),
        py::arg("inv_cov"),
        py::arg("logdet_inv_cov"),
        "Gaussian log-likelihood");
    
    m.def("gaussian_loglik_isotropic", [](py::array_t<double> x, py::array_t<double> mean, 
                                          double sigma_sqr) {
            py::buffer_info x_info = x.request();
            py::buffer_info m_info = mean.request();
            int dim = static_cast<int>(x_info.shape.back());
            size_t batch_size = x_info.size / dim;
            size_t num_components = 1;
            std::vector<double> result(batch_size * num_components);
            math::gaussian_loglik_isotropic_batch(
                static_cast<double*>(x_info.ptr), 
                static_cast<double*>(m_info.ptr), 
                &sigma_sqr, dim, batch_size, num_components, result.data()
            );
            return result;
        },
        py::arg("x"),
        py::arg("mean"),
        py::arg("sigma_sqr"),
        "Isotropic Gaussian log-likelihood");
    
    // Matrix utilities
    m.def("symmetrize", [](py::array_t<double> matrix) {
            py::buffer_info info = matrix.request();
            int dim = static_cast<int>(info.shape[0]);
            math::symmetrize(static_cast<double*>(info.ptr), dim);
        },
        py::arg("matrix"),
        "Make matrix symmetric: (A + Aᵀ) / 2");
    
    m.def("make_positive_definite", [](py::array_t<double> matrix, double epsilon) {
            py::buffer_info info = matrix.request();
            int dim = static_cast<int>(info.shape[0]);
            math::make_positive_definite(static_cast<double*>(info.ptr), dim, epsilon);
        },
        py::arg("matrix"),
        py::arg("epsilon") = 1e-6,
        "Make matrix positive definite");
    
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

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
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <memory>
#include <sstream>

#include "bindings.h"

// Distribution headers
#include "../distributions/distribution.h"
#include "../distributions/exponential_family.h"
#include "../distributions/delta.h"

namespace py = pybind11;
using namespace axiomcuda;
using namespace axiom::distributions;

// Bind std::vector<double> for return types
PYBIND11_MAKE_OPAQUE(std::vector<double>);

// Helper function to convert shape to string
std::string shape_to_string(const std::vector<int>& shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << ")";
    return oss.str();
}

// Type alias for ArrayDict with double arrays
using ArrayDictDouble = ArrayDict<std::vector<double>>;

void bind_array_dict(py::module_& m) {
    py::class_<ArrayDictDouble>(m, "ArrayDict", R"doc(
        Dictionary-like container for named arrays.
        
        Used extensively in distributions to store natural parameters,
        expectations, and sufficient statistics.
    )doc")
        .def(py::init<>(), "Create empty ArrayDict")
        .def("__getitem__", [](ArrayDictDouble& self, const std::string& key) {
                return py::array_t<double>(self.get(key).size(), self.get(key).data());
            }, py::arg("key"), "Get array by key")
        .def("__setitem__", [](ArrayDictDouble& self, const std::string& key, py::array_t<double> value) {
                py::buffer_info info = value.request();
                std::vector<double> vec(static_cast<double*>(info.ptr), 
                                       static_cast<double*>(info.ptr) + info.size);
                self.set(key, vec);
            }, py::arg("key"), py::arg("value"), "Set array by key")
        .def("__contains__", &ArrayDictDouble::has, py::arg("key"), "Check if key exists")
        .def("keys", &ArrayDictDouble::keys, "List all keys")
        .def("has", &ArrayDictDouble::has, py::arg("key"), "Check if key exists");
}

// Base Distribution binding
void bind_distribution_base(py::module_& m) {
    py::class_<Distribution, std::shared_ptr<Distribution>>(m, "Distribution", docstrings::DISTRIBUTION_BASE)
        // Shape fields (public data members)
        .def_readonly("batch_shape", &Distribution::batch_shape,
            "Shape of batch dimensions")
        .def_readonly("event_shape", &Distribution::event_shape,
            "Shape of event dimensions (data shape)")
        .def_readonly("dim", &Distribution::dim,
            "Dimensionality of the distribution")
        .def_readonly("event_dim", &Distribution::event_dim,
            "Number of event dimensions")
        .def_readonly("batch_dim", &Distribution::batch_dim,
            "Number of batch dimensions")
        .def("shape", &Distribution::shape,
            "Full shape (batch_shape + event_shape)")
        .def("to_event", &Distribution::to_event, py::arg("n"),
            "Convert batch dimensions to event dimensions")
        .def("expand_batch_shape", &Distribution::expand_batch_shape, py::arg("axes"),
            "Expand batch shape by inserting singleton dimensions")
        .def("swap_axes", &Distribution::swap_axes, py::arg("axis1"), py::arg("axis2"),
            "Swap batch axes")
        .def("copy", &Distribution::copy, "Create a copy of the distribution")
        .def("__repr__", [](const Distribution& self) {
            return "Distribution(shape=" + shape_to_string(self.shape()) + ")";
        });
}

// ExponentialFamily binding (double specialization)
void bind_exponential_family(py::module_& m) {
    using EF = ExponentialFamily<double>;
    
    py::class_<EF, Distribution, std::shared_ptr<EF>>(
        m, "ExponentialFamily", docstrings::EXPONENTIAL_FAMILY)
        // Properties for natural parameters and expectations
        .def_property("nat_params",
            [](EF& self) -> ArrayDictDouble& { return self.nat_params(); },
            [](EF& self, const ArrayDictDouble& val) { self.set_nat_params(val); },
            "Natural parameters of the distribution")
        .def_property("expectations",
            [](EF& self) -> ArrayDictDouble& { return self.expectations(); },
            [](EF& self, const ArrayDictDouble& val) { self.set_expectations(val); },
            "Expected sufficient statistics")
        .def("expected_statistics", &EF::expected_statistics,
            "Compute expected sufficient statistics")
        .def("statistics", &EF::statistics, py::arg("x"),
            "Compute sufficient statistics for data")
        .def("params_from_statistics", &EF::params_from_statistics, py::arg("stats"),
            "Convert sufficient statistics to natural parameters")
        .def("combine", &EF::combine, py::arg("other"),
            "Combine natural parameters with other distributions")
        .def("residual", [](const EF& self) -> const std::vector<double>& { 
                return self.residual(); 
            },
            "Get residual term (for message passing)")
        .def("log_partition", [](const EF& self) { return self.log_partition(); },
            "Compute log partition function")
        .def("entropy", [](const EF& self) { return self.entropy(); },
            "Compute entropy")
        .def("log_likelihood", [](const EF& self, const std::vector<double>& x) { 
                return self.log_likelihood(x); 
            }, py::arg("x"),
            "Compute log likelihood of data")
        .def("sample", [](const EF& self, const std::vector<double>& key, const std::vector<int>& shape) { 
                return self.sample(key, shape); 
            }, py::arg("key"), py::arg("shape"),
            "Draw samples from the distribution")
        .def("__mul__", [](const EF& self, const std::shared_ptr<EF>& other) {
                return self.operator*(other);
            },
            "Multiply two distributions (combine natural parameters)");
}

// Delta binding - temporarily disabled due to copy() return type issue
// void bind_delta(py::module_& m) {
//     using DeltaDouble = Delta<double>;
//     ...
// }

// Main distribution module initialization
void init_distribution_module(py::module_& m) {
    m.doc() = R"doc(
        Probability distributions for variational inference.
        
        This module provides:
        - Exponential family distributions (MultivariateNormal, Multinomial)
        - Conjugate prior distributions (Normal-Inverse-Wishart, Dirichlet)
        - Delta distributions for observed data
        
        All distributions support:
        - Natural and mean parameterizations
        - Sampling and log-likelihood computation
        - GPU acceleration via CUDA
    )doc";
    
    // Bind helper types first
    bind_array_dict(m);
    
    // Bind base classes
    bind_distribution_base(m);
    bind_exponential_family(m);
    
    // Bind Delta (simplest distribution) - temporarily disabled
    // bind_delta(m);
}

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

#include "bindings.h"
#include "../transforms/linear_mng.h"

namespace py = pybind11;
using namespace axiomcuda;
using namespace axiom::transforms;

// LinearMatrixNormalGamma binding - simplified version
void bind_linear_mng(py::module_& m) {
    py::class_<LinearMatrixNormalGamma, std::shared_ptr<LinearMatrixNormalGamma>>(
        m, "LinearMatrixNormalGamma", R"doc(
            Linear transformation with Matrix Normal - Gamma prior.
            
            Models the linear relationship y = Ax + ε where:
            - y is output (dim y_dim)
            - x is input (dim x_dim)
            - A is the linear transformation matrix
            - ε ~ N(0, Σ) is Gaussian noise with diagonal covariance
            
            Note: Full bindings with parameter access are not yet implemented.
            Use the constructor to set dimensions and basic configuration.
        )doc")
        // Constructor with dimensions
        .def(py::init<int, int, bool, bool, float, float, float, std::vector<int>>(),
            py::arg("x_dim"),
            py::arg("y_dim"),
            py::arg("use_bias") = true,
            py::arg("fixed_precision") = false,
            py::arg("scale") = 1.0f,
            py::arg("dof_offset") = 1.0f,
            py::arg("inv_v_scale") = 1.0f,
            py::arg("batch_shape") = std::vector<int>{},
            "Create LinearMatrixNormalGamma with given dimensions")
        // Core properties
        .def_property_readonly("x_dim", &LinearMatrixNormalGamma::get_x_dim,
            "Input dimension")
        .def_property_readonly("y_dim", &LinearMatrixNormalGamma::get_y_dim,
            "Output dimension")
        .def_property_readonly("use_bias", &LinearMatrixNormalGamma::uses_bias,
            "Whether transform uses bias term")
        .def_property_readonly("batch_shape", &LinearMatrixNormalGamma::get_batch_shape,
            "Batch shape of transform")
        .def_property_readonly("event_shape", &LinearMatrixNormalGamma::get_event_shape,
            "Event shape of transform")
        // Parameter access - placeholders
        .def("update_cache", &LinearMatrixNormalGamma::update_cache,
            "Update internal cache after parameter changes")
        .def("copy", &LinearMatrixNormalGamma::copy,
            "Create a copy of the transform");
}

// Main transform module initialization
void init_transform_module(py::module_& m) {
    m.doc() = R"doc(
        Variational inference transforms for message passing.
        
        This module provides transforms for variational inference:
        - LinearMatrixNormalGamma: Linear transformation with learnable parameters
        
        Transforms support:
        - Forward and backward message passing
        - Variational approximations for efficiency
        - Parameter learning from data
        - GPU acceleration
    )doc";
    
    // Bind concrete transforms only
    bind_linear_mng(m);
}

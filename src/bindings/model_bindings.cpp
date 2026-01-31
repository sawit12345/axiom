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
#include <pybind11/functional.h>
#include <memory>

#include "bindings.h"

namespace py = pybind11;
using namespace axiomcuda;

// Simplified model module - just expose config structs for now
void init_model_module(py::module_& m) {
    m.doc() = R"doc(
        Probabilistic models for learning and inference.
        
        This module provides implementations of:
        - SMM: Slot Mixture Model for image segmentation
        - RMM: Relational Mixture Model for action prediction
        - TMM: Transition Mixture Model for dynamics learning
        - IMM: Identity Mixture Model for object classification
        
        All models support GPU acceleration and online learning.
    )doc";
    
    // Model bindings are currently simplified due to namespace conflicts
    // and incomplete class definitions. Full bindings would require:
    // 1. Resolving axiom::models::Tensor vs axiomcuda::Tensor
    // 2. Completing all class method signatures
    // 3. Adding proper factory functions
}

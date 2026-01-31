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
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "bindings.h"

namespace py = pybind11;
using namespace axiomcuda;

// Forward declarations for submodule initialization
void init_tensor_module(py::module_& m);
void init_math_module(py::module_& m);
void init_distribution_module(py::module_& m);
void init_transform_module(py::module_& m);
void init_model_module(py::module_& m);

PYBIND11_MODULE(axiomcuda_backend, m) {
    m.doc() = R"doc(
        Axiom CUDA Backend - High-performance C++/CUDA kernels for Axiom
        
        This module provides accelerated implementations of:
        - Core mathematical operations (gamma, digamma, logsumexp, etc.)
        - Linear algebra operations (matrix operations, decompositions)
        - Probability distributions (MultivariateNormal, Multinomial, etc.)
        - Variational inference transforms (LinearMatrixNormalGamma)
        - Mixture models (SMM, RMM, TMM, IMM)
        
        All operations support both CPU and GPU execution with seamless
        numpy array interoperability.
    )doc";
    
    // Module metadata
    m.attr("__version__") = "0.1.0";
    m.attr("__cuda_version__") = "12.0";
    m.attr("__backend__") = "cuda";
    
    // Device management submodule
    py::module_ device = m.def_submodule("device", "Device management utilities");
    device.doc() = "GPU/CPU device management and memory utilities";
    
    device.def("get_device_count", &DeviceManager::getDeviceCount,
               "Get the number of available CUDA devices");
    
    device.def("get_current_device", &DeviceManager::getCurrentDevice,
               "Get the current CUDA device ID");
    
    device.def("set_device", &DeviceManager::setDevice,
               py::arg("device_id"),
               "Set the current CUDA device");
    
    device.def("synchronize", &DeviceManager::synchronize,
               py::arg("device_id") = -1,
               "Synchronize the specified CUDA device (or current if -1)");
    
    device.def("get_device_properties", &DeviceManager::getDeviceProperties,
               py::arg("device_id") = 0,
               "Get properties of the specified CUDA device");
    
    device.def("get_memory_info", &DeviceManager::getMemoryInfo,
               py::arg("device_id") = -1,
               "Get memory info for the specified device (free, total)");
    
    // Device properties class
    py::class_<DeviceProperties>(device, "DeviceProperties")
        .def_readonly("name", &DeviceProperties::name)
        .def_readonly("total_memory", &DeviceProperties::totalMemory)
        .def_readonly("major", &DeviceProperties::major)
        .def_readonly("minor", &DeviceProperties::minor)
        .def_readonly("multi_processor_count", &DeviceProperties::multiProcessorCount)
        .def_readonly("max_threads_per_block", &DeviceProperties::maxThreadsPerBlock)
        .def_readonly("warp_size", &DeviceProperties::warpSize)
        .def("__repr__", [](const DeviceProperties& p) {
            return "DeviceProperties(name='" + std::string(p.name) + 
                   "', memory=" + std::to_string(p.totalMemory / (1024*1024)) + " MB)";
        });
    
    // Initialize submodules
    py::module_ tensor = m.def_submodule("tensor", "Tensor and array operations");
    init_tensor_module(tensor);
    
    py::module_ math = m.def_submodule("math", "Mathematical functions");
    init_math_module(math);
    
    py::module_ distributions = m.def_submodule("distributions", "Probability distributions");
    init_distribution_module(distributions);
    
    py::module_ transforms = m.def_submodule("transforms", "Variational inference transforms");
    init_transform_module(transforms);
    
    py::module_ models = m.def_submodule("models", "Probabilistic models");
    init_model_module(models);
    
    // Convenience function to check CUDA availability
    m.def("cuda_available", []() {
        return DeviceManager::getDeviceCount() > 0;
    }, "Check if CUDA is available");
    
    m.def("get_cuda_version", []() {
        return std::string("12.0");  // Update based on actual version
    }, "Get the CUDA version");
}

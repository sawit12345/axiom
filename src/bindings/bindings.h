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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <memory>

// Forward declarations
namespace axiomcuda {
    class Tensor;
    class DeviceManager;
    struct DeviceProperties;
}

namespace py = pybind11;

// Type casters for numpy interoperability
namespace pybind11 { namespace detail {
    // Helper to convert between std::vector and numpy arrays
    template <typename T>
    struct type_caster<std::vector<T>> {
        using value_conv = make_caster<T>;
        PYBIND11_TYPE_CASTER(std::vector<T>, _("List[") + value_conv::name + _("]"));
        
        bool load(handle src, bool convert) {
            if (!isinstance<sequence>(src))
                return false;
            auto s = reinterpret_borrow<sequence>(src);
            value.reserve(s.size());
            for (auto it : s) {
                value_conv conv;
                if (!conv.load(it, convert))
                    return false;
                value.push_back(cast_op<T&&>(std::move(conv)));
            }
            return true;
        }
        
        template <typename U>
        static handle cast(U&& src, return_value_policy policy, handle parent) {
            list l(src.size());
            size_t index = 0;
            for (auto&& value : src) {
                auto value_ = reinterpret_steal<object>(
                    value_conv::cast(forward_like<U>(value), policy, parent));
                if (!value_)
                    return handle();
                PyList_SET_ITEM(l.ptr(), (ssize_t)index++, value_.release().ptr());
            }
            return l.release();
        }
    };
}}

// Helper macros for binding getters/setters
#define BIND_PROPERTY(cls, name, getter, setter, doc) \
    .def_property(name, &cls::getter, &cls::setter, doc)

#define BIND_READONLY(cls, name, getter, doc) \
    .def_property_readonly(name, &cls::getter, doc)

#define BIND_METHOD(cls, name, doc) \
    .def(#name, &cls::name, doc)

#define BIND_METHOD_OVERLOAD(cls, name, ret, args, doc) \
    .def(#name, static_cast<ret (cls::*)args>(&cls::name), doc)

#define BIND_STATIC_METHOD(cls, name, ret, args, doc) \
    .def_static(#name, static_cast<ret (*)(args)>(&cls::name), doc)

// Docstring helpers
namespace docstrings {
    const char* DISTRIBUTION_BASE = R"doc(
        Base class for all probability distributions.
        
        Provides common functionality for:
        - Shape management (batch_shape, event_shape)
        - Parameter access (natural parameters, mean parameters)
        - Statistical operations (entropy, log_partition, etc.)
    )doc";
    
    const char* EXPONENTIAL_FAMILY = R"doc(
        Exponential family distribution base class.
        
        Distributions in the exponential family have the form:
        p(x|theta) = h(x) * exp(eta(theta) . T(x) - A(eta))
        
        where:
        - h(x) is the base measure
        - eta(theta) are the natural parameters
        - T(x) are the sufficient statistics
        - A(eta) is the log partition function
    )doc";
    
    const char* CONJUGATE_BASE = R"doc(
        Conjugate prior distribution base class.
        
        Conjugate priors have the property that the posterior distribution
        is in the same family as the prior. This enables efficient Bayesian
        updates.
    )doc";
}

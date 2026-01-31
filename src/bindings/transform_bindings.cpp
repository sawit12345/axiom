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
#include "../transforms/transform.h"
#include "../transforms/linear_mng.h"

namespace py = pybind11;
using namespace axiomcuda;
using namespace axiomcuda::transforms;
using namespace axiomcuda::vi;

// Transform base class binding
void bind_transform_base(py::module_& m) {
    py::class_<Transform, std::shared_ptr<Transform>>(
        m, "Transform", R"doc(
            Base class for variational inference transforms.
            
            Transforms model p(y|x,θ) where y is the output, x is the input,
            and θ are parameters. They support:
            - Forward/backward message passing
            - Variational approximations
            - Parameter updates from data
            - KL divergence computation
            
            This is the base class for all transforms including
            LinearMatrixNormalGamma.
        )doc")
        .def_property_readonly("x_dim", &Transform::getXDim,
            "Input dimension")
        .def_property_readonly("y_dim", &Transform::getYDim,
            "Output dimension")
        .def_property_readonly("use_bias", &Transform::usesBias,
            "Whether transform uses bias term")
        .def_property_readonly("batch_shape", &Transform::getBatchShape,
            "Batch shape of transform")
        .def_property_readonly("event_shape", &Transform::getEventShape,
            "Event shape of transform")
        .def_property("posterior_params",
            &Transform::getPosteriorParams,
            &Transform::setPosteriorParams,
            "Posterior canonical parameters")
        .def_property("prior_params",
            &Transform::getPriorParams,
            &Transform::setPriorParams,
            "Prior canonical parameters")
        .def("update_from_data", [](Transform& self,
                                   std::tuple<py::array_t<double>, py::array_t<double>> data,
                                   py::object weights,
                                   float lr,
                                   float beta) {
                auto [X_arr, Y_arr] = data;
                Tensor X = Tensor::from_numpy(X_arr);
                Tensor Y = Tensor::from_numpy(Y_arr);
                
                Tensor* weights_ptr = nullptr;
                Tensor weights_t;
                if (!weights.is_none()) {
                    weights_t = Tensor::from_numpy(weights.cast<py::array_t<double>>());
                    weights_ptr = &weights_t;
                }
                
                self.updateFromData({X, Y}, weights_ptr, lr, beta);
            },
            py::arg("data"),
            py::arg("weights") = py::none(),
            py::arg("lr") = 1.0f,
            py::arg("beta") = 0.0f,
            "Update parameters from data (X, Y)")
        .def("update_from_probabilities", [](Transform& self,
                                            std::tuple<std::shared_ptr<Distribution>,
                                                      std::shared_ptr<Distribution>> pXY,
                                            py::object weights,
                                            float lr,
                                            float beta) {
                Tensor* weights_ptr = nullptr;
                Tensor weights_t;
                if (!weights.is_none()) {
                    weights_t = Tensor::from_numpy(weights.cast<py::array_t<double>>());
                    weights_ptr = &weights_t;
                }
                
                self.updateFromProbabilities(pXY, weights_ptr, lr, beta);
            },
            py::arg("pXY"),
            py::arg("weights") = py::none(),
            py::arg("lr") = 1.0f,
            py::arg("beta") = 0.0f,
            "Update from probability distributions")
        .def("update_from_statistics", &Transform::updateFromStatistics,
            py::arg("stats"),
            py::arg("lr") = 1.0f,
            py::arg("beta") = 0.0f,
            "Update from sufficient statistics")
        .def("expected_likelihood_params", &Transform::expectedLikelihoodParams,
            "Get expected natural parameters")
        .def("expected_log_likelihood", [](Transform& self,
                                          std::tuple<py::array_t<double>, py::array_t<double>> data) {
                auto [X_arr, Y_arr] = data;
                Tensor X = Tensor::from_numpy(X_arr);
                Tensor Y = Tensor::from_numpy(Y_arr);
                
                return self.expectedLogLikelihood({X, Y}).to_numpy();
            },
            py::arg("data"),
            "Compute expected log likelihood")
        .def("average_energy", [](Transform& self,
                                 std::tuple<std::shared_ptr<Distribution>,
                                           std::shared_ptr<Distribution>> inputs) {
                auto [pX, pY] = inputs;
                return self.averageEnergy({pX, pY}).to_numpy();
            },
            py::arg("inputs"),
            "Compute average energy for distribution inputs")
        .def("forward_from_normal", [](Transform& self,
                                      std::shared_ptr<ExponentialFamily> pX,
                                      bool pass_residual) {
                return self.forwardFromNormal(pX, pass_residual);
            },
            py::arg("pX"),
            py::arg("pass_residual") = false,
            "Forward message passing (exact, marginalizes over input)")
        .def("backward_from_normal", [](Transform& self,
                                       std::shared_ptr<ExponentialFamily> pY,
                                       bool pass_residual) {
                return self.backwardFromNormal(pY, pass_residual);
            },
            py::arg("pY"),
            py::arg("pass_residual") = false,
            "Backward message passing (exact)")
        .def("variational_forward", [](Transform& self,
                                      std::shared_ptr<Distribution> pX,
                                      bool pass_residual) {
                return self.variationalForward(pX, pass_residual);
            },
            py::arg("pX"),
            py::arg("pass_residual") = false,
            "Variational forward (fast approximation)")
        .def("variational_backward", [](Transform& self,
                                       std::shared_ptr<Distribution> pY,
                                       bool pass_residual) {
                return self.variationalBackward(pY, pass_residual);
            },
            py::arg("pY"),
            py::arg("pass_residual") = false,
            "Variational backward (fast approximation)")
        .def("predict", [](Transform& self, py::array_t<double> x) {
                Tensor x_t = Tensor::from_numpy(x);
                return self.predict(x_t);
            },
            py::arg("x"),
            "Predict output distribution given input")
        .def("kl_divergence", &Transform::klDivergence,
            "KL divergence between posterior and prior")
        .def("elbo", [](Transform& self,
                       std::tuple<py::array_t<double>, py::array_t<double>> data,
                       py::object weights) {
                auto [X_arr, Y_arr] = data;
                Tensor X = Tensor::from_numpy(X_arr);
                Tensor Y = Tensor::from_numpy(Y_arr);
                
                Tensor* weights_ptr = nullptr;
                Tensor weights_t;
                if (!weights.is_none()) {
                    weights_t = Tensor::from_numpy(weights.cast<py::array_t<double>>());
                    weights_ptr = &weights_t;
                }
                
                return self.elbo({X, Y}, weights_ptr).to_numpy();
            },
            py::arg("data"),
            py::arg("weights") = py::none(),
            "Compute ELBO for data")
        .def("elbo_contrib", [](Transform& self,
                               std::tuple<std::shared_ptr<Distribution>,
                                         std::shared_ptr<Distribution>> pXY,
                               py::object weights) {
                Tensor* weights_ptr = nullptr;
                Tensor weights_t;
                if (!weights.is_none()) {
                    weights_t = Tensor::from_numpy(weights.cast<py::array_t<double>>());
                    weights_ptr = &weights_t;
                }
                
                return self.elboContrib(pXY, weights_ptr).to_numpy();
            },
            py::arg("pXY"),
            py::arg("weights") = py::none(),
            "Compute ELBO contribution from distributions")
        .def("to_natural_params", &Transform::toNaturalParams,
            py::arg("params"),
            "Convert canonical to natural parameters")
        .def("copy", &Transform::copy,
            "Create a copy of the transform");
}

// LinearMatrixNormalGamma binding
void bind_linear_mng(py::module_& m) {
    py::class_<LinearMatrixNormalGamma, Transform, std::shared_ptr<LinearMatrixNormalGamma>>(
        m, "LinearMatrixNormalGamma", R"doc(
            Linear transformation with Matrix Normal - Gamma prior.
            
            Models the linear relationship y = Ax + ε where:
            - y is output (dim y_dim)
            - x is input (dim x_dim)
            - A is the linear transformation matrix
            - ε ~ N(0, Σ) is Gaussian noise with diagonal covariance
            
            The conjugate prior is Matrix Normal for A given Σ, and
            independent Gamma priors for the diagonal elements of Σ.
            
            Parameters (canonical):
            - mu: Mean of A (y_dim, x_dim)
            - inv_v: Inverse covariance of columns of A (x_dim, x_dim)
            - a: Shape parameters for Gamma (y_dim, 1)
            - b: Scale parameters for Gamma (y_dim, 1)
            
            Examples:
                >>> import numpy as np
                >>> from axiomcuda_backend import transforms, distributions
                >>> # Create with default parameters
                >>> lmg = transforms.LinearMatrixNormalGamma(
                ...     batch_shape=(1, 32),  # 1 batch, 32 slots
                ...     event_shape=(5, 3),   # y_dim=5, x_dim=3 (includes bias)
                ...     use_bias=True)
                >>> 
                >>> # Predict from input
                >>> x = np.random.randn(3, 1)  # (x_dim, 1)
                >>> pY = lmg.predict(x)
                >>> 
                >>> # Update from data
                >>> X = np.random.randn(100, 3, 1)
                >>> Y = np.random.randn(100, 5, 1)
                >>> lmg.update_from_data((X, Y), lr=0.1)
                >>> 
                >>> # Message passing
                >>> from axiomcuda_backend import distributions
                >>> # Forward: pY = ∫ p(y|x) p(x) dx
                >>> nat_params = distributions.ArrayDict()
                >>> nat_params['inv_sigma_mu'] = np.zeros((3, 1))
                >>> nat_params['inv_sigma'] = np.eye(3)
                >>> pX = distributions.MultivariateNormal(nat_params=nat_params)
                >>> pY = lmg.forward_from_normal(pX)
                >>> 
                >>> # Backward: pX = ∫ p(y|x) p(y) dy
                >>> pX_back = lmg.backward_from_normal(pY)
        )doc")
        .def(py::init([](py::object params, py::object prior_params,
                        std::vector<int> batch_shape, std::vector<int> event_shape,
                        int event_dim, bool use_bias, bool fixed_precision,
                        double scale, double dof_offset, double inv_v_scale,
                        py::object init_key) {
                ArrayDict p, pp;
                if (!params.is_none()) {
                    p = ArrayDict::from_dict(params.cast<py::dict>());
                }
                if (!prior_params.is_none()) {
                    pp = ArrayDict::from_dict(prior_params.cast<py::dict>());
                }
                
                return std::make_shared<LinearMatrixNormalGamma>(
                    p.is_empty() ? nullptr : &p,
                    pp.is_empty() ? nullptr : &pp,
                    batch_shape.empty() ? nullptr : &batch_shape,
                    event_shape.empty() ? nullptr : &event_shape,
                    event_dim, use_bias, fixed_precision, scale,
                    dof_offset, inv_v_scale, nullptr
                );
            }),
            py::arg("params") = py::none(),
            py::arg("prior_params") = py::none(),
            py::arg("batch_shape") = std::vector<int>{},
            py::arg("event_shape") = std::vector<int>{},
            py::arg("event_dim") = 2,
            py::arg("use_bias") = true,
            py::arg("fixed_precision") = false,
            py::arg("scale") = 1.0,
            py::arg("dof_offset") = 1.0,
            py::arg("inv_v_scale") = 1.0,
            py::arg("init_key") = py::none(),
            "Create LinearMatrixNormalGamma transform")
        .def_property_readonly("mu", &LinearMatrixNormalGamma::getMu,
            "Posterior mean of A")
        .def_property_readonly("prior_mu", &LinearMatrixNormalGamma::getPriorMu,
            "Prior mean of A")
        .def_property_readonly("inv_v", &LinearMatrixNormalGamma::getInvV,
            "Posterior inv_v (inverse covariance of columns)")
        .def_property_readonly("prior_inv_v", &LinearMatrixNormalGamma::getPriorInvV,
            "Prior inv_v")
        .def_property_readonly("v", &LinearMatrixNormalGamma::getV,
            "Posterior v (covariance of columns)")
        .def_property_readonly("prior_v", &LinearMatrixNormalGamma::getPriorV,
            "Prior v")
        .def_property_readonly("a", &LinearMatrixNormalGamma::getA,
            "Posterior Gamma shape a")
        .def_property_readonly("prior_a", &LinearMatrixNormalGamma::getPriorA,
            "Prior Gamma shape")
        .def_property_readonly("b", &LinearMatrixNormalGamma::getB,
            "Posterior Gamma scale b")
        .def_property_readonly("prior_b", &LinearMatrixNormalGamma::getPriorB,
            "Prior Gamma scale")
        .def_property_readonly("weights", &LinearMatrixNormalGamma::getWeights,
            "Weight matrix (mu without bias)")
        .def_property_readonly("bias", &LinearMatrixNormalGamma::getBias,
            "Bias vector")
        .def("expected_inv_sigma", &LinearMatrixNormalGamma::expectedInvSigma,
            "E[Σ⁻¹] = diag(a/b)")
        .def("expected_sigma", &LinearMatrixNormalGamma::expectedSigma,
            "E[Σ] = diag(b/(a-1))")
        .def("expected_logdet_inv_sigma", &LinearMatrixNormalGamma::expectedLogDetInvSigma,
            "E[log|Σ⁻¹|] = sum(digamma(a) - log(b))")
        .def("logdet_expected_inv_sigma", &LinearMatrixNormalGamma::logdetExpectedInvSigma,
            "log|E[Σ⁻¹]| = sum(log(a/b))")
        .def("expected_inv_sigma_x", &LinearMatrixNormalGamma::expectedInvSigmaX,
            "E[Σ⁻¹A]")
        .def("expected_x_inv_sigma_x", &LinearMatrixNormalGamma::expectedXInvSigmaX,
            "E[AᵀΣ⁻¹A]")
        .def("update_cache", &LinearMatrixNormalGamma::updateCache,
            "Update internal cache after parameter changes");
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
        
        Examples:
            >>> from axiomcuda_backend import transforms
            >>> # Create linear transform
            >>> linear = transforms.LinearMatrixNormalGamma(
            ...     batch_shape=(10,),
            ...     event_shape=(5, 3),
            ...     use_bias=True)
            >>> 
            >>> # Use in message passing
            >>> pY = linear.variational_forward(pX)
            >>> pX_new = linear.variational_backward(pY)
    )doc";
    
    // Bind base transform class
    bind_transform_base(m);
    
    // Bind concrete transforms
    bind_linear_mng(m);
}

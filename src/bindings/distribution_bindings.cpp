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
#include <memory>
#include <sstream>

#include "bindings.h"

// Distribution headers
#include "../vi/distribution.h"
#include "../vi/exponential/base.h"
#include "../vi/exponential/mvn.h"
#include "../vi/exponential/multinomial.h"
#include "../vi/exponential/delta.h"
#include "../vi/conjugate/base.h"
#include "../vi/conjugate/mvn.h"
#include "../vi/conjugate/multinomial.h"

namespace py = pybind11;
using namespace axiomcuda;
using namespace axiomcuda::vi;

// Helper to bind ArrayDict
void bind_array_dict(py::module_& m) {
    py::class_<ArrayDict>(m, "ArrayDict", R"doc(
        Dictionary-like container for named arrays.
        
        Used extensively in distributions to store natural parameters,
        expectations, and sufficient statistics. Supports both CPU and GPU
        storage with automatic numpy interop.
    )doc")
        .def(py::init<>(), "Create empty ArrayDict")
        .def("__getitem__", [](const ArrayDict& self, const std::string& key) {
            return self.get(key);
        }, py::arg("key"), "Get array by key")
        .def("__setitem__", [](ArrayDict& self, const std::string& key, py::array_t<double> value) {
            self.set(key, Tensor::from_numpy(value));
        }, py::arg("key"), py::arg("value"), "Set array by key")
        .def("__contains__", &ArrayDict::contains, py::arg("key"), "Check if key exists")
        .def("keys", &ArrayDict::keys, "Get list of keys")
        .def("items", &ArrayDict::items, "Get list of (key, value) pairs")
        .def("__len__", &ArrayDict::size, "Number of entries")
        .def("__repr__", [](const ArrayDict& self) {
            std::ostringstream oss;
            oss << "ArrayDict(";
            auto keys = self.keys();
            for (size_t i = 0; i < keys.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << keys[i] << "=...";
            }
            oss << ")";
            return oss.str();
        })
        .def("to_numpy", [](const ArrayDict& self) {
            py::dict result;
            for (const auto& key : self.keys()) {
                result[key.c_str()] = self.get(key).to_numpy();
            }
            return result;
        }, "Convert to Python dict of numpy arrays")
        .def_static("from_numpy", [](py::dict dict) {
            ArrayDict result;
            for (auto item : dict) {
                std::string key = item.first.cast<std::string>();
                py::array_t<double> arr = item.second.cast<py::array_t<double>>();
                result.set(key, Tensor::from_numpy(arr));
            }
            return result;
        }, py::arg("dict"), "Create from Python dict of numpy arrays");
}

// Base Distribution binding
void bind_distribution_base(py::module_& m) {
    py::class_<Distribution, std::shared_ptr<Distribution>>(m, "Distribution", docstrings::DISTRIBUTION_BASE)
        .def_property_readonly("batch_shape", &Distribution::batchShape, 
            "Shape of batch dimensions")
        .def_property_readonly("event_shape", &Distribution::eventShape, 
            "Shape of event dimensions (data shape)")
        .def_property_readonly("shape", &Distribution::shape, 
            "Full shape (batch_shape + event_shape)")
        .def_property_readonly("dim", &Distribution::dim, 
            "Dimensionality of the distribution")
        .def_property_readonly("event_dim", &Distribution::eventDim, 
            "Number of event dimensions")
        .def_property_readonly("batch_dim", &Distribution::batchDim, 
            "Number of batch dimensions")
        .def("to_event", &Distribution::toEvent, py::arg("n"), 
            "Convert batch dimensions to event dimensions")
        .def("expand_batch_shape", &Distribution::expandBatchShape, py::arg("axes"), 
            "Expand batch shape by inserting singleton dimensions")
        .def("swap_axes", &Distribution::swapAxes, py::arg("axis1"), py::arg("axis2"), 
            "Swap batch axes")
        .def("copy", &Distribution::copy, "Create a copy of the distribution")
        .def("entropy", &Distribution::entropy, "Compute entropy")
        .def("log_partition", &Distribution::logPartition, "Compute log partition function")
        .def("sample", &Distribution::sample, py::arg("key"), py::arg("shape") = std::vector<int>{1}, 
            "Draw samples from the distribution")
        .def("log_likelihood", &Distribution::logLikelihood, py::arg("x"), 
            "Compute log likelihood of data")
        .def("__mul__", &Distribution::multiply, py::arg("other"), 
            "Multiply two distributions (combine natural parameters)")
        .def("__repr__", [](const Distribution& self) {
            return "Distribution(shape=" + shape_to_string(self.shape()) + ")";
        });
}

// ExponentialFamily binding
void bind_exponential_family(py::module_& m) {
    py::class_<ExponentialFamily, Distribution, std::shared_ptr<ExponentialFamily>>(
        m, "ExponentialFamily", docstrings::EXPONENTIAL_FAMILY)
        .def_property("nat_params", 
            &ExponentialFamily::getNaturalParams, 
            &ExponentialFamily::setNaturalParams,
            "Natural parameters of the distribution")
        .def_property("expectations", 
            &ExponentialFamily::getExpectations, 
            &ExponentialFamily::setExpectations,
            "Expected sufficient statistics")
        .def("expected_statistics", &ExponentialFamily::expectedStatistics, 
            "Compute expected sufficient statistics")
        .def("statistics", &ExponentialFamily::statistics, py::arg("x"), 
            "Compute sufficient statistics for data")
        .def("params_from_statistics", &ExponentialFamily::paramsFromStatistics, py::arg("stats"), 
            "Convert sufficient statistics to natural parameters")
        .def("combine", &ExponentialFamily::combine, py::arg("others"), 
            "Combine natural parameters with other distributions")
        .def("residual", &ExponentialFamily::getResidual, 
            "Get residual term (for message passing)");
}

// MultivariateNormal (exponential family) binding
void bind_mvn_exponential(py::module_& m) {
    py::class_<exponential::MultivariateNormal, ExponentialFamily, 
               std::shared_ptr<exponential::MultivariateNormal>>(
        m, "MultivariateNormal", R"doc(
            Multivariate normal distribution in exponential family form.
            
            Natural parameters:
            - inv_sigma_mu: precision-weighted mean (Σ⁻¹μ)
            - inv_sigma: precision matrix (Σ⁻¹)
            
            Mean parameters:
            - mu: mean vector
            - sigma: covariance matrix
            
            Examples:
                >>> import numpy as np
                >>> from axiomcuda_backend import distributions
                >>> # Create with natural parameters
                >>> nat_params = distributions.ArrayDict()
                >>> nat_params['inv_sigma_mu'] = np.array([[1.0], [2.0]])
                >>> nat_params['inv_sigma'] = np.array([[1.0, 0.5], [0.5, 1.0]])
                >>> dist = distributions.MultivariateNormal(nat_params=nat_params)
                >>> # Get mean parameters
                >>> dist.mu
                array([[...], [...]])
        )doc")
        .def(py::init([](py::object nat_params, py::object expectations,
                        std::vector<int> batch_shape, std::vector<int> event_shape,
                        int event_dim, py::object init_key, double scale) {
                ArrayDict np, exp;
                if (!nat_params.is_none()) {
                    np = ArrayDict::from_dict(nat_params.cast<py::dict>());
                }
                if (!expectations.is_none()) {
                    exp = ArrayDict::from_dict(expectations.cast<py::dict>());
                }
                return std::make_shared<exponential::MultivariateNormal>(
                    np.is_empty() ? nullptr : &np,
                    batch_shape.empty() ? nullptr : &batch_shape,
                    event_shape.empty() ? nullptr : &event_shape,
                    event_dim, nullptr, scale  // key not implemented
                );
            }),
            py::arg("nat_params") = py::none(),
            py::arg("expectations") = py::none(),
            py::arg("batch_shape") = std::vector<int>{},
            py::arg("event_shape") = std::vector<int>{},
            py::arg("event_dim") = 2,
            py::arg("init_key") = py::none(),
            py::arg("scale") = 1.0,
            "Create MultivariateNormal distribution")
        .def_property_readonly("mu", &exponential::MultivariateNormal::getMean, 
            "Mean vector μ")
        .def_property_readonly("sigma", &exponential::MultivariateNormal::getCovariance, 
            "Covariance matrix Σ")
        .def_property_readonly("inv_sigma", &exponential::MultivariateNormal::getPrecision, 
            "Precision matrix Σ⁻¹")
        .def_property_readonly("inv_sigma_mu", &exponential::MultivariateNormal::getPrecisionWeightedMean, 
            "Precision-weighted mean Σ⁻¹μ")
        .def_property_readonly("logdet_inv_sigma", 
            &exponential::MultivariateNormal::getLogDetPrecision, 
            "Log determinant of precision")
        .def("expected_x", &exponential::MultivariateNormal::expectedX, 
            "E[x] = μ")
        .def("expected_xx", &exponential::MultivariateNormal::expectedXX, 
            "E[xxᵀ] = Σ + μμᵀ")
        .def("shift", &exponential::MultivariateNormal::shift, py::arg("deltax"), 
            "Shift distribution by delta");
}

// Multinomial binding
void bind_multinomial(py::module_& m) {
    py::class_<exponential::Multinomial, ExponentialFamily,
               std::shared_ptr<exponential::Multinomial>>(
        m, "Multinomial", R"doc(
            Multinomial (categorical) distribution in exponential family form.
            
            Natural parameters:
            - logits: unnormalized log probabilities
            
            Mean parameters:
            - mean: probability vector (softmax of logits)
            
            Examples:
                >>> import numpy as np
                >>> from axiomcuda_backend import distributions
                >>> nat_params = distributions.ArrayDict()
                >>> nat_params['logits'] = np.array([0.0, 0.0, 0.0])  # uniform
                >>> dist = distributions.Multinomial(nat_params=nat_params)
                >>> dist.mean
                array([0.333..., 0.333..., 0.333...])
        )doc")
        .def(py::init([](py::object nat_params, py::object expectations,
                        std::vector<int> batch_shape, std::vector<int> event_shape,
                        int event_dim, double input_logZ) {
                ArrayDict np, exp;
                if (!nat_params.is_none()) {
                    np = ArrayDict::from_dict(nat_params.cast<py::dict>());
                }
                if (!expectations.is_none()) {
                    exp = ArrayDict::from_dict(expectations.cast<py::dict>());
                }
                return std::make_shared<exponential::Multinomial>(
                    np.is_empty() ? nullptr : &np,
                    batch_shape.empty() ? nullptr : &batch_shape,
                    event_shape.empty() ? nullptr : &event_shape,
                    event_dim, input_logZ
                );
            }),
            py::arg("nat_params") = py::none(),
            py::arg("expectations") = py::none(),
            py::arg("batch_shape") = std::vector<int>{},
            py::arg("event_shape") = std::vector<int>{},
            py::arg("event_dim") = 1,
            py::arg("input_logZ") = 0.0)
        .def_property_readonly("logits", &exponential::Multinomial::getLogits, 
            "Logit parameters")
        .def_property_readonly("mean", &exponential::Multinomial::getMean, 
            "Probability vector (softmax of logits)")
        .def_property_readonly("log_mean", &exponential::Multinomial::getLogMean, 
            "Log probability vector")
        .def_property_readonly("log_normalizer", &exponential::Multinomial::getLogNormalizer, 
            "Log partition function")
        .def("expected_x", &exponential::Multinomial::expectedX, 
            "E[x] = mean vector")
        .def("expected_xx", &exponential::Multinomial::expectedXX, 
            "E[xxᵀ] = diag(mean)");
}

// Delta binding
void bind_delta(py::module_& m) {
    py::class_<Delta, Distribution, std::shared_ptr<Delta>>(
        m, "Delta", R"doc(
            Dirac delta distribution (point mass).
            
            A degenerate distribution concentrated at a single point.
            Useful for representing observed data in variational inference.
            
            Examples:
                >>> import numpy as np
                >>> from axiomcuda_backend import distributions
                >>> values = np.array([[1.0], [2.0], [3.0]])
                >>> delta = distributions.Delta(values)
                >>> delta.mean
                array([[1.], [2.], [3.]])
        )doc")
        .def(py::init<py::array_t<double>, int>(),
            py::arg("values"),
            py::arg("event_dim") = 1,
            "Create Delta distribution from values")
        .def_property_readonly("values", &Delta::getValues, 
            "Point mass location")
        .def_property_readonly("p", &Delta::getP, 
            "Alias for values");
}

// Conjugate base binding
void bind_conjugate_base(py::module_& m) {
    py::class_<Conjugate, Distribution, std::shared_ptr<Conjugate>>(
        m, "Conjugate", docstrings::CONJUGATE_BASE)
        .def_property_readonly("posterior_params", &Conjugate::getPosteriorParams, 
            "Posterior canonical parameters")
        .def_property_readonly("prior_params", &Conjugate::getPriorParams, 
            "Prior canonical parameters")
        .def_property_readonly("likelihood_type", &Conjugate::getLikelihoodType, 
            "Type of likelihood distribution")
        .def("to_natural_params", &Conjugate::toNaturalParams, py::arg("params"), 
            "Convert canonical to natural parameters")
        .def("expected_likelihood_params", &Conjugate::expectedLikelihoodParams, 
            "Get expected natural parameters <S(θ)>_q(θ)")
        .def("expected_posterior_statistics", &Conjugate::expectedPosteriorStatistics, 
            "Get expected posterior statistics")
        .def("expected_log_partition", &Conjugate::expectedLogPartition, 
            "Get expected log partition <A(θ)>_q(θ)")
        .def("log_prior_partition", &Conjugate::logPriorPartition, 
            "Log partition of prior")
        .def("log_posterior_partition", &Conjugate::logPosteriorPartition, 
            "Log partition of posterior")
        .def("update_from_statistics", &Conjugate::updateFromStatistics, 
            py::arg("stats"), py::arg("lr") = 1.0, py::arg("beta") = 0.0,
            "Update from sufficient statistics")
        .def("update_from_probabilities", &Conjugate::updateFromProbabilities,
            py::arg("pX"), py::arg("weights") = py::none(),
            py::arg("lr") = 1.0, py::arg("beta") = 0.0,
            "Update from probability distributions")
        .def("kl_divergence", &Conjugate::klDivergence, 
            "KL divergence between posterior and prior");
}

// MultivariateNormalConjugate (NIW) binding
void bind_mvn_conjugate(py::module_& m) {
    py::class_<conjugate::MultivariateNormal, Conjugate,
               std::shared_ptr<conjugate::MultivariateNormal>>(
        m, "MultivariateNormalConjugate", R"doc(
            Normal-Inverse-Wishart conjugate prior for multivariate normal.
            
            Canonical parameters:
            - mean: prior mean m
            - kappa: concentration parameter
            - u: scale matrix
            - n: degrees of freedom
            
            This is the conjugate prior for the multivariate normal with
            unknown mean and covariance.
            
            Examples:
                >>> import numpy as np
                >>> from axiomcuda_backend import distributions
                >>> params = distributions.ArrayDict()
                >>> params['mean'] = np.zeros((2, 1))
                >>> params['kappa'] = np.array([[1.0]])
                >>> params['u'] = np.eye(2)
                >>> params['n'] = np.array([[4.0]])
                >>> niw = distributions.MultivariateNormalConjugate(
                ...     params=params, prior_params=params)
        )doc")
        .def(py::init([](py::object params, py::object prior_params,
                        std::vector<int> batch_shape, std::vector<int> event_shape,
                        int event_dim, bool fixed_precision, double scale,
                        double dof_offset, py::object init_key) {
                ArrayDict p, pp;
                if (!params.is_none()) {
                    p = ArrayDict::from_dict(params.cast<py::dict>());
                }
                if (!prior_params.is_none()) {
                    pp = ArrayDict::from_dict(prior_params.cast<py::dict>());
                }
                return std::make_shared<conjugate::MultivariateNormal>(
                    p.is_empty() ? nullptr : &p,
                    pp.is_empty() ? nullptr : &pp,
                    batch_shape.empty() ? nullptr : &batch_shape,
                    event_shape.empty() ? nullptr : &event_shape,
                    event_dim, fixed_precision, scale, dof_offset, nullptr
                );
            }),
            py::arg("params") = py::none(),
            py::arg("prior_params") = py::none(),
            py::arg("batch_shape") = std::vector<int>{},
            py::arg("event_shape") = std::vector<int>{},
            py::arg("event_dim") = 2,
            py::arg("fixed_precision") = false,
            py::arg("scale") = 1.0,
            py::arg("dof_offset") = 0.0,
            py::arg("init_key") = py::none())
        .def_property_readonly("mean", &conjugate::MultivariateNormal::getMean, 
            "Posterior mean m")
        .def_property_readonly("prior_mean", &conjugate::MultivariateNormal::getPriorMean, 
            "Prior mean")
        .def_property_readonly("kappa", &conjugate::MultivariateNormal::getKappa, 
            "Posterior concentration")
        .def_property_readonly("prior_kappa", &conjugate::MultivariateNormal::getPriorKappa, 
            "Prior concentration")
        .def_property_readonly("n", &conjugate::MultivariateNormal::getN, 
            "Posterior degrees of freedom")
        .def_property_readonly("prior_n", &conjugate::MultivariateNormal::getPriorN, 
            "Prior degrees of freedom")
        .def_property_readonly("u", &conjugate::MultivariateNormal::getScaleMatrix, 
            "Posterior scale matrix U")
        .def_property_readonly("inv_u", &conjugate::MultivariateNormal::getInvScaleMatrix, 
            "Inverse scale matrix U⁻¹")
        .def_property_readonly("expected_sigma", &conjugate::MultivariateNormal::expectedSigma, 
            "E[Σ] = U⁻¹/(n-d-1)")
        .def_property_readonly("expected_inv_sigma", &conjugate::MultivariateNormal::expectedInvSigma, 
            "E[Σ⁻¹] = nU")
        .def("expected_logdet_inv_sigma", &conjugate::MultivariateNormal::expectedLogDetInvSigma, 
            "E[log|Σ⁻¹|]")
        .def("expected_log_likelihood", &conjugate::MultivariateNormal::expectedLogLikelihood, 
            py::arg("data"), "Expected log likelihood")
        .def("average_energy", &conjugate::MultivariateNormal::averageEnergy, 
            py::arg("x"), "Average energy for distribution input")
        .def("variational_residual", &conjugate::MultivariateNormal::variationalResidual, 
            "Variational residual term")
        .def("kl_divergence_wishart", &conjugate::MultivariateNormal::klDivergenceWishart, 
            "KL divergence of Wishart component");
}

// MultinomialConjugate (Dirichlet) binding
void bind_multinomial_conjugate(py::module_& m) {
    py::class_<conjugate::Multinomial, Conjugate,
               std::shared_ptr<conjugate::Multinomial>>(
        m, "MultinomialConjugate", R"doc(
            Dirichlet conjugate prior for multinomial/categorical.
            
            Canonical parameters:
            - alpha: concentration parameters (pseudo-counts)
            
            Mean parameters:
            - mean: E[π] = α / sum(α)
            
            Examples:
                >>> import numpy as np
                >>> from axiomcuda_backend import distributions
                >>> params = distributions.ArrayDict()
                >>> params['alpha'] = np.array([1.0, 1.0, 1.0])  # uniform prior
                >>> dirichlet = distributions.MultinomialConjugate(
                ...     params=params, prior_params=params)
        )doc")
        .def(py::init([](py::object params, py::object prior_params,
                        std::vector<int> batch_shape, std::vector<int> event_shape,
                        int event_dim, double initial_count, py::object init_key) {
                ArrayDict p, pp;
                if (!params.is_none()) {
                    p = ArrayDict::from_dict(params.cast<py::dict>());
                }
                if (!prior_params.is_none()) {
                    pp = ArrayDict::from_dict(prior_params.cast<py::dict>());
                }
                return std::make_shared<conjugate::Multinomial>(
                    p.is_empty() ? nullptr : &p,
                    pp.is_empty() ? nullptr : &pp,
                    batch_shape.empty() ? nullptr : &batch_shape,
                    event_shape.empty() ? nullptr : &event_shape,
                    event_dim, initial_count, nullptr
                );
            }),
            py::arg("params") = py::none(),
            py::arg("prior_params") = py::none(),
            py::arg("batch_shape") = std::vector<int>{},
            py::arg("event_shape") = std::vector<int>{},
            py::arg("event_dim") = 1,
            py::arg("initial_count") = 1.0,
            py::arg("init_key") = py::none())
        .def_property_readonly("alpha", &conjugate::Multinomial::getAlpha, 
            "Posterior concentration α")
        .def_property_readonly("prior_alpha", &conjugate::Multinomial::getPriorAlpha, 
            "Prior concentration")
        .def_property_readonly("mean", &conjugate::Multinomial::getMean, 
            "E[π] = α / sum(α)")
        .def_property_readonly("log_mean", &conjugate::Multinomial::getLogMean, 
            "E[log π] = ψ(α) - ψ(sum(α))")
        .def_property_readonly("variance", &conjugate::Multinomial::getVariance, 
            "Var[π] = α(sum(α)-α)/(sum(α)²(sum(α)+1))")
        .def("expected_log_likelihood", &conjugate::Multinomial::expectedLogLikelihood, 
            py::arg("x"), "Expected log likelihood")
        .def("forward", &conjugate::Multinomial::forward, 
            "Get forward message distribution");
}

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
    bind_conjugate_base(m);
    
    // Bind exponential family distributions
    bind_mvn_exponential(m);
    bind_multinomial(m);
    bind_delta(m);
    
    // Bind conjugate priors
    bind_mvn_conjugate(m);
    bind_multinomial_conjugate(m);
}

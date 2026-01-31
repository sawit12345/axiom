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

// Model headers
#include "../models/smm.h"
#include "../models/rmm.h"
#include "../models/tmm.h"
#include "../models/imm.h"
#include "../vi/models/base.h"
#include "../vi/models/mixture.h"
#include "../vi/models/hybrid_mixture_model.h"

namespace py = pybind11;
using namespace axiomcuda;
using namespace axiomcuda::models;
using namespace axiomcuda::vi;

// SMM Config binding
void bind_smm_config(py::module_& m) {
    py::class_<SMMConfig>(m, "SMMConfig", R"doc(
        Configuration for Slot Mixture Model.
        
        Attributes:
            width: Image width (default: 160)
            height: Image height (default: 210)
            input_dim: Input feature dimension (default: 5)
            slot_dim: Latent slot dimension (default: 2)
            num_slots: Number of slots (default: 32)
            use_bias: Whether to use bias (default: True)
            ns_a: Noise scale for transform mean (default: 1.0)
            ns_b: Noise scale for bias (default: 1.0)
            dof_offset: Degrees of freedom offset (default: 10.0)
            mask_prob: Mask template probabilities (default: [0,0,0,0,1])
            scale: Scaling factors (default: [0.075, 0.075, 0.75, 0.75, 0.75])
            transform_inv_v_scale: Transform prior scale (default: 100.0)
            bias_inv_v_scale: Bias prior scale (default: 0.001)
            num_e_steps: Number of E-step iterations (default: 2)
            learning_rate: Learning rate (default: 1.0)
            beta: Momentum for updates (default: 0.0)
            eloglike_threshold: ELBO threshold for growing (default: 5.0)
            max_grow_steps: Maximum grow iterations (default: 20)
    )doc")
        .def(py::init<>())
        .def_readwrite("width", &SMMConfig::width)
        .def_readwrite("height", &SMMConfig::height)
        .def_readwrite("input_dim", &SMMConfig::inputDim)
        .def_readwrite("slot_dim", &SMMConfig::slotDim)
        .def_readwrite("num_slots", &SMMConfig::numSlots)
        .def_readwrite("use_bias", &SMMConfig::useBias)
        .def_readwrite("ns_a", &SMMConfig::nsA)
        .def_readwrite("ns_b", &SMMConfig::nsB)
        .def_readwrite("dof_offset", &SMMConfig::dofOffset)
        .def_readwrite("mask_prob", &SMMConfig::maskProb)
        .def_readwrite("scale", &SMMConfig::scale)
        .def_readwrite("transform_inv_v_scale", &SMMConfig::transformInvVScale)
        .def_readwrite("bias_inv_v_scale", &SMMConfig::biasInvVScale)
        .def_readwrite("num_e_steps", &SMMConfig::numESteps)
        .def_readwrite("learning_rate", &SMMConfig::learningRate)
        .def_readwrite("beta", &SMMConfig::beta)
        .def_readwrite("eloglike_threshold", &SMMConfig::eloglikeThreshold)
        .def_readwrite("max_grow_steps", &SMMConfig::maxGrowSteps);
}

// SMM binding
void bind_smm(py::module_& m) {
    // SMM class
    py::class_<SlotMixtureModel, std::shared_ptr<SlotMixtureModel>>(
        m, "SMM", R"doc(
            Slot Mixture Model (SMM) for image segmentation.
            
            The SMM uses slots to represent objects in an image. Each slot has
            a latent representation and a linear transformation that maps from
            latent space to pixel features (position + color).
            
            The model alternates between:
            1. E-step: Infer latent slots given pixel assignments
            2. M-step: Update transformations given inferred slots
            
            Examples:
                >>> from axiomcuda_backend import models
                >>> config = models.SMMConfig()
                >>> config.width = 160
                >>> config.height = 210
                >>> config.num_slots = 32
                >>> smm = models.SMM.create(config)
                >>> 
                >>> # Process an image (H, W, 3) -> (H*W, 5)
                >>> import numpy as np
                >>> img = np.random.rand(210, 160, 3)
                >>> smm.initialize_from_data(img)
                >>> qx, qz, used = smm.infer_and_update(img)
        )doc")
        .def_static("create", &createSMM,
                   py::arg("config"),
                   "Factory method to create SMM from config")
        .def("initialize_from_data", &SlotMixtureModel::initializeFromData,
            py::arg("inputs"),
            "Initialize slots from data (forces all data to first slot)")
        .def("infer_and_update", [](SlotMixtureModel& self,
                                    py::array_t<double> inputs,
                                    py::array_t<double> qx_prev,
                                    int num_slots,
                                    int num_e_steps,
                                    double eloglike_threshold,
                                    int max_grow_steps,
                                    double learning_rate,
                                    double beta) {
                // Convert numpy arrays to Tensors
                Tensor inputs_t = Tensor::from_numpy(inputs);
                Tensor qx_prev_mu = Tensor::from_numpy(qx_prev);
                
                // Run inference
                struct {
                    Tensor qx_mu;
                    Tensor qx_inv_sigma;
                    Tensor qz;
                    Tensor used_mask;
                    Tensor ell_max;
                } result;
                
                self.inferAndUpdate(inputs_t, qx_prev_mu,
                                   num_e_steps, eloglike_threshold,
                                   max_grow_steps, learning_rate, beta,
                                   result.qx_mu, result.qx_inv_sigma,
                                   result.qz, result.used_mask, result.ell_max);
                
                // Return as tuple
                return py::make_tuple(
                    result.qx_mu.to_numpy(),
                    result.qx_inv_sigma.to_numpy(),
                    result.qz.to_numpy(),
                    result.used_mask.to_numpy(),
                    result.ell_max.to_numpy()
                );
            },
            py::arg("inputs"),
            py::arg("qx_prev"),
            py::arg("num_slots"),
            py::arg("num_e_steps") = 2,
            py::arg("eloglike_threshold") = 5.0,
            py::arg("max_grow_steps") = 10,
            py::arg("learning_rate") = 1.0,
            py::arg("beta") = 0.0,
            "Run inference and update model")
        .def("e_step", [](SlotMixtureModel& self,
                         py::array_t<double> inputs,
                         py::array_t<double> qx_mu,
                         py::array_t<double> qx_inv_sigma,
                         int num_iterations) {
                Tensor inputs_t = Tensor::from_numpy(inputs);
                Tensor qx_mu_t = Tensor::from_numpy(qx_mu);
                Tensor qx_inv_sigma_t = Tensor::from_numpy(qx_inv_sigma);
                
                Tensor out_qx_mu, out_qx_inv_sigma, out_qz, out_ell;
                
                self.eStep(inputs_t, qx_mu_t, qx_inv_sigma_t, num_iterations,
                          out_qx_mu, out_qx_inv_sigma, out_qz, out_ell);
                
                return py::make_tuple(
                    out_qx_mu.to_numpy(),
                    out_qx_inv_sigma.to_numpy(),
                    out_qz.to_numpy(),
                    out_ell.to_numpy()
                );
            },
            py::arg("inputs"),
            py::arg("qx_mu"),
            py::arg("qx_inv_sigma"),
            py::arg("num_iterations") = 1,
            "Perform E-step inference")
        .def("m_step", [](SlotMixtureModel& self,
                         py::array_t<double> inputs,
                         py::array_t<double> qx_mu,
                         py::array_t<double> qx_inv_sigma,
                         py::array_t<double> qz,
                         double lr,
                         double beta,
                         py::object grow_mask) {
                Tensor inputs_t = Tensor::from_numpy(inputs);
                Tensor qx_mu_t = Tensor::from_numpy(qx_mu);
                Tensor qx_inv_sigma_t = Tensor::from_numpy(qx_inv_sigma);
                Tensor qz_t = Tensor::from_numpy(qz);
                
                Tensor* grow_mask_ptr = nullptr;
                Tensor grow_mask_t;
                if (!grow_mask.is_none()) {
                    grow_mask_t = Tensor::from_numpy(grow_mask.cast<py::array_t<double>>());
                    grow_mask_ptr = &grow_mask_t;
                }
                
                self.mStep(inputs_t, qx_mu_t, qx_inv_sigma_t, qz_t,
                          lr, beta, grow_mask_ptr);
            },
            py::arg("inputs"),
            py::arg("qx_mu"),
            py::arg("qx_inv_sigma"),
            py::arg("qz"),
            py::arg("lr") = 1.0,
            py::arg("beta") = 0.0,
            py::arg("grow_mask") = py::none(),
            "Perform M-step parameter update")
        .def("variational_forward", [](SlotMixtureModel& self,
                                      py::array_t<double> qx_mu,
                                      py::array_t<double> qx_inv_sigma) {
                Tensor qx_mu_t = Tensor::from_numpy(qx_mu);
                Tensor qx_inv_sigma_t = Tensor::from_numpy(qx_inv_sigma);
                
                Tensor out_py_mu, out_py_inv_sigma;
                self.variationalForward(qx_mu_t, qx_inv_sigma_t,
                                       out_py_mu, out_py_inv_sigma);
                
                return py::make_tuple(out_py_mu.to_numpy(),
                                    out_py_inv_sigma.to_numpy());
            },
            py::arg("qx_mu"),
            py::arg("qx_inv_sigma"),
            "Forward prediction from latent slots")
        .def_property_readonly("likelihood", &SlotMixtureModel::getLikelihood,
            "Linear transformation likelihood")
        .def_property_readonly("pi", &SlotMixtureModel::getPi,
            "Prior over slot assignments")
        .def_property_readonly("px", &SlotMixtureModel::getPx,
            "Prior over slot latents")
        .def_property_readonly("num_slots", &SlotMixtureModel::getNumSlots,
            "Number of slots")
        .def_property_readonly("slot_dim", &SlotMixtureModel::getSlotDim,
            "Dimension of each slot");
    
    // Utility functions
    m.def("format_observation", [](py::array_t<double> obs,
                                  std::vector<double> offset,
                                  std::vector<double> stdevs) {
            Tensor obs_t = Tensor::from_numpy(obs);
            Tensor out_t;
            formatObservation(obs_t, out_t, offset, stdevs);
            return out_t.to_numpy();
        },
        py::arg("obs"),
        py::arg("offset"),
        py::arg("stdevs"),
        "Format observation with position encoding");
    
    m.def("add_position_encoding", [](py::array_t<double> image,
                                     double width_scale,
                                     double height_scale) {
            Tensor img_t = Tensor::from_numpy(image);
            Tensor out_t;
            addPositionEncoding(img_t, out_t, width_scale, height_scale);
            return out_t.to_numpy();
        },
        py::arg("image"),
        py::arg("width_scale") = 1.0,
        py::arg("height_scale") = 1.0,
        "Add position encoding to image");
}

// RMM Config binding
void bind_rmm_config(py::module_& m) {
    py::class_<RMMConfig>(m, "RMMConfig", R"doc(
        Configuration for Relational Mixture Model.
        
        Attributes:
            num_components_per_switch: Components per TMM switch (default: 25)
            num_switches: Number of TMM switches (default: 100)
            num_object_types: Number of object identity types (default: 32)
            num_features: Number of features (default: 5)
            num_continuous_dims: Continuous dimensions (default: 7)
            interact_with_static: Allow interaction with static objects (default: False)
            r_ell_threshold: RMM log-likelihood threshold (default: -100)
            i_ell_threshold: IMM log-likelihood threshold (default: -500)
            cont_scale_identity: Scale for identity model (default: 0.5)
            cont_scale_switch: Scale for switch model (default: 25.0)
            discrete_alphas: Dirichlet prior for discrete features
            r_interacting: Interaction radius (default: 0.6)
            forward_predict: Predict future interactions (default: False)
            stable_r: Stable interaction radius mode (default: False)
            relative_distance: Use relative distances (default: True)
            reward_prob_threshold: Reward probability threshold (default: 0.45)
    )doc")
        .def(py::init<>())
        .def_readwrite("num_components_per_switch", &RMMConfig::numComponentsPerSwitch)
        .def_readwrite("num_switches", &RMMConfig::numSwitches)
        .def_readwrite("num_object_types", &RMMConfig::numObjectTypes)
        .def_readwrite("num_features", &RMMConfig::numFeatures)
        .def_readwrite("num_continuous_dims", &RMMConfig::numContinuousDims)
        .def_readwrite("interact_with_static", &RMMConfig::interactWithStatic)
        .def_readwrite("r_ell_threshold", &RMMConfig::rEllThreshold)
        .def_readwrite("i_ell_threshold", &RMMConfig::iEllThreshold)
        .def_readwrite("cont_scale_identity", &RMMConfig::contScaleIdentity)
        .def_readwrite("cont_scale_switch", &RMMConfig::contScaleSwitch)
        .def_readwrite("discrete_alphas", &RMMConfig::discreteAlphas)
        .def_readwrite("r_interacting", &RMMConfig::rInteracting)
        .def_readwrite("forward_predict", &RMMConfig::forwardPredict)
        .def_readwrite("stable_r", &RMMConfig::stableR)
        .def_readwrite("relative_distance", &RMMConfig::relativeDistance)
        .def_readwrite("reward_prob_threshold", &RMMConfig::rewardProbThreshold);
}

// RMM binding
void bind_rmm(py::module_& m) {
    py::class_<RelationalModel, std::shared_ptr<RelationalModel>>(
        m, "RMM", R"doc(
            Relational Mixture Model (RMM) for action prediction.
            
            The RMM models relationships between objects using:
            - Object identities (from IMM)
            - Continuous features (position, velocity)
            - Discrete features (object type, action, reward, TMM switch)
            
            It predicts the TMM switch and reward given object states.
            
            Examples:
                >>> from axiomcuda_backend import models
                >>> config = models.RMMConfig()
                >>> rmm = models.RMM.create(config, action_dim=6)
                >>> 
                >>> # Predict TMM switch and reward
                >>> tmm_switch, reward = rmm.predict(
                ...     continuous_obs, discrete_obs)
        )doc")
        .def_static("create", &createRMM,
                   py::arg("config"),
                   py::arg("action_dim"),
                   py::arg("reward_dim") = 3,
                   "Create RMM from configuration")
        .def("predict", [](RelationalModel& self,
                          py::array_t<double> c_sample,
                          std::vector<py::array_t<double>> d_sample,
                          py::object key,
                          double reward_prob_threshold) {
                Tensor c_t = Tensor::from_numpy(c_sample);
                std::vector<Tensor> d_t;
                for (auto& arr : d_sample) {
                    d_t.push_back(Tensor::from_numpy(arr));
                }
                
                Tensor out_switch, out_reward, out_ell, out_qz;
                int mix_slot;
                
                self.predict(c_t, d_t, nullptr,  // key not implemented
                           reward_prob_threshold,
                           out_switch, out_reward, out_ell, out_qz, mix_slot);
                
                return py::make_tuple(
                    out_switch.to_numpy(),
                    out_reward.to_numpy(),
                    out_ell.to_numpy(),
                    out_qz.to_numpy(),
                    mix_slot
                );
            },
            py::arg("c_sample"),
            py::arg("d_sample"),
            py::arg("key") = py::none(),
            py::arg("reward_prob_threshold") = 0.45,
            "Predict TMM switch and reward")
        .def("infer_and_update", [](RelationalModel& self,
                                   py::object imm,  // IMM object
                                   py::array_t<double> obs,
                                   int tmm_switch,
                                   int object_idx,
                                   py::object action,
                                   py::object reward,
                                   double r_ell_threshold,
                                   double i_ell_threshold) {
                Tensor obs_t = Tensor::from_numpy(obs);
                
                Tensor out_ell;
                self.inferAndUpdate(obs_t, tmm_switch, object_idx,
                                  r_ell_threshold, out_ell);
                
                return out_ell.to_numpy();
            },
            py::arg("imm"),
            py::arg("obs"),
            py::arg("tmm_switch"),
            py::arg("object_idx"),
            py::arg("action") = py::none(),
            py::arg("reward") = py::none(),
            py::arg("r_ell_threshold") = 1.0,
            py::arg("i_ell_threshold") = 1.0,
            "Infer and update RMM with new observation")
        .def_property_readonly("model", &RelationalModel::getModel,
            "Underlying hybrid mixture model")
        .def_property_readonly("used_mask", &RelationalModel::getUsedMask,
            "Mask of used components");
}

// TMM Config binding
void bind_tmm_config(py::module_& m) {
    py::class_<TMMConfig>(m, "TMMConfig", R"doc(
        Configuration for Transition Mixture Model.
        
        Attributes:
            n_total_components: Maximum number of dynamics components (default: 200)
            state_dim: State dimension (position coordinates) (default: 2)
            dt: Time step (default: 1.0)
            vu: Velocity uncertainty (default: 0.05)
            use_bias: Use bias in transitions (default: True)
            sigma_sqr: Gaussian likelihood variance (default: 2.0)
            logp_threshold: Log probability threshold (default: -1e-5)
            position_threshold: Position teleport threshold (default: 0.15)
            use_unused_counter: Track unused components (default: True)
            use_velocity: Use velocity in dynamics (default: True)
            clip_value: Minimum value for transitions (default: 5e-4)
    )doc")
        .def(py::init<>())
        .def_readwrite("n_total_components", &TMMConfig::nTotalComponents)
        .def_readwrite("state_dim", &TMMConfig::stateDim)
        .def_readwrite("dt", &TMMConfig::dt)
        .def_readwrite("vu", &TMMConfig::vu)
        .def_readwrite("use_bias", &TMMConfig::useBias)
        .def_readwrite("sigma_sqr", &TMMConfig::sigmaSqr)
        .def_readwrite("logp_threshold", &TMMConfig::logpThreshold)
        .def_readwrite("position_threshold", &TMMConfig::positionThreshold)
        .def_readwrite("use_unused_counter", &TMMConfig::useUnusedCounter)
        .def_readwrite("use_velocity", &TMMConfig::useVelocity)
        .def_readwrite("clip_value", &TMMConfig::clipValue);
}

// TMM binding
void bind_tmm(py::module_& m) {
    py::class_<TransitionMixtureModel, std::shared_ptr<TransitionMixtureModel>>(
        m, "TMM", R"doc(
            Transition Mixture Model (TMM) for dynamics.
            
            The TMM learns multiple linear dynamics components for object motion.
            It creates new components on-the-fly when observations don't match
            existing dynamics.
            
            Components include:
            - Default dynamics (constant velocity)
            - Keep unused (maintain unused state)
            - Become unused (transition to unused)
            - Custom velocity/bias components (learned from data)
            
            Examples:
                >>> from axiomcuda_backend import models
                >>> config = models.TMMConfig()
                >>> config.state_dim = 2  # x, y
                >>> tmm = models.TMM.create(config)
                >>> 
                >>> # Update with state transition
                >>> x_prev = np.array([0.0, 0.0, 1.0, 0.0])  # x, y, vx, vy
                >>> x_curr = np.array([1.0, 0.0, 1.0, 0.0])
                >>> tmm.update(x_prev, x_curr)
                >>> 
                >>> # Predict next state
                >>> x_next = tmm.forward(x_prev)
        )doc")
        .def_static("create", &createTMM,
                   py::arg("config"),
                   "Create TMM from configuration")
        .def("update", [](TransitionMixtureModel& self,
                         py::array_t<double> x_prev,
                         py::array_t<double> x_curr) {
                Tensor x_prev_t = Tensor::from_numpy(x_prev);
                Tensor x_curr_t = Tensor::from_numpy(x_curr);
                
                Tensor out_logprobs;
                self.update(x_prev_t, x_curr_t, out_logprobs);
                return out_logprobs.to_numpy();
            },
            py::arg("x_prev"),
            py::arg("x_curr"),
            "Update model with state transition")
        .def("forward", [](TransitionMixtureModel& self,
                          py::array_t<double> x) {
                Tensor x_t = Tensor::from_numpy(x);
                Tensor out_t = self.forward(x_t);
                return out_t.to_numpy();
            },
            py::arg("x"),
            "Predict next state given current state")
        .def("compute_logprobs", [](TransitionMixtureModel& self,
                                   py::array_t<double> x_prev,
                                   py::array_t<double> x_curr) {
                Tensor x_prev_t = Tensor::from_numpy(x_prev);
                Tensor x_curr_t = Tensor::from_numpy(x_curr);
                
                Tensor out_logprobs = self.computeLogProbs(x_prev_t, x_curr_t);
                return out_logprobs.to_numpy();
            },
            py::arg("x_prev"),
            py::arg("x_curr"),
            "Compute log probabilities for transition")
        .def_property_readonly("transitions", &TransitionMixtureModel::getTransitions,
            "Transition matrices (K, 2*state_dim, 2*state_dim+use_bias)")
        .def_property_readonly("used_mask", &TransitionMixtureModel::getUsedMask,
            "Boolean mask of used components")
        .def_property_readonly("num_components", &TransitionMixtureModel::getNumComponents,
            "Number of components in model")
        .def_property_readonly("state_dim", &TransitionMixtureModel::getStateDim,
            "State dimension");
}

// IMM Config binding
void bind_imm_config(py::module_& m) {
    py::class_<IMMConfig>(m, "IMMConfig", R"doc(
        Configuration for Identity Mixture Model.
        
        Attributes:
            num_object_types: Number of object identity classes (default: 32)
            num_features: Number of features for identity (default: 5)
            i_ell_threshold: Log-likelihood threshold for new class (default: -500)
            cont_scale_identity: Scale for continuous features (default: 0.5)
            color_precision_scale: Scale for color features (default: 1.0)
            color_only_identity: Use only color for identity (default: False)
    )doc")
        .def(py::init<>())
        .def_readwrite("num_object_types", &IMMConfig::numObjectTypes)
        .def_readwrite("num_features", &IMMConfig::numFeatures)
        .def_readwrite("i_ell_threshold", &IMMConfig::iEllThreshold)
        .def_readwrite("cont_scale_identity", &IMMConfig::contScaleIdentity)
        .def_readwrite("color_precision_scale", &IMMConfig::colorPrecisionScale)
        .def_readwrite("color_only_identity", &IMMConfig::colorOnlyIdentity);
}

// IMM binding
void bind_imm(py::module_& m) {
    py::class_<IdentityMixtureModel, std::shared_ptr<IdentityMixtureModel>>(
        m, "IMM", R"doc(
            Identity Mixture Model (IMM) for object classification.
            
            The IMM classifies objects based on their visual features.
            It uses a Gaussian mixture model over the feature space with
            a Normal-Inverse-Wishart conjugate prior.
            
            Examples:
                >>> from axiomcuda_backend import models
                >>> config = models.IMMConfig()
                >>> config.num_object_types = 32
                >>> imm = models.IMM.create(config)
                >>> 
                >>> # Infer identity from features
                >>> features = np.random.rand(5, 1)  # (features, 1)
                >>> class_label = imm.infer_identity(features)
                >>> 
                >>> # Update model
                >>> imm.infer_and_update(features, object_idx=0)
        )doc")
        .def_static("create", &createIMM,
                   py::arg("config"),
                   "Create IMM from configuration")
        .def("infer_identity", [](IdentityMixtureModel& self,
                                 py::array_t<double> x,
                                 bool color_only) {
                Tensor x_t = Tensor::from_numpy(x);
                return self.inferIdentity(x_t, color_only);
            },
            py::arg("x"),
            py::arg("color_only_identity") = false,
            "Infer object identity from features")
        .def("infer_and_update", [](IdentityMixtureModel& self,
                                   py::array_t<double> obs,
                                   int object_idx,
                                   double i_ell_threshold,
                                   bool color_only) {
                Tensor obs_t = Tensor::from_numpy(obs);
                self.inferAndUpdate(obs_t, object_idx, i_ell_threshold, color_only);
            },
            py::arg("obs"),
            py::arg("object_idx"),
            py::arg("i_ell_threshold") = 1.0,
            py::arg("color_only_identity") = false,
            "Infer identity and update model")
        .def_property_readonly("model", &IdentityMixtureModel::getModel,
            "Underlying mixture model")
        .def_property_readonly("used_mask", &IdentityMixtureModel::getUsedMask,
            "Mask of used identity classes");
}

// Mixture model bindings
void bind_mixture_models(py::module_& m) {
    py::class_<MixtureModel, std::shared_ptr<MixtureModel>>(
        m, "Mixture", R"doc(
            Generic mixture model with conjugate priors.
            
            Supports both continuous (Gaussian) and discrete (Multinomial)
            mixture components with conjugate priors.
        )doc")
        .def(py::init<int, int, const std::vector<int>&, double>(),
            py::arg("num_components"),
            py::arg("continuous_dim") = 0,
            py::arg("discrete_dims") = std::vector<int>{},
            py::arg("cont_scale") = 1.0,
            "Create mixture model")
        .def("e_step", &MixtureModel::eStep,
            py::arg("data"),
            "Perform E-step")
        .def("m_step", &MixtureModel::mStep,
            py::arg("qz"), py::arg("lr") = 1.0, py::arg("beta") = 0.0,
            "Perform M-step")
        .def_property_readonly("prior", &MixtureModel::getPrior,
            "Mixture prior (Dirichlet)")
        .def_property_readonly("continuous_likelihood", &MixtureModel::getContinuousLikelihood,
            "Continuous likelihood (NIW)")
        .def_property_readonly("discrete_likelihoods", &MixtureModel::getDiscreteLikelihoods,
            "Discrete likelihoods (Dirichlet-Multinomial)");
    
    py::class_<HybridMixture, std::shared_ptr<HybridMixture>>(
        m, "HybridMixture", R"doc(
            Hybrid mixture with continuous and discrete components.
            
            Used by RMM for modeling relationships with mixed feature types.
        )doc")
        .def(py::init<>())
        .def("e_step", &HybridMixture::eStep,
            py::arg("c_data"), py::arg("d_data"), py::arg("w_disc"),
            "E-step for hybrid data")
        .def("m_step", &HybridMixture::mStep,
            py::arg("c_data"), py::arg("d_data"), py::arg("qz"),
            py::arg("lr") = 1.0, py::arg("beta") = 0.0,
            "M-step for hybrid data")
        .def_property_readonly("continuous_likelihood", &HybridMixture::getContinuousLikelihood,
            "Shared continuous likelihood")
        .def_property_readonly("discrete_likelihoods", &HybridMixture::getDiscreteLikelihoods,
            "List of discrete likelihoods");
}

// Main model module initialization
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
    
    // Config classes
    bind_smm_config(m);
    bind_rmm_config(m);
    bind_tmm_config(m);
    bind_imm_config(m);
    
    // Model classes
    bind_smm(m);
    bind_rmm(m);
    bind_tmm(m);
    bind_imm(m);
    
    // Generic mixture models
    bind_mixture_models(m);
}

# Copyright 2025 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the "License");
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/axiom/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AxiomCUDA Models Module
=======================

GPU-accelerated model implementations wrapping C++ CUDA kernels.

This module provides CUDA-accelerated versions of the AXIOM models with the same
API as axiom.models. The models are designed to work with JAX's GPU backend and
can optionally call into C++ CUDA kernels when the `_axiomcuda` backend is available.

Models
------
- **SMM** (Slot Mixture Model): Object-centric slot attention model
- **RMM** (Reward Mixture Model): Reward and switch prediction model
- **TMM** (Transition Mixture Model): Object dynamics and transition model
- **IMM** (Identity Mixture Model): Object identity recognition model
- **HierarchSMM**: Hierarchical multi-layer SMM

Hybrid Models
-------------
- **Mixture**: Generic mixture model
- **HybridMixture**: Mixture with both continuous and discrete likelihoods

Configuration Classes
---------------------
- SMMConfig, RMMConfig, TMMConfig, IMMConfig

Usage
-----
>>> from axiomcuda.models import create_smm, SMMConfig
>>> import jax.random as jr
>>>
>>> key = jr.PRNGKey(0)
>>> config = SMMConfig(num_slots=32, slot_dim=2)
>>> smm = create_smm(key, **config.__dict__)

Device Management
-----------------
All models support explicit device placement via the `device` argument:
>>> from jax import devices
>>> gpu = devices('gpu')[0]
>>> smm = create_smm(key, device=gpu)

C++ Backend
-----------
When the `_axiomcuda` C++ extension is built and available, models will automatically
utilize CUDA kernels for compute-intensive operations. Otherwise, they fall back to
JAX implementations which still leverage GPU acceleration through JAX.

See Also
--------
- axiom.models: CPU/JAX reference implementations
- axiom.vi: Variational inference primitives
"""

# Model classes
from .smm import (
    SMM,
    SMMConfig,
    create_smm,
    initialize_smm_model,
    infer_and_update,
    add_position_encoding,
    format_single_frame,
    _compute_qx_given_qz,
    _compute_qz_given_qx,
    _update_optimized,
    create_e_step_fn,
    HierarchicalSMM,
    create_hierarch_smm,
    initialize_hierarch_smm,
    augment_state_with_velocity,
)

from .rmm import (
    RMM,
    RMMConfig,
    create_rmm,
    predict,
    forward_default,
    _in_ellipse,
    _object_interactions,
    get_interacting_objects_ellipse,
    get_interacting_objects_closest,
    _to_distance_obs_hybrid,
    mark_dirty,
    infer_and_update as rmm_infer_and_update,
    rollout_sample,
    rollout,
    _find_pairs,
    consider_merge,
    compute_elbo,
    run_bmr,
)

from .tmm import (
    TMM,
    TMMConfig,
    generate_default_dynamics_component,
    generate_default_become_unused_component,
    generate_default_keep_unused_component,
    generate_default_stop_component,
    create_velocity_component,
    create_bias_component,
    create_position_velocity_component,
    create_position_bias_component,
    add_component,
    forward,
    gaussian_loglike,
    compute_logprobs,
    add_vel_or_bias_component,
    update_transitions,
    create_tmm,
    update_model,
)

from .imm import (
    IMM,
    IMMConfig,
    infer_identity,
    create_imm,
    infer_remapped_color_identity,
    infer_and_update_identity,
)

from .utils_hybrid import (
    create_mm,
    train_step_fn,
    Mixture,
    HybridMixture,
)

from .base import (
    ModelConfig,
    ModelState,
    BaseModel,
    check_cuda_available,
    get_device,
    to_device,
)

__all__ = [
    # SMM and Hierarchical SMM
    "SMM",
    "SMMConfig",
    "create_smm",
    "initialize_smm_model",
    "infer_and_update",
    "add_position_encoding",
    "format_single_frame",
    "_compute_qx_given_qz",
    "_compute_qz_given_qx",
    "_update_optimized",
    "create_e_step_fn",
    "HierarchicalSMM",
    "create_hierarch_smm",
    "initialize_hierarch_smm",
    "augment_state_with_velocity",
    
    # RMM
    "RMM",
    "RMMConfig",
    "create_rmm",
    "predict",
    "forward_default",
    "_in_ellipse",
    "_object_interactions",
    "get_interacting_objects_ellipse",
    "get_interacting_objects_closest",
    "_to_distance_obs_hybrid",
    "mark_dirty",
    "rmm_infer_and_update",
    "rollout_sample",
    "rollout",
    "_find_pairs",
    "consider_merge",
    "compute_elbo",
    "run_bmr",
    
    # TMM
    "TMM",
    "TMMConfig",
    "generate_default_dynamics_component",
    "generate_default_become_unused_component",
    "generate_default_keep_unused_component",
    "generate_default_stop_component",
    "create_velocity_component",
    "create_bias_component",
    "create_position_velocity_component",
    "create_position_bias_component",
    "add_component",
    "forward",
    "gaussian_loglike",
    "compute_logprobs",
    "add_vel_or_bias_component",
    "update_transitions",
    "create_tmm",
    "update_model",
    
    # IMM
    "IMM",
    "IMMConfig",
    "infer_identity",
    "create_imm",
    "infer_remapped_color_identity",
    "infer_and_update_identity",
    
    # Hybrid utilities
    "create_mm",
    "train_step_fn",
    "Mixture",
    "HybridMixture",
    
    # Base classes
    "ModelConfig",
    "ModelState",
    "BaseModel",
    "check_cuda_available",
    "get_device",
    "to_device",
]

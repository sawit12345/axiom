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
AxiomCUDA - High-performance CUDA-accelerated tensor operations for Axiom

This package provides a CUDA-accelerated API layer that mirrors the original
Axiom API while providing GPU acceleration through custom CUDA kernels.
"""

__version__ = "0.1.0"

# Core imports
from .device import Device, get_device, set_default_device, cuda_available
from .tensor import Tensor
from .config import (
    PlannerConfig,
    SMMConfig,
    RMMConfig,
    TMMConfig,
    IMMConfig,
    ExperimentConfig,
)
from .utils import (
    tree_map,
    tree_flatten,
    tree_unflatten,
    tree_structure,
    tree_transpose,
    to_numpy,
    from_numpy,
    batch_apply,
    vmap,
)
from .random import (
    PRNGKey,
    split,
    normal,
    uniform,
    categorical,
    bernoulli,
    randint,
    seed,
)
from .nn import (
    softmax,
    log_softmax,
    one_hot,
    relu,
    gelu,
    sigmoid,
    tanh,
    silu,
)

# Submodules
from . import infer
from . import planner
from . import visualize
from . import models
from . import vi

__all__ = [
    # Version
    "__version__",
    # Device management
    "Device",
    "get_device",
    "set_default_device",
    "cuda_available",
    # Tensor
    "Tensor",
    # Config classes
    "PlannerConfig",
    "SMMConfig",
    "RMMConfig",
    "TMMConfig",
    "IMMConfig",
    "ExperimentConfig",
    # Tree operations
    "tree_map",
    "tree_flatten",
    "tree_unflatten",
    "tree_structure",
    "tree_transpose",
    # Array utilities
    "to_numpy",
    "from_numpy",
    "batch_apply",
    "vmap",
    # Random
    "PRNGKey",
    "split",
    "normal",
    "uniform",
    "categorical",
    "bernoulli",
    "randint",
    "seed",
    # Neural network
    "softmax",
    "log_softmax",
    "one_hot",
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "silu",
    # Submodules
    "infer",
    "planner",
    "visualize",
    "models",
    "vi",
]


def init(key, config, observation, action_dim):
    """Initialize the AxiomCUDA models and state.
    
    Args:
        key: PRNGKey for random initialization
        config: ExperimentConfig with model configurations
        observation: Initial observation
        action_dim: Dimension of action space
        
    Returns:
        Initial carry dictionary containing all model states
    """
    return infer.init(key, config, observation, action_dim)


def step_fn(carry, config, obs, reward, action, num_tracked=0, update=True, remap_color=False):
    """Execute a single step of inference and learning.
    
    Args:
        carry: Current state dictionary
        config: ExperimentConfig
        obs: Current observation
        reward: Reward signal
        action: Action taken
        num_tracked: Number of tracked objects
        update: Whether to update models
        remap_color: Whether to remap colors
        
    Returns:
        Tuple of (next_carry, records)
    """
    return infer.step_fn(carry, config, obs, reward, action, num_tracked, update, remap_color)


def plan_fn(key, carry, config, action_dim):
    """Plan the next action using the planner.
    
    Args:
        key: PRNGKey
        carry: Current state dictionary
        config: ExperimentConfig
        action_dim: Dimension of action space
        
    Returns:
        Tuple of (action, carry, info)
    """
    return infer.plan_fn(key, carry, config, action_dim)


def reduce_fn_rmm(key, rmm_model, cxm=None, dxm=None, n_samples=2000, n_pairs=2000):
    """Reduce the RMM model using Bayesian Model Reduction.
    
    Args:
        key: PRNGKey
        rmm_model: RMM model state
        cxm: Optional continuous samples
        dxm: Optional discrete samples
        n_samples: Number of samples
        n_pairs: Number of pairs
        
    Returns:
        Tuple of (rmm_model, merged_pairs, cxm, dxm)
    """
    return infer.reduce_fn_rmm(key, rmm_model, cxm, dxm, n_samples, n_pairs)

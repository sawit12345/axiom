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
Configuration classes for AxiomCUDA models.

These dataclasses define hyperparameters for all model components:
- PlannerConfig: MPPI planner configuration
- SMMConfig: Slot Mixture Model configuration
- TMMConfig: Transition Mixture Model configuration
- RMMConfig: Reward Mixture Model configuration
- IMMConfig: Identity Mixture Model configuration
- ExperimentConfig: Top-level experiment configuration
"""

from dataclasses import dataclass, field
from typing import Sequence, Optional


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for the MPPI (Model Predictive Path Integral) planner.
    
    Attributes:
        num_steps: Planning horizon (number of steps to plan ahead)
        num_policies: Number of action sequences to sample
        num_samples_per_policy: Number of samples per policy for MC estimation
        topk_ratio: Fraction of top samples to use for refitting distribution
        random_ratio: Fraction of samples to draw randomly (vs from distribution)
        alpha: Learning rate for distribution update (0-1)
        temperature: Temperature for softmax in policy selection
        normalize: Whether to normalize rewards
        iters: Number of optimization iterations per planning step
        gamma: Discount factor for rewards
        repeat_prob: Probability of repeating the previous action
        info_gain: Weight for information gain in objective
        lazy_reward: Whether to penalize non-lazy policies
        sample_action: Whether to sample or take argmax action
    """
    num_steps: int = 24
    num_policies: int = 512
    num_samples_per_policy: int = 1
    topk_ratio: float = 0.1
    random_ratio: float = 0.5
    alpha: float = 1.0
    temperature: float = 10.0
    normalize: bool = True
    iters: int = 1
    gamma: float = 0.99
    repeat_prob: float = 0.0
    info_gain: float = 1.0
    lazy_reward: bool = False
    sample_action: bool = False


@dataclass(frozen=True)
class SMMConfig:
    """Configuration for the Slot Mixture Model (SMM).
    
    SMM performs object discovery and segmentation by learning slot-based
    representations of the visual scene.
    
    Attributes:
        width: Width of input image
        height: Height of input image
        input_dim: Dimension of input feature vector (e.g., 5 for x, y, r, g, b)
        slot_dim: Dimension of each slot latent
        num_slots: Number of slots (objects) to discover
        use_bias: Whether to use bias in linear transformations
        ns_a: Noise parameter for transform mean
        ns_b: Noise parameter for bias
        dof_offset: Offset for degrees of freedom in prior
        mask_prob: Probability of using each linear template
        scale: Scaling factors for prior on bias elements
        transform_inv_v_scale: Scaling for inverse variance of transform
        bias_inv_v_scale: Scaling for inverse variance of bias
        num_e_steps: Number of E-step iterations
        learning_rate: Learning rate for model updates
        beta: Beta value for updates (momentum-like)
        eloglike_threshold: Expected log-likelihood threshold for model growth
        max_grow_steps: Maximum number of growth steps per update
    """
    width: int = 160
    height: int = 210
    input_dim: int = 5
    slot_dim: int = 2
    num_slots: int = 32
    use_bias: bool = True
    ns_a: float = 1.0
    ns_b: float = 1.0
    dof_offset: float = 10.0
    mask_prob: tuple[float] = field(
        default_factory=lambda: tuple([0.0, 0.0, 0.0, 0.0, 1.0])
    )
    scale: tuple[float] = field(
        default_factory=lambda: tuple([0.075, 0.075, 0.75, 0.75, 0.75])
    )
    transform_inv_v_scale: float = 100.0
    bias_inv_v_scale: float = 0.001
    num_e_steps: int = 2
    learning_rate: float = 1.0
    beta: float = 0.0
    eloglike_threshold: float = 5.0
    max_grow_steps: int = 20


@dataclass(frozen=True)
class TMMConfig:
    """Configuration for the Transition Mixture Model (TMM).
    
    TMM models object dynamics by learning transition matrices for different
    motion patterns (e.g., constant velocity, static, etc.).
    
    Attributes:
        n_total_components: Total number of dynamic components (switches)
        state_dim: Dimension of state space (position coordinates)
        dt: Time step for dynamics
        vu: Unused counter value scale
        use_bias: Whether to use bias in transition matrices
        sigma_sqr: Variance of Gaussian likelihood
        logp_threshold: Threshold for log probability
        position_threshold: Position change threshold for velocity clipping
        use_unused_counter: Whether to track unused counter
        use_velocity: Whether to model velocity
        clip_value: Threshold to clip small velocities to zero
    """
    n_total_components: int = 200
    state_dim: int = 2
    dt: float = 1.0
    vu: float = 0.05
    use_bias: bool = True
    sigma_sqr: float = 2.0
    logp_threshold: float = -0.00001
    position_threshold: float = 0.15
    use_unused_counter: bool = True
    use_velocity: bool = True
    clip_value: float = 5e-4


@dataclass(frozen=True)
class RMMConfig:
    """Configuration for the Reward Mixture Model (RMM).
    
    RMM learns reward functions based on object interactions and identities.
    
    Attributes:
        num_components_per_switch: Number of components per TMM switch
        num_switches: Number of TMM switches
        num_object_types: Number of distinct object types
        num_features: Number of features per object
        num_continuous_dims: Number of continuous dimensions
        interact_with_static: Whether to model interactions with static objects
        r_ell_threshold: Reward log-likelihood threshold
        i_ell_threshold: Identity log-likelihood threshold
        cont_scale_identity: Scale for identity continuous prior
        cont_scale_switch: Scale for switch continuous prior
        discrete_alphas: Dirichlet alphas for discrete dimensions
        r_interacting: Radius for object interaction
        r_interacting_predict: Radius for prediction-time interaction
        forward_predict: Whether to use forward prediction
        stable_r: Whether to use stable radius
        relative_distance: Whether to use relative distances
        absolute_distance_scale: Whether to use absolute distance scale
        reward_prob_threshold: Probability threshold for positive reward
        color_precision_scale: Scale for color precision
        color_only_identity: Whether to use color only for identity
        exclude_background: Whether to exclude background from modeling
        use_ellipses_for_interaction: Whether to use ellipses for interaction regions
        velocity_scale: Scale factor for velocity features
    """
    num_components_per_switch: int = 25
    num_switches: int = 100
    num_object_types: int = 32
    num_features: int = 5
    num_continuous_dims: int = 7
    interact_with_static: bool = False
    r_ell_threshold: float = -100.0
    i_ell_threshold: float = -500.0
    cont_scale_identity: float = 0.5
    cont_scale_switch: float = 25.0
    discrete_alphas: tuple[float] = field(
        default_factory=lambda: tuple([1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1e-4])
    )
    r_interacting: float = 0.6
    r_interacting_predict: float = 0.6
    forward_predict: bool = False
    stable_r: bool = False
    relative_distance: bool = True
    absolute_distance_scale: bool = False
    reward_prob_threshold: float = 0.45
    color_precision_scale: float = 1.0
    color_only_identity: bool = False
    exclude_background: bool = True
    use_ellipses_for_interaction: bool = True
    velocity_scale: float = 10.0


@dataclass(frozen=True)
class IMMConfig:
    """Configuration for the Identity Mixture Model (IMM).
    
    IMM learns to identify object types based on their visual features.
    
    Attributes:
        num_object_types: Number of distinct object types to identify
        num_features: Number of features per object
        i_ell_threshold: Identity log-likelihood threshold
        cont_scale_identity: Scale for continuous prior
        color_precision_scale: Scale for color precision
        color_only_identity: Whether to use color features only
    """
    num_object_types: int = 32
    num_features: int = 5
    i_ell_threshold: float = -500.0
    cont_scale_identity: float = 0.5
    color_precision_scale: float = 1.0
    color_only_identity: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration for an AxiomCUDA experiment.
    
    This combines all model configurations and experiment parameters.
    
    Attributes:
        name: Experiment name
        id: Unique run ID
        group: Experiment group name
        seed: Random seed
        game: Game/environment name
        num_steps: Number of training steps
        smm: SMM configuration (single or sequence for hierarchical)
        imm: IMM configuration
        tmm: TMM configuration
        rmm: RMM configuration
        planner: Planner configuration (None for random actions)
        moving_threshold: Velocity threshold for object tracking
        used_threshold: Usage threshold for object tracking
        min_track_steps: Minimum steps to track an object
        max_steps_tracked_unused: Max steps to track unused object
        prune_every: Steps between BMR pruning
        use_unused_counter: Whether to use unused counter
        project: Wandb project name
        precision_type: Numeric precision ('float32' or 'float64')
        layer_for_dynamics: Which SMM layer to use for dynamics
        warmup_smm: Whether to warmup SMM
        num_warmup_steps: Number of warmup steps
        velocity_clip_value: Velocity clipping threshold
        perturb: Perturbation type (None for no perturbation)
        perturb_step: Step to apply perturbation
        remap_color: Whether to remap colors on perturbation
        bmr_samples: Number of samples for BMR
        bmr_pairs: Number of pairs for BMR
    """
    name: str = "axiom"
    id: str = "default"
    group: str = "axiom"
    seed: int = 0
    game: str = "Explode"
    num_steps: int = 10000
    smm: SMMConfig | Sequence[SMMConfig] = field(
        default_factory=lambda: SMMConfig()
    )
    imm: IMMConfig = field(default_factory=IMMConfig)
    tmm: TMMConfig = field(default_factory=TMMConfig)
    rmm: RMMConfig = field(default_factory=RMMConfig)
    planner: Optional[PlannerConfig] = field(default_factory=PlannerConfig)
    moving_threshold: float | Sequence[float] = 1e-2
    used_threshold: float | Sequence[float] = 0.2
    min_track_steps: Sequence[int] = (1, 1)
    max_steps_tracked_unused: int = 10
    prune_every: int = 500
    use_unused_counter: bool = True
    project: str = "axiom"
    precision_type: str = "float32"
    layer_for_dynamics: int = 0
    warmup_smm: bool = False
    num_warmup_steps: int = 50
    velocity_clip_value: float = 7.5e-4
    perturb: Optional[str] = None
    perturb_step: int = 5000
    remap_color: bool = False
    bmr_samples: int = 2000
    bmr_pairs: int = 2000

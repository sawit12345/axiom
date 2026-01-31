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
Reward Mixture Model (RMM) - CUDA accelerated wrapper.

Wraps the C++ backend for GPU-accelerated reward and switch modeling.
"""

from dataclasses import dataclass, field
from typing import NamedTuple, List, Tuple, Optional

import numpy as np

# Use ONLY the C++ backend
try:
    import axiomcuda_backend as backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

from .base import ModelConfig


@dataclass(frozen=True)
class RMMConfig(ModelConfig):
    """
    Configuration for the RMM
    
    Attributes:
        num_components_per_switch: Number of mixture components per TMM switch.
        num_switches: Number of TMM switches.
        num_object_types: Number of object identity classes.
        num_features: Number of continuous features.
        num_continuous_dims: Number of continuous dimensions.
        interact_with_static: Whether to interact with static objects.
        r_ell_threshold: Threshold for reward model log-likelihood.
        i_ell_threshold: Threshold for identity model log-likelihood.
        cont_scale_identity: Scale for continuous identity features.
        cont_scale_switch: Scale for continuous switch features.
        discrete_alphas: Prior alphas for discrete features.
        r_interacting: Radius factor for ellipse interaction.
        r_interacting_predict: Radius factor for prediction interactions.
        forward_predict: Whether to forward predict interactions.
        stable_r: Use stable radius calculation.
        relative_distance: Use relative distance features.
        absolute_distance_scale: Use absolute distance scaling.
        reward_prob_threshold: Threshold for reward prediction.
        color_precision_scale: Scale for color feature precision.
        color_only_identity: Use only color for identity.
        exclude_background: Exclude background from interactions.
        use_ellipses_for_interaction: Use ellipses for interaction detection.
        velocity_scale: Scale factor for velocity features.
    """

    num_components_per_switch: int = 25
    num_switches: int = 100
    num_object_types: int = 32
    num_features: int = 5
    num_continuous_dims: int = 7
    interact_with_static: bool = False

    r_ell_threshold: float = -100
    i_ell_threshold: float = -500

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


class RMM(NamedTuple):
    """Reward Mixture Model container.
    
    Attributes:
        model: The underlying C++ RewardMixtureModel handle.
        used_mask: Mask indicating which components are used.
        dirty_mask: Mask indicating components needing updates.
        max_switches: Maximum number of TMM switches.
    """
    model: object  # C++ backend model handle
    used_mask: np.ndarray
    dirty_mask: np.ndarray
    max_switches: int


def _check_backend():
    """Check if C++ backend is available."""
    if not BACKEND_AVAILABLE:
        raise RuntimeError("C++ backend (axiomcuda_backend) is not available. Cannot create RMM model.")


def predict(
    rmm: RMM,
    c_sample: np.ndarray,
    d_sample: List[np.ndarray],
    key: Optional[np.ndarray] = None,
    reward_prob_threshold: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Predict TMM switch and reward given observations.
    
    Args:
        rmm: Reward Mixture Model.
        c_sample: Continuous features.
        d_sample: Discrete features.
        key: Random key for sampling (numpy random state or None).
        reward_prob_threshold: Threshold for reward prediction.
        
    Returns:
        Tuple of (tmm_slot, reward, elogp, qz, mix_slot).
    """
    _check_backend()
    
    # Overwrite the TMM switch and reward with uniform priors
    d_sample_modified = d_sample[:-2] + [
        np.ones_like(d_sample[-2]) * (1.0 / d_sample[-2].flatten().shape[0]),
        np.ones_like(d_sample[-1]) * (1.0 / d_sample[-1].flatten().shape[0]),
    ]
    
    # Weight vector for discrete features (mask out reward and tmm switch)
    w_disc = np.array([1.0] * len(d_sample_modified))
    w_disc[-2:] = 0.0
    
    # Convert to backend tensor format
    c_t = c_sample.astype(np.float64).reshape(1, -1, 1)
    d_t = [d.astype(np.float64).reshape(1, -1, 1) for d in d_sample_modified]
    
    # Call backend prediction
    out_tmm_slot = np.array([0], dtype=np.int32)
    out_reward = np.array([0.0], dtype=np.float64)
    out_elogp = np.zeros(rmm.used_mask.shape[0], dtype=np.float64)
    out_qz = np.zeros(rmm.used_mask.shape[0], dtype=np.float64)
    out_mix_slot = np.array([0], dtype=np.int32)
    
    rmm.model.predict(
        c_t, d_t, w_disc.tolist(),
        out_tmm_slot, out_reward, out_elogp, out_qz, out_mix_slot
    )
    
    # Mask out unused components in elogp
    out_elogp = out_elogp * rmm.used_mask + (1 - rmm.used_mask) * (-1e10)
    
    # Softmax to get qz
    exp_elogp = np.exp(out_elogp - out_elogp.max())
    qz = exp_elogp / exp_elogp.sum()
    
    # Get mix_slot
    if key is not None:
        # Sample from distribution
        mix_slot = np.random.choice(qz.shape[0], p=qz)
        tmm_slot = out_tmm_slot[0]
    else:
        mix_slot = qz.argmax()
        tmm_slot = out_tmm_slot[0]
    
    # Compute reward
    reward = out_reward[0]
    
    return (
        np.array([tmm_slot], dtype=np.int32),
        np.array([reward], dtype=np.float64),
        out_elogp,
        qz,
        int(mix_slot)
    )


def get_interacting_objects_ellipse(
    data: np.ndarray,
    tracked_obj_mask: np.ndarray,
    object_idx: int,
    r_interacting: float,
    forward_predict: bool,
    exclude_background: bool = True,
    interact_with_static: bool = True,
) -> Tuple[int, np.ndarray]:
    """Get interacting objects using ellipse intersection method.
    
    Args:
        data: Object states.
        tracked_obj_mask: Mask of tracked objects.
        object_idx: Index of object to check.
        r_interacting: Radius factor for ellipse.
        forward_predict: Whether to forward predict interactions.
        exclude_background: Whether to exclude background.
        interact_with_static: Whether to interact with static objects.
        
    Returns:
        Tuple of (other_idx, distances).
    """
    _check_backend()
    
    # Use C++ backend for ellipse interaction detection
    result = backend.models.ObjectInteraction()
    
    backend.models.detectEllipseInteractions(
        data.astype(np.float64),
        object_idx,
        r_interacting,
        forward_predict,
        exclude_background,
        interact_with_static,
        tracked_obj_mask.astype(np.float64),
        result
    )
    
    return result.other_idx, np.array([result.distance_x, result.distance_y])


def get_interacting_objects_closest(
    data: np.ndarray,
    tracked_obj_mask: np.ndarray,
    object_idx: int,
    r_interacting: float = 2.0,
    exclude_background: bool = True,
    interact_with_static: bool = True,
    absolute_distance_scale: bool = False,
) -> Tuple[int, np.ndarray]:
    """Get interacting objects using closest distance method.
    
    Args:
        data: Object states.
        tracked_obj_mask: Mask of tracked objects.
        object_idx: Index of object to check.
        r_interacting: Radius factor.
        exclude_background: Whether to exclude background.
        interact_with_static: Whether to interact with static objects.
        absolute_distance_scale: Use absolute distance scaling.
        
    Returns:
        Tuple of (other_idx, distances).
    """
    _check_backend()
    
    # Use C++ backend for closest interaction detection
    result = backend.models.ObjectInteraction()
    
    backend.models.detectClosestInteractions(
        data.astype(np.float64),
        object_idx,
        r_interacting,
        exclude_background,
        interact_with_static,
        absolute_distance_scale,
        tracked_obj_mask.astype(np.float64),
        result
    )
    
    return result.other_idx, np.array([result.distance_x, result.distance_y])


def _to_distance_obs_hybrid(
    imm,
    data: np.ndarray,
    object_idx: int,
    action: int,
    reward: int,
    tmm_switch: int,
    tracked_obj_mask: Optional[np.ndarray],
    interact_with_static: bool,
    max_switches: int,
    action_dim: int = 6,
    object_identities: Optional[np.ndarray] = None,
    num_object_classes: int = 32,
    reward_dim: int = 3,
    forward_predict: bool = False,
    r_interacting: float = 0.6,
    stable_r: bool = False,
    relative_distance: bool = False,
    color_only_identity: bool = False,
    exclude_background: bool = True,
    use_ellipses_for_interaction: bool = False,
    velocity_scale: float = 10.0,
    absolute_distance_scale: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Convert observation to hybrid model features.
    
    Args:
        imm: Identity Mixture Model (IMM instance).
        data: Object states.
        object_idx: Index of object.
        action: Action taken.
        reward: Reward received.
        tmm_switch: TMM switch value.
        tracked_obj_mask: Mask of tracked objects.
        interact_with_static: Whether to interact with static objects.
        max_switches: Maximum number of TMM switches.
        action_dim: Dimension of action space.
        object_identities: Object identity labels.
        num_object_classes: Number of object classes.
        reward_dim: Dimension of reward space.
        forward_predict: Whether to forward predict.
        r_interacting: Radius factor for interaction.
        stable_r: Use stable radius.
        relative_distance: Use relative distance.
        color_only_identity: Use only color for identity.
        exclude_background: Exclude background.
        use_ellipses_for_interaction: Use ellipses for interaction.
        velocity_scale: Scale for velocity features.
        absolute_distance_scale: Use absolute distance scaling.
        
    Returns:
        Tuple of (c_feat, d_feat) - continuous and discrete features.
    """
    _check_backend()
    
    if tracked_obj_mask is None:
        tracked_obj_mask = np.array([True] * data.shape[0])
    
    # Get interacting objects
    if use_ellipses_for_interaction:
        other_idx, distances = get_interacting_objects_ellipse(
            data,
            tracked_obj_mask,
            object_idx,
            r_interacting,
            forward_predict,
            exclude_background=exclude_background,
            interact_with_static=interact_with_static,
        )
    else:
        other_idx, distances = get_interacting_objects_closest(
            data,
            tracked_obj_mask,
            object_idx,
            r_interacting,
            exclude_background=exclude_background,
            interact_with_static=interact_with_static,
            absolute_distance_scale=absolute_distance_scale,
        )
    
    if stable_r:
        if tmm_switch == 0:
            other_idx = -1
    
    # Infer identities if not provided
    if object_identities is None:
        # Use IMM to infer identity
        from . import imm as imm_tools
        features = data[[object_idx, other_idx], -5:, None] if other_idx >= 0 else data[[object_idx], -5:, None]
        class_labels = imm_tools.infer_identity(imm, features, color_only_identity)
        
        self_id = np.zeros(num_object_classes + 1, dtype=np.float64)
        self_id[class_labels[0]] = 1.0
        
        if other_idx >= 0 and class_labels[0] != class_labels[1]:
            other_id = np.zeros(num_object_classes + 1, dtype=np.float64)
            other_id[class_labels[1]] = 1.0
        else:
            other_id = np.zeros(num_object_classes + 1, dtype=np.float64)
            other_id[num_object_classes] = 1.0  # Unknown/other class
    else:
        self_id = np.zeros(num_object_classes + 1, dtype=np.float64)
        self_id[object_identities[object_idx]] = 1.0
        
        other_id = np.zeros(num_object_classes + 1, dtype=np.float64)
        if other_idx >= 0:
            other_id[object_identities[other_idx]] = 1.0
        else:
            other_id[num_object_classes] = 1.0
    
    # Continuous features: (x, y, u, vx, vy) + relative distances
    c_feat = data[object_idx, :5].copy()
    if relative_distance and other_idx >= 0:
        d = distances
        c_feat = np.concatenate([c_feat, d])
    elif relative_distance:
        c_feat = np.concatenate([c_feat, np.array([1.2, 1.2])])
    
    # Scale velocity features
    c_feat[3:5] *= velocity_scale
    
    # Discrete features: (self_id, other_id, used/unused, action, reward, tmm_switch)
    d_feat = [
        self_id,
        other_id,
        np.array([1.0, 0.0] if data[object_idx, 2] == 0 else [0.0, 1.0]),  # used/unused
        np.array([1.0 if i == action else 0.0 for i in range(action_dim)]),  # action
        np.array([1.0 if i == (reward + 1) else 0.0 for i in range(reward_dim)]),  # reward
        np.array([1.0 if i == tmm_switch else 0.0 for i in range(max_switches)]),  # tmm_switch
    ]
    
    return c_feat, d_feat


def create_rmm(
    action_dim: int,
    num_components_per_switch: int,
    num_switches: int,
    num_object_types: int,
    num_continuous_dims: int = 5,
    reward_dim: int = 3,
    cont_scale_switch: float = 25.0,
    discrete_alphas: Optional[List[float]] = None,
    **kwargs,
) -> RMM:
    """Create a Reward Mixture Model using the C++ backend.
    
    Args:
        action_dim: Dimension of action space.
        num_components_per_switch: Components per TMM switch.
        num_switches: Number of TMM switches.
        num_object_types: Number of object types.
        num_continuous_dims: Number of continuous dimensions.
        reward_dim: Dimension of reward space.
        cont_scale_switch: Scale for continuous features.
        discrete_alphas: Prior alphas for discrete features.
        **kwargs: Additional arguments.
        
    Returns:
        RMM: Initialized Reward Mixture Model.
    """
    _check_backend()
    
    if discrete_alphas is None:
        discrete_alphas = [1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1e-4]
    
    # Create the C++ backend model
    model = backend.models.createRMM(
        action_dim,
        num_components_per_switch,
        num_switches,
        num_object_types,
        num_continuous_dims,
        reward_dim,
        cont_scale_switch,
        discrete_alphas
    )
    
    num_components = num_components_per_switch * num_switches
    r_used_mask = np.zeros(num_components, dtype=np.float64)
    dirty_mask = np.zeros_like(r_used_mask)
    
    return RMM(
        model=model,
        used_mask=r_used_mask,
        dirty_mask=dirty_mask,
        max_switches=num_switches,
    )


def mark_dirty(rmm: RMM, elogp: np.ndarray, dx: List[np.ndarray], r_ell_threshold: float) -> RMM:
    """Mark components as dirty if predictions are wrong.
    
    Args:
        rmm: Reward Mixture Model.
        elogp: Log probabilities.
        dx: Discrete observations.
        r_ell_threshold: Threshold for marking dirty.
        
    Returns:
        RMM with updated dirty mask.
    """
    _check_backend()
    
    # Softmax to get qz
    exp_elogp = np.exp(elogp - elogp.max())
    qz = exp_elogp / exp_elogp.sum()
    
    z = qz.argmax()
    
    # Check predictions
    tmm_predict = (rmm.model.state_.disc_alpha[-1].data[z].argmax() == dx[-1].argmax())
    reward_predict = (rmm.model.state_.disc_alpha[-2].data[z].argmax() == dx[-2].argmax())
    
    # Mark dirty if well explained but wrong prediction
    if elogp.max() < r_ell_threshold and (not tmm_predict or not reward_predict):
        dirty_mask = rmm.dirty_mask.copy()
        dirty_mask[z] += 1.0
        return RMM(
            model=rmm.model,
            used_mask=rmm.used_mask,
            dirty_mask=dirty_mask,
            max_switches=rmm.max_switches,
        )
    
    return rmm


def infer_and_update(
    rmm: RMM,
    imm,
    obs: np.ndarray,
    tmm_switch: int,
    object_idx: int,
    action: Optional[int] = None,
    reward: Optional[int] = None,
    r_ell_threshold: float = 1.0,
    i_ell_threshold: float = 1.0,
    tracked_obj_ids: Optional[np.ndarray] = None,
    interact_with_static: bool = True,
    num_switches: int = 100,
    num_features: int = 5,
    r_interacting: float = 0.6,
    forward_predict: bool = False,
    stable_r: bool = False,
    relative_distance: bool = False,
    color_only_identity: bool = False,
    exclude_background: bool = True,
    use_ellipses_for_interaction: bool = True,
    absolute_distance_scale: bool = False,
    **kwargs,
) -> Tuple[RMM, np.ndarray]:
    """Infer switch and update RMM.
    
    Args:
        rmm: Reward Mixture Model.
        imm: Identity Mixture Model.
        obs: Observations.
        tmm_switch: TMM switch value.
        object_idx: Object index.
        action: Action taken.
        reward: Reward received.
        r_ell_threshold: Threshold for reward model.
        i_ell_threshold: Threshold for identity model.
        tracked_obj_ids: Tracked object IDs.
        interact_with_static: Whether to interact with static objects.
        num_switches: Number of TMM switches.
        num_features: Number of features.
        r_interacting: Radius factor for interaction.
        forward_predict: Whether to forward predict.
        stable_r: Use stable radius.
        relative_distance: Use relative distance.
        color_only_identity: Use only color for identity.
        exclude_background: Exclude background.
        use_ellipses_for_interaction: Use ellipses for interaction.
        absolute_distance_scale: Use absolute distance scaling.
        
    Returns:
        Tuple of (updated_rmm, elogp).
    """
    _check_backend()
    
    # Convert observation to hybrid features
    cx, dx = _to_distance_obs_hybrid(
        imm,
        obs,
        object_idx,
        action if action is not None else 0,
        reward if reward is not None else 0,
        tmm_switch,
        tracked_obj_ids,
        interact_with_static,
        num_switches,
        action_dim=dx[-3].shape[0] if 'dx' in locals() else 6,
        object_identities=None,
        num_object_classes=rmm.model.state_.num_components // rmm.max_switches - 1,
        r_interacting=r_interacting,
        forward_predict=forward_predict,
        stable_r=stable_r,
        relative_distance=relative_distance,
        color_only_identity=color_only_identity,
        exclude_background=exclude_background,
        use_ellipses_for_interaction=use_ellipses_for_interaction,
        absolute_distance_scale=absolute_distance_scale,
    )
    
    # Format for backend
    cx = cx.astype(np.float64).reshape(1, -1, 1)
    dx_formatted = [d.astype(np.float64).reshape(1, -1, 1) for d in dx]
    
    # Training step
    qz = np.zeros(rmm.used_mask.shape[0], dtype=np.float64)
    grew_component = np.array([False])
    
    rmm.model.trainStep(cx, dx_formatted, r_ell_threshold, qz, grew_component)
    
    # Get ELBO/log probabilities
    c_ell = np.zeros(rmm.used_mask.shape[0], dtype=np.float64)
    d_ell = np.zeros(rmm.used_mask.shape[0], dtype=np.float64)
    
    w_disc = [1.0] * len(dx_formatted)
    rmm.model.eStep(cx, dx_formatted, w_disc, qz, c_ell, d_ell)
    
    elogp = c_ell + d_ell
    elogp = elogp * rmm.used_mask + (1 - rmm.used_mask) * (-1e10)
    
    # Update used mask
    used_mask = rmm.model.getUsedMask().data.astype(np.float64)
    
    return (
        RMM(
            model=rmm.model,
            used_mask=used_mask,
            dirty_mask=rmm.dirty_mask,
            max_switches=rmm.max_switches,
        ),
        elogp
    )


def run_bmr(rmm: RMM, n_samples: int, pairs: Optional[np.ndarray] = None) -> Tuple[RMM, np.ndarray, np.ndarray, np.ndarray]:
    """Run Bayesian Model Reduction.
    
    Args:
        rmm: Reward Mixture Model.
        n_samples: Number of samples.
        pairs: Pairs to consider for merging (optional).
        
    Returns:
        Tuple of (new_rmm, elbo_hist, num_components_hist, merged_pairs).
    """
    _check_backend()
    
    # Run BMR using C++ backend
    rmm.model.runBMR(n_samples, 0.0)  # elbo_threshold=0.0
    
    # Get updated state
    used_mask = rmm.model.getUsedMask().data.astype(np.float64)
    
    return (
        RMM(
            model=rmm.model,
            used_mask=used_mask,
            dirty_mask=rmm.dirty_mask,
            max_switches=rmm.max_switches,
        ),
        np.array([]),  # elbo_hist not directly available from C++
        np.array([]),  # num_components_hist not directly available
        pairs if pairs is not None else np.array([]),
    )

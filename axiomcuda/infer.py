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

"""Main inference engine for axiomcuda - uses C++ backend only."""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import asdict

import axiomcuda_backend as backend

from axiomcuda.models import (
    smm as smm_tools,
    hsmm as hsmm_tools,
    tmm as tmm_tools,
    rmm as rmm_tools,
    imm as imm_tools,
)

from axiomcuda import planner as mppi


def init(key, config, observation, action_dim):
    """Initialize all models using the C++ backend.
    
    Args:
        key: Random key (numpy array or int)
        config: Experiment configuration
        observation: Initial observation (numpy array)
        action_dim: Dimension of action space
        
    Returns:
        initial_carry: Dictionary containing all initialized models and state
    """
    # Convert key to backend format if needed
    if isinstance(key, int):
        key = np.array([key], dtype=np.int32)
    
    # SMM - use C++ backend
    key, subkey = backend.split_key(key)
    smm_model = hsmm_tools.create_hierarch_smm(subkey, layer_configs=config.smm)

    used, moving = [], []
    for l in range(len(config.smm)):
        # check which slots are actually used and contain moving objects
        used.append(np.zeros(config.smm[l].num_slots, dtype=np.float32))
        moving.append(np.zeros(config.smm[l].num_slots, dtype=np.float32))

    ## TMM - use C++ backend
    key, subkey = backend.split_key(key)
    tmm_model = tmm_tools.create_tmm(subkey, **asdict(config.tmm))

    ## IMM - use C++ backend
    key, subkey = backend.split_key(key)
    imm_model = imm_tools.create_imm(
        subkey,
        **asdict(config.imm),
    )

    ## RMM - use C++ backend
    key, subkey = backend.split_key(key)
    rmm_model = rmm_tools.create_rmm(
        subkey,
        action_dim=action_dim,
        **asdict(config.rmm),
    )

    # Initialize the model using the first observation
    obs = smm_tools.format_single_frame(
        observation,
        offset=smm_model.stats["offset"],
        stdevs=smm_model.stats["stdevs"],
    )
    smm_model, py, qx, qz, u, _ = hsmm_tools.initialize_hierarch_smm(
        subkey, smm_model, init_inputs=obs, layer_configs=config.smm
    )

    num_tracked_steps = []
    objects_mean, objects_sigma, used, x, moving = [], [], [], [], []
    for l in range(len(config.smm)):
        if l == 0:
            objects_mean_l, objects_sigma_l = (
                py[l].mean[0, 0, :, :, 0],  # shape (num_slots_layer_l, 5)
                py[l].sigma[0, 0],  # shape (num_slots_layer_l, 5, 5)
            )

        else:
            # compute assignment mask normalized over the input slots (over the N_tokens dimension)
            safe_denom = qz[l][0].sum(0, keepdims=True) + 1e-8
            qz_normalized_over_inputs = qz[l][0] / safe_denom
            # (num_slots_layer_l-1, 3, 1) , (num_slots_layer_l-1, 1, num_slots_layer_l) --> (3, num_slots_layer_l)
            macro_slot_average_color = (
                py[l - 1].mean[0, 0, :, 2:, :] * qz_normalized_over_inputs[:, None, :]
            ).sum(0)
            # (num_slots_layer_l-1, 3, 3) , (num_slots_layer_l-1, 1, 1, num_slots_layer_l) --> (3,3, num_slots_layer_l)
            macro_slot_average_color_sigma = (
                np.expand_dims(py[l - 1].sigma[0, 0, :, 2:, 2:], -1)
                * qz_normalized_over_inputs[:, None, None, :]
            ).sum(0)

            macro_slot_position = py[l].mean[0, 0, :, :2, 0]
            macro_slot_shape = py[l].sigma[0, 0, :, :2, :2]

            objects_mean_l = np.concatenate(
                [
                    macro_slot_position,
                    np.transpose(macro_slot_average_color, axes=(1, 0)),
                ],
                axis=-1,
            )

            objects_sigma_l = backend.block_diag_vmap(
                macro_slot_shape,
                np.transpose(macro_slot_average_color_sigma, axes=(2, 0, 1)),
            )
        objects_mean.append(objects_mean_l)
        objects_sigma.append(objects_sigma_l)
        used.append(0.01 * u[l])

        # Initialize x with zeros and fill in positions, first velocity will be 0
        x_l = np.zeros(
            (
                objects_mean_l.shape[0],
                2 * (2 + int(config.use_unused_counter)) + config.rmm.num_features,
            ),
            dtype=np.float32
        )
        x_l[:, :2] = objects_mean_l[:, :2]
        if config.use_unused_counter:
            # Counter is at 0 if used, positive otherwise
            x_l[:, 2] = (1 - u[l]) * config.tmm.vu

        # Let's assume this is 5 for now, sigma x, sigma y, r, g, b
        safe_shape_variances = objects_sigma_l[:, np.arange(2), np.arange(2)] + 1e-8
        shape = np.sqrt(safe_shape_variances) * 3
        color = objects_mean_l[:, 2:]
        x_l[:, -config.rmm.num_features : -config.rmm.num_features + 2] = shape
        x_l[:, -config.rmm.num_features + 2 :] = color
        x.append(x_l)
        # track moving average of the component's velocity when used
        abs_vels_l = np.sqrt(
            x_l[:, 2 + int(config.use_unused_counter)] ** 2
            + x_l[:, 3 + int(config.use_unused_counter)] ** 2
        )
        moving_l = 0.01 * abs_vels_l * u[l]
        moving.append(moving_l)

        num_tracked_steps.append(np.zeros(config.smm[l].num_slots, dtype=np.int32))

    tracked_obj_ids = [
        np.array([False] * config.smm[l].num_slots) for l in range(len(config.smm))
    ]
    
    # initial carry
    num_plan_steps = 1 if config.planner is None else config.planner.num_steps
    fg_mask = qz[0][0].sum(0)
    fg_mask = fg_mask < fg_mask.max()
    fg_mask = [fg_mask] + [
        np.ones(config.smm[l].num_slots, dtype=bool) for l in range(1, len(config.smm))
    ]
    
    initial_carry = {
        "smm_model": smm_model,
        "imm_model": imm_model,
        "tmm_model": tmm_model,
        "rmm_model": rmm_model,
        "qx": qx,
        "used": used,
        "moving": moving,
        "tracked_obj_ids": tracked_obj_ids,
        "num_tracked_steps": num_tracked_steps,
        "foreground_mask": fg_mask,
        "x": x,
        "object_colors": [objects_mean[l][:, 2:] for l in range(len(config.smm))],
        "key": key,
        "mppi_probs": np.full((num_plan_steps, action_dim), 1.0 / action_dim, dtype=np.float32),
        "current_plan": np.zeros(num_plan_steps, dtype=np.int32),
    }

    return initial_carry


def step_fn(
    carry, config, obs, reward, action, num_tracked=0, update=True, remap_color=False
):
    """Main inference step using C++ backend.
    
    Args:
        carry: Current state dictionary containing all models
        config: Experiment configuration
        obs: Current observation (numpy array)
        reward: Current reward signal
        action: Current action taken
        num_tracked: Number of tracked objects
        update: Whether to update models
        remap_color: Whether to remap colors
        
    Returns:
        next_carry: Updated state dictionary
        records: Dictionary of visualization/debugging records
    """
    # unpacking the carry dictionary
    smm_model = carry["smm_model"]
    imm_model = carry["imm_model"]
    tmm_model = carry["tmm_model"]
    rmm_model = carry["rmm_model"]
    qx = carry["qx"]
    used_prev = carry["used"]
    moving_prev = carry["moving"]
    num_tracked_steps_prev = carry["num_tracked_steps"]
    prev_x = carry["x"]
    key = carry["key"]

    # Update the SMM model using C++ backend
    obs = smm_tools.format_single_frame(
        obs,
        offset=smm_model.stats["offset"],
        stdevs=smm_model.stats["stdevs"],
    )

    key, subkey = backend.split_key(key)
    hsmm_mask = carry["foreground_mask"][0]

    if num_tracked == 0:
        slots_to_pass_up = [backend.where(hsmm_mask, size=hsmm_mask.shape[0] - 1)[0]]
    else:
        slots_to_pass_up = [backend.where(carry["tracked_obj_ids"][0], size=num_tracked)[0]]
    
    (
        smm_model,
        py,
        qx,
        qz_for_vis,
        used_smm,
        smm_eloglike_for_vis,
    ) = hsmm_tools.infer_and_update(
        subkey,
        smm_model,
        obs,
        prev_qx=qx,
        layer_configs=config.smm,
        slots_to_pass_up=slots_to_pass_up,
    )

    (
        objects_mean,
        objects_sigma,
        used,
        unused_mask,
        x,
        moving,
        tracked_obj_ids,
        num_tracked_steps,
    ) = ([], [], [], [], [], [], [], [])
    
    for l in range(len(config.smm)):
        if l == 0:
            objects_mean_l, objects_sigma_l = (
                py[l].mean[0, 0, :, :, 0],  # shape (num_slots_layer_l, 5)
                py[l].sigma[0, 0],  # shape (num_slots_layer_l, 5, 5)
            )

        else:
            # compute assignment mask normalized over the input slots (over the N_tokens dimension)
            safe_denom = qz_for_vis[l][0].sum(0, keepdims=True) + 1e-8
            # compute assignment mask normalized over the input slots (over the N_tokens dimension)
            qz_normalized_over_inputs = qz_for_vis[l][0] / safe_denom
            # (num_slots_layer_l-1, 3, 1) , (num_slots_layer_l-1, 1, num_slots_layer_l) --> (3, num_slots_layer_l)
            macro_slot_average_color = (
                py[l - 1].mean[0, 0, slots_to_pass_up[l - 1], 2:, :]
                * qz_normalized_over_inputs[:, None, :]
            ).sum(0)
            # (num_slots_layer_l-1, 3, 3) , (num_slots_layer_l-1, 1, 1, num_slots_layer_l) --> (3,3, num_slots_layer_l)
            macro_slot_average_color_sigma = (
                np.expand_dims(
                    py[l - 1].sigma[0, 0, slots_to_pass_up[l - 1], 2:, 2:], -1
                )
                * qz_normalized_over_inputs[:, None, None, :]
            ).sum(0)

            macro_slot_position = py[l].mean[0, 0, :, :2, 0]
            macro_slot_shape = py[l].sigma[0, 0, :, :2, :2]

            objects_mean_l = np.concatenate(
                [
                    macro_slot_position,
                    np.transpose(macro_slot_average_color, axes=(1, 0)),
                ],
                axis=-1,
            )

            objects_sigma_l = backend.block_diag_vmap(
                macro_slot_shape,
                np.transpose(macro_slot_average_color_sigma, axes=(2, 0, 1)),
            )

        objects_mean.append(objects_mean_l)
        objects_sigma.append(objects_sigma_l)

        used_l = 0.99 * used_prev[l] + 0.01 * used_smm[l]
        used.append(used_l)

        # mask needed to pass on prev_x later
        unused_mask_l = 1 - used_smm[l][:, None]
        unused_mask.append(unused_mask_l)

        prev_x_l = prev_x[l]

        x_l = np.zeros_like(prev_x_l)

        # Repeat previous value of positions if unused
        x_l[:, :2] = (
            unused_mask_l * prev_x_l[:, :2]
            + (1 - unused_mask_l) * objects_mean_l[:, :2]
        )

        if config.use_unused_counter:
            # Increment if used & reset to 0 if now used
            # Scale by vu to avoid becoming dominant in the TMM ELL term
            count = unused_mask_l[:, 0] * config.tmm.vu
            x_l[:, 2] = prev_x_l[:, 2] * unused_mask_l[:, 0] + count

        # append velocity to get (num_obj, 4) state representations
        velocity_l = (
            x_l[:, : 2 + int(config.use_unused_counter)]
            - prev_x_l[:, : 2 + int(config.use_unused_counter)]
        )

        velocity_l = np.where(
            np.abs(velocity_l) < config.tmm.clip_value, 0.0, velocity_l
        )

        x_l[
            :,
            2 + int(config.use_unused_counter) : x_l.shape[1] - config.rmm.num_features,
        ] = velocity_l

        safe_shape_variances = objects_sigma_l[:, np.arange(2), np.arange(2)] + 1e-8
        shape = np.sqrt(safe_shape_variances) * 3
        color = objects_mean_l[:, 2:]
        x_l[:, -config.rmm.num_features : -config.rmm.num_features + 2] = shape
        x_l[:, -config.rmm.num_features + 2 :] = color

        # Repeat previous value of shape and colors if unused
        x_l[:, -config.rmm.num_features :] = (
            unused_mask_l * prev_x_l[:, -config.rmm.num_features :]
            + (1 - unused_mask_l) * x_l[:, -config.rmm.num_features :]
        )

        # track moving average of the component's velocity when used
        abs_vels_l = np.sqrt(
            x_l[:, 2 + int(config.use_unused_counter)] ** 2
            + x_l[:, 3 + int(config.use_unused_counter)] ** 2
        )

        # in case of large velocities (i.e. teleportations),
        # set the velocity part to 0
        mask_l = (np.abs(abs_vels_l) < config.tmm.position_threshold)[:, None]
        zeros_for_vel = np.ones(
            2 * (2 + int(config.use_unused_counter)) + config.rmm.num_features
        )
        zeros_for_vel[
            2
            + int(config.use_unused_counter) : 2
            * (2 + int(config.use_unused_counter))
        ] = 0
        x_l = x_l * mask_l + x_l * (1 - mask_l) * zeros_for_vel
        
        if config.use_unused_counter:
            # We can't have negative velocities for unused
            x_l[:, 5] = 0

        x.append(x_l)

        moving_l = (
            0.99 * moving_prev[l] * used_smm[l]
            + (1 - used_smm[l]) * moving_prev[l]
            + 0.01 * abs_vels_l * used_smm[l]
        )

        moving.append(moving_l)

        # Determine if the object meets the instantaneous tracking criteria in this step
        meets_thresholds_l = (
            (moving_l > config.moving_threshold[l])
            & (used_l > config.used_threshold[l])
            & (
                x_l[:, 2] < config.max_steps_tracked_unused * config.tmm.vu
            )  # stop tracking if currently unused for more than max_steps_tracked_unused steps
        )

        # Update the consecutive tracked steps counter
        # Increment the counter if thresholds are met, otherwise reset to 0
        num_tracked_steps_l = (num_tracked_steps_prev[l] + 1) * meets_thresholds_l
        num_tracked_steps.append(num_tracked_steps_l)

        # Track object
        tracked_obj_ids_l = num_tracked_steps_l >= config.min_track_steps[l]
        if l == 0:
            # Don't track the bg slot even if it moves
            tracked_obj_ids_l[0] = False

        tracked_obj_ids.append(tracked_obj_ids_l)

    # compute a foreground mask, using the zero-th layer assignments
    fg_mask = qz_for_vis[0][0].sum(0)
    fg_mask = fg_mask < fg_mask.max()
    fg_mask = [fg_mask] + [
        np.ones(config.smm[l].num_slots, dtype=bool) for l in range(1, len(config.smm))
    ]

    # Update the TMM model using C++ backend
    dyn_layer_id = config.layer_for_dynamics

    def tmm_over_objects(carry, k):
        tmm_model = carry

        def _no_op(_model):
            return _model, np.full(config.tmm.n_total_components, 0.0, dtype=np.float32)

        def _update_with_k(_model):
            return tmm_tools.update_model(
                _model,
                prev_x[dyn_layer_id][
                    k, : prev_x[dyn_layer_id].shape[1] - config.rmm.num_features
                ],
                x[dyn_layer_id][
                    k, : x[dyn_layer_id].shape[1] - config.rmm.num_features
                ],
                **asdict(config.tmm),
            )

        tmm_model_updated, logprobs_k = backend.cond(
            update & tracked_obj_ids[dyn_layer_id][k], _update_with_k, _no_op, tmm_model
        )

        return tmm_model_updated, logprobs_k

    # Scan over objects using C++ backend scan
    tmm_model, logprobs = backend.scan(
        tmm_over_objects, tmm_model, np.arange(config.smm[dyn_layer_id].num_slots, dtype=np.int32)
    )
    switches = np.argmax(logprobs, axis=-1)

    # Update the rMM and iMM model using C++ backend
    if remap_color:
        # we are triggered that colors might be remapped ... try to reassign object identities based
        # on other features, e.g. shape
        def _infer_remapped_color_id(rmm_model, k):
            identity = imm_tools.infer_remapped_color_identity(
                imm_model, obs=x[dyn_layer_id], object_idx=k, **asdict(config.rmm)
            )
            return imm_model, identity

        imm_model, identities = backend.scan(
            _infer_remapped_color_id,
            imm_model,
            np.arange(config.smm[dyn_layer_id].num_slots, dtype=np.int32),
        )

        mask = (x[dyn_layer_id][:, 2] == 0) & fg_mask[dyn_layer_id][:]
        identities = identities * mask[:, None]

        def do_remap(rmm_model):
            # TODO update model, remapping old slots on the new data
            model_updated = backend.tree_map(lambda x: x, imm_model.model)

            # first reset the slots to remap to prior params
            slots_to_wipe = identities.sum(axis=0)
            model_updated.prior.posterior_params = backend.tree_map(
                lambda post, pri: pri * slots_to_wipe + post * (1 - slots_to_wipe),
                model_updated.prior.posterior_params,
                model_updated.prior.prior_params,
            )

            # and then update i_model
            object_features = x[dyn_layer_id][
                :, None, x[dyn_layer_id].shape[-1] - config.rmm.num_features :, None
            ]
            object_features[:, :, 2:, :] = object_features[:, :, 2:, :] * 100

            def remap_model(model, k):
                def _update_model(model, k):
                    model._m_step_keep_unused(
                        object_features[k], [], identities[k][None, :]
                    )
                    return model, None

                def _do_nothing(model, k):
                    return model, None

                model = backend.cond(mask[k], _update_model, _do_nothing, model, k)
                return model

            # and update rmm_model
            model_updated, _ = backend.scan(
                remap_model,
                model_updated,
                np.arange(config.smm[dyn_layer_id].num_slots, dtype=np.int32),
            )
            imm_model = backend.tree_at(lambda x: x.model, imm_model, model_updated)
            return rmm_model

        def do_nothing(imm_model):
            return imm_model

        # if we can remap all objects, we do it, else we wait
        # TODO we have trouble inferring the correct slot if we already remapped few objects previously
        # in practice this waits until all objects are back in view
        imm_model = backend.cond(
            (imm_model.model.prior.alpha > 0.1).sum()
            == ((identities > 0.1).sum(axis=1) > 0.1).sum(),
            do_remap,
            do_nothing,
            imm_model,
        )
    else:

        def update_i_model(imm_model, k):
            def _update_model(imm):
                imm = imm_tools.infer_and_update_identity(
                    imm,
                    obs=prev_x[dyn_layer_id],
                    object_idx=k,
                    **asdict(config.imm),
                )
                return imm

            is_visible = (prev_x[dyn_layer_id][k][2] == 0) & fg_mask[dyn_layer_id][k]

            if not update:
                is_visible = np.zeros_like(is_visible, dtype=bool)

            imm_model_updated = backend.cond(
                is_visible,  # is_visible if config.rmm.interact_with_static else is_tracked,
                _update_model,
                lambda x: x,
                imm_model,
            )

            return imm_model_updated, None

        # Fit the identity model on background components first
        imm_model, _ = backend.scan(
            update_i_model, imm_model, np.arange(config.smm[dyn_layer_id].num_slots, dtype=np.int32)
        )

    def rmm_over_objects(carry, k):
        key, rmm_model = carry
        key, subkey = backend.split_key(key)

        def _no_op(model):
            return model, 0

        def _update_model(model):
            model, logprobs_k_rmm = rmm_tools.rmm_infer_and_update(
                subkey,
                model,
                imm_model,
                obs=prev_x[dyn_layer_id],
                tmm_switch=switches[k],
                object_idx=k,
                tracked_obj_ids=tracked_obj_ids[dyn_layer_id],
                action=action,
                reward=reward,
                **asdict(config.rmm),
            )
            return model, np.argmax(logprobs_k_rmm, axis=-1)[0]

        rmm_model_updated, rmm_slot = backend.cond(
            update & tracked_obj_ids[dyn_layer_id][k], _update_model, _no_op, rmm_model
        )

        return (key, rmm_model_updated), rmm_slot

    key, subkey = backend.split_key(key)
    (_, rmm_model), rmm_switches = backend.scan(
        rmm_over_objects,
        (subkey, rmm_model),
        np.arange(config.smm[dyn_layer_id].num_slots, dtype=np.int32),
    )

    # store some records
    records = {}
    records["decoded_mu"] = objects_mean
    records["decoded_sigma"] = objects_sigma
    records["qz"] = qz_for_vis
    records["smm_eloglike"] = smm_eloglike_for_vis
    records["switches"] = switches
    records["rmm_switches"] = rmm_switches
    records["tracked_obj_ids"] = tracked_obj_ids
    records["x"] = x
    records["moving"] = moving

    next_carry = {
        "smm_model": smm_model,
        "tmm_model": tmm_model,
        "imm_model": imm_model,
        "rmm_model": rmm_model,
        "qx": qx,
        "used": used,
        "moving": moving,
        "tracked_obj_ids": tracked_obj_ids,
        "num_tracked_steps": num_tracked_steps,
        "foreground_mask": fg_mask,
        "x": x,
        "object_colors": [objects_mean[l][:, 2:] for l in range(len(config.smm))],
        "key": key,
        "mppi_probs": carry["mppi_probs"],
        "current_plan": carry["current_plan"],
    }

    return next_carry, records


def plan_fn(key, carry, config, action_dim):
    """Planning wrapper using C++ backend.
    
    Args:
        key: Random key (numpy array or int)
        carry: Current state dictionary
        config: Experiment configuration
        action_dim: Dimension of action space
        
    Returns:
        action: Selected action
        carry: Updated state dictionary
        info: Planning information
    """
    imm_model = carry["imm_model"]
    tmm_model = carry["tmm_model"]
    rmm_model = carry["rmm_model"]

    probs = carry["mppi_probs"]
    current_plan = carry["current_plan"]

    x = carry["x"][config.layer_for_dynamics]
    tracked_obj_ids = carry["tracked_obj_ids"][config.layer_for_dynamics]

    def rollout_fn(k, x_input, actions):

        def rollout_action_seq(actions, key):
            object_identities = imm_tools.infer_identity(
                imm_model,
                x_input[..., None],
                config.imm.color_only_identity,
            )

            return rmm_tools.rollout(
                rmm_model,
                imm_model,
                tmm_model,
                x_input,
                actions,
                tracked_obj_ids,
                key=key,
                **asdict(config.rmm),
                object_identities=object_identities,
            )

        # planner gives actions as [num_steps, num_samples, action_dim]
        # whereas we vmap over num_samples
        actions = np.transpose(actions, (1, 0, 2))

        # pred_xs should be shape (num_policies, num_timesteps, num_slots, data_dim)
        split_keys = backend.split_key(k, actions.shape[0])
        pred_xs, pred_switches, pred_rmm_switches, info_gains, pred_rewards = backend.vmap(
            rollout_action_seq
        )(actions[:, :, 0], split_keys)

        # Planner expects predictions transposed appropriately.
        pred_xs = np.transpose(pred_xs, (1, 0, 2, 3))
        pred_switches = np.transpose(pred_switches, (1, 0, 2))
        pred_rmm_switches = np.transpose(pred_rmm_switches, (1, 0, 2))
        pred_rewards = np.transpose(pred_rewards, (1, 0, 2))
        info_gains = np.transpose(info_gains, (1, 0, 2))

        return pred_xs, pred_switches, pred_rmm_switches, pred_rewards, info_gains

    action, info = mppi.plan(
        x,
        rollout_fn,
        action_dim=action_dim,
        key=key,
        probs=probs,
        current_plan=current_plan,
        **asdict(config.planner),
    )

    carry["mppi_probs"] = info["probs"]
    carry["current_plan"] = info["current_plan"]
    return action, carry, info


def reduce_fn_rmm(key, rmm_model, cxm=None, dxm=None, n_samples=2000, n_pairs=2000):
    """BMR (Bayesian Model Reduction) for RMM using C++ backend.
    
    Reduces the number of components in the RMM model by merging similar components.
    
    Args:
        key: Random key (numpy array)
        rmm_model: Current RMM model
        cxm: Continuous samples (optional)
        dxm: Discrete samples (optional)
        n_samples: Number of samples to draw
        n_pairs: Number of pairs to consider for merging
        
    Returns:
        rmm_model: Updated RMM model after reduction
        merged_pairs: Information about merged pairs
        cxm: Continuous samples
        dxm: Discrete samples
    """
    key, subkey = backend.split_key(key)
    cxm_new, dxm_new = rmm_model.model.sample(subkey, n_samples)
    if cxm is None:
        cxm, dxm = cxm_new, dxm_new
    else:
        # Basically also keep optimizing for the old data points that were sampled
        cxm = np.concatenate([cxm, cxm_new], axis=0)
        dxm = backend.tree_map(
            lambda d1, d2: np.concatenate([d1, d2], axis=0), dxm, dxm_new
        )

    key, subkey = backend.split_key(key)
    rmm_model, _, _, merged_pairs = rmm_tools.run_bmr(
        subkey, rmm_model, n_pairs, cxm=cxm, dxm=dxm
    )
    return rmm_model, merged_pairs, cxm, dxm

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

"""Main entry point for axiomcuda - CUDA-accelerated version of Axiom.

This module provides a drop-in replacement for the original main.py that uses
the C++/CUDA backend (axiomcuda) instead of the JAX-based implementation (axiom).
It maintains full compatibility with the original API while providing GPU acceleration
through custom CUDA kernels.

Based on original work from VERSES AI, Inc. (https://github.com/VersesTech/axiom)
"""

import sys
import rich

import mediapy
import csv

from tqdm import tqdm

import wandb
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from axiomcuda import defaults

import gameworld.envs  # Triggers registration of the environments in Gymnasium
import gymnasium

from axiomcuda import infer as ax
from axiomcuda import visualize as vis


def main(config):
    """Main training loop for axiomcuda.
    
    This function runs the complete training and evaluation loop for an agent
    using the axiomcuda package. It handles:
    - Environment setup and interaction
    - Model initialization and updates
    - Planning and action selection
    - BMR (Bayesian Model Reduction) for model compression
    - Logging to CSV, video, and wandb
    
    Args:
        config: ExperimentConfig with all model and training configurations
    """
    # Initialize random seeds
    key = jr.PRNGKey(config.seed)
    np.random.seed(config.seed)

    # Set precision for numerical stability
    if config.precision_type == "float64":
        jax.config.update("jax_enable_x64", True)

    # Create environment
    env = gymnasium.make(
        f'Gameworld-{config.game}-v0', 
        perturb=config.perturb, 
        perturb_step=config.perturb_step
    )

    # Storage for tracking metrics
    observations = []
    rewards = []
    expected_utility = []
    expected_info_gain = []
    num_components = []

    # Reset environment
    obs, _ = env.reset()
    obs = obs.astype(np.uint8)
    observations.append(obs)
    reward = 0

    # Initialize models
    key, subkey = jr.split(key)
    carry = ax.init(subkey, config, obs, env.action_space.n)

    # Buffer for BMR (Bayesian Model Reduction)
    bmr_buffer = None, None

    # Main training loop
    for t in tqdm(range(config.num_steps)):
        # Action selection using planning
        key, subkey = jr.split(key)
        action, carry, plan_info = ax.plan_fn(subkey, carry, config, env.action_space.n)

        # Track planning metrics
        best = jnp.argsort(plan_info["rewards"][:, :, 0].sum(0))[-1]
        expected_utility.append(
            plan_info["expected_utility"][:, best, :].mean(-1).sum(0)
        )
        expected_info_gain.append(
            plan_info["expected_info_gain"][:, best, :].mean(-1).sum(0)
        )
        num_components.append(carry["rmm_model"].used_mask.sum())

        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        obs = obs.astype(np.uint8)
        observations.append(obs)
        rewards.append(reward)

        # Handle color remapping for perturbations
        update = True
        remap_color = False
        if (
            config.remap_color
            and config.perturb is not None
            and t + 1 >= config.perturb_step
            and t < config.perturb_step + 20
        ):
            update = False
            remap_color = True

        # Update models with new observation
        carry, rec = ax.step_fn(
            carry,
            config,
            obs,
            jnp.array(reward),
            action,
            num_tracked=0,
            update=update,
            remap_color=remap_color,
        )

        # Log observations
        observations.append(obs)

        # Handle episode reset
        if done:
            obs, _ = env.reset()
            obs = obs.astype(np.uint8)
            observations.append(obs)
            reward = 0

            # Update models without learning on reset
            carry, rec = ax.step_fn(
                carry,
                config,
                obs,
                jnp.array(reward),
                jnp.array(0),
                num_tracked=0,
                update=False,
            )

        # Bayesian Model Reduction (BMR) - merge similar components
        if (t + 1) % config.prune_every == 0:
            key, subkey = jr.split(key)
            new_rmm, pairs, *bmr_buffer = ax.reduce_fn_rmm(
                subkey,
                carry["rmm_model"],
                *bmr_buffer,
                n_samples=config.bmr_samples,
                n_pairs=config.bmr_pairs,
            )
            carry["rmm_model"] = new_rmm

    # Write results to CSV file
    with open(f"{config.game.lower()}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Step",
                "Reward",
                "Average Reward",
                "Cumulative Reward",
                "Expected Utility",
                "Expected Info Gain",
                "Num Components",
            ]
        )
        for i in range(len(rewards)):
            writer.writerow(
                [
                    i,
                    rewards[i],
                    jnp.mean(jnp.array(rewards[max(0, i - 1000) : max(i, 1)])),
                    sum(jnp.array(rewards[max(0, i - 1000) : i])),
                    expected_utility[i],
                    expected_info_gain[i],
                    num_components[i],
                ]
            )

    # Save gameplay video
    with mediapy.set_show_save_dir("."):
        mediapy.show_videos({f"{config.game.lower()}": observations}, fps=30)

    # Initialize wandb for logging
    wandb.init(
        reinit=True,
        group=config.group,
        project=config.project,
        config=config,
        resume="allow",
        id=config.id + "-" + config.game,
        name=config.name + "-" + config.game,
    )

    # Log metrics to wandb
    for i in range(len(rewards)):
        wandb.log(
            {
                "reward": rewards[i],
                "reward_1k_avg": jnp.mean(
                    jnp.array(rewards[max(0, i - 1000) : max(i, 1)])
                ),
                "cumulative_reward": sum(jnp.array(rewards[max(0, i - 1000) : i])),
                "expected_utility": expected_utility[i],
                "expected_info_gain": expected_info_gain[i],
                "num_components": num_components[i],
            }
        )

    # Log final visualizations
    logs = {
        "play": wandb.Video(
            np.asarray(observations)[-1000:].transpose(0, 3, 1, 2),
            fps=30,
            format="mp4",
        ),
        "rmm": wandb.Image(
            vis.plot_rmm(carry["rmm_model"], carry["imm_model"], colorize="cluster")
        ),
        "plan": wandb.Image(
            vis.plot_plan(
                observations[-2],
                plan_info,
                carry["tracked_obj_ids"][config.layer_for_dynamics],
                carry["smm_model"].stats,
                topk=1,
            )
        ),
        "identities": wandb.Image(vis.plot_identity_model(carry["imm_model"])),
    }
    
    # Log perturbation video if applicable
    if config.perturb is not None:
        logs["perturb"] = wandb.Video(
            np.asarray(observations)[
                config.perturb_step - 100 : config.perturb_step + 100
            ].transpose(0, 3, 1, 2),
            fps=30,
            format="mp4",
        )
    
    wandb.log(logs)


if __name__ == "__main__":
    config = defaults.parse_args(sys.argv[1:])
    rich.print(config)
    main(config)

# Copyright 2025 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
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

import defaults

import gameworld.envs # Triggers registration of the environments in Gymnasium
import gymnasium

from axiom import infer as ax
from axiom import visualize as vis


def _compute_reward_stats(rewards, window=1000):
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.size == 0:
        return rewards, rewards

    cumulative = np.concatenate(([0.0], np.cumsum(rewards)))
    end_idxs = np.arange(1, rewards.size + 1)
    start_idxs = np.maximum(0, end_idxs - window)

    reward_sums = cumulative[end_idxs] - cumulative[start_idxs]
    reward_avgs = reward_sums / (end_idxs - start_idxs)
    return reward_avgs, reward_sums


def main(config):
    key = jr.PRNGKey(config.seed)
    np.random.seed(config.seed)

    if config.precision_type == "float64":
        jax.config.update("jax_enable_x64", True)

    # Create environment.
    env = gymnasium.make(f'Gameworld-{config.game}-v0', perturb=config.perturb, perturb_step=config.perturb_step)

    observations = []
    rewards = []
    expected_utility = []
    expected_info_gain = []
    num_components = []
    plan_info = None

    # reset
    obs, _ = env.reset()
    obs = obs.astype(np.uint8)
    observations.append(obs)
    reward = 0

    # initialize
    key, subkey = jr.split(key)
    carry = ax.init(subkey, config, obs, env.action_space.n)

    bmr_buffer = None, None

    # main loop
    for t in tqdm(range(config.num_steps)):
        # action selection
        key, subkey = jr.split(key)
        if config.planner is None:
            action = jr.randint(subkey, shape=(), minval=0, maxval=env.action_space.n)
            expected_utility.append(0.0)
            expected_info_gain.append(0.0)
        else:
            action, carry, plan_info = ax.plan_fn(subkey, carry, config, env.action_space.n)

            best = jnp.argmax(plan_info["rewards"][:, :, 0].sum(0))
            expected_utility.append(
                float(
                    jax.device_get(
                        plan_info["expected_utility"][:, best, :].mean(-1).sum(0)
                    )
                )
            )
            expected_info_gain.append(
                float(
                    jax.device_get(
                        plan_info["expected_info_gain"][:, best, :].mean(-1).sum(0)
                    )
                )
            )

        num_components.append(int(jax.device_get(carry["rmm_model"].used_mask.sum())))

        action_int = int(jax.device_get(action))
        action_arr = jnp.asarray(action_int, dtype=jnp.int32)

        # step env
        obs, reward, done, truncated, info = env.step(action_int)
        obs = obs.astype(np.uint8)
        observations.append(obs)
        rewards.append(reward)

        # wandb.log({"reward": reward})

        # update models
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

        carry, rec = ax.step_fn(
            carry,
            config,
            obs,
            jnp.array(reward),
            action_arr,
            num_tracked=0,
            update=update,
            remap_color=remap_color,
        )

        if done:
            obs, _ = env.reset()
            obs = obs.astype(np.uint8)
            observations.append(obs)
            reward = 0

            carry, rec = ax.step_fn(
                carry,
                config,
                obs,
                jnp.array(reward),
                jnp.array(0, dtype=jnp.int32),
                num_tracked=0,
                update=False,
            )

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

    rewards_np = np.asarray(rewards, dtype=np.float64)
    reward_1k_avg, reward_1k_sum = _compute_reward_stats(rewards_np, window=1000)
    expected_utility_np = np.asarray(expected_utility, dtype=np.float64)
    expected_info_gain_np = np.asarray(expected_info_gain, dtype=np.float64)
    num_components_np = np.asarray(num_components, dtype=np.int32)

    # Write results to file: a csv file iwth the rewards adn a video of the gameplay
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
        for i in range(len(rewards_np)):
            writer.writerow(
                [
                    i,
                    rewards_np[i],
                    reward_1k_avg[i],
                    reward_1k_sum[i],
                    expected_utility_np[i],
                    expected_info_gain_np[i],
                    num_components_np[i],
                ]
            )

    observations_np = np.asarray(observations)
    mediapy.write_video(f"{config.game.lower()}.mp4", observations_np, fps=30)

    # Do wandb logging after the job to avoid performance impact
    wandb.init(
        reinit=True,
        group=config.group,
        project=config.project,
        config=config,
        resume="allow",
        id=config.id + "-" + config.game,
        name=config.name + "-" + config.game,
    )

    for i in range(len(rewards_np)):
        wandb.log(
            {
                "reward": rewards_np[i],
                "reward_1k_avg": reward_1k_avg[i],
                "cumulative_reward": reward_1k_sum[i],
                "expected_utility": expected_utility_np[i],
                "expected_info_gain": expected_info_gain_np[i],
                "num_components": num_components_np[i],
            }
        )

    # finally log a sample of final gameplay
    logs = {
        "play": wandb.Video(
            observations_np[-1000:].transpose(0, 3, 1, 2),
            fps=30,
            format="mp4",
        ),
        "rmm": wandb.Image(
            vis.plot_rmm(carry["rmm_model"], carry["imm_model"], colorize="cluster")
        ),
        "identities": wandb.Image(vis.plot_identity_model(carry["imm_model"])),
    }

    if plan_info is not None and len(observations) >= 2:
        logs["plan"] = wandb.Image(
            vis.plot_plan(
                observations[-2],
                plan_info,
                carry["tracked_obj_ids"][config.layer_for_dynamics],
                carry["smm_model"].stats,
                topk=1,
            )
        )

    if config.perturb is not None:
        start = max(0, config.perturb_step - 100)
        end = min(observations_np.shape[0], config.perturb_step + 100)
        logs["perturb"] = wandb.Video(
            observations_np[start:end].transpose(0, 3, 1, 2),
            fps=30,
            format="mp4",
        )
    wandb.log(logs)


if __name__ == "__main__":
    config = defaults.parse_args(sys.argv[1:])
    rich.print(config)
    main(config)

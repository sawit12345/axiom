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

"""MPPI (Model Predictive Path Integral) planning for axiomcuda - C++ backend only."""

import numpy as np
from typing import Callable
from dataclasses import dataclass

import axiomcuda_backend as backend


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for MPPI planner.
    
    Attributes:
        num_steps: Planning horizon (number of timesteps)
        num_policies: Number of policy samples to evaluate
        num_samples_per_policy: Number of samples per policy
        topk_ratio: Fraction of top samples to use for refitting
        random_ratio: Fraction of samples to draw randomly
        alpha: Mixing coefficient for policy update
        temperature: Temperature for softmax weighting
        normalize: Whether to normalize rewards
        iters: Number of optimization iterations
        gamma: Discount factor for rewards
        repeat_prob: Probability of repeating actions
        info_gain: Weight for information gain in objective
        lazy_reward: Whether to penalize active policies
        sample_action: Whether to sample action or take argmax
    """
    num_steps: int = 24
    num_policies: int = 512
    num_samples_per_policy: int = 1
    topk_ratio: float = 0.1  # fraction of the top samples to calc probs
    random_ratio: float = 0.5  # fraction of the samples to draw randomly
    alpha: float = 1.0
    temperature: float = 10.0
    normalize: bool = True
    iters: int = 1
    gamma: float = 0.99
    repeat_prob: float = 0.0
    info_gain: float = 1.0
    lazy_reward: bool = False
    sample_action: bool = False


def _smooth_actions(key, num_steps, num_samples, action_dim, bias_prob=0.6):
    """Generate smoothed action sequences using C++ backend.
    
    Creates action sequences where consecutive actions have a tendency to repeat,
    creating smoother policy samples.
    
    Args:
        key: Random key (numpy array)
        num_steps: Length of action sequence
        num_samples: Number of sequences to generate
        action_dim: Number of possible actions
        bias_prob: Probability of keeping previous action
        
    Returns:
        Action sequences of shape (num_steps, num_samples)
    """
    # Initialize the action sequence array
    key, _key = backend.split_key(key)
    action_sequences = backend.randint(
        _key, shape=(num_steps, num_samples), minval=0, maxval=action_dim
    )
    key, _key = backend.split_key(key)
    rv = backend.beta(_key, 1.0, 1.0, shape=(num_steps, num_samples))
    idx = rv < bias_prob

    def step_fn(carry, xs):
        actions = carry
        idx, rand_actions = xs
        new_actions = np.where(idx, actions, rand_actions)

        return new_actions, actions

    _, action_sequences = backend.scan(
        step_fn, action_sequences[0], (idx, action_sequences)
    )
    return action_sequences


def plan(
    state,
    rollout_fn: Callable,
    action_dim: int,
    key,
    probs: np.ndarray = None,
    current_plan: np.ndarray = None,
    num_steps: int = 10,
    num_policies: int = 128,
    num_samples_per_policy: int = 5,
    topk_ratio: float = 0.1,
    random_ratio: float = 0.5,
    alpha: float = 1.0,
    temperature: float = 10,
    normalize: bool = True,
    iters: int = 1,
    gamma: float = 0.99,
    repeat_prob: float = 0.0,
    info_gain: float = 0.0,
    lazy_reward: bool = False,
    sample_action: bool = False,
    **kwargs,
):
    """Main MPPI planning loop using C++ backend.
    
    Args:
        state: Current state of the system
        rollout_fn: Function to rollout actions and return predictions
        action_dim: Number of possible actions
        key: Random key (numpy array)
        probs: Current policy probabilities (optional)
        current_plan: Current best plan (optional)
        num_steps: Planning horizon
        num_policies: Number of policy samples
        num_samples_per_policy: Samples per policy
        topk_ratio: Fraction of top samples to keep
        random_ratio: Fraction of random samples
        alpha: Policy update mixing coefficient
        temperature: Softmax temperature
        normalize: Whether to normalize rewards
        iters: Optimization iterations
        gamma: Reward discount factor
        repeat_prob: Action repetition probability
        info_gain: Information gain weight
        lazy_reward: Whether to use lazy reward
        sample_action: Whether to sample final action
        
    Returns:
        action: Selected action
        info: Dictionary with planning information
    """
    topk = int(num_policies * topk_ratio)
    num_random_samples = int(num_policies * random_ratio)

    if probs is None:
        probs = np.full((num_steps, action_dim), 1.0 / action_dim, dtype=np.float32)
    else:
        probs = np.roll(probs, -1, axis=0)
        probs[-1, :] = 1 / probs.shape[-1]

    if current_plan is not None:
        current_plan = np.roll(current_plan, -1, axis=0)

    for i in range(iters):
        # sample from the given probs
        key, subkey = backend.split_key(key)
        actions = backend.categorical(
            subkey,
            np.expand_dims(np.log(probs + 1e-16), -2),
            shape=(num_steps, num_policies),
        )

        # overwrite final x actions with random smoothed actions
        if num_random_samples > 0:
            key, subkey = backend.split_key(key)
            random_actions = _smooth_actions(
                subkey, num_steps, num_random_samples, action_dim, repeat_prob
            )
            actions[:, -num_random_samples:] = random_actions

        # if we have a current plan, keep that in the sample set always
        if current_plan is not None:
            actions[:, 0] = current_plan

        # force constant action policies in first k items after current plan
        for k in range(action_dim):
            actions[:, k + 1] = k

        # draw a number of samples for each policy
        repeated_actions = np.repeat(actions, num_samples_per_policy, axis=1)

        key, subkey = backend.split_key(key)
        states, switches, rmm_switches, expected_utility, expected_info_gain = (
            rollout_fn(subkey, state, repeated_actions[..., None])
        )

        # average utility/info gain over policy samples
        expected_utility = expected_utility.reshape(
            num_steps, num_policies, num_samples_per_policy, 1
        ).mean(axis=2)
        expected_info_gain = expected_info_gain.reshape(
            num_steps, num_policies, num_samples_per_policy, 1
        ).mean(axis=2)

        # reshape to have num_steps, num_policies, num_samples_per_policy
        states = states.reshape(
            num_steps,
            num_policies,
            num_samples_per_policy,
            states.shape[-2],
            states.shape[-1],
        )
        switches = switches.reshape(
            num_steps, num_policies, num_samples_per_policy, switches.shape[-1]
        )
        rmm_switches = rmm_switches.reshape(
            num_steps, num_policies, num_samples_per_policy, rmm_switches.shape[-1]
        )

        rewards = expected_utility + info_gain * expected_info_gain

        if lazy_reward:
            # prefer lazy policies with action sequences that are biased towards noop at end
            lazy_rewards = (-np.arange(rewards.shape[1]) * 1e-2)[None, :]
            lazy_rewards * (actions > 0)
            rewards += lazy_rewards

        probs, _ = _refit(
            probs, actions, rewards, topk, alpha, temperature, gamma, normalize
        )

    idx = np.argsort(rewards.sum(axis=0)[:, 0])[-1]
    new_plan = actions[:, idx]

    if not sample_action:
        action = new_plan[0]
    else:
        # sample from the probs or rather pick the top rewarding action sequence?
        key, subkey = backend.split_key(key)
        action = backend.choice(subkey, probs.shape[-1], p=probs[0])

    return action, {
        "states": states,
        "switches": switches,
        "rmm_switches": rmm_switches,
        "rewards": rewards,
        "actions": actions,
        "probs": probs,
        "current_plan": new_plan,
        "expected_utility": expected_utility,
        "expected_info_gain": expected_info_gain,
    }


def _refit(
    probs, actions, rewards, topk, alpha, temperature=10, gamma=0.99, normalize=True
):
    """Refit policy based on top-performing samples.
    
    Args:
        probs: Current policy probabilities
        actions: Sampled actions
        rewards: Obtained rewards
        topk: Number of top samples to use
        alpha: Mixing coefficient
        temperature: Softmax temperature
        gamma: Discount factor
        normalize: Whether to normalize rewards
        
    Returns:
        probs_updated: Updated policy probabilities
        r_top: Rewards of top samples
    """
    T, A = probs.shape

    num_steps = rewards.shape[0]
    discounts = gamma ** np.arange(num_steps)
    discounted_rewards = rewards * np.expand_dims(discounts, (-1, -2))
    cum_rewards = discounted_rewards.sum(0).squeeze(-1)[None]

    if normalize:
        cum_rewards = min_max_normalization(cum_rewards)

    topk_indices = np.argsort(-cum_rewards, axis=-1)[..., :topk]
    a_top = np.take_along_axis(actions, topk_indices, axis=-1)
    r_top = np.take_along_axis(cum_rewards, topk_indices, axis=-1)

    # match moment
    w = backend.softmax(temperature * r_top, axis=-1)[..., None]
    one_hot_actions = backend.one_hot(a_top, A)
    counts = np.sum(w * one_hot_actions, axis=1)
    probs_new = counts / counts.sum(axis=-1, keepdims=True)
    probs_updated = alpha * probs_new + (1 - alpha) * probs
    return probs_updated, r_top


def min_max_normalization(v: np.ndarray, dim: int = -1):
    """Min-max normalize array.
    
    Args:
        v: Input array
        dim: Dimension to normalize over
        
    Returns:
        Normalized array
    """
    v_max = v.max(dim, keepdims=True)
    v_min = v.min(dim, keepdims=True)

    v_norm = np.where(v_max != v_min, (v - v_min) / (v_max - v_min + 1e-16), v)
    return v_norm

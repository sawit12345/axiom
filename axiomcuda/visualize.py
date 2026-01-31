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

"""Visualization utilities for axiomcuda - using numpy only, C++ backend for models."""

import io

import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import norm

from axiomcuda.models import rmm as rmm_tools
from axiomcuda.models import imm as imm_tools


def fig2img(fig):
    """Convert matplotlib figure to numpy array image.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Numpy array of shape (height, width, 3)
    """
    with io.BytesIO() as buff:
        fig.savefig(buff, facecolor="white", format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close(fig)
    return im[:, :, :3]


def draw_ellipse(position, covariance, ax=None, nsigs=None, **kwargs):
    """Draw an ellipse with a given position and covariance.
    
    Args:
        position: Center position (2,)
        covariance: Covariance matrix (2, 2) or variances (2,)
        ax: Matplotlib axis (optional)
        nsigs: List of sigma levels to draw (default: [1, 2, 3])
        **kwargs: Additional arguments for Ellipse patch
    """
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    kwargs["angle"] = angle
    kwargs["edgecolor"] = "black"
    kwargs["facecolor"] = kwargs.get("color", None)
    if "color" in kwargs:
        del kwargs["color"]
    if nsigs is None:
        nsigs = list(range(1, 4))

    for nsig in nsigs:
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, **kwargs))


def draw_ellipses(means, covars, ax=None, nsigs=None, zorder=1, scatter=True, **kwargs):
    """Draw multiple ellipses with given means and covariances.
    
    Args:
        means: Array of positions (N, 2)
        covars: Array of covariances (N, 2, 2)
        ax: Matplotlib axis (optional)
        nsigs: List of sigma levels
        zorder: Z-order for drawing
        scatter: Whether to scatter means
        **kwargs: Additional arguments
    """
    ax = ax or plt.gca()
    nsig = nsigs or range(1, 4)

    # Compute SVD for all covariances using numpy
    U_list, s_list, Vt_list = [], [], []
    for cov in covars:
        U, s, Vt = np.linalg.svd(cov)
        U_list.append(U)
        s_list.append(s)
        Vt_list.append(Vt)
    
    U = np.array(U_list)
    s = np.array(s_list)
    
    vals = 2 * np.sqrt(s)
    widths, heights = vals[:, 0], vals[:, 1]
    angles = np.array([np.degrees(np.arctan2(u[1, 0], u[0, 0])) for u in U])

    colors = kwargs.get("colors", [None] * means.shape[0])

    if kwargs.get("edgecolors", None) is not None:
        edgecolors = kwargs["edgecolors"]
        del kwargs["edgecolors"]
    else:
        edgecolors = [kwargs.get("edgecolor", "black")] * means.shape[0]

    if kwargs.get("colors", None) is not None:
        del kwargs["colors"]

    for color, angle, position, width, height, edgecolor in zip(
        colors, angles, means, widths, heights, edgecolors
    ):
        kwargs["edgecolor"] = edgecolor

        if color is not None:
            kwargs["facecolor"] = color

        kwargs["angle"] = angle
        for nsig in nsigs:
            ax.add_patch(
                Ellipse(position, nsig * width, nsig * height, **kwargs, zorder=zorder)
            )

    if scatter:
        ax.scatter(
            means[:, 0],
            means[:, 1],
            color=colors,
            marker=".",
            alpha=kwargs.get("alpha", 1.0),
        )


def add_colorbar(fig, ax, plot_im):
    """Add colorbar to plot.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axis
        plot_im: Image to add colorbar for
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(plot_im, cax=cax, orientation="vertical")


def transform_mvn(scale, offset, mean, cova):
    """Transform multivariate normal parameters.
    
    Args:
        scale: Scale factors
        offset: Offset values
        mean: Mean vector
        cova: Covariance matrix
        
    Returns:
        (new_mean, new_cova): Transformed parameters
    """
    A = np.diag(scale)
    new_mean = A.dot(mean) + offset
    new_cova = np.dot(A, np.dot(cova, A.T))
    return new_mean, new_cova


def plot_rmm(
    rmm: rmm_tools.RMM,
    imm: imm_tools.IMM = None,
    width=160,
    height=210,
    colorize="switch",
    indices=None,
    return_ax=False,
    highlight_idcs=None,
    fig_ax=None,
    scatter=True,
    color_only_identity=False,
):
    """Plot RMM (Reward Mixture Model) state.
    
    Visualizes the components of the RMM as ellipses colored by different attributes.
    
    Args:
        rmm: RMM model
        imm: IMM model (optional, for coloring by identity)
        width: Plot width
        height: Plot height
        colorize: How to color components ('switch', 'reward', 'cluster', 'infogain')
        indices: Which indices to plot (default: used components)
        return_ax: Whether to return axis objects
        highlight_idcs: Indices to highlight
        fig_ax: Pre-existing figure and axis tuple
        scatter: Whether to scatter means
        color_only_identity: Whether to use only color for identity
        
    Returns:
        Image array or (fig, ax) tuple if return_ax=True
    """
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=((8.4 / height) * width, 8.4))
    else:
        fig, ax = fig_ax
    ax.set_xlim([-1, 1])
    ax.set_ylim([1, -1])
    ax.set_xticks([])
    ax.set_yticks([])

    cmap = plt.get_cmap("viridis")
    tab20 = plt.get_cmap("tab20")

    if indices is None:
        indices = rmm.used_mask > 0

    mu = rmm.model.continuous_likelihood.mean[indices, :2, 0]
    si = rmm.model.continuous_likelihood.expected_sigma()[indices, :2, :2]

    edgecolors = None
    if colorize == "switch":
        sw = rmm.model.discrete_likelihoods[4].mean()[indices, ..., 0]
        colors = np.argmax(sw, axis=-1)
        colors = colors / colors.max() if colors.max() > 0 else colors
        colors = tab20(colors)
    elif colorize == "reward":
        rw = rmm.model.discrete_likelihoods[3].mean()[indices, ..., 0]
        rw = np.argmax(rw, axis=-1) / rw.shape[-1] if rw.shape[-1] > 0 else rw
        c = {
            -1: (1, 0, 0),  # Red
            0: (1, 1, 0),  # Yellow
            1: (0, 1, 0),  # Green
        }
        cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", [c[-1], c[0], c[1]])
        colors = cmap_custom(rw)
    elif colorize == "cluster":
        identities = np.argmax(
            rmm.model.discrete_likelihoods[0].mean()[indices, ..., 0], axis=-1
        )

        interact_identities = np.argmax(
            rmm.model.discrete_likelihoods[1].mean()[indices, ..., 0], axis=-1
        )
        
        if imm is not None:
            colors = (
                (
                    imm.model.continuous_likelihood.mean[
                        identities, (2 - 2 * int(color_only_identity)) :, 0
                    ]
                    / 100
                )
                * 0.5
                + 0.5
            )
            colors = np.clip(colors, 0, 1).tolist()

            edgecolors = (
                (
                    imm.model.continuous_likelihood.mean[
                        interact_identities, (2 - 2 * int(color_only_identity)) :, 0
                    ]
                    / 100
                )
                * 0.5
                + 0.5
            )
            edgecolors = np.clip(edgecolors, 0, 1).tolist()
        else:
            colors = tab20(identities % 20)
            edgecolors = tab20(interact_identities % 20)
    elif colorize == "infogain":
        ig = 1 / rmm.model.prior.alpha - 1 / rmm.model.prior.alpha.sum()
        ig = ig[indices]
        colors = cmap(ig)
    else:
        colors = cmap(np.ones(mu.shape[0]))

    draw_ellipses(
        mu,
        si,
        ax,
        alpha=0.65,
        colors=colors,
        linewidth=2,
        nsigs=[3],
        scatter=scatter,
        edgecolors=edgecolors if edgecolors is not None else colors,
    )

    if highlight_idcs is not None:
        for j, highlight_idx in enumerate(highlight_idcs):
            idx_match = np.where(indices)[0]
            match_positions = np.where(idx_match == highlight_idx)[0]
            if len(match_positions) > 0:
                i = match_positions[0]
                draw_ellipses(
                    mu[i : i + 1],
                    si[i : i + 1],
                    ax,
                    alpha=0.10,
                    colors=["black"],
                    edgecolor=tab20(2 * i),
                    linewidth=2,
                    nsigs=[3],
                    zorder=2,
                    scatter=scatter,
                    edgecolors=edgecolors,
                )
            else:
                plt.scatter([0], [0], marker="x", color=tab20(2 * j), s=50)

    if return_ax:
        return fig, ax
    return fig2img(fig)


def plot_identity_model(imm, return_ax=False, color_only_identity=False):
    """Plot IMM (Identity Mixture Model) state.
    
    Visualizes learned object shapes and colors.
    
    Args:
        imm: IMM model
        return_ax: Whether to return axis objects
        color_only_identity: Whether to use only color features
        
    Returns:
        Image array or (fig, ax) tuple if return_ax=True
    """
    num_object_types = imm.model.continuous_likelihood.mean.shape[0]

    # ensure that the number of subplots are compatible with the total number of object types
    if num_object_types <= 8:
        n_rows, n_cols = 1, num_object_types
    else:
        n_rows = int(np.ceil(np.sqrt(num_object_types)))
        n_cols = int(np.ceil(num_object_types / n_rows))
        while (n_rows * n_cols) < num_object_types:
            n_rows += 1
            n_cols = int(np.ceil(num_object_types / n_rows))

    fig, ax = plt.subplots(n_rows, n_cols)
    if num_object_types == 1:
        ax = np.array([ax])
    ax_flat = ax.flatten() if hasattr(ax, 'flatten') else [ax]

    for object_label, a in enumerate(ax_flat[:num_object_types]):
        if not color_only_identity:
            width, height = imm.model.continuous_likelihood.mean[object_label, :2, 0]
            co = (
                imm.model.continuous_likelihood.mean[object_label, 2:, 0] / 100
            ) * 0.5 + 0.5
        else:
            width, height = 0.3, 0.3
            co = (
                imm.model.continuous_likelihood.mean[object_label, :, 0] / 100
            ) * 0.5 + 0.5

        a.add_patch(
            Ellipse(
                (0.0, 0.0),
                width,
                height,
                facecolor=np.clip(co, 0, 1).tolist(),
                edgecolor="black",
            )
        )
        a.set_xlim([-0.25, 0.25])
        a.set_ylim([-0.25, 0.25])
        a.set_title(f"{bool(imm.used_mask[object_label])}", fontsize=8)
        a.set_xticks([])
        a.set_yticks([])

    # Hide unused subplots
    for a in ax_flat[num_object_types:]:
        a.axis("off")

    plt.suptitle("Learned shapes")
    if return_ax:
        return fig, ax
    return fig2img(fig)


def plot_smm(decoded_mu, decoded_sigma, offsets, stdevs, width, height, qz=None):
    """Plot SMM (Slot Mixture Model) state.
    
    Visualizes slot assignments as ellipses with colors.
    
    Args:
        decoded_mu: Decoded means from SMM
        decoded_sigma: Decoded covariances from SMM
        offsets: Offset for coordinate transformation
        stdevs: Standard deviations for coordinate transformation
        width: Image width
        height: Image height
        qz: Assignment probabilities (optional)
        
    Returns:
        Image array
    """
    mu_list, cova_list = [], []
    for mu, si in zip(decoded_mu, decoded_sigma):
        new_mu, new_cova = transform_mvn(
            stdevs.flatten(),
            offsets.flatten(),
            mu,
            si,
        )
        mu_list.append(new_mu)
        cova_list.append(new_cova)
    
    mu = np.array(mu_list)
    cova = np.array(cova_list)

    pos, col = mu[..., :2], np.clip(mu[..., 2:], 0, 255.0) / 255.0

    # just pick the first 3 dims randomly as a color dim now
    if col.shape[-1] > 3:
        col = np.clip(col[..., :3], 0, 1)

    cova = cova[..., :2, :2]

    if qz is not None:
        assignments = np.argmax(qz[0, ...], axis=-1)
        heights = [(assignments == i).sum() for i in range(qz.shape[-1])]
        indices = np.argwhere(np.array(heights) != 0)[:, 0]
    else:
        indices = np.arange(cova.shape[0])

    fig, ax = plt.subplots(figsize=((2.1 / height) * width, 2.1))
    for i in indices:
        draw_ellipse(
            pos[i],
            covariance=cova[i],
            alpha=0.125,
            color=tuple(col[i].tolist()),
            ax=ax,
        )

    ax.scatter(pos[indices, 0], pos[indices, 1], c=col[indices])
    ax.set_aspect("equal")
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return fig2img(fig)


def plot_tmm(transitions, used_mask, width=160, height=210, return_ax=False):
    """Plot TMM (Transition Mixture Model) state.
    
    Visualizes the learned transition dynamics components.
    
    Args:
        transitions: Transition matrices
        used_mask: Mask for used components
        width: Plot width
        height: Plot height
        return_ax: Whether to return axis objects
        
    Returns:
        Image array or (fig, ax) tuple if return_ax=True
    """
    n_components = transitions.shape[0]
    n_used = int(used_mask.sum())
    
    # Plot transition matrices as heatmaps
    n_cols = min(5, n_used) if n_used > 0 else 1
    n_rows = int(np.ceil(n_used / n_cols)) if n_used > 0 else 1
    
    if n_used > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        
        used_indices = np.where(used_mask)[0]
        
        for idx, comp_idx in enumerate(used_indices):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows == 1:
                a = axes[col]
            else:
                a = axes[row, col]
            
            im = a.imshow(transitions[comp_idx], cmap='RdBu_r', vmin=-1, vmax=1)
            a.set_title(f'Component {int(comp_idx)}')
            a.set_xlabel('State dim')
            a.set_ylabel('State dim')
            plt.colorbar(im, ax=a)
        
        # Hide unused subplots
        for idx in range(n_used, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows == 1:
                axes[col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.suptitle('TMM Transition Components')
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=((8.4 / height) * width, 8.4))
        ax.text(0.5, 0.5, 'No used components', ha='center', va='center')
        ax.axis('off')
    
    if return_ax:
        return fig, ax
    return fig2img(fig)


def plot_plan(
    obs,
    plan_info,
    tracked_obj_ids,
    stats,
    decoded_mu=None,
    topk=5,
    descending=True,
    indices=None,
):
    """Plot planning results.
    
    Visualizes the top-k planning rollouts on top of the observation.
    
    Args:
        obs: RGB observation (numpy array)
        plan_info: Planning information dictionary from planner
        tracked_obj_ids: Boolean array of tracked object indices
        stats: SMM statistics for coordinate transformation
        decoded_mu: Decoded means for coloring (optional)
        topk: Number of top trajectories to show
        descending: Whether to sort in descending order
        indices: Specific indices to plot (optional)
        
    Returns:
        Image array
    """
    obj_ids = np.argwhere(tracked_obj_ids).squeeze()
    if obj_ids.shape == ():
        obj_ids = obj_ids[None]

    if decoded_mu is None:
        if plan_info["states"].shape[-1] > 6:
            colors = (plan_info["states"][0, 0, 0, obj_ids, -3:] * 128 + 128).astype(
                np.uint8
            )
        else:
            colors = []
    else:
        colors = (decoded_mu[obj_ids, 2:] * 128 + 128).astype(np.uint8)

    rewards = plan_info["rewards"]
    if indices is not None:
        idx = indices
    else:
        if descending:
            idx = np.argsort(-rewards.sum(axis=0)[:, 0])[:topk]
        else:
            idx = np.argsort(rewards.sum(axis=0)[:, 0])[:topk]
    states_topk = plan_info["states"][:, idx]
    rewards_topk = rewards[:, idx]
    utility_topk = plan_info["expected_utility"][:, idx]
    info_gain_topk = plan_info["expected_info_gain"][:, idx]

    r = rewards_topk.sum(axis=0).mean()
    u = utility_topk.sum(axis=0).mean()
    ig = info_gain_topk.sum(axis=0).mean()
    title = f"r: {r:.1f}, u: {u:.1f}, ig: {ig:.1f}"

    return plot_rollouts(
        obs,
        states_topk[:, :, 0, obj_ids, :],
        rewards_topk,
        colors,
        stats=stats,
        title=title,
    )


def plot_rollouts(
    obs, states, rewards, colors, horizon=-1, stats=None, title=None
):
    """Plot rollouts on observation.
    
    Args:
        obs: RGB observation (numpy array)
        states: States array [T, B, O, D]
        rewards: Rewards array [T, B, 1]
        colors: Colors for objects
        horizon: Truncation horizon
        stats: SMM statistics (optional)
        title: Plot title (optional)
        
    Returns:
        Image array
    """
    # plot results in a 160 x 210 image
    my_dpi = 50
    fig = plt.figure(figsize=(160 / my_dpi, 210 / my_dpi), dpi=my_dpi, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    if title is not None:
        ax.text(
            0.5,
            0.97,
            title,
            verticalalignment="top",
            horizontalalignment="center",
            transform=ax.transAxes,
            color="white",
            fontsize=16,
        )

    ax.imshow(obs)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # Clip rewards into a sensible range for cmap
    rewards = np.clip(rewards, -1, 1)
    for i in range(states.shape[1]):
        time_plot(
            ax,
            states[:horizon, i],
            rewards[:, i, 0],
            colors,
            stats=stats,
            alpha=0.9 / states.shape[1],
        )
    ax.set_xlim(0, 160)
    ax.set_ylim(210, 0)
    frame = fig2img(fig)
    ax.clear()
    return frame


def time_plot(ax, states, rewards, colors, stats=None, alpha=1.0):
    """Plot a trajectory over time.
    
    Args:
        ax: Matplotlib axis
        states: States array [T, N, D]
        rewards: Rewards array [T]
        colors: Colors for each object
        stats: SMM statistics (optional)
        alpha: Alpha scale
        
    Returns:
        ax: Matplotlib axis
    """
    time_horizon = len(states)
    if stats is None:
        states = states[:, :, :2]
        # shape is now T, Object, 2
    else:
        # SMM DATA
        states = smm_states_to_coords(states[..., :2], stats)

    if len(colors) == 0:
        colors = ["#FFFFFF"] * states.shape[-2]

    for i in range(states.shape[-2]):
        state = states[:, i]
        color = colors[i]
        if not isinstance(color, str):
            color = col_triplet_to_str(color)
        lc, hc = hue_shift(color, amount=0.02)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "SMM-plot", [lc, color, hc]
        )
        alpha_range = alpha * (np.linspace(1, 0, time_horizon) ** 2)
        color = cmap(rewards[i])
        alpha_range = np.broadcast_to(alpha_range[:, None], (time_horizon, 4))
        alpha_range[:, :3] = color[:3]
        ax.scatter(state[:, 0], state[:, 1], c=alpha_range)
    ax.set_xlim(0, 210)
    ax.set_ylim(160, 0)
    return ax


def smm_states_to_coords(positions, stats):
    """Convert SMM state coordinates to image coordinates.
    
    Args:
        positions: Positions array
        stats: SMM statistics
        
    Returns:
        Transformed positions
    """
    return positions * stats["stdevs"][None, None, :2] + stats["offset"][None, None, :2]


def str_to_col_triplet(c):
    """Convert hex color string to RGB triplet.
    
    Args:
        c: Hex color string (e.g., '#FF0000')
        
    Returns:
        RGB array [R, G, B]
    """
    h = c.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)])


def col_triplet_to_str(c):
    """Convert RGB triplet to hex color string.
    
    Args:
        c: RGB array [R, G, B]
        
    Returns:
        Hex color string
    """
    return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"


def hue_shift(color, amount=np.pi / 2):
    """Shift hue of a color.
    
    Args:
        color: Color (hex string or RGB array)
        amount: Amount to shift (radians)
        
    Returns:
        (low_color, high_color): Tuple of shifted colors
    """
    if isinstance(color, str):
        color = str_to_col_triplet(color)
    if color.dtype is not np.floating:
        color = color / 255
    hsv = np.array(matplotlib.colors.rgb_to_hsv(color))
    low = hsv.copy()
    low[0] = (hsv[0] - amount) % (1.0)
    high = hsv.copy()
    high[0] = (hsv[0] + amount) % (1.0)
    low_rgb = (matplotlib.colors.hsv_to_rgb(low) * 255).astype(np.int32)
    high_rgb = (matplotlib.colors.hsv_to_rgb(high) * 255).astype(np.int32)
    return col_triplet_to_str(low_rgb), col_triplet_to_str(high_rgb)


def make_empty_figure(width, height, figsize=(2.1, 2.1)):
    """Create empty figure.
    
    Args:
        width: Image width
        height: Image height
        figsize: Figure size
        
    Returns:
        Image array
    """
    fig, ax = plt.subplots(figsize=((figsize[0] / height) * width, figsize[1]))
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig2img(fig)


def plot_elbo_smm(elbo, width, height):
    """Plot SMM ELBO heatmap.
    
    Args:
        elbo: ELBO values (numpy array)
        width: Image width
        height: Image height
        
    Returns:
        Image array
    """
    fig, ax = plt.subplots(figsize=((2.1 / height) * width, 2.1))
    ax.imshow(elbo.reshape(height, width))
    idx = np.argmin(elbo)
    ax.scatter(idx % width, idx // width, color="red", marker="x")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig2img(fig)


def plot_qz_smm(qz, width, height):
    """Plot SMM slot assignment heatmap.
    
    Args:
        qz: Assignment probabilities (numpy array)
        width: Image width
        height: Image height
        
    Returns:
        Image array
    """
    fig, ax = plt.subplots(figsize=((2.1 / height) * width, 2.1))
    cmap = plt.get_cmap("tab20", qz.shape[-1])
    assignments = cmap(np.argmax(qz[0], axis=-1).reshape(height, width))[:, :, :3]
    ax.imshow(assignments)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig2img(fig)

"""Visualization tools for BNEATEST: network topology, training curves,
and uncertainty analysis via uncertainty-toolbox."""
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .genome import Genome
    from .reporting import StatisticsReporter


def _plot_line_or_point(ax, x: np.ndarray, y: np.ndarray, **kwargs) -> bool:
    """Plot a line for 2+ points or a marker for a single point."""
    if x.size == 0 or y.size == 0:
        return False

    if x.size == 1:
        point_kwargs = dict(kwargs)
        point_kwargs.pop('linewidth', None)
        point_kwargs.setdefault('marker', 'o')
        point_kwargs['linestyle'] = 'None'
        ax.plot(x, y, **point_kwargs)
    else:
        ax.plot(x, y, **kwargs)
    return True


def _show_no_data(ax, title: str) -> None:
    """Render a readable placeholder when training stats are absent."""
    ax.set_title(title)
    ax.text(0.5, 0.5, 'No training data yet',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=10, color='#666666')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# Network topology visualization
# ---------------------------------------------------------------------------

def draw_genome(genome: 'Genome',
                node_radius: float = 0.05,
                vertical_distance: float = 0.25,
                horizontal_distance: float = 0.8,
                show_weights: bool = True,
                weight_label_fontsize: float = 6.0,
                colormap: str = 'RdBu',
                figsize: Tuple[float, float] = (10, 6)) -> None:
    """Draw the evolved network topology with Bayesian weight info.

    Connections are color-coded by mu (mean weight):
      - Blue for positive, red for negative.
    Connection width is proportional to |mu|.
    Connection alpha (transparency) is inversely proportional to sigma
    (high uncertainty → more transparent).

    Args:
        genome: The genome to visualize.
        node_radius: Radius of node circles.
        vertical_distance: Vertical spacing between nodes in same layer.
        horizontal_distance: Horizontal spacing between layers.
        show_weights: If True, annotate connections with mu/sigma.
        weight_label_fontsize: Font size for weight annotations.
        colormap: Matplotlib colormap for weight coloring.
        figsize: Figure size (width, height).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    from .node import group_nodes, NodeType

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('auto')

    # Collect weight stats for normalization
    enabled_conns = [c for c in genome.connections if c.enabled]
    if enabled_conns:
        mus = [c.weight.mu.item() for c in enabled_conns]
        sigmas = [c.weight.sigma.item() for c in enabled_conns]
        max_abs_mu = max(abs(m) for m in mus) if mus else 1.0
        max_sigma = max(sigmas) if sigmas else 1.0
    else:
        max_abs_mu = 1.0
        max_sigma = 1.0

    cmap = plt.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=-max_abs_mu, vmax=max_abs_mu)

    # Position nodes
    positions = {}
    node_groups = group_nodes(genome.nodes, 'depth')

    node_type_colors = {
        NodeType.INPUT: '#4CAF50',
        NodeType.BIAS: '#FF9800',
        NodeType.HIDDEN: '#2196F3',
        NodeType.OUTPUT: '#F44336',
    }

    for group_idx, nodes in enumerate(node_groups):
        y_position = -vertical_distance * (len(nodes) - 1) / 2
        for i, node in enumerate(nodes):
            pos = (group_idx * horizontal_distance,
                   y_position + i * vertical_distance)
            positions[node.id] = pos

            color = node_type_colors.get(node.type, '#9E9E9E')
            circle = plt.Circle(pos, node_radius,
                                facecolor=color, alpha=0.3, linewidth=1.5,
                                edgecolor=color)
            ax.add_artist(circle)
            ax.text(*pos, str(node.id),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8, fontweight='bold')

    if positions:
        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]
        min_x, max_x = min(xs) - node_radius, max(xs) + node_radius
        min_y, max_y = min(ys) - node_radius, max(ys) + node_radius
        pad_x = max(horizontal_distance, node_radius * 4.0, 0.2)
        pad_y = max(vertical_distance, node_radius * 4.0, 0.2)
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)
        # Keep a consistent bounding box when saving with bbox_inches='tight'.
        ax.plot([min_x - pad_x, max_x + pad_x],
                [min_y - pad_y, max_y + pad_y],
                alpha=0.0)

    # Draw connections
    for connection in genome.connections:
        if not connection.enabled:
            continue

        mu = connection.weight.mu.item()
        sigma = connection.weight.sigma.item()

        node1_x, node1_y = positions[connection.in_node.id]
        node2_x, node2_y = positions[connection.out_node.id]

        angle = math.atan2(node2_x - node1_x, node2_y - node1_y)
        x_adj = node_radius * math.sin(angle)
        y_adj = node_radius * math.cos(angle)

        # Color by mu, width by |mu|, alpha by certainty
        color = cmap(norm(mu))
        linewidth = 0.5 + 2.5 * min(abs(mu) / max_abs_mu, 1.0)
        alpha = max(0.2, 1.0 - 0.8 * (sigma / max_sigma)) if max_sigma > 0 else 1.0

        arrow = patches.FancyArrowPatch(
            (node1_x + x_adj, node1_y + y_adj),
            (node2_x - x_adj, node2_y - y_adj),
            arrowstyle="Simple,tail_width=0.5,head_width=3,head_length=5",
            color=color, alpha=alpha, linewidth=linewidth,
            antialiased=True)
        ax.add_patch(arrow)

        # Annotate with mu/sigma
        if show_weights:
            mid_x = (node1_x + node2_x) / 2
            mid_y = (node1_y + node2_y) / 2
            label = f'\u03bc={mu:.2f}\n\u03c3={sigma:.3f}'
            ax.text(mid_x, mid_y, label,
                    fontsize=weight_label_fontsize,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', alpha=0.7,
                              edgecolor='gray', linewidth=0.5))

    # Legend for node types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=c, markersize=8, label=t.name.capitalize())
        for t, c in node_type_colors.items()
        if any(n.type == t for n in genome.nodes)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7)

    # Colorbar for weight mu
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Weight \u03bc (mean)', fontsize=8)

    ax.set_title('Bayesian Network Topology', fontsize=11)
    ax.axis('off')
    plt.tight_layout()


def draw_weight_distributions(genome: 'Genome',
                              figsize: Tuple[float, float] = (12, 5)) -> None:
    """Plot the distribution of mu and sigma across all enabled connections.

    Shows:
    - Left: histogram of mu values
    - Right: histogram of sigma values
    """
    import matplotlib.pyplot as plt

    enabled = [c for c in genome.connections if c.enabled]
    if not enabled:
        print("No enabled connections to visualize.")
        return

    mus = [c.weight.mu.item() for c in enabled]
    sigmas = [c.weight.sigma.item() for c in enabled]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.hist(mus, bins=max(10, len(mus) // 3), color='#2196F3',
             alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Weight \u03bc (mean)')
    ax1.set_ylabel('Count')
    ax1.set_title('Weight Mean Distribution')

    ax2.hist(sigmas, bins=max(10, len(sigmas) // 3), color='#FF9800',
             alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Weight \u03c3 (uncertainty)')
    ax2.set_ylabel('Count')
    ax2.set_title('Weight Uncertainty Distribution')

    fig.suptitle(f'Weight Distributions ({len(enabled)} connections)', fontsize=12)
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Training performance plots
# ---------------------------------------------------------------------------

def plot_fitness(stats: 'StatisticsReporter',
                 figsize: Tuple[float, float] = (10, 5)) -> None:
    """Plot fitness curves over generations.

    Shows mean fitness with stdev band, best fitness, and median.
    """
    import matplotlib.pyplot as plt

    gens, means, stdevs = stats.get_fitness_arrays()
    bests = np.array(stats.best_fitnesses)
    medians = np.array(stats.median_fitnesses)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if gens.size == 0:
        _show_no_data(ax, 'Fitness over Generations')
        plt.tight_layout()
        return

    if gens.size >= 2:
        ax.fill_between(gens, means - stdevs, means + stdevs,
                        alpha=0.2, color='#2196F3', label='Mean \u00b1 stdev')
    else:
        ax.errorbar(gens, means, yerr=stdevs, fmt='o', color='#2196F3',
                    capsize=4, label='Mean \u00b1 stdev')

    _plot_line_or_point(ax, gens, means, color='#2196F3', linewidth=1.5, label='Mean')
    _plot_line_or_point(ax, gens, bests, color='#4CAF50', linewidth=1.5, label='Best')
    _plot_line_or_point(ax, gens, medians, color='#FF9800', linewidth=1.0,
                        linestyle='--', alpha=0.7, label='Median')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness over Generations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_complexity(stats: 'StatisticsReporter',
                    figsize: Tuple[float, float] = (10, 4)) -> None:
    """Plot network complexity (nodes and connections) over generations."""
    import matplotlib.pyplot as plt

    gens = np.array(stats.generations)
    nodes = np.array(stats.n_nodes)
    conns = np.array(stats.n_connections)

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    if gens.size == 0:
        _show_no_data(ax1, 'Network Complexity over Generations')
        plt.tight_layout()
        return

    color1, color2 = '#2196F3', '#F44336'
    _plot_line_or_point(ax1, gens, nodes, color=color1, linewidth=1.5,
                        label='Nodes (mean)')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Nodes', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    _plot_line_or_point(ax2, gens, conns, color=color2, linewidth=1.5,
                        label='Connections (mean)')
    ax2.set_ylabel('Connections', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title('Network Complexity over Generations')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_uncertainty_evolution(stats: 'StatisticsReporter',
                              figsize: Tuple[float, float] = (10, 4)) -> None:
    """Plot how weight uncertainty (sigma) evolves over training."""
    import matplotlib.pyplot as plt

    gens, mean_sigmas, max_sigmas = stats.get_sigma_arrays()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if gens.size == 0:
        _show_no_data(ax, 'Weight Uncertainty over Generations')
        plt.tight_layout()
        return

    _plot_line_or_point(ax, gens, mean_sigmas, color='#FF9800', linewidth=1.5,
                        label='Mean \u03c3')
    _plot_line_or_point(ax, gens, max_sigmas, color='#F44336', linewidth=1.0,
                        linestyle='--', alpha=0.7, label='Max \u03c3')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Weight \u03c3')
    ax.set_title('Weight Uncertainty over Generations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_training_summary(stats: 'StatisticsReporter',
                          figsize: Tuple[float, float] = (14, 10)) -> None:
    """Combined 2x2 dashboard with fitness, complexity, and uncertainty plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Fitness
    ax = axes[0, 0]
    gens, means, stdevs = stats.get_fitness_arrays()
    bests = np.array(stats.best_fitnesses)
    if gens.size == 0:
        _show_no_data(ax, 'Fitness')
    else:
        if gens.size >= 2:
            ax.fill_between(gens, means - stdevs, means + stdevs,
                            alpha=0.2, color='#2196F3')
        else:
            ax.errorbar(gens, means, yerr=stdevs, fmt='o', color='#2196F3',
                        capsize=4)
        _plot_line_or_point(ax, gens, means, color='#2196F3', linewidth=1.5,
                            label='Mean')
        _plot_line_or_point(ax, gens, bests, color='#4CAF50', linewidth=1.5,
                            label='Best')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 2. Best fitness (cumulative)
    ax = axes[0, 1]
    if gens.size == 0:
        _show_no_data(ax, 'Best Fitness (cumulative)')
    else:
        cum_best = np.maximum.accumulate(bests)
        _plot_line_or_point(ax, gens, cum_best, color='#4CAF50', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Best Fitness (cumulative)')
        ax.grid(True, alpha=0.3)

    # 3. Complexity
    ax = axes[1, 0]
    nodes = np.array(stats.n_nodes)
    conns = np.array(stats.n_connections)
    if gens.size == 0:
        _show_no_data(ax, 'Network Complexity')
    else:
        _plot_line_or_point(ax, gens, nodes, color='#2196F3', linewidth=1.5,
                            label='Nodes')
        _plot_line_or_point(ax, gens, conns, color='#F44336', linewidth=1.5,
                            label='Connections')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Count (mean)')
        ax.set_title('Network Complexity')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 4. Uncertainty
    ax = axes[1, 1]
    _, mean_sigmas, max_sigmas = stats.get_sigma_arrays()
    if gens.size == 0:
        _show_no_data(ax, 'Weight Uncertainty')
    else:
        _plot_line_or_point(ax, gens, mean_sigmas, color='#FF9800', linewidth=1.5,
                            label='Mean \u03c3')
        _plot_line_or_point(ax, gens, max_sigmas, color='#F44336', linewidth=1.0,
                            linestyle='--', alpha=0.7, label='Max \u03c3')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Weight \u03c3')
        ax.set_title('Weight Uncertainty')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('BNEATEST Training Summary', fontsize=13, fontweight='bold')
    plt.tight_layout()


# ---------------------------------------------------------------------------
# uncertainty-toolbox integration
# ---------------------------------------------------------------------------

def plot_uncertainty_toolbox(
        predictions_mean: np.ndarray,
        predictions_std: np.ndarray,
        targets: np.ndarray,
        figsize: Tuple[float, float] = (14, 10)) -> None:
    """Generate uncertainty analysis plots using uncertainty-toolbox.

    Creates a 2x2 figure with:
      - Calibration plot
      - Sharpness (histogram of predicted stds)
      - Residuals vs predicted stds
      - Prediction intervals ordered by true value

    Args:
        predictions_mean: Array of predicted means, shape (n_samples,).
        predictions_std: Array of predicted standard deviations, shape (n_samples,).
        targets: Array of true values, shape (n_samples,).
        figsize: Figure size.

    Requires: pip install uncertainty-toolbox
    """
    try:
        import uncertainty_toolbox as uct
    except ImportError:
        raise ImportError(
            "uncertainty-toolbox is required for this plot. "
            "Install it with: pip install uncertainty-toolbox")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Calibration plot
    uct.plot_calibration(predictions_mean, predictions_std, targets, ax=axes[0, 0])
    axes[0, 0].set_title('Calibration')

    # 2. Sharpness
    uct.plot_sharpness(predictions_std, ax=axes[0, 1])
    axes[0, 1].set_title('Sharpness')

    # 3. Residuals vs stds
    uct.plot_residuals_vs_stds(predictions_mean, predictions_std, targets,
                               ax=axes[1, 0])
    axes[1, 0].set_title('Residuals vs Predicted Std')

    # 4. Intervals ordered
    uct.plot_intervals_ordered(predictions_mean, predictions_std, targets,
                               ax=axes[1, 1])
    axes[1, 1].set_title('Prediction Intervals')

    fig.suptitle('Uncertainty Analysis (uncertainty-toolbox)', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()


def compute_uncertainty_metrics(
        predictions_mean: np.ndarray,
        predictions_std: np.ndarray,
        targets: np.ndarray) -> dict:
    """Compute uncertainty quality metrics using uncertainty-toolbox.

    Returns dict with calibration, sharpness, and accuracy metrics.

    Requires: pip install uncertainty-toolbox
    """
    try:
        import uncertainty_toolbox as uct
    except ImportError:
        raise ImportError(
            "uncertainty-toolbox is required. "
            "Install it with: pip install uncertainty-toolbox")

    return uct.get_all_metrics(predictions_mean, predictions_std, targets)

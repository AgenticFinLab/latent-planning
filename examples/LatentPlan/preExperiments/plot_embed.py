"""
To plot the embeddings in various types.
"""

from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_2d(
    encodings: np.ndarray,
    encoding_labels: list,
    label_legend: str,
    save_path: str = None,
    filename: str = None,
    tsne_config={
        "n_components": 2,
        "random_state": 42,
        "perplexity": 5,
        "n_iter": 1000,
    },
):
    """Plot the embedding in the 2D space."""

    ## Apply t-SNE for dimensionality reduction
    n_samples = len(encodings)
    if n_samples <= tsne_config["perplexity"]:
        tsne_config["perplexity"] = n_samples - 1 if n_samples - 1 > 0 else 1

    tsne = TSNE(**tsne_config)
    embeddings_2d = tsne.fit_transform(encodings)

    df = pd.DataFrame(
        {
            "Dimension 1": embeddings_2d[:, 0],
            "Dimension 2": embeddings_2d[:, 1],
            label_legend: encoding_labels,
        }
    )
    # Identify unique categories and build a palette
    unique_categories = df[label_legend].unique()
    palette = sns.color_palette("tab10", n_colors=len(unique_categories))

    # Map each category to its assigned color
    cat_to_color = dict(zip(unique_categories, palette))
    # Plot the features
    # 4: Plot using Seaborn and Matplotlib axis
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(
        data=df,
        x="Dimension 1",
        y="Dimension 2",
        hue=label_legend,
        palette=palette,
        s=100,
        alpha=0.7,
        ax=ax,
    )

    # For each category, compute its center and plot it with the same color
    for cat, group_df in df.groupby(label_legend):
        center_x = group_df["Dimension 1"].mean()
        center_y = group_df["Dimension 2"].mean()

        ax.scatter(
            center_x,
            center_y,
            s=200,
            c=[cat_to_color[cat]],  # same color as the main scatter
            marker="X",  # choose a marker that stands out
            edgecolor="black",  # optional: outline the center marker
            linewidth=1.5,
            #           label=f"Center of {cat}",
        )

    ax.legend(title=label_legend)
    # Remove x and y axes and their labels
    ax.axis("off")
    # Save the figure to a PNG file
    plt.tight_layout()
    output_path = f"{save_path}/{filename}"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_distance_heatmap(
    distances: np.ndarray,
    labels: list,
    save_path: str = None,
    filename: str = None,
    heatmap_config: dict = None,
):
    """
    Plot the heatmap to present the distance relation between questions.
    Each (i,j) rectangle of the figure present the distance between the question i and question j.
    """
    if heatmap_config is None:
        heatmap_config = {"cbar_kws": {"label": "Cosine Distance"}}

    # Step 1: Setup the figure with subplots
    fig, ax = plt.subplots(figsize=(12, 10))

    # Step 2: Add Tick Labels for Categories
    # Generate labels for each category
    u_labels = list(dict.fromkeys(labels))
    tick_labels = []
    tick_positions = []
    end_pos = 0
    for category in u_labels:
        first_pos = labels.index(category) + 1
        last_pos = len(labels) - labels[::-1].index(category)
        tick_positions.append((first_pos + last_pos) / 2)
        tick_labels.append(first_pos - 1)
        end_pos = last_pos
    tick_labels.append(end_pos)

    # Step 2: Plot the heatmap using the provided axes
    print(tick_labels)
    sns.heatmap(
        distances,
        annot=False,  # No numerical values
        cmap="YlGnBu",  # Colormap
        ax=ax,  # Pass the axis to the heatmap
        **heatmap_config,
    )
    # Set positions and labels for X-axis
    ax.set_xticks(tick_labels)
    ax.set_xticklabels(tick_labels)

    # Set positions and labels for Y-axis
    ax.set_yticks(tick_labels)
    ax.set_yticklabels(tick_labels)

    tick_length = tick_labels[-1] - tick_labels[0]
    for idx, pos in enumerate(tick_positions):
        # Add x
        distance = tick_length / 20

        ax.text(
            pos,
            ax.get_xlim()[1] + distance,
            u_labels[idx],
            rotation=0,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            fontweight="bold",
        )
        # Add y
        ax.text(
            -distance,
            pos,
            u_labels[idx],
            rotation=10,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    # Step 4: Display the plot
    plt.tight_layout()
    output_path = f"{save_path}/{filename}"
    plt.savefig(output_path + ".pdf", format="pdf", bbox_inches="tight")
    plt.savefig(output_path + ".png", format="png", bbox_inches="tight", dpi=300)

    plt.close()


def plot_question_step_relation(
    step_distance_distribution: Dict[int, np.ndarray],
    step_question_distribution: Dict[int, np.ndarray],
    save_path,
    filename,
    plot_config: dict = None,
):
    """
    Plot the relation between questions and steps.

    :param step_distance_distribution: A Dict with the step index as the key
     while the value is the similarity scores of this step in all questions and the central step encoding.
    :param step_question_distribution: A Dict with the step index as the key
     while the value is the similarity scores of all questions and the central
     question encoding
    """

    # Prepare data for plotting
    all_data = []
    key_spacing = 3  # Set larger spacing between keys
    x_offsets = {
        key: i * key_spacing for i, key in enumerate(step_distance_distribution.keys())
    }
    # Store mean y-values for each key
    means_y = {}
    for step_idx, step_distr in step_distance_distribution.items():
        question_distr = step_question_distribution[step_idx]
        x_offset = x_offsets[step_idx]
        means_y[step_idx] = np.mean(question_distr)
        for q_idx, step_v in enumerate(step_distr):
            q_v = question_distr[q_idx]
            all_data.append({"step_idx": step_idx, "x": step_v + x_offset, "y": q_v})

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Add vertical lines and small range lines around each key
    for key, x_offset in x_offsets.items():
        mean_y = means_y[key]
        plt.axvline(
            x=x_offset + 1, color="gray", linestyle="--", linewidth=0.8
        )  # Vertical line at 0
        # Add horizontal small range lines (-1 to 1) centered at mean_y
        plt.plot(
            [x_offset, x_offset + 2],
            [mean_y, mean_y],
            color="black",
            linestyle="--",
            linewidth=0.8,
        )  # Horizontal line
        plt.plot(
            [x_offset, x_offset],
            [mean_y - 0.05, mean_y + 0.05],
            color="black",
            linestyle="--",
            linewidth=0.8,
        )  # Start marker
        plt.plot(
            [x_offset + 2, x_offset + 2],
            [mean_y - 0.05, mean_y + 0.05],
            color="black",
            linestyle="--",
            linewidth=0.8,
        )  # End marker
        # Add annotations for -1 and 1
        plt.text(
            x_offset,
            mean_y - 0.06,
            "0",
            ha="center",
            va="top",
            fontsize=10,
            color="blue",
        )  # Lower bound
        plt.text(
            x_offset + 2,
            mean_y - 0.06,
            "2",
            ha="center",
            va="top",
            fontsize=10,
            color="blue",
        )  # Upper bound

        # Add tick labels for the range
        tick_positions = np.linspace(x_offset - 1, x_offset + 1, 5)
        tick_labels = [f"{tick - x_offset:.1f}" for tick in tick_positions]
        plt.xticks(
            list(plt.xticks()[0]) + list(tick_positions),
            list(plt.xticks()[1]) + tick_labels,
        )

    sns.scatterplot(data=df, x="x", y="y", hue="step_idx", legend="brief", s=30)
    plt.grid(False)
    # Set manual x-ticks for keys
    key_ticks = [x_offsets[key] + 1 for key in step_distance_distribution.keys()]
    plt.xticks(key_ticks, step_distance_distribution.keys())

    plot_config = (
        {"xlabel": "Reasoning Step ID", "ylabel": "Question Distance"}
        if plot_config is None
        else plot_config
    )

    plt.xlabel(plot_config["xlabel"])
    plt.ylabel(plot_config["ylabel"])
    # Remove the upper and right borders
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    # Step 4: Display the plot
    plt.tight_layout()
    output_path = f"{save_path}/{filename}"
    plt.savefig(output_path, dpi=300)
    plt.close()

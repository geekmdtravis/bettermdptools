# -*- coding: utf-8 -*-

from typing import Literal
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import LinearSegmentedColormap


class Plots:
    @staticmethod
    def values_heat_map(
        data: np.ndarray,
        title: str,
        size: tuple[int, int],
        show: bool = True,
        filename: str = None,
    ) -> None:
        data = np.around(np.array(data).reshape(size), 2)
        df = pd.DataFrame(data=data)
        sns.heatmap(df, annot=True).set_title(title)

        if show and filename:
            warnings.warn(
                "Both show and filename are present. Saving to file takes precedence."
            )

        if show and not filename:
            plt.show()
        if filename:
            plt.savefig(filename)

    @staticmethod
    def v_iters_plot(
        data: np.ndarray | list[np.ndarray],
        title: str,
        show: bool = True,
        filename: str | None = None,
        watermark: str | None = None,
        size: tuple[int, int] = (8, 6),
        dpi: int = 100,
        legend: bool = True,
        title_size: int = 16,
        label_size: int = 12,
        axis_size: int = 12,
        log_x: bool = False,
        log_y: bool = False,
        legend_labels: list[str] | None = None,
        plot_type: Literal["line", "bar", "step"] = "line",
        delta_convergence: np.ndarray | None = None,
    ) -> None:
        """
        Plot the values of an MDP over iterations with optional delta convergence on secondary axis.
        """
        if legend_labels is not None and len(legend_labels) != len(data):
            raise ValueError("legend_labels must be the same length as data")

        sns.set_theme(style="whitegrid")

        # Convert single array to list for consistent handling
        if isinstance(data, np.ndarray):
            data = [data]

        if len(data) > 5:
            warnings.warn("More than 5 lines may reduce plot legibility")

        fig, ax1 = plt.subplots(figsize=size, dpi=dpi)

        labels = (
            legend_labels
            if legend_labels is not None
            else [f"Line {i}" for i in range(len(data))]
        )

        for line, label in zip(data, labels):
            if plot_type == "line":
                sns.lineplot(data=line, label=label, ax=ax1)
            elif plot_type == "bar":
                ax1.bar(range(len(line)), line, label=label)
            elif plot_type == "step":
                ax1.step(range(len(line)), line, label=label, where="post")

        ax1.set_xlabel("Iterations", fontsize=label_size)
        ax1.set_ylabel("Value", fontsize=label_size)
        ax1.tick_params(axis="both", which="major", labelsize=axis_size)

        if delta_convergence is not None:
            ax2 = ax1.twinx()
            ax2.plot(delta_convergence, "--", color="red", label="Delta Convergence")
            ax2.set_ylabel("Delta Convergence", fontsize=label_size)
            ax2.tick_params(axis="y", labelsize=axis_size)

        plt.title(title, fontsize=title_size)

        if log_x:
            ax1.set_xscale("log")
        if log_y:
            ax1.set_yscale("log")

        if legend:
            lines1, labels1 = ax1.get_legend_handles_labels()
            if delta_convergence is not None:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=label_size)
            else:
                ax1.legend(fontsize=label_size)

        if watermark:
            plt.figtext(
                0.5,
                0.5,
                "BETTER MDP TOOLS",
                ha="center",
                va="center",
                alpha=0.2,
                fontsize=72,
            )

        if show and filename:
            warnings.warn(
                "Both show and filename are present. Saving to file takes precedence."
            )
        plt.tight_layout()
        if show and not filename:
            plt.show()
        if filename:
            plt.savefig(filename)

    # modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def get_policy_map(pi, val_max, actions, map_size):
        """Map the best learned action to arrows."""
        # convert pi to numpy array
        best_action = np.zeros(val_max.shape[0], dtype=np.int32)
        for idx, val in enumerate(val_max):
            best_action[idx] = pi[idx]
        policy_map = np.empty(best_action.flatten().shape, dtype=str)
        for idx, val in enumerate(best_action.flatten()):
            policy_map[idx] = actions[val]
        policy_map = policy_map.reshape(map_size[0], map_size[1])
        val_max = val_max.reshape(map_size[0], map_size[1])
        return val_max, policy_map

    # modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def plot_policy(val_max, directions, map_size, title, show=True, filename=None):
        """Plot the policy learned."""
        sns.heatmap(
            val_max,
            annot=directions,
            fmt="",
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
        ).set(title=title)
        img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"

        if show and filename:
            warnings.warn(
                "Both show and filename are present. Saving to file takes precedence."
            )

        if show and not filename:
            plt.show()
        if filename:
            plt.savefig(filename)

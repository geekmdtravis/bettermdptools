# -*- coding: utf-8 -*-

import math
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
        size: tuple[int, int] = (10, 5),
        dpi: int = 100,
        legend: bool = True,
        title_size: int = 16,
        label_size: int = 12,
        axis_size: int = 12,
    ) -> None:
        sns.set_theme(style="whitegrid")

        # Convert single array to list for consistent handling
        if isinstance(data, np.ndarray):
            data = [data]

        if len(data) > 5:
            warnings.warn("More than 5 lines may reduce plot legibility")

        plt.figure(figsize=size, dpi=dpi)

        for line in data:
            sns.lineplot(data=line, legend=legend)

        plt.title(title, fontsize=title_size)
        plt.xlabel("Iterations", fontsize=label_size)
        plt.ylabel("Value", fontsize=label_size)
        plt.tick_params(axis="both", which="major", labelsize=axis_size)

        if watermark:
            plt.figtext(
                0.5, 0.5, watermark, ha="center", va="center", alpha=0.2, fontsize=36
            )

        if show and filename:
            warnings.warn(
                "Both show and filename are present. Saving to file takes precedence."
            )

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

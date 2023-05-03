import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

from data import MyDataset


def array_statistics(arr, ax_kde: Axes, title: str):
    # calculate statistics
    arr = np.array(arr)
    mean = np.mean(arr)
    median = np.median(arr)
    stdev = np.std(arr)
    percentile_90th = np.percentile(arr, 90)
    min_val = np.min(arr)
    max_val = np.max(arr)

    stats_data = {
        "Mean": [mean],
        "Median": [median],
        "Standard Deviation": [stdev],
        "90th Percentile": [percentile_90th],
        "Min": [min_val],
        "Max": [max_val],
    }

    # Kernel Density Estimation (KDE) plot
    sns.kdeplot(arr, ax=ax_kde, color="blue")
    ax_kde.axvline(
        percentile_90th, color="red", linestyle="dashed", label="90th Percentile"
    )
    ax_kde.legend()
    ax_kde.set_title(title)
    ax_kde.set_xlabel("Length")
    ax_kde.set_ylabel("Density")

    return ax_kde, stats_data


def display_dataset_statistics(dataset: MyDataset):
    # turn statistics in terms of chars
    turn_lengths = []
    for dialogue in dataset[0]:
        for turn in dialogue:
            turn_lengths.append(len(turn))

    # dialogue statistics in terms of tokens
    dialogue_lengths = []
    for dialogue in dataset[0]:
        concat_dialogue = "[SEP]".join(dialogue)
        dialogue_lengths.append(len(concat_dialogue) / 4)

    # dialogue statistics in terms of turns
    dialogue_lengths_turns = []
    for dialogue in dataset[0]:
        dialogue_lengths_turns.append(len(dialogue))

    # Create the array data for each plot
    arrays = [
        (turn_lengths, "Turn Lengths (char)"),
        (dialogue_lengths, "Dialogue Lengths (tokens)"),
        (dialogue_lengths_turns, "Dialogue Lengths (turns)"),
    ]

    # Set up the grid for the plots and tables
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 7), gridspec_kw={"height_ratios": [20, 3]}
    )
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    column_labels = [
        "Mean",
        "Median",
        "S.D.",
        "90th %",
        "Min",
        "Max",
    ]

    # Create the KDE plots and tables
    for index, (arr, title) in enumerate(arrays):
        ax_kde, stats_data = array_statistics(
            arr,
            axes[0, index],
            title,
        )

        for key, value in stats_data.items():
            stats_data[key] = round(value[0], 1)

        table_data = [list(stats_data.values())]

        axes[1, index].axis("off")
        table = axes[1, index].table(
            cellText=table_data,
            colLabels=column_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)

    plt.show()

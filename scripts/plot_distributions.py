import os
from datasets import load_dataset, load_from_disk
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes


def draw_distribution(data):
    # Calculate the percentage of integers that are less than or equal to each threshold
    max_threshold = np.percentile(data, 98)
    thresholds = np.arange(0, max_threshold, 16)
    percentages = [np.sum(data <= t) / len(data) * 100 for t in thresholds]

    # Plot the results
    plt.plot(thresholds, percentages)
    plt.xlabel("Threshold")
    plt.ylabel("Percentage Included")
    plt.title("Percentage of examples Included by Threshold")

    # Add vertical and horizontal lines that stop at the plot line
    for t, p in zip(thresholds, percentages):
        plt.hlines(y=p, xmin=0, xmax=t, color="grey", alpha=0.5)
        plt.vlines(x=t, ymin=0, ymax=p, color="grey", alpha=0.5)

    # Set the x and y axis labels and ticks
    plt.xticks(thresholds, rotation=60)
    plt.yticks(percentages)

    plt.show()


def array_statistics(arr, title="Array Statistics"):
    # calculate statistics
    arr = np.array(arr)
    mean = np.mean(arr)
    median = np.median(arr)
    variance = np.var(arr)
    stdev = np.std(arr)
    percentile_25th = np.percentile(arr, 25)
    percentile_50th = np.percentile(arr, 50)
    percentile_75th = np.percentile(arr, 75)
    percentile_90th = np.percentile(arr, 90)
    percentile_95th = np.percentile(arr, 95)
    min_val = np.min(arr)
    max_val = np.max(arr)

    stats_data = {
        "Mean": [mean],
        "Median": [median],
        "Variance": [variance],
        "Standard Deviation": [stdev],
        "25th Percentile": [percentile_25th],
        "50th Percentile": [percentile_50th],
        "75th Percentile": [percentile_75th],
        "90th Percentile": [percentile_90th],
        "95th Percentile": [percentile_95th],
        "Min": [min_val],
        "Max": [max_val],
    }

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 3)

    # Add subplots for plots
    ax_hist: Axes = fig.add_subplot(gs[0, 0])
    ax_box: Axes = fig.add_subplot(gs[0, 1])
    ax_kde: Axes = fig.add_subplot(gs[1, 0])
    ax_cumulative: Axes = fig.add_subplot(gs[1, 1])

    # Histogram
    sns.histplot(arr, kde=False, ax=ax_hist, color="blue")
    ax_hist.axvline(
        percentile_90th, color="red", linestyle="dashed", label="90th Percentile"
    )
    ax_hist.legend()
    ax_hist.set_title("Histogram")
    ax_hist.set_xlabel("Length")
    ax_hist.set_ylabel("Frequency")

    # Box plot
    sns.boxplot(y=arr, ax=ax_box, color="blue")
    ax_box.axhline(
        percentile_90th, color="red", linestyle="dashed", label="90th Percentile"
    )
    ax_box.legend()
    ax_box.set_title("Box Plot")
    ax_box.set_ylabel("Length")

    # Kernel Density Estimation (KDE) plot
    sns.kdeplot(arr, ax=ax_kde, color="blue")
    ax_kde.axvline(
        percentile_90th, color="red", linestyle="dashed", label="90th Percentile"
    )
    ax_kde.legend()
    ax_kde.set_title("Kernel Density Estimation Plot")
    ax_kde.set_xlabel("Length")
    ax_kde.set_ylabel("Density")

    # Cumulative Density Plot
    sns.kdeplot(arr, cumulative=True, ax=ax_cumulative, color="blue")
    ax_cumulative.axvline(
        percentile_90th, color="red", linestyle="dashed", label="90th Percentile"
    )
    ax_cumulative.legend()
    ax_cumulative.set_title("Cumulative Density Plot")
    ax_cumulative.set_xlabel("Length")
    ax_cumulative.set_ylabel("Cumulative Density")

    # Prepare cell_text for the table with two columns
    cell_text = []
    for key, value in stats_data.items():
        cell_text.append([key, f"{value[0]:.2f}"])

    # Display statistics table as a subplot
    ax_table: Axes = fig.add_subplot(gs[:, 2])
    table = ax_table.table(
        cellText=cell_text, colLabels=["Statistic", "Value"], loc="center"
    )
    ax_table.axis("off")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    ax_table.axis("off")

    fig.tight_layout()

    # Add the title for all figures
    fig.suptitle(title, fontsize=16, y=1.05)

    plt.show()

    return fig


if __name__ == "__main__":
    dataset_dir = "data/swda"

    # Load the dataset
    if os.path.exists(dataset_dir):
        # load the dataset from disk
        dataset = load_from_disk(dataset_dir)
        print("Dataset loaded from disk")
    else:
        # load the dataset from Hugging Face and save it to disk
        dataset = load_dataset("silicone", "swda")
        dataset.save_to_disk(dataset_dir)
        print("Dataset loaded from Hugging Face and saved to disk")

    train = dataset["train"].to_pandas()
    len_turns = [len(x) for x in dataset["train"]["Utterance"]]
    len_dialogues = train.groupby("Dialogue_ID").agg(len)["Utterance"].tolist()

    draw_distribution(len_turns)
    draw_distribution(len_dialogues)

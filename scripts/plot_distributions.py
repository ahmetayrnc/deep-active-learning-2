import os
from datasets import load_dataset, load_from_disk
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def array_statistics(arr):
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

    stats_df = pd.DataFrame(stats_data)
    print("Statistics Table:")
    print(stats_df.to_string(index=False))

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # Line plot with mean and median
    ax[0, 0].plot(arr, color="blue")
    ax[0, 0].axhline(mean, color="red", linestyle="dashed", label="Mean")
    ax[0, 0].axhline(median, color="green", linestyle="dashed", label="Median")
    ax[0, 0].legend()
    ax[0, 0].set_title("Line Plot with Mean and Median")
    ax[0, 0].set_xlabel("Index")
    ax[0, 0].set_ylabel("Value")

    # Histogram
    sns.histplot(arr, kde=False, ax=ax[0, 1], color="blue")
    ax[0, 1].set_title("Histogram")
    ax[0, 1].set_xlabel("Value")
    ax[0, 1].set_ylabel("Frequency")

    # Box plot
    sns.boxplot(y=arr, ax=ax[1, 0], color="blue")
    ax[1, 0].set_title("Box Plot")
    ax[1, 0].set_ylabel("Value")

    # Kernel Density Estimation (KDE) plot
    sns.kdeplot(arr, ax=ax[1, 1], color="blue")
    ax[1, 1].set_title("Kernel Density Estimation Plot")
    ax[1, 1].set_xlabel("Value")
    ax[1, 1].set_ylabel("Density")

    fig.tight_layout()
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

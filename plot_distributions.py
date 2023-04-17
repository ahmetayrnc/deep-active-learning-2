import os
from datasets import load_dataset, load_from_disk
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def draw_distribution(data):
    # Calculate the percentage of integers that are less than or equal to each threshold
    thresholds = np.arange(0, max(data), 16)
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

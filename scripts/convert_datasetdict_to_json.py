import pandas as pd
import json
import os
import argparse
from datasets import load_dataset, load_from_disk
from pprint import pprint
from datasets import DatasetDict
from typing import List, TypedDict


class Turn(TypedDict):
    idx: str
    content: str
    user: str


class Dialogue(TypedDict):
    dialogue_id: str
    turns: List[Turn]
    labels: List[str]


def convert_to_json(dataset_dict: DatasetDict, dataset_dir: str, split: str):
    df = pd.DataFrame(dataset_dict[split])

    # group the dataframe by Dialogue_ID
    grouped = df.groupby("Dialogue_ID")

    # create a list to hold the dialogue JSON objects
    dialogues = []

    # loop over the dialogue groups and construct the JSON object for each dialogue
    for dialogue_id, group in grouped:
        # create a dictionary to hold the dialogue JSON object
        dialogue: Dialogue = {
            "dialogue_id": str(dialogue_id),
            "turns": [],
            "labels": [],
        }

        # loop over the rows in the group to construct the turns list
        for i, row in group.iterrows():
            turn = {
                "idx": str(row["Idx"]),
                "content": row["Utterance"],
                "user": str(row["Idx"] % 2),
            }
            label = row["Label"]

            dialogue["turns"].append(turn)
            dialogue["labels"].append(label)

        # add the dialogue object to the list of dialogues
        dialogues.append(dialogue)

    # convert the list of dialogues to JSON
    json_data = json.dumps(dialogues)

    # create the directory to hold the JSON files
    os.makedirs(f"{dataset_dir}/json", exist_ok=True)

    # write the JSON data to a file
    with open(f"{dataset_dir}/json/{split}.json", "w") as f:
        f.write(json_data)

    # print the output file directory
    print(f"JSON file saved to {dataset_dir}/json/{split}.json")


# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    type=str,
    default="train",
    choices=["train", "validation", "test"],
    help="split",
)

args = parser.parse_args()
pprint(vars(args))
print()

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

convert_to_json(dataset_dict=dataset, dataset_dir=dataset_dir, split=args.split)

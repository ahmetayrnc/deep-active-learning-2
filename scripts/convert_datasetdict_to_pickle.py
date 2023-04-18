import pickle
import pandas as pd
import os
import argparse
from datasets import load_dataset, load_from_disk
from pprint import pprint
from datasets import DatasetDict
from typing import List, TypedDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding


class Dialogue(TypedDict):
    dialogue_id: str
    turns: List[BatchEncoding]
    labels: List[int]


def convert_to_json(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    dataset_dir: str,
    split: str,
):
    df = pd.DataFrame(dataset_dict[split])

    # group the dataframe by Dialogue_ID
    grouped = df.groupby("Dialogue_ID")

    # create a list to hold the dialogue JSON objects
    dialogues = []

    # loop over the dialogue groups and construct the JSON object for each dialogue
    for dialogue_id, group in grouped:
        if len(group) > 512:
            print(f"skipped dialogue: {dialogue_id}")
            continue

        # create a dictionary to hold the dialogue JSON object
        dialogue: Dialogue = {
            "dialogue_id": str(dialogue_id),
            "turns": [],
            "labels": [],
        }

        # loop over the rows in the group to construct the turns list
        for i, row in group.iterrows():
            turn = tokenizer(
                row["Utterance"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            label = row["Label"]

            dialogue["turns"].append(turn)
            dialogue["labels"].append(label)

        # add the dialogue object to the list of dialogues
        dialogues.append(dialogue)

    # create the directory to hold the JSON files
    os.makedirs(f"{dataset_dir}/pickle", exist_ok=True)

    # Save the object to a file
    with open(f"{dataset_dir}/pickle/{split}.pickle", "wb") as f:
        pickle.dump(dialogues, f)

    # print the output file directory
    print(f"pickle file saved to {dataset_dir}/pickle/{split}.pickle")


def main(args_dict):
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

    pretrained_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=True)

    convert_to_json(
        dataset_dict=dataset,
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        split=args_dict["split"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="split",
    )

    args = parser.parse_args()
    args_dict = vars(args)
    pprint(args_dict)

    main(args_dict)

import json
from typing import List, Tuple, TypedDict
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
import os
import argparse
from pprint import pprint
import tqdm
from convert_datasetdict_to_json import Dialogue


class DialogueTorch(TypedDict):
    dialogue_id: str
    input_ids: torch.FloatTensor
    attention_masks: torch.FloatTensor
    labels: torch.IntTensor


def convert_json_to_torch(
    tokenizer: PreTrainedTokenizerFast, max_length: int, dialogue: Dialogue
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    turns = dialogue["turns"]
    labels = dialogue["labels"]

    input_ids = []
    attention_masks = []

    num_turns = len(turns)
    for i, turn in enumerate(turns):
        turn_contents = list(map(lambda turn: turn["content"], turns))
        turn_content = turn["content"]

        context_prev = turn_contents[
            max(0, i - 3) : i
        ]  # Include the previous 3 turns as context
        context_next = turn_contents[
            i + 1 : min(i + 4, num_turns)
        ]  # Include the next 3 turns as context
        text = " [SEP] ".join(
            context_prev + [turn_content] + context_next
        )  # Concatenate turns with a separator token
        encoded = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    # input_ids = input_ids.cpu().numpy().tolist()
    # attention_masks = attention_masks.cpu().numpy().tolist()
    # labels = labels.cpu().numpy().tolist()

    return input_ids, attention_masks, labels


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

dataset_dir = "data/swda"

# Open the JSON file and load the data into a dictionary
with open(f"{dataset_dir}/json/{args.split}.json") as f:
    data = json.load(f)

# Load the tokenizer and model
model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=46)

# Convert the data to PyTorch tensors
data_torch: List[DialogueTorch] = []
for i in tqdm.tqdm(range(len(data))):
    conversation: Dialogue = data[i]
    input_ids, attention_masks, labels = convert_json_to_torch(
        tokenizer=tokenizer,
        dialogue=conversation,
        max_length=512,
    )
    conversation_torch: DialogueTorch = {
        "dialogue_id": conversation["dialogue_id"],
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "labels": labels,
    }
    data_torch.append(conversation_torch)


dialogue_ids = []
input_ids = []
attention_masks = []
labels = []

for item in data_torch:
    dialogue_ids.append(item["dialogue_id"])
    input_ids.append(item["input_ids"])
    attention_masks.append(item["attention_masks"])
    labels.append(item["labels"])

# Convert lists to tensors
input_ids_tensor = torch.cat(input_ids)
attention_masks_tensor = torch.cat(attention_masks)
labels_tensor = torch.cat(labels)

# create the directory to hold the JSON files
os.makedirs(f"{dataset_dir}/torch", exist_ok=True)

# Save tensors
torch.save(input_ids_tensor, f"{dataset_dir}/torch/{args.split}_input_ids.pt")
torch.save(
    attention_masks_tensor, f"{dataset_dir}/torch/{args.split}_attention_masks.pt"
)
torch.save(labels_tensor, f"{dataset_dir}/torch/{args.split}_labels.pt")

print(f"tensors saved to {dataset_dir}/torch")

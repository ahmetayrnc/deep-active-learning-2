import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import argparse
from pprint import pprint
import tqdm


def convert_json_to_torch(tokenizer, max_length: int, conversation):
    turns = conversation["turns"]
    labels = conversation["labels"]

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

    input_ids = input_ids.cpu().numpy().tolist()
    attention_masks = attention_masks.cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()

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
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=46)

# Convert the data to PyTorch tensors
data_torch = []
for i in tqdm.tqdm(range(len(data))):
    conversation = data[i]
    input_ids, attention_masks, labels = convert_json_to_torch(
        tokenizer=tokenizer,
        max_length=512,
        conversation=conversation,
    )
    conversation_torch = {
        "dialogue_id": conversation["dialogue_id"],
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "labels": labels,
    }
    data_torch.append(conversation_torch)

# Convert the data to JSON
json_data_torch = json.dumps(data_torch)

# create the directory to hold the JSON files
os.makedirs(f"{dataset_dir}/json_torch", exist_ok=True)

# write the JSON data to a file
with open(f"{dataset_dir}/json_torch/{args.split}.json", "w") as f:
    f.write(json_data_torch)

# print the output file directory
print(f"JSON file saved to {dataset_dir}/json_torch/{args.split}.json")

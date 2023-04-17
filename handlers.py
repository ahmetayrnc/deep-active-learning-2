from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from data import Dialogue
from transformers import BatchEncoding


# class Conversation_Handler(Dataset):
#     def __init__(self, data: JSONDataset) -> None:
#         self.data = data

#         self.input_ids = data["input_ids"]
#         self.attention_masks = data["attention_masks"]
#         self.labels = data["labels"]

#     def __len__(self) -> int:
#         return len(self.labels)

#     def __getitem__(
#         self, idx: int
#     ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor, int]:
#         input_ids: torch.FloatTensor = self.input_ids[idx]
#         attention_mask: torch.FloatTensor = self.attention_masks[idx]
#         label: torch.IntTensor = self.labels[idx]

#         return input_ids, attention_mask, label, idx


class DialogueDataset(Dataset):
    def __init__(self, dialogues: List[Dialogue]):
        self.dialogues = dialogues

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Tuple[List[BatchEncoding], List[int]]:
        return self.dialogues[idx]["turns"], self.dialogues[idx]["labels"]


def string_collator(batch):
    dialogues, labels = zip(*batch)
    return dialogues, torch.stack([torch.tensor(label) for label in labels])

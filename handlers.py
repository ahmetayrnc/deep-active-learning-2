from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding

from data import MyDataset


class DialogueDataset(Dataset):
    def __init__(self, dialogues: MyDataset):
        self.dialogues = dialogues

    def __len__(self) -> int:
        return len(self.dialogues[1])

    def __getitem__(self, idx: int) -> Tuple[List[BatchEncoding], List[int]]:
        return self.dialogues[0][idx], self.dialogues[1][idx]


def string_collator(batch):
    dialogues, labels = zip(*batch)
    return dialogues, torch.stack([torch.tensor(label) for label in labels])

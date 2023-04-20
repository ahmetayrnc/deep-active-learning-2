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

    # Find the maximum dialogue length in the batch
    max_len = max(len(label) for label in labels)

    # Pad the labels and stack them as a tensor
    padded_labels = []
    for label in labels:
        if len(label) < max_len:
            pad_len = max_len - len(label)
            padded_label = label + [-1] * pad_len
        else:
            padded_label = label
        padded_labels.append(torch.tensor(padded_label))

    return dialogues, torch.stack(padded_labels)

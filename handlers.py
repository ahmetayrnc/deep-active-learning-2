from typing import Tuple
import numpy as np
from torch.utils.data import Dataset
from data import JSONDataset


class Conversation_Handler(Dataset):
    def __init__(self, data: JSONDataset) -> None:
        self.data = data

        self.input_ids = data["input_ids"]
        self.attention_masks = data["attention_masks"]
        self.labels = data["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        input_ids: np.ndarray = self.input_ids[idx]
        attention_mask: np.ndarray = self.attention_masks[idx]
        label: np.ndarray = self.labels[idx]

        return input_ids, attention_mask, label, idx

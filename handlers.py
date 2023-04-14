import torch
from torch.utils.data import Dataset


class Conversation_Handler(Dataset):
    def __init__(self, data: "dict[str, torch.Tensor]") -> None:
        self.data = data

        self.input_ids: torch.FloatTensor = data["input_ids"]
        self.attention_masks: torch.FloatTensor = data["attention_masks"]
        self.labels: torch.IntTensor = data["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> "tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor, int]":
        input_ids: torch.FloatTensor = self.input_ids[idx]
        attention_mask: torch.FloatTensor = self.attention_masks[idx]
        label: torch.IntTensor = self.labels[idx]

        return input_ids, attention_mask, label, idx

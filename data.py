import os
from typing import Tuple, Type, TypedDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.data import Dataset


class JSONDataset(TypedDict):
    input_ids: torch.FloatTensor  # shape: (n_samples, max_seq_len)
    attention_masks: torch.FloatTensor  # shape: (n_samples, max_seq_len)
    labels: torch.IntTensor  # shape: (n_samples,)


class Metrics(TypedDict):
    accuracy: float
    f1: float
    precision: float
    recall: float


class Data:
    def __init__(
        self,
        train: JSONDataset,
        validation: JSONDataset,
        test: JSONDataset,
        handler: Type[Dataset],
    ):
        self.train = train
        self.test = test
        self.validation = validation

        self.handler = handler
        self.n_pool = len(train["input_ids"])
        self.n_test = len(test["input_ids"])

        self.labeled_idxs: np.ndarray = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num: int) -> None:
        # generate initial labeled pool
        tmp_idxs: np.ndarray = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_data(self) -> Tuple[np.ndarray, Dataset]:
        labeled_idxs: np.ndarray = np.arange(self.n_pool)[self.labeled_idxs]
        indexed: JSONDataset = {k: v[labeled_idxs] for k, v in self.train.items()}
        return labeled_idxs, self.handler(indexed)

    def get_unlabeled_data(self) -> Tuple[np.ndarray, Dataset]:
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        indexed: JSONDataset = {k: v[unlabeled_idxs] for k, v in self.train.items()}
        return unlabeled_idxs, self.handler(indexed)

    def get_train_data(self) -> Tuple[np.ndarray, Dataset]:
        return self.labeled_idxs.copy(), self.handler(self.train)

    def get_test_data(self) -> Dataset:
        return self.handler(self.test)

    def cal_test_acc(self, preds) -> float:
        y_true = self.test["labels"]
        y_pred = preds

        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def cal_test_metrics(self, preds) -> Metrics:
        y_true = self.test["labels"]
        y_pred = preds

        accuracy = accuracy_score(y_true, y_pred)

        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return {
            "accuracy": accuracy,
            "f1": macro_f1,
            "recall": macro_recall,
            "precision": macro_precision,
        }


def get_SWDA() -> Tuple[JSONDataset, JSONDataset, JSONDataset]:
    torch_data_dir = "data/swda/torch"

    if not os.path.isfile(f"{torch_data_dir}/train_input_ids.pt"):
        raise FileNotFoundError(
            f"{torch_data_dir}/train_input_ids.pt does not exist. Please run convert scripts first."
        )

    datasets: list[JSONDataset] = []
    for split in ["train", "validation", "test"]:
        input_ids = torch.load(f"{torch_data_dir}/{split}_input_ids.pt")
        attention_masks = torch.load(f"{torch_data_dir}/{split}_attention_masks.pt")
        labels = torch.load(f"{torch_data_dir}/{split}_labels.pt")
        dataset = JSONDataset(
            input_ids=input_ids, attention_masks=attention_masks, labels=labels
        )
        datasets.append(dataset)

    train, validation, test = datasets
    return train, validation, test

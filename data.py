import os
from typing import Tuple, Type, TypedDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from torch.utils.data import Dataset


class JSONDataset(TypedDict):
    input_ids: np.ndarray  # shape: (n_samples, max_seq_len)
    attention_masks: np.ndarray  # shape: (n_samples, max_seq_len)
    labels: np.ndarray  # shape: (n_samples,)


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
    json_torch_data_dir = "data/swda/json_torch"

    if not os.path.isfile(f"{json_torch_data_dir}/train.json"):
        raise FileNotFoundError(
            f"{json_torch_data_dir}/train.json does not exist. Please run convert scripts first."
        )

    with open(f"{json_torch_data_dir}/train.json") as f:
        train: JSONDataset = json.load(f)

    print(train)
    with open(f"{json_torch_data_dir}/validation.json") as f:
        validation: JSONDataset = json.load(f)

    with open(f"{json_torch_data_dir}/test.json") as f:
        test: JSONDataset = json.load(f)

    def concatanate_turns(data) -> JSONDataset:
        labels = []
        input_ids = []
        attention_masks = []

        for i in range(len(data)):
            conversation = data[i]

            labels.extend(conversation["labels"])
            input_ids.extend(conversation["input_ids"])
            attention_masks.extend(conversation["attention_masks"])

        labels = np.array(labels)
        input_ids = np.array(input_ids)
        attention_masks = np.array(attention_masks)
        return {
            "labels": labels,
            "input_ids": input_ids,
            "attention_masks": attention_masks,
        }

    train = concatanate_turns(train)
    validation = concatanate_turns(validation)
    test = concatanate_turns(test)

    return train, validation, test

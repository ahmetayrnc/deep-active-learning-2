import os
from typing import List, Tuple, Type, TypedDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
import pickle
from transformers import BatchEncoding


class Metrics(TypedDict):
    accuracy: float
    f1: float
    precision: float
    recall: float


class Dialogue(TypedDict):
    dialogue_id: str
    turns: List[BatchEncoding]
    labels: List[int]


PickledDataset = List[Dialogue]


class Data:
    def __init__(
        self,
        train: PickledDataset,
        test: PickledDataset,
        handler: Type[Dataset],
    ):
        self.train = np.array(train)
        self.test = np.array(test)

        self.handler = handler
        self.n_pool = len(train)
        self.n_test = len(test)

        self.labeled_idxs: np.ndarray = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num: int) -> None:
        # generate initial labeled pool
        tmp_idxs: np.ndarray = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_data(self) -> Tuple[np.ndarray, Dataset]:
        labeled_idxs: np.ndarray = np.arange(self.n_pool)[self.labeled_idxs]
        indexed: PickledDataset = self.train[labeled_idxs]
        return labeled_idxs, self.handler(indexed)

    def get_unlabeled_data(self) -> Tuple[np.ndarray, Dataset]:
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        indexed: PickledDataset = self.train[unlabeled_idxs]
        return unlabeled_idxs, self.handler(indexed)

    def get_train_data(self) -> Tuple[np.ndarray, Dataset]:
        return self.labeled_idxs.copy(), self.handler(self.train)

    def get_test_data(self) -> Dataset:
        return self.handler(self.test)

    def cal_test_acc(self, preds: np.ndarray) -> float:
        v_extract_labels = np.vectorize(lambda x: x["labels"], otypes=[List[int]])
        labels_list = v_extract_labels(self.test)
        y_true = np.concatenate(labels_list)
        y_pred = preds

        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def cal_test_metrics(self, preds: np.ndarray) -> Metrics:
        v_extract_labels = np.vectorize(lambda x: x["labels"], otypes=[List[int]])
        labels_list = v_extract_labels(self.test)
        y_true = np.concatenate(labels_list)
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


def get_SWDA() -> Tuple[List[Dialogue], List[Dialogue]]:
    pickle_data_dir = "data/swda/pickle"

    if not os.path.isfile(f"{pickle_data_dir}/train.pickle"):
        raise FileNotFoundError(
            f"{pickle_data_dir}/train.pickle does not exist. Please run convert scripts first."
        )

    with open(f"{pickle_data_dir}/train.pickle", "rb") as f:
        train = pickle.load(f)

    with open(f"{pickle_data_dir}/test.pickle", "rb") as f:
        test = pickle.load(f)

    return train, test

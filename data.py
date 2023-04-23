import os
from typing import List, Tuple, Type, TypedDict
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, Dataset as HF_Dataset
import pandas as pd
from sklearn.metrics import classification_report


class Metrics(TypedDict):
    accuracy: float
    f1: float
    precision: float
    recall: float


MyDataset = Tuple[List[List[str]], List[List[int]]]


class Data:
    def __init__(
        self,
        train: MyDataset,
        test: MyDataset,
        handler: Type[Dataset],
    ):
        self.train = np.array(train[0], dtype=object), np.array(train[1], dtype=object)
        self.test = np.array(test[0], dtype=object), np.array(test[1], dtype=object)

        self.handler = handler
        self.n_pool = len(train[1])
        self.n_test = len(test[1])

        self.labeled_idxs: np.ndarray = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num: int) -> None:
        # generate initial labeled pool
        tmp_idxs: np.ndarray = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_data(self) -> Tuple[np.ndarray, Dataset]:
        labeled_idxs: np.ndarray = np.arange(self.n_pool)[self.labeled_idxs]
        indexed: MyDataset = self.train[0][labeled_idxs], self.train[1][labeled_idxs]
        return labeled_idxs, self.handler(indexed)

    def get_unlabeled_data(self) -> Tuple[np.ndarray, Dataset]:
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        indexed: MyDataset = (
            self.train[0][unlabeled_idxs],
            self.train[1][unlabeled_idxs],
        )
        return unlabeled_idxs, self.handler(indexed)

    def get_train_data(self) -> Tuple[np.ndarray, Dataset]:
        return self.labeled_idxs.copy(), self.handler(self.train)

    def get_test_data(self) -> Dataset:
        return self.handler(self.test)

    def cal_test_metrics(self, preds: np.ndarray) -> Metrics:
        y_true = np.concatenate(self.test[1])
        y_pred = preds

        metrics = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        # also print the metrics
        print(classification_report(y_true, y_pred, zero_division=0))

        return {
            "accuracy": metrics["accuracy"],
            "f1": metrics["macro avg"]["f1-score"],
            "recall": metrics["macro avg"]["recall"],
            "precision": metrics["macro avg"]["precision"],
        }


def convert(dataset: HF_Dataset) -> MyDataset:
    def process_group(group):
        group_df = pd.DataFrame(group[1])
        if len(group_df) > 512:
            print(f"skipped dialogue: {group[0]}")
            return None

        turns = group_df["Utterance"].tolist()
        labels = group_df["Label"].tolist()

        return turns, labels

    df = dataset.to_pandas()
    grouped = df.groupby("Dialogue_ID")
    results = list(map(process_group, grouped))
    all_turns, all_labels = zip(*[r for r in results if r is not None])
    return all_turns, all_labels


def get_silicone_dataset(dataset_name: str):
    main_dataset_name = "silicone"
    dataset_dir = f"data/{dataset_name}"

    # Load the dataset
    if os.path.exists(dataset_dir):
        # load the dataset from disk
        dataset = load_from_disk(dataset_dir)
        print("Dataset loaded from disk")
    else:
        # load the dataset from Hugging Face and save it to disk
        dataset = load_dataset(main_dataset_name, dataset_name)
        dataset.save_to_disk(dataset_dir)
        print("Dataset loaded from Hugging Face and saved to disk")

    train = convert(dataset["train"])
    test = convert(dataset["test"])
    return train, test


def get_SWDA() -> Tuple[MyDataset, MyDataset]:
    return get_silicone_dataset("swda")


def get_DYDA() -> Tuple[MyDataset, MyDataset]:
    return get_silicone_dataset("dyda_da")


def get_CSABS():
    def convert(dataset: HF_Dataset) -> MyDataset:
        return dataset["sentences"], dataset["labels"]

    dataset_name = "allenai/csabstruct"
    dataset_dir = f"data/{dataset_name}"

    # Load the dataset
    if os.path.exists(dataset_dir):
        # load the dataset from disk
        dataset = load_from_disk(dataset_dir)
        print("Dataset loaded from disk")
    else:
        # load the dataset from Hugging Face and save it to disk
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(dataset_dir)
        print("Dataset loaded from Hugging Face and saved to disk")

    train = convert(dataset["train"])
    test = convert(dataset["test"])
    return train, test

import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, A_train, A_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.A_train = A_train
        self.A_test = A_test

        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(
            self.X_train[labeled_idxs],
            self.Y_train[labeled_idxs],
            self.A_train[labeled_idxs],
        )

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(
            self.X_train[unlabeled_idxs],
            self.Y_train[unlabeled_idxs],
            self.A_train[unlabeled_idxs],
        )

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(
            self.X_train, self.Y_train, self.A_train
        )

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test, self.A_test)

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test

    def cal_test_metrics(self, preds):
        y_true = self.Y_test
        y_pred = preds

        accuracy = accuracy_score(y_true, y_pred)

        micro_recall = recall_score(y_true, y_pred, average="micro")
        micro_precision = precision_score(y_true, y_pred, average="micro")
        micro_f1 = f1_score(y_true, y_pred, average="micro")

        macro_recall = recall_score(y_true, y_pred, average="macro")
        macro_precision = precision_score(y_true, y_pred, average="macro")
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        return {
            "accuracy": accuracy,
            "micro_recall": micro_recall,
            "micro_precision": micro_precision,
            "micro_f1": micro_f1,
            "macro_recall": macro_recall,
            "macro_precision": macro_precision,
            "macro_f1": macro_f1,
        }


def get_MNIST(handler):
    raw_train = datasets.MNIST("./data/MNIST", train=True, download=True)
    raw_test = datasets.MNIST("./data/MNIST", train=False, download=True)
    return Data(
        raw_train.data[:40000],
        raw_train.targets[:40000],
        raw_test.data[:40000],
        raw_test.targets[:40000],
        handler,
    )


def get_SWDA():
    torch_data_dir = "data/swda/torch"

    if os.path.isfile(f"{torch_data_dir}/X_tr.pt"):
        X_tr = torch.load(f"{torch_data_dir}/X_tr.pt")
        Y_tr = torch.load(f"{torch_data_dir}/Y_tr.pt")
        X_te = torch.load(f"{torch_data_dir}/X_te.pt")
        Y_te = torch.load(f"{torch_data_dir}/Y_te.pt")
        A_tr = torch.load(f"{torch_data_dir}/A_tr.pt")
        A_te = torch.load(f"{torch_data_dir}/A_te.pt")

        return X_tr, Y_tr, X_te, Y_te, A_tr, A_te

    return None
    # data path
    path = "data"

    # dataset directory
    dataset_dir = os.path.join(path, "swda")
    print(f"Dataset: {dataset_dir}")
    print(f"Loading dataset...")

    # Load the dataset
    if os.path.exists(dataset_dir):
        # load the dataset from disk
        dataset = load_from_disk(dataset_dir)
        print("Dataset loaded from disk")
    else:
        # load the dataset from Hugging Face and save it to disk
        dataset = load_dataset("silicone", "swda")
        dataset.save_to_disk(dataset_dir)
        print("Dataset loaded from Hugging Face and saved to disk")

    # tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", use_fast=True)
    print("Tokenizer loaded")

    max_len = 128

    # Define preprocessing function
    def preprocess(example):
        text = example["Utterance"]
        label = example["Label"]
        tokens = tokenizer.encode_plus(
            text,  # Sentence to tokenize
            add_special_tokens=True,  # Add special tokens [CLS] and [SEP]
            # padding="max_length",   # Pad to fixed length
            # max_length=20,          # Max length of the sequence
            # truncation=True,        # Truncate to max length if needed
            return_attention_mask=True,  # Generate attention mask
            return_tensors="pt",  # Return PyTorch tensors
        )
        return (
            tokens["input_ids"][0],  # because we only have one sentence
            torch.tensor(label),
            tokens["attention_mask"][0],  # because we only have one sentence
        )

    print("Preprocessing training data...")

    # Apply preprocessing function to dataset
    X_tr = []
    Y_tr = []
    A_tr = []
    for example in dataset["train"]:
        x, y, a = preprocess(example)
        X_tr.append(x)
        Y_tr.append(y)
        A_tr.append(a)

    # pad first element to 128
    X_tr[0] = nn.ConstantPad1d((0, max_len - X_tr[0].shape[0]), 0)(X_tr[0])
    A_tr[0] = nn.ConstantPad1d((0, max_len - A_tr[0].shape[0]), 0)(A_tr[0])

    # Pad sequences and create dataset
    X_tr = pad_sequence(X_tr, batch_first=True, padding_value=0)
    Y_tr = torch.tensor(Y_tr)
    A_tr = pad_sequence(A_tr, batch_first=True, padding_value=0)

    print("Training data preprocessed")

    print("Preprocessing test data...")
    # Apply preprocessing function to dataset
    X_te = []
    Y_te = []
    A_te = []
    for example in dataset["test"]:
        x, y, a = preprocess(example)
        X_te.append(x)
        Y_te.append(y)
        A_te.append(a)

    # pad first element to 128
    X_te[0] = nn.ConstantPad1d((0, max_len - X_te[0].shape[0]), 0)(X_te[0])
    A_te[0] = nn.ConstantPad1d((0, max_len - A_te[0].shape[0]), 0)(A_te[0])

    # Pad sequences and create dataset
    X_te = pad_sequence(X_te, batch_first=True, padding_value=0)
    Y_te = torch.tensor(Y_te)
    A_te = pad_sequence(A_te, batch_first=True, padding_value=0)

    print("Test data preprocessed")

    print(
        f"Dataset shapes: {X_tr.shape}, {Y_tr.shape}, {X_te.shape}, {Y_te.shape} {A_tr.shape}, {A_te.shape}"
    )

    os.makedirs(torch_data_dir, exist_ok=True)

    torch.save(X_tr, f"{torch_data_dir}/X_tr.pt")
    torch.save(Y_tr, f"{torch_data_dir}/Y_tr.pt")
    torch.save(X_te, f"{torch_data_dir}/X_te.pt")
    torch.save(Y_te, f"{torch_data_dir}/Y_te.pt")
    torch.save(A_tr, f"{torch_data_dir}/A_tr.pt")
    torch.save(A_te, f"{torch_data_dir}/A_te.pt")

    return X_tr, Y_tr, X_te, Y_te, A_tr, A_te

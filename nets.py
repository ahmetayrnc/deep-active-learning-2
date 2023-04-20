from typing import Type, TypedDict, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from handlers import string_collator
from tqdm.auto import tqdm
import torch.nn.functional as F


# Define the type of each field
class TrainArgs(TypedDict):
    batch_size: int
    num_workers: int


class TestArgs(TypedDict):
    batch_size: int
    num_workers: int


class OptimizerArgs(TypedDict):
    lr: float
    momentum: float


class DatasetArgs(TypedDict):
    n_epoch: int
    n_labels: int
    model_name: str
    train_args: TrainArgs
    test_args: TestArgs
    optimizer_args: OptimizerArgs


Params = Dict[str, DatasetArgs]


class Net:
    def __init__(
        self, net: Type[nn.Module], params: DatasetArgs, device: str, n_epoch: int
    ):
        self.net = net
        self.params = params
        self.device = device
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        self.n_epoch = n_epoch

    def train(self, data: Dataset, epoch_callback=None):
        n_epoch = self.n_epoch
        self.model = self.net(
            self.params["model_name"],
            self.params["n_labels"],
            # self.params["max_turn_length"],
        )
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(), **self.params["optimizer_args"]
        )
        loader = DataLoader(
            data, shuffle=True, collate_fn=string_collator, **self.params["train_args"]
        )

        for epoch in range(n_epoch):
            epoch_loss = 0.0

            for batch_dialogues, batch_labels in tqdm(loader, ncols=100):
                batch_labels = batch_labels.to(self.device)
                logits, _ = self.model(batch_dialogues)
                loss = self.loss_function(
                    logits.view(-1, self.params["n_labels"]), batch_labels.view(-1)
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

            if epoch_callback:
                epoch_callback()
            # print(f"Epoch {epoch + 1}/{n_epoch} - Loss: {epoch_loss / len(loader)}")

    def predict(self, data: Dataset) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(
            data, shuffle=False, collate_fn=string_collator, **self.params["test_args"]
        )

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_dialogues, batch_labels in tqdm(loader):
                logits, _ = self.model(batch_dialogues)
                preds = torch.argmax(logits, dim=2).cpu().numpy()
                labels = batch_labels.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)
        all_labels = np.concatenate(all_labels, axis=None)
        all_preds = np.concatenate(all_preds, axis=None)

        return all_preds

    def predict_prob(self, data: Dataset) -> List[np.ndarray]:
        self.model.eval()

        loader = DataLoader(
            data, shuffle=False, collate_fn=string_collator, **self.params["test_args"]
        )

        all_probs = []
        with torch.no_grad():
            for batch_dialogues, batch_labels in tqdm(loader):
                logits, _ = self.model(batch_dialogues)
                probs = F.softmax(logits, dim=2)
                probs = probs.cpu().numpy()
                all_probs.extend(probs)

        # all_probs = np.concatenate(all_probs, axis=None)

        return all_probs

    # def predict_prob_dropout(self, data: Dataset, n_drop: int = 10):
    #     self.clf.train()
    #     probs = torch.zeros([len(data.labels), self.clf.n_class])
    #     loader = DataLoader(data, shuffle=False, **self.params["test_args"])
    #     for i in range(n_drop):
    #         with torch.no_grad():
    #             for input_ids, attention_mask, label, idxs in loader:
    #                 input_ids, attention_mask, label = (
    #                     input_ids.to(self.device),
    #                     attention_mask.to(self.device),
    #                     label.to(self.device),
    #                 )
    #                 logits, embeddings = self.clf(input_ids, attention_mask)
    #                 prob = F.softmax(logits, dim=1)
    #                 probs[idxs] += prob.cpu()
    #     probs /= n_drop
    #     return probs

    # def predict_prob_dropout_split(self, data: Dataset, n_drop: int = 10):
    #     self.clf.train()
    #     probs = torch.zeros([n_drop, len(data.labels), self.clf.n_class])
    #     loader = DataLoader(data, shuffle=False, **self.params["test_args"])
    #     for i in range(n_drop):
    #         with torch.no_grad():
    #             for input_ids, attention_mask, label, idxs in loader:
    #                 input_ids, attention_mask, label = (
    #                     input_ids.to(self.device),
    #                     attention_mask.to(self.device),
    #                     label.to(self.device),
    #                 )
    #                 logits, embeddings = self.clf(input_ids, attention_mask)
    #                 prob = F.softmax(logits, dim=1)
    #                 probs[i][idxs] += F.softmax(logits, dim=1).cpu()
    #     return probs

    # def get_embeddings(self, data: Dataset):
    #     self.clf.eval()
    #     embeddings = torch.zeros([len(data.labels), self.clf.get_embedding_dim()])
    #     loader = DataLoader(data, shuffle=False, **self.params["test_args"])
    #     with torch.no_grad():
    #         for input_ids, attention_mask, label, idxs in loader:
    #             input_ids, attention_mask, label = (
    #                 input_ids.to(self.device),
    #                 attention_mask.to(self.device),
    #                 label.to(self.device),
    #             )
    #             logits, embeddings1 = self.clf(input_ids, attention_mask)
    #             embeddings[idxs] = embeddings1.cpu()
    #     return embeddings

from typing import Type, TypedDict, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from torch.utils.data import Dataset
import numpy as np
from data import MyDataset
from handlers import string_collator
from sklearn.metrics import classification_report


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


class HierarchicalDialogueActClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(HierarchicalDialogueActClassifier, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, use_fast=True
        )
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.dialogue_transformer = AutoModel.from_pretrained(pretrained_model_name)

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        # move everything to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.dialogue_transformer.to(self.device)
        self.classifier.to(self.device)

    def forward(self, dialogues: List[List[str]]):
        dialogue_turns = dialogues[0]

        # Tokenize and move the encoding data to the device
        turn_encodings = [
            self.tokenizer(
                turn,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            ).to(self.device)
            for turn in dialogue_turns
        ]

        # Obtain turn embeddings
        turn_outputs = [self.encoder(**encoding) for encoding in turn_encodings]
        turn_hidden_states = [
            output.last_hidden_state[:, 0, :] for output in turn_outputs
        ]
        turn_embeddings = torch.stack(turn_hidden_states)

        # Permute the dimensions of the turn_embeddings tensor
        turn_embeddings = turn_embeddings.permute(1, 0, 2)

        # Pass the turn embeddings through the dialogue-level transformer
        dialogue_outputs = self.dialogue_transformer(inputs_embeds=turn_embeddings)
        dialogue_hidden_states = dialogue_outputs.last_hidden_state

        # Use the hidden state of each turn for classification
        logits = self.classifier(dialogue_hidden_states)

        # dialogue_hidden_states: hidden states
        return logits, dialogue_hidden_states


class Net:
    def __init__(self, net: Type[nn.Module], params: DatasetArgs, device: str):
        self.net = net
        self.params = params
        self.device = device
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    def train(self, data: Dataset):
        n_epoch = self.params["n_epoch"]
        self.model: HierarchicalDialogueActClassifier = self.net(
            self.params["model_name"], self.params["n_labels"]
        )
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), **self.params["optimizer_args"])
        loader = DataLoader(
            data, shuffle=True, collate_fn=string_collator, **self.params["train_args"]
        )

        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            epoch_loss = 0.0

            for batch_dialogues, batch_labels in loader:
                batch_labels = batch_labels.to(self.device)
                logits, _ = self.model(batch_dialogues)
                loss = self.loss_function(
                    logits.view(-1, self.params["n_labels"]), batch_labels.view(-1)
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{n_epoch} - Loss: {epoch_loss / len(loader)}")

    def predict(self, data: Dataset):
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

        mask = [label != -1 for label in all_labels]
        all_preds = [all_preds[i] for i, mask_value in enumerate(mask) if mask_value]
        all_labels = [all_labels[i] for i, mask_value in enumerate(mask) if mask_value]

        print("\nTest set classification report:")
        print(classification_report(all_labels, all_preds, zero_division=0))

        return all_preds

    # def predict_prob(self, data: Dataset):
    #     self.clf.eval()
    #     probs = torch.zeros([len(data.labels), self.clf.n_class])
    #     loader = DataLoader(data, shuffle=False, **self.params["test_args"])
    #     with torch.no_grad():
    #         for input_ids, attention_mask, label, idxs in loader:
    #             input_ids, attention_mask, label = (
    #                 input_ids.to(self.device),
    #                 attention_mask.to(self.device),
    #                 label.to(self.device),
    #             )
    #             logits, embeddings = self.clf(input_ids, attention_mask)
    #             prob = F.softmax(logits, dim=1)
    #             probs[idxs] = prob.cpu()
    #     return probs

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

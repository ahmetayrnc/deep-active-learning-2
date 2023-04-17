from typing import Type, TypedDict, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    AutoModel,
    BatchEncoding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import Dataset
from data import Dialogue
import numpy as np
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
    train_args: TrainArgs
    test_args: TestArgs
    optimizer_args: OptimizerArgs


Params = Dict[str, DatasetArgs]


class SWDA_Net(nn.Module):
    def __init__(self, n_class: int = 46) -> None:
        super(SWDA_Net, self).__init__()
        self.n_class = n_class
        model_name = "distilbert-base-cased"
        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=n_class,
            )
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        outputs: SequenceClassifierOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden_states: list[torch.FloatTensor] = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        return logits, last_hidden_state

    def get_embedding_dim(self):
        return self.model.config.hidden_dim


class HierarchicalDialogueActClassifier(nn.Module):
    def __init__(self):
        super(HierarchicalDialogueActClassifier, self).__init__()
        pretrained_model_name = "distilbert-base-cased"
        num_classes = 46

        self.encoder: PreTrainedModel = AutoModel.from_pretrained(pretrained_model_name)
        self.dialogue_transformer: PreTrainedModel = AutoModel.from_pretrained(
            pretrained_model_name
        )

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.dialogue_transformer.to(self.device)
        self.classifier.to(self.device)

    # Batch size 1
    def forward(self, dialogues: List[List[BatchEncoding]]):
        dialogue = dialogues[0]

        # Move the encoding data to the device
        turn_encodings = [encoding.to(self.device) for encoding in dialogue]

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

        return logits, dialogue_hidden_states


class Net:
    def __init__(self, net: Type[nn.Module], params: DatasetArgs, device: str):
        self.net = net
        self.params = params
        self.device = device
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        self.num_classes = 46

    def train(self, data: Dataset):
        n_epoch = self.params["n_epoch"]
        self.model: HierarchicalDialogueActClassifier = self.net()
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
                    logits.view(-1, self.num_classes), batch_labels.view(-1)
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{n_epoch} - Loss: {epoch_loss / len(loader)}")

    def predict(self, data: Dialogue):
        self.model.eval()
        preds = torch.zeros(len(data.labels), dtype=int)
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])

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

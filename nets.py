import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification


class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def train(self, data):
        # print("Training")
        n_epoch = self.params["n_epoch"]
        # print(f"Training for {n_epoch} epochs")
        # print("Net init")
        self.clf = self.net().to(self.device)
        # print("Net init done")
        self.clf.train()
        # print("Optimizer init")
        optimizer = optim.SGD(self.clf.parameters(), **self.params["optimizer_args"])

        # print("DataLoader init")
        loader = DataLoader(data, shuffle=True, **self.params["train_args"])
        # print("DataLoader init done")
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, a, idxs) in enumerate(loader):
                print(f"batch_idx: {batch_idx}")
                x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
                # print("moved to device")
                optimizer.zero_grad()
                # print("zero_grad")
                logits, embeddings = self.clf(x, a)
                # print("through net")
                loss = F.cross_entropy(logits, y)
                # print("loss")
                loss.backward()
                # print("backward")
                optimizer.step()
                # print("step")

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        with torch.no_grad():
            for x, y, a, idxs in loader:
                x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
                logits, embeddings = self.clf(x, a)
                pred = torch.argmax(logits, dim=1)
                preds[idxs] = pred.cpu()
        return preds

    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        with torch.no_grad():
            for x, y, a, idxs in loader:
                x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
                logits, embeddings = self.clf(x, a)
                prob = F.softmax(logits, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, a, idxs in loader:
                    x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
                    logits, embeddings = self.clf(x, a)
                    prob = F.softmax(logits, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, a, idxs in loader:
                    x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
                    logits, embeddings = self.clf(x)
                    prob = F.softmax(logits, dim=1)
                    probs[i][idxs] += F.softmax(logits, dim=1).cpu()
        return probs

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        with torch.no_grad():
            for x, y, a, idxs in loader:
                x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
                logits, embeddings1 = self.clf(x)
                embeddings[idxs] = embeddings1.cpu()
        return embeddings


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class SWDA_Net(nn.Module):
    def __init__(self, n_class=46):
        super(SWDA_Net, self).__init__()
        self.n_class = n_class
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-model",
            num_labels=n_class,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        return logits, last_hidden_state

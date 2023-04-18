from nets import Net
from data import Data
from torch.utils.data import Dataset


class Strategy:
    def __init__(self, dataset, net):
        self.dataset: Data = dataset
        self.net: Net = net

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def predict(self, data: Dataset):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data: Dataset):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data: Dataset, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data: Dataset, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs

    def get_embeddings(self, data: Dataset):
        embeddings = self.net.get_embeddings(data)
        return embeddings

from torch.utils.data import Dataset


class SWDA_Handler(Dataset):
    def __init__(self, X, Y, A, transform=None):
        self.X = X
        self.Y = Y
        self.A = A
        self.transform = transform

    def __getitem__(self, index):
        x, y, a = self.X[index], self.Y[index], self.A[index]
        return x, y, a, index

    def __len__(self):
        return len(self.X)


class Conversation_Handler(Dataset):
    def __init__(self, data):
        self.data = data

        self.labels = data["labels"]
        self.input_ids = data["input_ids"]
        self.attention_masks = data["attention_masks"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        label = self.labels[idx]

        return input_ids, attention_mask, label, idx

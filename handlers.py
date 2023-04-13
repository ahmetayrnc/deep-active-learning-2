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

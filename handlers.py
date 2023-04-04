from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class MNIST_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x.numpy(), mode="L")
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


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

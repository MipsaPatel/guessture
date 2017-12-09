import torch.nn as nn
import torch.nn.functional as func


class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.c1 = nn.Conv2d(1, 5, 5)
        self.p1 = nn.MaxPool2d(3)
        self.c2 = nn.Conv2d(5, 10, 5)
        self.p2 = nn.MaxPool2d(3)
        self.c3 = nn.Conv2d(10, 20, 5)
        self.p3 = nn.MaxPool2d(2)
        self.loss = func.nll_loss

    def forward(self, x):
        x = func.relu(self.p1(self.c1(x)))
        x = func.relu(self.p2(self.c2(x)))
        x = func.relu(self.p3(self.c3(x)))
        x = x.view(-1, 320)

        return x

import torch.nn as nn
import torch.nn.functional as func


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(1, 5, 5)
        self.p1 = nn.MaxPool2d(3)
        self.c2 = nn.Conv2d(5, 10, 5)
        self.p2 = nn.MaxPool2d(3)
        self.c3 = nn.Conv2d(10, 20, 5)
        self.p3 = nn.MaxPool2d(2)
        self.l1 = nn.Linear(320, 100)
        self.l2 = nn.Linear(100, 63)
        self.loss = func.nll_loss

    def forward(self, x):
        x = func.relu(self.p1(self.c1(x)))
        x = func.relu(self.p2(self.c2(x)))
        x = func.relu(self.p3(self.c3(x)))
        x = x.view(-1, 320)
        x = func.relu(self.l1(x))
        x = func.dropout(x, training=self.training)
        x = self.l2(x)
        return func.log_softmax(x)

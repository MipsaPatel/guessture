import torch.nn.functional as F
from torch import nn


class NeuralNet(nn.Module):
    """
    The neural network being used.
    Layer       Parameters    Output size
    Input                   :  1 x 128 x 128
    Convolution (1, 5, 5)   :  4 x 124 x 124
    MaxPool     (3, 3)      :  4 x  41 x  41
    ReLU                    :  4 x  41 x  41
    Convolution (5, 10, 5)  : 16 x  37 x  37
    MaxPool     (3, 3)      : 16 x  12 x  12
    ReLU                    : 16 x  12 x  12
    Convolution (10, 16, 5) : 64 x   8 x   8
    MaxPool     (4, 4)      : 64 x   2 x   2
    ReLU                    : 64 x   2 x   2
             reshape        :  1 x 256
    LSTM        (256, 256)  :  1 x 256
    Linear      (256, 128)  :  1 x 128
    Linear      (128, 63)   :  1 x  63
    """

    def __init__(self):
        """
        Initialize the layers.
        """
        super().__init__()
        # Set 1
        self.c1 = nn.Conv2d(1, 4, 5)
        self.p1 = nn.MaxPool2d(3)
        # Set 2
        self.c2 = nn.Conv2d(4, 16, 5)
        self.p2 = nn.MaxPool2d(3)
        # Set 3
        self.c3 = nn.Conv2d(16, 64, 5)
        self.p3 = nn.MaxPool2d(4)
        # Set 4
        self.lstm = nn.LSTM(256, 256)
        # Set 5
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 63)
        # loss function to be used
        self.loss = F.nll_loss

    def forward(self, x, rnn=True):
        # Using ReLU as activation function
        x = F.relu(self.p1(self.c1(x)))
        x = F.relu(self.p2(self.c2(x)))
        x = F.relu(self.p3(self.c3(x)))
        # reshape
        x = x.view(-1, 1, 256)
        if rnn:  # skip RNN while training CNN
            _, (x, _) = self.lstm(x)
        x = F.relu(self.l1(x))
        x = F.dropout(x, training=self.training)
        # Reduce the dimensions in torch (3 to 2)
        x = F.log_softmax(self.l2(x).squeeze(1))
        return x

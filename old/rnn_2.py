import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as func


class RNN_2(nn.Module):
    def __init__(self):
        super(RNN_2, self).__init__()
        self.hidden_size = 256
        self.num_layers = 1

        self.lstm = nn.LSTM(320, self.hidden_size, self.num_layers, batch_first=True)

        self.l1 = nn.Linear(self.hidden_size, 128)
        self.l2 = nn.Linear(128, 63)

        self.loss = nn.MSELoss

    def step(self, input, hidden=None):
        input = input.view(1, -1).unsqueeze(1)
        output, hidden = self.lstm(input, hidden)

        output = func.relu(self.l1(output.squeeze(1)))
        output = self.l2(output.squeeze(1))

        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 63))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return output

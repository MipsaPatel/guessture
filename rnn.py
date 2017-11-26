import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = 256
        self.num_layers = 6

        self.input = nn.Linear(63, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 63)

        self.loss = nn.CrossEntropyLoss

    def step(self, input, hidden=None):
        input = self.input(input.view(1, -1)).unsqueeze(1)
        # print("Input: ", input)
        output, hidden = self.lstm(input, hidden)
        # print("Output: ", output, hidden)
        output = self.fc(output.squeeze(1))
        # print("FInal output: ", output)

        return output, hidden

    # def forward(self, x):
    #     h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
    #     c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
    #
    #     out, _ = self.lstm(x, (h0, c0))
    #
    #     out = self.fc(out[:, -1, :])
    #     return out

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 63))
        # print("Steps: ", steps)
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, None)
            outputs[i] = output
        # print("Outputs: ", outputs)
        return output

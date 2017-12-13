import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from helper import get_action
from load import SignVideo
from params import Parameters

P = Parameters(cuda=torch.cuda.is_available())


def predict(model, loader):
    for data in loader:
        if P.cuda:
            data = data.cuda()

        data = Variable(data)
        output = model(data)
        return output.max(1)[1].data[0]


if __name__ == '__main__':
    path = os.path.abspath(input('Enter the path to the video: ').strip())
    print('Loading video...')
    data = SignVideo(path, frame_skip=P.frame_skip, start=P.frame_interval[0],
                     end=P.frame_interval[1], transform=P.transforms, target=False)
    data_loader = DataLoader(data, batch_size=len(data), **P.kwargs)

    print('Loading model...')
    NN_model = torch.load(P.model_path)
    if P.cuda:
        print('Using CUDA...')
        NN_model = NN_model.cuda()

    output = predict(NN_model, data_loader)

    print(output, get_action(output))

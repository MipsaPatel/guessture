import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .helper import get_action, is_video
from .load import SignVideo
from .models import NeuralNet
from .params import Parameters

P = Parameters(cuda=torch.cuda.is_available())


def load_model(path=P.model_path):
    """
    Load the model from path.
    :param path: The path to the model.
    :return: The loaded model.
    """
    print("Loading model from '%s'..." % path)
    model = NeuralNet()
    if P.cuda:
        print('Using CUDA...')
        model = model.cuda()
    model.load_state_dict(torch.load(path))
    return model


def load_video(path):
    """
    Creates a data loader instance for the given video.
    :param path: The path to the video.
    :return: The data loader instance.
    """
    data = SignVideo(path, frame_skip=P.frame_skip, start=P.frame_interval[0],
                     end=P.frame_interval[1], transform=P.transforms, target=False)

    return DataLoader(data, batch_size=len(data), **P.kwargs)


def predict(model, path):
    """
    Compute the output of the model for the given video.
    :param model: The model to be used.
    :param path: The path to the video.
    :return: The output of the model.
    """
    # Ensure that the file is a video
    if not is_video(path):
        return 'Error: Not a video file.'
    # create a loader for the video
    loader = load_video(path)
    # compute the output
    for data in loader:
        if P.cuda:
            data = data.cuda()

        data = Variable(data)
        output = model(data)
        return get_action(output.max(1)[1].data[0])

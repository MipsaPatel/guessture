import os

from torchvision import transforms


class Parameters:
    """
    A class to hold all parameters used by the model.
    """

    def __init__(self, cuda=False):
        # is CUDA available?
        self.cuda = cuda

        # root directory that contains all data files.
        self.data_dir = os.path.abspath('../sample')
        self.train_data = os.path.join(self.data_dir, 'train')
        self.test_data = os.path.join(self.data_dir, 'test')

        # arguments to be used if CUDA is available
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        # Data parameters
        self.frame_interval = 0.25, 0.75
        self.frame_skip = 2

        self.test_size = 0.3

        # transformations to be applied to each frame
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),                        # for crop and scale
            transforms.CenterCrop((240, 240)),              # change resolution, eliminate sides
            transforms.Scale(128),                          # scale it down
            transforms.Lambda(lambda x: x.convert('L')),    # gray scale
            transforms.ToTensor()                           # make it easier for pyTorch
        ])

        # Training parameters
        self.learning_rate = 0.001
        self.momentum = 0.5

        # Number of epochs
        self.cnn_epochs = 30
        self.rnn_epochs = 50

        # Number of batches between 2 output logs
        self.log_interval = 50
        # Number of epochs between 2 copies of model
        self.save_interval = 5

        # Load a saved model
        self.model_path = 'model.pth'
        self.load_model = False

        # Training
        self.train_cnn = True
        self.train_rnn = True

        self.train = self.train_cnn or self.train_rnn

        # cnn batch size
        self.cnn_train_batch_size = 128
        self.cnn_test_batch_size = 128

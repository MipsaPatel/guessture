import torch
from torch.autograd import Variable
from torchvision import transforms
from torch import nn

from rnn import RNN
from cnn import CNN
from frameloader import train_test_loader


# # # # # # # # # # # # CUDA PARAMETERS # # # # # # # # # # # #

CUDA = True  # Set this to False if you don't want CUDA to be used
NUM_WORKERS = 1

CUDA = CUDA and torch.cuda.is_available()


# # # # # # # # # # # DATA SET PARAMETERS # # # # # # # # # # #

# data set to use
DATA_DIR = '../sample'

# image parameters
CENTER_CROP_SIZE = 240, 240
IMAGE_SCALE = 128

# batch sizes for train and test
BATCH_SIZE = 32
TEST_BATCH_SIZE = BATCH_SIZE

# start and end of video, as ratio
FRAME_INTERVAL = 0.25, 0.75

# number of frames to skip between 2 frames (default 0)
FRAME_SKIP = 9

# test split size (default 0.3 for 70:30 split)
TEST_SIZE = 0.3

# transformations to be applied to each frame
transform_list = transforms.Compose([
    transforms.ToPILImage(),                        # for crop and scale
    transforms.CenterCrop(CENTER_CROP_SIZE),        # change resolution, eliminate sides
    transforms.Scale(IMAGE_SCALE),                  # scale it down
    transforms.Lambda(lambda x: x.convert('L')),    # gray scale
    transforms.ToTensor()                           # make it easier for pyTorch
])

# Additional args to DataLoader
KWArgs = {'num_workers': NUM_WORKERS, 'pin_memory': True} if CUDA else {}

print('Loading data...', flush=True, end=' ')

train_loader, test_loader = train_test_loader(DATA_DIR,
                                              batch_size=BATCH_SIZE,
                                              frame_skip=FRAME_SKIP,
                                              frame_interval=FRAME_INTERVAL,
                                              transform=transform_list,
                                              test_batch_size=TEST_BATCH_SIZE,
                                              test_size=TEST_SIZE,
                                              **KWArgs)

print('Done')


# # # # # # # # # # # # MODEL PARAMETERS # # # # # # # # # # # #

LEARNING_RATE = 0.01
MOMENTUM = 0.5

LOG_INTERVAL = 100

EPOCHS = 2

# Parameters for RNN
num_classes = 63
hidden_size = 256
input_size = 63
num_layers = 3

rnn_model = RNN(input_size, num_layers, num_classes, hidden_size)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)


# # # # # # # # # # # # # # DETAILS # # # # # # # # # # # # # #

print('Directory:', DATA_DIR)
print('Length: Train: (%d); Test: (%d)' % (len(train_loader.batch_sampler.sampler),
                                           len(test_loader.batch_sampler.sampler)))
print('Batch size:', BATCH_SIZE)
print('Frames: (%.2f, %.2f) with' % FRAME_INTERVAL, FRAME_SKIP, 'skip')
print('Learning Rate:', LEARNING_RATE)
print('Epochs:', EPOCHS)


# # # # # # # # # # # TRAINING AND TESTING # # # # # # # # # # #

def rnn_train(model, rnn_model, epoch):
    rnn_model.train()

    for batch, (video, target) in enumerate(data_loader):
        for frame in video:
            pass


def train(model, epoch):
    model.train()
    print('Training:')
    for batch, (data, target) in enumerate(train_loader):
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).long()
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()
        if not batch % LOG_INTERVAL:
            print('\nTrain:')
            print('Epoch:', epoch)
            print('Batch:', batch)
            print('Loss:', loss.data[0])


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    print('\nTesting:')
    for batch, (data, target) in enumerate(train_loader):
        if not batch % LOG_INTERVAL:
            print('Batch:', batch)
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += model.loss(output, target, size_average=False).data[0]
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

    test_length = len(test_loader.batch_sampler.sampler)
    test_loss /= test_length
    print('\nTest:')
    print('Average loss:', round(test_loss, 4))
    print('Accuracy: ', correct, '/', test_length, ' (', round(100.0 * correct / test_length, 1), '%)', sep='')


# # # # # # # # # # # # # LOAD OR TRAIN # # # # # # # # # # # #

# change to the file you want to use
LOAD_PATH = 'model.pth'

# set to true while training RNN
LOAD_FROM_PATH = False


# # # # # # # # # # # # # RUN CNN MODEL # # # # # # # # # # # #

if LOAD_FROM_PATH:
    CNN_model = torch.load(LOAD_PATH)
else:
    CNN_model = CNN()
    if CUDA:
        CNN_model = CNN_model.cuda()

    optimizer = torch.optim.SGD(CNN_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    for e in range(EPOCHS):
        print('\n\nEpoch:', e)
        train(CNN_model, e)
        test(CNN_model)

    import time  # avoid overwriting an existing file
    path = 'model' + str(int(time.time() * 1000)) + '.pth'
    print("Saving model to '%s'..." % path, flush=True, end=' ')
    with open(path, 'wb') as f:
        torch.save(CNN_model, f)
    print('Done')


# # # # # # # # # # # # # TEST CNN MODEL # # # # # # # # # # # #

test(CNN_model)

# # # # # # # # # # # # # RUN RNN MODEL # # # # # # # # # # # #

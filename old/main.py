import time

import torch
from rnn import RNN
from torch.autograd import Variable
from torchvision import transforms
from videoloader import RandomFrameLoader, train_test_data_loader, VideoLoader

from old.cnn import CNN

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
FRAME_SKIP = 2

# test split size (default 0.3 for 70:30 split)
TEST_SIZE = 0.3

# number of batches to process between two logs
LOG_INTERVAL = 100

# transformations to be applied to each frame
TRANSFORM_LIST = transforms.Compose([
    transforms.ToPILImage(),                        # for crop and scale
    transforms.CenterCrop(CENTER_CROP_SIZE),        # change resolution, eliminate sides
    transforms.Scale(IMAGE_SCALE),                  # scale it down
    transforms.Lambda(lambda x: x.convert('L')),    # gray scale
    transforms.ToTensor()                           # make it easier for pyTorch
])

# Additional args to DataLoader
KWArgs = {'num_workers': NUM_WORKERS, 'pin_memory': True} if CUDA else {}

print('Loading data...', flush=True, end=' ')

random_train_loader, random_test_loader = train_test_data_loader(RandomFrameLoader(DATA_DIR,
                                                                                   frame_skip=FRAME_SKIP,
                                                                                   frame_interval=FRAME_INTERVAL,
                                                                                   transform=TRANSFORM_LIST),
                                                                 batch_size=BATCH_SIZE,
                                                                 test_batch_size=TEST_BATCH_SIZE,
                                                                 test_size=TEST_SIZE, **KWArgs)

video_train_loader, video_test_loader = train_test_data_loader(VideoLoader(DATA_DIR,
                                                                           frame_skip=FRAME_SKIP,
                                                                           frame_interval=FRAME_INTERVAL,
                                                                           transform=TRANSFORM_LIST,
                                                                           batch_size=BATCH_SIZE,
                                                                           **KWArgs),
                                                               batch_size=1,
                                                               test_batch_size=1,
                                                               test_size=TEST_SIZE, collate_fn=lambda x: x, **KWArgs)

print('Done')


# # # # # # # # # # # CNN MODEL PARAMETERS # # # # # # # # # # #

CNN_LEARNING_RATE = 0.01
CNN_MOMENTUM = 0.5
CNN_EPOCHS = 20


# # # # # # # # # # # RNN MODEL PARAMETERS # # # # # # # # # # #

RNN_LEARNING_RATE = 0.01
RNN_MOMENTUM = 0.5
RNN_EPOCHS = 20


# # # # # # # # # # # # # # DETAILS # # # # # # # # # # # # # #

print('Directory:', DATA_DIR)
print('Length: Train: (%d); Test: (%d)' % (len(random_train_loader.batch_sampler.sampler),
                                           len(random_test_loader.batch_sampler.sampler)))
print('Batch size:', BATCH_SIZE)
print('Frames: (%.2f, %.2f) with' % FRAME_INTERVAL, FRAME_SKIP, 'skip')
print('CNN:')
print('Learning Rate:', CNN_LEARNING_RATE)
print('Epochs:', CNN_EPOCHS)
print('RNN:')
print('Learning Rate:', RNN_LEARNING_RATE)
print('Epochs:', RNN_EPOCHS)


# # # # # # # # # # TRAINING AND TESTING CNN  # # # # # # # # # #

def train_cnn(model, epoch, loader):
    model.train()
    print('Training:')
    for batch, (data, target) in enumerate(loader):
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


def test_cnn(model, epoch, loader):
    model.eval()
    test_loss = 0
    correct = 0
    print('\nTesting:')
    for batch, (data, target) in enumerate(loader):
        if not batch % LOG_INTERVAL:
            print('Batch:', batch)
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += model.loss(output, target, size_average=False).data[0]
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

    test_length = len(loader.batch_sampler.sampler)
    test_loss /= test_length
    accuracy = round(100.0 * correct / test_length, 1)
    print('\nTest:')
    print('Average loss:', round(test_loss, 4))
    print('Accuracy: ', correct, '/', test_length, ' (', round(100.0 * correct / test_length, 1), '%)', sep='')

    if epoch and epoch % 5 == 0:
        path = 'cnn_accuracy' + '_' + str(epoch) + '_' + str(int(time.time() * 1000)) + '.txt'
        print("Saving accuracy to '%s'..." % path, flush=True, end=' ')
        with open(path, 'w') as f:
            f.write(str(accuracy))


# # # # # # # # # # TRAINING AND TESTING RNN # # # # # # # # # #

def train_rnn(model, cnn, epoch, loader):
    model.train()
    for batch, videos in enumerate(loader):
        for video in videos:
            cnn_output = None
            for data, target in video:
                if CUDA:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target).long()

                if cnn_output is None:
                    cnn_output = cnn(data)
                else:
                    torch.cat((cnn_output, cnn(data)), dim=0)

            RNN_optimizer.zero_grad()
            rnn_output = model(cnn_output)
            loss = RNN_criterion(rnn_output, target[0])
            loss.backward()
            RNN_optimizer.step()

        if not batch % LOG_INTERVAL:
            print('\nRNN Train:')
            print('Epoch:', epoch)
            print('Batch:', batch)
            print('Loss:', loss.data[0])


def test_rnn(model, cnn, loader):
    model.eval()
    test_loss = 0
    correct = 0

    for batch, videos in enumerate(loader):
        for video in videos:
            cnn_output = None
            for data, target in video:
                if CUDA:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target).long()

                if cnn_output is None:
                    cnn_output = cnn(data)
                else:
                    torch.cat((cnn_output, cnn(data)), dim=0)

            rnn_output = model(cnn_output)
            test_loss += RNN_criterion(rnn_output, target[0]).data[0]
            prediction = rnn_output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(target[0].data.view_as(prediction)).cpu().sum()

    test_length = len(loader.batch_sampler.sampler)
    test_loss /= test_length
    print('\nTest:')
    print('Average loss:', round(test_loss, 4))
    print('Accuracy: ', correct, '/', test_length, ' (', round(100.0 * correct / test_length, 1), '%)', sep='')


# # # # # # # # # # # # # LOAD OR TRAIN # # # # # # # # # # # #

# change to the file you want to use
CNN_LOAD_PATH = 'cnn_model.pth'
RNN_LOAD_PATH = 'model.pth'

# set to true while training RNN
CNN_LOAD_FROM_PATH = False

# set to true when reusing RNN
RNN_LOAD_FROM_PATH = False


# # # # # # # # # # # # # RUN CNN MODEL # # # # # # # # # # # #

if CNN_LOAD_FROM_PATH:
    CNN_model = torch.load(CNN_LOAD_PATH)
else:
    CNN_model = CNN()
    if CUDA:
        CNN_model = CNN_model.cuda()

    optimizer = torch.optim.SGD(CNN_model.parameters(), lr=CNN_LEARNING_RATE, momentum=CNN_MOMENTUM)

    for e in range(CNN_EPOCHS):
        print('\n\nEpoch:', e)
        train_cnn(CNN_model, e, random_train_loader)
        test_cnn(CNN_model, e, random_test_loader)

        if e and e % 5 == 0:
            path = 'cnn_model' + '_' + str(e) + '_' + str(int(time.time() * 1000)) + '.pth'
            print("Saving model to '%s'..." % path, flush=True, end=' ')
            with open(path, 'wb') as f:
                torch.save(CNN_model, f)

    print('Done')


# # # # # # # # # # # # # TEST CNN MODEL # # # # # # # # # # # #

test_cnn(CNN_model, CNN_EPOCHS, random_test_loader)


# # # # # # # # # # # # # RUN RNN MODEL # # # # # # # # # # # #

if RNN_LOAD_FROM_PATH:
    RNN_model = torch.load(RNN_LOAD_PATH)
else:
    RNN_model = RNN()
    if CUDA:
        RNN_model = RNN_model.cuda()

    RNN_criterion = torch.nn.CrossEntropyLoss()
    RNN_optimizer = torch.optim.Adam(RNN_model.parameters(), lr=RNN_LEARNING_RATE)

    print("RNN")

    for e in range(RNN_EPOCHS):
        print('\n\nEpoch:', e)
        train_rnn(RNN_model, CNN_model, e, video_train_loader)
        test_rnn(RNN_model, CNN_model, video_test_loader)

        if e and e % 5 == 0:
            path = 'rnn_model' + str(int(time.time() * 1000)) + '.pth'
            print("Saving model to '%s'..." % path, flush=True, end=' ')
            with open(path, 'wb') as f:
                torch.save(RNN_model, f)

    print('Done')

# # # # # # # # # # # # # TEST RNN MODEL # # # # # # # # # # # #

# test_rnn(RNN_model, RNN_EPOCHS, random_test_loader)
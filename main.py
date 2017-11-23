import torch
from torch.autograd import Variable
from torchvision import transforms

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

# number of frames to skip between 2 frames (default 0)
FRAME_SKIP = 29  # 9

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


train_loader, test_loader = train_test_loader(DATA_DIR,
                                              batch_size=BATCH_SIZE,
                                              frame_skip=FRAME_SKIP,
                                              transform=transform_list,
                                              test_batch_size=TEST_BATCH_SIZE,
                                              test_size=TEST_SIZE,
                                              **KWArgs)


# # # # # # # # # # # Model PARAMETERS # # # # # # # # # # #
LEARNING_RATE = 0.01
MOMENTUM = 0.5

LOG_INTERVAL = 10

EPOCHS = 10

model = CNN()
if CUDA:
    model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


def train(epoch):
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        print('Batch:', batch)
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
            print('Loss:', loss.data[0])


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for batch, (data, target) in enumerate(train_loader):
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


for e in range(EPOCHS):
    print('\n\nEpoch:', e)
    train(e)
    test()

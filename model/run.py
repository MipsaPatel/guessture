import os
import time

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from .load import SignData, SignDataRandom
from .models import NeuralNet
from .params import Parameters

# A container for all parameters required by the model, training etc.
P = Parameters(torch.cuda.is_available())


def train(model, optimizer, loader, epoch, rnn=True):
    """
    Train the model for one epoch.
    :param model: The model instance to be trained.
    :param optimizer: The optimizer to be used.
    :param loader: Provides data in batches.
    :param epoch: Training epoch count.
    :param rnn: A boolean indicating whether the recurrent layer must be used for training.
    """
    # initiate training
    model.train()
    print("\nTraining [%d]..." % epoch)
    for batch, (data, target) in enumerate(loader):
        # data returned has an extra dimension which is the number of videos.
        data = data.squeeze(0)
        if P.cuda:  # convert to CUDA instances if available
            data, target = data.cuda(), target.cuda()

        # Create a Variable instance for each. PyTorch requires all values to be Variables
        data, target = Variable(data), Variable(target)

        # PyTorch accumulates gradients. Clear them whenever a new pass is done.
        optimizer.zero_grad()

        # Get the output of the net
        output = model(data, rnn=rnn)
        # When CNN is trained with normal data, an output is generated per frame instead of one for the entire video
        # All have same target. So, create a tensor of matching dimensions
        # target = target.repeat(len(output))

        # compute the loss and back-propagate
        # Remove size averaging for large batch sizes
        loss = model.loss(output, target, size_average=False)
        loss.backward()

        # Update the weights from gradients
        optimizer.step()

        # Log
        if not (batch + 1) % P.log_interval:
            print('Train [%d] (%d)... %.6f' % (epoch, batch + 1, loss.data[0]))


def test(model, loader, epoch, rnn=True):
    """
    Test the model.
    :param model: The model instance to be tested.
    :param loader: Provides data in batches.
    :param epoch: Test epoch count.
    :param rnn: A boolean indicating whether the recurrent layer must be used for testing.
    """
    # initiate testing
    model.eval()

    # aggregate the predictions and losses
    correct = 0
    loss = 0
    count = 0

    print("\nTesting [%d]..." % epoch)
    for batch, (data, target) in enumerate(loader):
        # data has an extra dimension
        data = data.squeeze(0)
        if P.cuda:  # convert to CUDA instances if available
            data, target = data.cuda(), target.cuda()

        # Create a Variable instance for each. PyTorch requires all values to be Variables
        data, target = Variable(data, volatile=True), Variable(target)

        # Get the output of the net
        output = model(data, rnn=rnn)
        # While testing CNN with normal loader, an output is generated per frame instead of one for the entire video
        # All have same target. So, create a tensor of matching dimensions
        # target = target.repeat(len(output))

        # count the number of tests done
        count += len(target)

        # accumulate loss
        loss += model.loss(output, target, size_average=False).data[0]
        # count the number of correct classifications
        correct += output.max(1)[1].eq(target).sum().data[0]

        # Log
        if not (batch + 1) % P.log_interval:
            print('Test [%d] (%d)... %.4f @%.2f%% (%d/%d)' %
                  (epoch, batch + 1, loss / count, 100.0 * correct / count, correct, count))

    # Final log
    print('\nTest [%d]:' % epoch,
          'Loss: %.4f' % (loss / count),
          'Accuracy: %.2f%% (%d/%d)\n' % (100.0 * correct / count, correct, count),
          sep='\n')


def save(model, epoch=0, prefix='', filename=None):
    """
    Save the model.
    :param model: The model to be saved.
    :param epoch: The number of epochs this model is trained for.
    :param prefix: The string to append before the file name.
    :param filename: The filename to save the model.
                <prefix>model_<epoch>_<timestamp>.pth is used as the default filename.
    """
    if filename is None:
        filename = prefix + 'model_%03d_%d.pth' % (epoch, time.time())

    print("\nSaving to '%s'..." % filename, end=' ', flush=True)
    # File is opened as a Binary file.
    with open(os.path.join(P.model_dir, filename), 'wb') as f:
        torch.save(model.state_dict(), f)
    print('Done\n')


def run():
    print('Loading training data...')
    train_loader = DataLoader(SignData(P.train_data,
                                       P.frame_skip,
                                       P.frame_interval,
                                       P.transforms,
                                       **P.kwargs),
                              shuffle=True,
                              **P.kwargs)
    print('Training data has', len(train_loader), 'videos.\n')

    print('Randomizing training data...')
    random_train_loader = DataLoader(SignDataRandom(train_loader.dataset),
                                     batch_size=P.cnn_train_batch_size,
                                     shuffle=True,
                                     **P.kwargs)
    print('Training data has', len(random_train_loader.dataset), 'frames.\n')

    print('Loading testing data...')
    test_loader = DataLoader(SignData(P.test_data,
                                      P.frame_skip,
                                      P.frame_interval,
                                      P.transforms,
                                      **P.kwargs),
                             shuffle=True,
                             **P.kwargs)
    print('Test data has', len(test_loader), 'videos.\n')

    print('Randomizing test data...')
    random_test_loader = DataLoader(SignDataRandom(test_loader.dataset),
                                    batch_size=P.cnn_test_batch_size,
                                    shuffle=True,
                                    **P.kwargs)
    print('Test data has', len(random_test_loader.dataset), 'frames.\n')

    # Create model
    print('Generating model...')
    nn_model = NeuralNet()
    if P.cuda:  # convert to CUDA instance if available
        nn_model = nn_model.cuda()
    if P.load_model:
        nn_model.load_state_dict(torch.load(P.model_path))
        # Sanity check
        test(nn_model, test_loader, 0)

    # Training
    if P.train:
        print('Setting up parameters...')
        model_optimizer = Adam(nn_model.parameters(), lr=P.learning_rate)

        print('\nStarting training...')
        if P.train_cnn:
            print('\nTraining only CNN...')
            for e in range(P.cnn_epochs):
                train(nn_model, model_optimizer, random_train_loader, e, rnn=False)
                test(nn_model, random_test_loader, e, rnn=False)

                if not (e + 1) % P.save_interval:
                    save(nn_model, e, 'cnn_')

            # Save model after training CNN
            save(nn_model, filename='cnn_model.pth')

        if P.train_rnn:
            print('\nTraining the entire network...')
            for e in range(P.rnn_epochs):
                train(nn_model, model_optimizer, train_loader, e)
                test(nn_model, test_loader, e)

                if not (e + 1) % P.save_interval:
                    save(nn_model, e, 'rnn_')

            # Save model after training RNN
            save(nn_model, filename='model.pth')

        # Save the final trained model
        save(nn_model, filename='model.pth')

    print('Done!')


if __name__ == '__main__':
    run()

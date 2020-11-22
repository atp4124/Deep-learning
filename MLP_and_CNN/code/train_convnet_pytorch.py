"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import cifar10_utils
from convnet_pytorch import ConvNet
import argparse
import os

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    a = (predictions.max(dim=1)[1] == targets).sum().item()
    b = targets.shape[0]
    accuracy =  a / b
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy

def init_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def train():
    """
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    loss_list = []
    batch_list = []
    accuracy_list = []
    batch_list_test = []
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Running in GPU model')

    device = torch.device('cuda' if use_cuda else 'cpu')
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    for iter in range(313):
        x, y = cifar10['test'].next_batch(FLAGS.batch_size)
        y = np.argmax(y, axis=1)
        batch_list_test.append((x, y))

    for iter in range(FLAGS.max_steps):
        x, y = cifar10['train'].next_batch(FLAGS.batch_size)
        y = np.argmax(y, axis=1)
        batch_list.append((x, y))
    print('Batch list completed')
    in_features = 3
    out_features = 10 #num_classes
    net = ConvNet(in_features, out_features).to(device)
    lossfunc = nn.CrossEntropyLoss()
    optimiser = optim.Adam(net.parameters(), lr=FLAGS.learning_rate)
    net.train()
    for i in range(FLAGS.max_steps):
        inputs, labels = batch_list[i]
        # inputs = torch.from_numpy(inputs)
        inputs = torch.tensor(inputs)
        labels = torch.from_numpy(labels).long()
        inputs, labels = inputs.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = net.forward(inputs.float())

        loss = lossfunc(outputs, labels)

        loss_list.append(loss)

        loss.backward()
        optimiser.step()
        if (i + 1) % FLAGS.eval_freq == 0:
            net.eval()
            total = 0
            for data in batch_list_test:
                x_test, y_test = data
                x_test = torch.from_numpy(x_test).type(dtype).to(device)
                y_test = torch.from_numpy(y_test).long().to(device)
                predicted = net.forward(x_test)
                acc = accuracy(predicted, y_test)
                total += acc

            accuracy_list.append(total / len(batch_list_test))
            print('Accuracy on test set at step {} is {}'.format(i, total / len(batch_list_test)))
            print('Loss of training is {}'.format(loss.item()))

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(accuracy_list) * FLAGS.eval_freq, step=FLAGS.eval_freq), accuracy_list, 'o-')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    #
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel('Step')
    plt.ylabel('Loss')

    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()

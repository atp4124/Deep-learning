"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
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
from mlp_pytorch import MLP
import argparse
import os
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '200, 200'
LEARNING_RATE_DEFAULT = 0.01
MAX_STEPS_DEFAULT = 3200
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


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
    a = (predictions.max(dim=1)[1] == targets).sum()
    print(a)
    b = targets.shape[0]
    print(b)
    accuracy = a.item()/b
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy

def var_init(model, std=0.01):
    for name, param in model.named_parameters():
        param.data.normal_(std=std)


def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    loss_list = []
    batch_list = []
    accuracy_list = []
    # load the batches and reshape the samples


    for iter in range(FLAGS.max_steps):
        x, y = cifar10['train'].next_batch(FLAGS.batch_size)
        y = np.argmax(y, axis=1)
        # transform sample into vector
        x = np.reshape(x, (FLAGS.batch_size, -1))
        batch_list.append((x, y))
    print('Batch list completed')
    in_features = batch_list[0][0].shape[1]
    out_features = 10 #num_classes

    x_test, y_test = cifar10['test'].images, cifar10['test'].labels
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    y_test = np.argmax(y_test, axis=1)
    print(y_test.shape)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).long()
    print(y_test)
    net = MLP(in_features, dnn_hidden_units, out_features)
    #var_init(net, sd=0.0001)
    lossfunc = nn.CrossEntropyLoss()
    optimiser = optim.SGD(net.parameters(), lr=FLAGS.learning_rate)
    print(net)
    net.train()
    for i in range(FLAGS.max_steps):
        inputs, labels = batch_list[i]
        #inputs = torch.from_numpy(inputs)
        inputs = torch.tensor(inputs)
        labels = torch.from_numpy(labels).long()

        optimiser.zero_grad()
        outputs = net.forward(inputs.float())

        loss = lossfunc(outputs, labels)

        loss_list.append(loss)

        loss.backward()
        optimiser.step()


        if (i+1) % FLAGS.eval_freq == 0:
            net.eval()
            predicted = net.forward(x_test)
            accuracy_val = accuracy(predicted, y_test)
            accuracy_list.append(accuracy_val)
            print('Accuracy on test set at step {} is {}'.format(i, accuracy_val))
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
    #
    ########################
    # PUT YOUR CODE HERE  #
    #######################
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
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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

"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
import argparse
import cifar10_utils
from modules import LinearModule, SoftMaxModule, CrossEntropyModule
from modules import ELUModule
from mlp_numpy import MLP
import numpy as np
import os
import matplotlib.pyplot as plt
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 0.001
MAX_STEPS_DEFAULT = 1400
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
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))/targets.shape[0]
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # load the dataset
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    print('Dataset is loaded')
    #loss function initialisation
    lossfunc = CrossEntropyModule()
    #list initialisation
    loss_list = []
    batch_list = []
    accuracy_list = []
    #load the batches and reshape the samples
    for iter in range(FLAGS.max_steps):
         x, y = cifar10['train'].next_batch(FLAGS.batch_size)
         #transform sample into vector
         x = np.reshape(x, (FLAGS.batch_size, -1))
         batch_list.append((x, y))
    print('Batch list is completed')
    in_features = batch_list[0][0].shape[1]
    out_features = batch_list[0][1].shape[1]

    x_test, y_test = cifar10['test'].images, cifar10['test'].labels
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    net = MLP(in_features, dnn_hidden_units, out_features)
    print('Begin iterations')
    for i in range(FLAGS.max_steps):
        #print('Iteration {}'.format(i))
        inputs, labels = batch_list[i]
        #output of the network
        outputs = net.forward(inputs)

        #calculate loss of the outputs
        #the input of the cross entropy module should be the output of the network
        loss = lossfunc.forward(outputs, labels)

        loss_list.append(loss)
        #gradient of loss
        gradient = lossfunc.backward(outputs, labels)

        #backpropagate through the network
        net.backward(gradient)
        #update parameters
        for layer in net.layers:
            if isinstance(layer, LinearModule):
            #update weights
               layer.params['weight'] -= FLAGS.learning_rate * layer.grads['weight']
            #update biases
               layer.params['bias'] -= FLAGS.learning_rate * layer.grads['bias']

        #evaluation on the test set
        if (i+1) % FLAGS.eval_freq == 0:
            predicted = net.forward(x_test)
            accuracy_val = accuracy(predicted, y_test)
            accuracy_list.append(accuracy_val)
            print('Accuracy on test set at step {} is {}'.format(i, accuracy_val))
            print('Loss of training is {}'.format(loss))

    # # plot accuracies and losses
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(accuracy_list)*FLAGS.eval_freq, step=FLAGS.eval_freq), accuracy_list, 'o-')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    #
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    #






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

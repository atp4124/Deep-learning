"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""

from modules import LinearModule, SoftMaxModule, CrossEntropyModule
from modules import ELUModule, ReluModule, SigmoidModule


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.layers = []
        in_features = n_inputs
        if len(n_hidden) > 0:
           for hidden_layer in n_hidden:
               self.layers.append(LinearModule(in_features, hidden_layer))
               #self.layers.append(ELUModule())
               self.layers.append(ReluModule())
               in_features = hidden_layer

        self.layers.append(LinearModule(in_features, n_classes))
        self.layers.append(SoftMaxModule())
        #self.layers.append(SigmoidModule())


        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.layers:
            out = layer.forward(x)
            x = out
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in reversed(self.layers):
            dx = layer.backward(dout)
            dout = dx
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx

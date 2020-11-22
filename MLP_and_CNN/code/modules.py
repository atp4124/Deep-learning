"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.params = {'weight': np.random.normal(loc=0, scale=0.0001, size=(out_features, in_features)),
                       'bias': np.zeros((out_features, 1))}
        self.grads = {'weight': np.zeros((out_features, in_features)), 'bias': np.zeros((out_features, 1))}
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        out = x @ self.params['weight'].T + self.params['bias'].T #(s,n_out) = (s,n_in)(n_in,n_out)
        self.input = x
        ########################
        # END OF YOUR CODE    #
        #######################
        return out

    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.grads['weight'] = dout.T @ self.input # (n_out, n_in) = (n_out,s)(s,n_in)
        self.grads['bias'] = np.sum(dout, axis=0)[:, np.newaxis]# (n_out,)
        dx = dout @ self.params['weight'] # (s,n_in) = (s,n_out)(n_out,n_in)

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        b = x.max(axis=1, keepdims=True)
        y = np.exp(x-b)
        out = y/y.sum(axis=1, keepdims=True)
        self.prob = out
        #we need the softmax probabilities to calculate the gradients in the backpropagation

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        ds = np.einsum('ij,ik->ijk', self.prob, self.prob)
        diag = np.apply_along_axis(np.diag, 1, self.prob)
        dh_dx = diag-ds
        dx = np.einsum('ij,ijk->ik', dout, dh_dx)
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        loss_sample = np.einsum('ij,ij->i', y, np.log(x))
        out = -loss_sample.mean()
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dx = -(y/x)/len(y)

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        out = np.where(x >= 0, x, np.exp(x)-1)
        self.input = x

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        gradient = np.where(self.input >= 0, np.ones_like(self.input), np.exp(self.input))
        dx = dout * gradient

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx



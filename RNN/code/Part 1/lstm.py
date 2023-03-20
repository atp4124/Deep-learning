"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import os

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.params = nn.ParameterDict()

        for gate in ['g', 'i', 'f', 'o']:
            self.params['W_' + gate + 'x'] = nn.Parameter(torch.empty(input_dim, hidden_dim))
            self.params['W_' + gate + 'h'] = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            nn.init.kaiming_normal_(self.params['W_' + gate + 'x'], nonlinearity='linear')
            nn.init.kaiming_normal_(self.params['W_' + gate + 'h'], nonlinearity='linear')
            self.params['b_' + gate] = nn.Parameter(torch.zeros(1, hidden_dim))

        self.params['W_ph'] = nn.Parameter(torch.empty(hidden_dim, num_classes))
        nn.init.kaiming_normal_(self.params['W_ph'], nonlinearity='linear')
        self.params['b_p'] = nn.Parameter(torch.zeros(1, num_classes))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.embedding = nn.Embedding(3, input_dim)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        #if x.shape != (self.batch_size, self.seq_length, self.input_dim):
            #raise ValueError('The dimension of x is not as expected!')
        x = x.squeeze()
        h_t = torch.zeros(self.batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(self.batch_size, self.hidden_dim, device=self.device)
        for t in range(self.seq_length):
            x_em = self.embedding(x[:, t].long())
            g_t = self.tanh(x_em @ self.params['W_gx'] + h_t @ self.params['W_gh'] + self.params['b_g'])
            i_t = self.sigmoid(x_em @ self.params['W_ix'] + h_t @ self.params['W_ih'] + self.params['b_i'])
            f_t = self.sigmoid(x_em @ self.params['W_fx'] + h_t @ self.params['W_fh'] + self.params['b_f'])
            o_t = self.sigmoid(x_em @ self.params['W_ox'] + h_t @ self.params['W_oh'] + self.params['b_o'])
            c_t = g_t * i_t + c_t * f_t
            h_t = self.tanh(c_t) * o_t

        p_t = h_t @ self.params['W_ph'] + self.params['b_p']
        out = self.softmax(p_t)

        return out
        ########################
        # END OF YOUR CODE    #
        #######################

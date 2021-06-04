"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(peepLSTM, self).__init__()

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

        for gate in ['i', 'f', 'o']:
            self.params['W_' + gate + 'h'] = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            self.params['W_' + gate + 'x'] = nn.Parameter(torch.empty(input_dim, hidden_dim))
            nn.init.kaiming_normal_(self.params['W_' + gate + 'x'], nonlinearity='linear')
            nn.init.kaiming_normal_(self.params['W_' + gate + 'h'], nonlinearity='linear')
            self.params['b_' + gate] = nn.Parameter(torch.zeros(1, hidden_dim))

        self.params['W_cx'] = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.params['b_c'] = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.kaiming_normal_(self.params['W_cx'], nonlinearity='linear')
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
        x = x.squeeze()
        #h_t = torch.zeros(self.batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(self.batch_size, self.hidden_dim, device=self.device)
        for t in range(self.seq_length):
            x_em = self.embedding(x[:, t].long())
            i_t = self.sigmoid(x_em @ self.params['W_ix'] + c_t @ self.params['W_ih'] + self.params['b_i'])
            f_t = self.sigmoid(x_em @ self.params['W_fx'] + c_t @ self.params['W_fh'] + self.params['b_f'])
            o_t = self.sigmoid(x_em @ self.params['W_ox'] + c_t @ self.params['W_oh'] + self.params['b_o'])
            c_t = self.sigmoid(x_em @ self.params['W_cx'] + self.params['b_c']) * i_t + c_t * f_t
            h_t = self.tanh(c_t) * o_t

        p_t = h_t @ self.params['W_ph'] + self.params['b_p']
        out = self.softmax(p_t)

        return out
        ########################
        # END OF YOUR CODE    #
        #######################


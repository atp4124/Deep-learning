###############################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Adapted: 2020-11-09
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
# datasets
import datasets

# models
from bi_lstm import biLSTM
from lstm import LSTM
from gru import GRU
from peep_lstm import peepLSTM
import matplotlib.pyplot as plt
import numpy as np

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

###############################################################################


def train(config):
    #np.random.seed(24)
    #torch.manual_seed(24)


    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print(device)

    # Load dataset
    if config.dataset == 'randomcomb':
        print('Load random combinations dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        dataset = datasets.RandomCombinationsDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

    elif config.dataset == 'bss':
        print('Load bss dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = 2
        config.input_dim = 3
        dataset = datasets.BaumSweetSequenceDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = 4 * config.input_length

    elif config.dataset == 'bipalindrome':
        print('Load binary palindrome dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        dataset = datasets.BinaryPalindromeDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = config.input_length*4+2-1



    # Setup the model that we are going to use
    if config.model_type == 'LSTM':
        print("Initializing LSTM model ...")
        model = LSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'biLSTM':
        print("Initializing bidirectional LSTM model...")
        model = biLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'GRU':
        print("Initializing GRU model ...")
        model = GRU(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'peepLSTM':
        print("Initializing peephole LSTM model ...")
        model = peepLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    # Setup the loss and optimizer
    loss_function = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    accuracy_list = []
    loss_list = []
    old_loss = 1.0
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Move to GPU

        batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        batch_targets = batch_targets.to(device)   # [batch_size]
        #print(batch_inputs[:,0,:].shape)
        #embedding = nn.Embedding(3, config.input_dim)
        #print(embedding(batch_inputs[:,0,:].long()).shape)
        # Reset for next iteration
        model.zero_grad()

        # Forward pass
        log_probs = model(batch_inputs)

        # Compute the loss, gradients and update network parameters
        loss = loss_function(log_probs, batch_targets)
        loss.backward()

        #######################################################################
        # Check for yourself: what happens here and why?
        #######################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        #######################################################################

        optimizer.step()

        predictions = torch.argmax(log_probs, dim=1)
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / log_probs.size(0)
        accuracy_list.append(accuracy)
        loss_list.append(loss.item())
        # print(predictions[0, ...], batch_targets[0, ...])

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 60 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                   Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        # Check if training is finished
        if step == config.train_steps or old_loss == loss.item():
            # If you receive a PyTorch data-loader error, check this bug report
            # https://github.com/pytorch/pytorch/pull/9655
            break
        else:
            old_loss = loss.item()
    print('Done training.')
    ###########################################################################
    ###########################################################################

    print('Evaluating...')
    acc = []
    for i in range(3):
        acc_sublist = []
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            model.eval()
            batch_inputs = batch_inputs.to(device)  # [batch_size, seq_length,1]
            batch_targets = batch_targets.to(device)
            pred = model(batch_inputs)
            predictions = torch.argmax(pred, dim=1)
            correct = (predictions == batch_targets).sum().item()
            accuracy = correct / pred.size(0)
            acc_sublist.append(accuracy)
            if step == 25:
                break
        acc.append(np.mean(acc_sublist))
    print('Mean accuracy is {} and standard deviation is {}'.format(np.mean(acc), np.std(acc)))
    return accuracy_list, loss_list
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='bipalindrome',
                        choices=['randomcomb', 'bss', 'bipalindrome'],
                        help='Dataset to be trained on.')
    # Model params
    parser.add_argument('--model_type', type=str, default='LSTM',
                        choices=['LSTM', 'biLSTM', 'GRU', 'peepLSTM'],
                        help='Model type: LSTM, biLSTM, GRU or peepLSTM')
    parser.add_argument('--input_length', type=int, default=10,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=64,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=256,
                        help='Number of hidden units in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=800,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    # Misc params
    parser.add_argument('--device', type=str, default="cpu",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')

    config = parser.parse_args()

    # Train the model
    for length in [10, 20]:
        print('Length ' + str(length))
        config.input_length = length
        accuracy_list, loss_list = train(config)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(accuracy_list)), accuracy_list, 'o-')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        #
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(loss_list)), loss_list)
        plt.xlabel('Step')
        plt.ylabel('Loss')





# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import TextDataset
from model import TextGenerationModel
import random
###############################################################################
def sample(pred_char, temperature):
    if temperature == 0:
        return pred_char.argmax()
    else:
        dist = pred_char / temperature
        out = torch.softmax(dist, dim=0)
        return torch.multinomial(out, 1)


def generate_sequence(dataset, model, device, temperature, length):
    random_chars = dataset._char_to_ix[random.choice(dataset._chars)]
    random_chars = torch.tensor(random_chars, device=device)
    random_chars = random_chars.view(1, 1)
    predictions = []
    h_0c_0 = None
    for i in range(length):
        pred_seq, h_0c_0 = model.forward(random_chars, h_0c_0)
        random_chars[0, 0] = sample(pred_seq.squeeze(), temperature)
        predictions.append(random_chars.item())
    return dataset.convert_to_string(predictions)

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    config.txt_file = './assets/book_EN_grimms_fairy_tails.txt'
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, 64, config.dropout_keep_prob,
                                config.lstm_num_hidden, config.lstm_num_layers, device)  # FIXME

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss() # FIXME
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)  # FIXME
    step = 0
    loss_list = []
    accuracy_list = []
    while step < 33600:
        for (batch_inputs, batch_targets) in data_loader:
            model.train()
            step += 1
            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################
            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_targets = torch.stack(batch_targets, dim=1).to(device)

            # loss = np.inf   # fixme
            # accuracy = 0.0  # fixme
            model.zero_grad()
            pred, _ = model(batch_inputs)
            pred = pred.view(-1, dataset.vocab_size)
            batch_targets = batch_targets.view(-1)
            loss = criterion(pred, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            predictions = torch.argmax(pred, dim=1)
            correct = (predictions == batch_targets).sum().item()

            accuracy = correct / pred.size(0)
            accuracy_list.append(accuracy)
            loss_list.append(loss.item())
            # Just for time measurement
            t2 = time.time()
            examples_per_second = 64 / float(t2 - t1)

            if (step + 1) % 60 == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                           Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    1000000, 64, examples_per_second,
                    accuracy, loss
                ))

            if step % 11200 == 0:
                # Generate some sentences by sampling from the model
                model.eval()
                for i in range(5):
                    for temperature in [0, 0.5, 1.0, 2.0]:
                        for length in [30, 40, 60]:
                            sentence = generate_sequence(dataset, model, device, temperature, length)
                            with open('./summaries.txt', 'a', encoding='utf-8') as file:
                                file.write("{};{};{};{};{}\n".format(i, step, temperature, length, sentence))

            if step == 33600:
                # If you receive a PyTorch data-loader error,
                # check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(accuracy_list)), accuracy_list, 'o-')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    #
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='./assets/book_EN_grimms_fairy_tails.txt',
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=60,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)

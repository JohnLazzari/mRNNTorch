"""
examples/torch/run_FlipFlop.py
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import LeakmRNN, EmRNN
from flip_flop_data import FlipFlopData


def train_FlipFlop(model_type):
    """Train an RNN to solve the N-bit memory task.
    Args:
        None.
    Returns:
        model: FlipFlop object.
                The trained RNN model.
        valid_predictions: dict.
                The model's predictions on a set of held-out validation trials.
    """

    # Data specifications
    n_bits = 3
    n_train = 512
    epochs = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model hyperparameters
    n_exc = 21
    n_inhib = 9
    num_units = 30

    data_gen = FlipFlopData()
    train_data = data_gen.generate_data(n_trials=n_train)

    if model_type == "leaky":
        model = LeakmRNN(n_bits, n_exc, n_inhib, n_bits).to(device)
    elif model_type == "elman":
        model = EmRNN(n_bits, n_exc, n_inhib, n_bits).to(device)
    else:
        raise ValueError

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        running_loss = 0
        optimizer.zero_grad()
        # Gather inputs and targets
        input = torch.from_numpy(train_data["inputs"]).to(device)
        target = torch.from_numpy(train_data["targets"]).to(device)

        # forward pass through model
        if model_type == "leaky":
            x0 = torch.zeros(size=(n_train, num_units)).to(device)
            h0 = torch.zeros(size=(n_train, num_units)).to(device)
            out, _, _ = model(input, x0, h0)
        elif model_type == "elman":
            h0 = torch.zeros(size=(n_train, num_units)).to(device)
            out, _ = model(input, h0)
        else:
            raise ValueError

        # loss and update parameters
        loss_ = criterion(out, target)
        loss_.backward()
        optimizer.step()

        running_loss += loss_.item()

        if epoch % 10 == 0:
            running_loss /= 10
            print(f"Epoch: {epoch}, Avg. Loss: {running_loss}")
            running_loss = 0

        # Save model
        torch.save(
            model.state_dict(),
            f"flip_flop_rnn_{model_type}.pth",
        )


def main():
    # Step 1: Train an RNN to solve the N-bit memory task
    train_FlipFlop("leaky")
    train_FlipFlop("elman")


if __name__ == "__main__":
    main()

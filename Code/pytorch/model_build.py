import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.optim as optim
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import time

import load_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

INPUT_DIM = 256 * 256 * 3
OUTPUT_DIM = 3


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.layers = nn.Sequential(
            nn.Linear(3000000, 192),
            nn.ReLU(),
            nn.Linear(192, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        logits = self.layers(x)
        return logits


def main():
    train_dataloader, test_dataloader = load_data.buildLoaders(
        annotations_file=r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\preprocess\out.csv",
        img_dir=r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG", size=1000,
        color="rgb")  # Wenn color = grayscale ist funktioniert es schon
    model = NeuralNetwork()
    learning_rate = 1e-3
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    torch.save(model, 'model.pth')


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    main()

import os

import torch
import torch.nn as nn
from azureml.core import Run

import load_data

run = Run.get_context()
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

INPUT_DIM = 256 * 256 * 3
OUTPUT_DIM = 3


class NeuralNetwork(nn.Module):
    """
    Neurale Network Klasse entsprechend Pytorch. Erbt von nn.module
    """

    def __init__(self):
        """
        Init methode für die klasse. Hier werden verschiedene Layer für das Netzwerk gebaut
        """
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000 * 1000 * 3, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 4, bias=True)
        )
        self.conv = nn.Sequential(  # Das müsste in etwa unsere idee mit dem 5 kernel abbilden?
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            nn.Flatten()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(396010, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 4, bias=True)
        )

    def forward(self, x):
        """
        Forward methode entsprechend pytorch. Hier entsteht das eigentliche Model
        :param x:
        :return:
        """
        # logits = self.layers(x)
        x = self.conv(x)
        # print(x)
        logits = self.conv2(x)
        return logits


def main(path, out):
    """
    Main methode. Hier werden die Data loader abgerufen, das Neurale netzwerk auf das Device transferiert. Als Loss Function wird CrossEntropyLoss gewählt und also
    optimizer Adam mit einer Learning rate von 1*10^-3. Schließlich wird für eine gegebene epoch zahl jedes mal eine Trainings und Test loop ausgeführt.
    """
    train_dataloader, test_dataloader = load_data.buildLoaders(
        annotations_file="image_data.csv",
        img_dir=path, size=1000,
        color="rgb", batch_size=32)  # Wenn color = grayscale ist funktioniert es schon
    model = NeuralNetwork().to(device)
    learning_rate = 1e-3
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), os.path.join(out, 'model.pth'))


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Code stammt aus der Tutorial seite von Pytorch: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    Iteriert über das datenset und Konvergiert zu Optimalen Parametern
    :param dataloader: Data Loader für die Trainings daten
    :param model: Das eigentliche mdel
    :param loss_fn: Die vordefinierte Loss function
    :param optimizer Den Vordefinierten Optimizer
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            run.log("loss", loss)
            run.log("current", current)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    """
    Code stammt aus der Tutorial seite von Pytorch: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    Iteriert über das Datenset um die model Performance zu bewerten
    :param dataloader: Data Loader für die Test daten
    :param model: Das eigentliche Model
    :param loss_fn: Die vordefinierte Loss function
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    run.log("accuracy", (100 * correct))
    run.log("Avg Loss", test_loss)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='Path for the output'
    )
    args = parser.parse_args()
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")
    with open(os.path.join(args.output_path, "readme.txt"), 'w') as f:
        f.write('readme')
    main(args.data_path, args.output_path)

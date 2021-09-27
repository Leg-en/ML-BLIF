import dataset as ds
from model import NeuralNetwork
import torch
import torch.nn as nn

def main():
    """
    Main methode. Hier werden die Data loader abgerufen, das Neurale netzwerk auf das Device transferiert. Als Loss Function wird CrossEntropyLoss gewählt und also
    optimizer Adam mit einer Learning rate von 1*10^-3. Schließlich wird für eine gegebene epoch zahl jedes mal eine Trainings und Test loop ausgeführt.
    """
    train_dataloader, test_dataloader = ds.Builder(1)  # Wenn color = grayscale ist funktioniert es schon
    model = NeuralNetwork()
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
    torch.save(model, model.pth)


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
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    main()
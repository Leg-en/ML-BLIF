import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
mpl.use('Qt5Agg')

import load_data

path = r"C:\Users\Emily\Documents\Bachelor_Artefakte\modelle\model.pth"
image_idx = 0
img_size = 1000


def load(path):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def buildLoaders(annotations_file=r"C:\Users\Emily\Documents\Bachelor_Artefakte\image_data.csv",
                 img_dir=r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG", size=img_size, color="rgb", ):
    training_data = load_data.CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, size=size, color=color)
    loader = DataLoader(training_data, batch_size=1, shuffle=True)
    return loader

def predict(model, image):
    with torch.no_grad():
        res = model(image)
    prediction = np.argmax(res)
    prediction = prediction.item()
    return prediction


def main():
    global path
    model = load(path)
    loader = buildLoaders()
    l = len(loader)
    c = 0
    for idx,(img,label) in enumerate(loader):
        print(f"Iteration: {idx} von: {l}")
        prediction = predict(model,img)
        lab = label[0].item()
        if prediction == lab:
            c += 1
    print(f"Getroffene = {c}")
    print(f"Laenge = {l}")
    print(f"Treffer in prozent: {(c/l)*100}")





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


if __name__ == '__main__':
    main()




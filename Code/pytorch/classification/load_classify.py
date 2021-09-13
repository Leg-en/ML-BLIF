import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
mpl.use('Qt5Agg')

import load_data

path = r"/Artefakte/modelle/model.pth"
image_idx = 0
img_size = 1000


def load(path):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def buildLoaders(annotations_file=r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Artefakte\image_data.csv",
                 img_dir=r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG", size=img_size, color="rgb", ):
    training_data = load_data.CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, size=size, color=color)
    loader = DataLoader(training_data, batch_size=2, shuffle=True)
    return loader

def predict(model, image):
    output = model(image)
    prediction = torch.max(output.data, 1)[1].numpy()
    print(prediction)
    prediction = max(prediction)
    if prediction == 0:
        print("Himmel")
    elif prediction == 1:
        print("Strand")
    elif prediction == 2:
        print("Wasser")
    else:
        print("Sonstiges")

def show(label, image):
    print(label)
    img = image[0].squeeze()
    img = np.array(img)
    img = img.reshape((img_size,img_size,3))
    plt.imshow(img)
    plt.show()

def main():
    global path
    model = load(path)
    loader = buildLoaders()
    feature, label = next(iter(loader))
    show(label,feature)
    predict(model,feature)



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




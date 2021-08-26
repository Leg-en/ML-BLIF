import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
mpl.use('Qt5Agg')

import load_data

path = None
image_idx = 0
img_size = 1000


def load(path):
    model = torch.load(path)
    return model

def buildLoaders(annotations_file=r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Artefakte\image_data.csv",
                 img_dir=r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG", size=img_size, color="rgb", ):
    training_data = load_data.CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, size=size, color=color)
    loader = DataLoader(training_data, batch_size=2, shuffle=True)
    return loader

def predict(model, image):
    x = model.predict(image)
    print(x)

def show(label, image):
    print(label)
    img = image[0].squeeze()
    img = np.array(img)
    img = img.reshape((img_size,img_size,3))
    plt.imshow(img)
    plt.show()

def main():
    global path
    #model = load(path)
    loader = buildLoaders()
    feature, label = next(iter(loader))
    show(label,feature)

if __name__ == '__main__':
    main()
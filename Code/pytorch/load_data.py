import os

import numpy as np
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, size=1000):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.size = size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = ImageOps.grayscale(image)
        ymin, ymax, xmin, xmax = (
            self.img_labels.iloc[idx, 5], self.img_labels.iloc[idx, 7], self.img_labels.iloc[idx, 4],
            self.img_labels.iloc[idx, 6])
        image = image.crop((xmin, ymin, xmax, ymax))
        image = image.resize((self.size, self.size))
        #Todo: Image to tensor
        image = transforms.ToTensor()(image).unsqueeze(0)
        label = self.img_labels.iloc[idx, 3]
        if label == "Himmel":
            label = 0
        elif label == "Strand":
            label = 1
        elif label == "Wasser":
            label = 3
        label = torch.tensor(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def buildLoaders(annotations_file=r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\preprocess\out.csv",
                 img_dir=r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG",size=28):
    training_data = CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, size=size)
    train_data, test_data = torch.utils.data.random_split(training_data, [350, 98])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    return train_loader, test_loader


def test(train_loader):
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    #plt.imshow(img.permute(1,2,0))
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")


def main():
    train_loader, test_loader = buildLoaders()
    test(train_loader)


if __name__ == '__main__':
    main()

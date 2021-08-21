import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, size=None, color=None):
        """
        Init methode für ein Custom Image Dataset für pytorch. Kompatibel mit pytorch dataloader.
        :param annotations_file:  PFad zu einem CSV File welches nach dem Schema filename,width,height,class,xmin,ymin,xmax,ymax und n datensätze enthält
        :param img_dir: Pfad zu einem Ordner mit verschiedenen Bildern
        :param size: Ein Integer wert der die Image größe wiederspiegelt. ALle images werden resized auf (size,size)
        :param color: grayscale oder rgb.
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.size = size
        self.color = color

    def __len__(self):
        """
        Returnt die länge des Datensatzes
        :return: Die länge des Datensatzes
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Returnt das Image und das Label für den gegebenen Index
        :param idx: Index
        :return: Image und Label
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        if self.color == "grayscale":
            image = ImageOps.grayscale(image)
        elif self.color == "rgb":
            pass
        ymin, ymax, xmin, xmax = (
            self.img_labels.iloc[idx, 5], self.img_labels.iloc[idx, 7], self.img_labels.iloc[idx, 4],
            self.img_labels.iloc[idx, 6])
        image = image.crop((xmin, ymin, xmax, ymax))
        image = image.resize((self.size, self.size))
        image = transforms.ToTensor()(image)
        label = self.img_labels.iloc[idx, 3]
        if label == "Himmel":
            label = 0
        elif label == "Strand":
            label = 1
        elif label == "Wasser":
            label = 3
        label = torch.tensor(label)
        return image, label


def buildLoaders(annotations_file=r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Artefakte\image_data.csv",
                 img_dir=r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG", size=28, color="rgb", batch_size=64):
    """
    Hilfs methode. Baut Loader für einen Trainings und einen Validierungs/Test daten
    :param annotations_file: Pfad zu dem CSV File in dem alle relevanten daten liegen
    :param img_dir: Pfad zu den Bildern
    :param size: Image size zu den die Bildern resized werden
    :param color: Grayscale oder rgb
    :param batch_size: Batch Size fuer die Bilder
    :return:
    """
    training_data = CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, size=size, color=color)
    train_data, test_data = torch.utils.data.random_split(training_data, [350, 98])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def test(train_loader):
    """
    Test methode. Nimmt einen Loader entgegen und zeigt das erste ausgegebene Label und das Image
    :param train_loader:
    """
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    # plt.imshow(img.permute(1,2,0))
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")


def main():
    train_loader, test_loader = buildLoaders()
    test(train_loader)


if __name__ == '__main__':
    main()

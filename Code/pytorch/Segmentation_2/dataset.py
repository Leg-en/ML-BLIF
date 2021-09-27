import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ["himmel", "strand", "wasser", "unlabelled"] #class_value 0 = himmel
    look_up = {
        0: 76,
        1: 149,
        2: 29,
        3:0
    }
    #CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
    #           'tree', 'signsymbol', 'fence', 'car',
    #           'pedestrian', 'bicyclist', 'unlabelled']


    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        try:
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        except TypeError:
            self.class_values = [0,1,2]
        for idx, val in enumerate(self.class_values):
            self.class_values[idx] = self.look_up[val]


        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        return torch.tensor(image, dtype=torch.float), torch.tensor(mask)

    def __len__(self):
        return len(self.ids)


def Builder(batch_size=32):
    train_ds = Dataset(r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\train\img",
                       r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\train\mask")
    valid_ds = Dataset(r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\valid\img",
                       r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\valid\mask")
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    return train_dl, valid_dl


def vis(dl):
    for (Image, Mask) in dl:
        print(Image)
        Image = Image[0]
        Mask = Mask[0]
        Image = Image.numpy()
        Mask = Mask.numpy()
        #cv2.imshow("Image", Image)
        #cv2.waitKey()
        #cv2.imshow("Mask", Mask)
        #cv2.waitKey()
        break


if __name__ == '__main__':
    train_dl, valid_dl = Builder(1)
    vis(train_dl)

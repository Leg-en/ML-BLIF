import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            preprocessing=None,
            augmentation=None,
            size=992

    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.to_tens = transforms.ToTensor()
        self.size = size
        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        # image = Image.open(self.images_fps[i])
        image = cv2.resize(cv2.cvtColor(cv2.imread(self.images_fps[i]), cv2.COLOR_BGR2RGB), (self.size, self.size))
        mask = cv2.resize(cv2.imread(self.masks_fps[i], 0), (self.size, self.size))

        masks = [(mask == v) for v in [0,149,29,76]]
        mask = np.stack(masks, axis=-1).astype('float')


        if self.augmentation is not None:
            res = self.augmentation(image=image, mask=mask)
            image = res["image"]
            mask = res["mask"]

        if self.preprocessing is not None:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            return image,mask

        return torch.tensor(image).permute(2,1,0).float(), torch.tensor(mask).permute(2,1,0)

    def __len__(self):
        return len(self.ids)


def Builder(batch_size=32, size=None, preprocess=None):
    train_ds = Dataset(r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\train\img",
                       r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\train\mask", size=size, )
    valid_ds = Dataset(r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\valid\img",
                       r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\valid\mask", size=size)
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
        cv2.imshow("Image", Image)
        cv2.waitKey()
        cv2.imshow("Mask", Mask)
        cv2.waitKey()
        break


if __name__ == '__main__':
    train_dl, valid_dl = Builder(1)
    vis(train_dl)

import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
import albumentations as albu
import numpy as np
import segmentation_models_pytorch as smp
import torch
from catalyst.dl import SupervisedRunner
import argparse
from azureml.core import Run
run = Run.get_context()



#https://colab.research.google.com/gist/Scitator/e3fd90eec05162e16b476de832500576/cars-segmentation-camvid.ipynb#scrollTo=AAQdydOw7A8n

parser = argparse.ArgumentParser()
parser.add_argument(
    '--x_train_dir',
    type=str,
    help='Path to the training data'
)
parser.add_argument(
    '--y_train_dir',
    type=str,
    help='Path for the output'
)
parser.add_argument(
    '--x_valid_dir',
    type=str,
    help='Path for the output'
)
parser.add_argument(
    '--y_valid_dir',
    type=str,
    help='Path for the output'
)
parser.add_argument(
    '--output_path',
    type=str,
    help='Path for the output'
)

args = parser.parse_args()


logdir = args.output_path

x_train_dir = args.x_train_dir
y_train_dir = args.y_train_dir

x_valid_dir = args.x_valid_dir
y_valid_dir = args.y_valid_dir

print("===== DATA =====")
print("x_train_dir PATH: " + x_train_dir)
print("y_train_dir PATH: " + y_train_dir)
print("x_valid_dir PATH: " + x_valid_dir)
print("y_valid_dir PATH: " + y_valid_dir)
print("LIST FILES IN DATA PATH...")
print(os.listdir(x_train_dir))
print("================")

#x_test_dir = os.path.join(DATA_DIR, 'test', "img")
#y_test_dir = os.path.join(DATA_DIR, 'test', "mask")


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

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

dataset = Dataset(x_train_dir, y_train_dir, classes=['himmel'])


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(3008, 4000)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

augmented_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    classes=["himmel"]
)

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ["himmel"]
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    raise ValueError

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    #augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

num_epochs = 10  # change me
loaders = {
    "train": train_loader,
    "valid": valid_loader
}

criterion = smp.utils.losses.DiceLoss()
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-4},

    # decrease lr for encoder in order not to permute
    # pre-trained weights with large gradients on training start
    {'params': model.encoder.parameters(), 'lr': 1e-6},
])
scheduler = None

runner = SupervisedRunner()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)

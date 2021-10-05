import segmentation_models_pytorch as smp
import torch
from catalyst import dl, utils
from torch.utils.data import DataLoader

import dataset as ds
import transformations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def catalyst(model, loss, optimizer, train_dl, valid_dl, epochs):
    loaders = {
        "train": train_dl,
        "valid": valid_dl
    }
    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )
    runner.train(
        model=model,
        loaders=loaders,
        criterion=loss,
        optimizer=optimizer,
        num_epochs=epochs,
        logdir="./logs4",
        verbose=True,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            dl.IOUCallback(input_key="logits", target_key="targets"),
            dl.DiceCallback(input_key="logits", target_key="targets"),
            dl.TrevskyCallback(input_key="logits", target_key="targets", alpha=0.2),

        ]
    )


def main(mode):
    """
    Main methode. Hier werden die Data loader abgerufen, das Neurale netzwerk auf das Device transferiert. Als Loss Function wird CrossEntropyLoss gewählt und also
    optimizer Adam mit einer Learning rate von 1*10^-3. Schließlich wird für eine gegebene epoch zahl jedes mal eine Trainings und Test loop ausgeführt.
    """

    ENCODER = "resnet152"
    ENCODER_WEIGHTS = "imagenet"

    model = smp.Unet(classes=4, activation="softmax2d", encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dl, valid_dl = getLoaders(preprocessing_fn)

    loss_fn = smp.utils.losses.DiceLoss()

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-4},
        {'params': model.encoder.parameters(), 'lr': 1e-6},
    ])

    epochs = 15*2
    if mode == "new":
        catalyst(model=model, loss=loss_fn, optimizer=optimizer, train_dl=train_dl, valid_dl=valid_dl, epochs=epochs)
    elif mode == "continue":
        checkpoint = utils.torch.load_checkpoint(
            r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\pytorch\Segmentation_2\logs3\checkpoints\best_full.pth")
        utils.torch.unpack_checkpoint(checkpoint=checkpoint, model=model, criterion=loss_fn, optimizer=optimizer)
        model.eval()
        catalyst(model=model, loss=loss_fn, optimizer=optimizer, train_dl=train_dl, valid_dl=valid_dl, epochs=epochs)


def getLoaders(func):
    train_ds = ds.Dataset(r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\train\img",
                          r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\train\mask", size=128 * 8,
                          augmentation=transformations.get_training_augmentation(),
                          preprocessing=transformations.get_preprocessing(preprocessing_fn=func)
                          )
    valid_ds = ds.Dataset(r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\valid\img",
                          r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\splitted\valid\mask", size=128 * 8,
                          augmentation=transformations.get_validation_augmentation(),
                          preprocessing=transformations.get_preprocessing(preprocessing_fn=func)
                          )

    train_dl = DataLoader(train_ds, batch_size=16)
    valid_dl = DataLoader(valid_ds, batch_size=4)
    return train_dl, valid_dl


if __name__ == '__main__':
    main(mode="continue")

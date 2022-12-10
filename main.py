import os
from Data.Dataloader import Images
from Models.Training import UNet_Train

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import matplotlib.pyplot as plt
import torch.nn as nn

if __name__ == "__main__":
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath="./saved_models/", save_top_k=1, monitor="val_loss", save_on_train_epoch_end=True)

    model = UNet_Train()

    logger = CSVLogger("logs", name="unet")

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=10,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=logger,
        log_every_n_steps=1
    )

    trainer.fit(
        model=model,
        datamodule=Images(batch_size=2)
    )

    samples, predictions = trainer.predict(dataloaders=Images(batch_size=1), ckpt_path='best')

    print(predictions[0].shape)

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(samples[0][0,0,:,:])
    axes[1].imshow(nn.Sigmoid()(predictions[0][0,1,:,:]))

    plt.savefig('pred.png')


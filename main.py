import os
from Data.Dataloader import ImageData
from Models.Training import UNet_Train

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath="./saved_models/", save_top_k=1, monitor="UNet_loss", save_on_train_epoch_end=True)

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=50,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    trainer.fit(
        model=UNet_Train(),
        datamodule=ImageData(
            batch_size=64)
    )

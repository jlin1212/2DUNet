from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
import numpy as np
from pytorch_lightning import LightningModule
import torch

class UNet_Train(LightningModule):
    def __init__(self, img_size=(1, 1, ??, ??, ??), batch_size=1, lr=??):
        super().__init__()

        self.save_hyperparameters()
        self.example_input_array = [torch.zeros(self.hparams.img_size)]

        self.model = BasicUNet(
            #specify model details here, including activation functions
            #documentation https://docs.monai.io/en/stable/networks.html#basicunet
                     )
        #consider changing loss types
        self.DSC_Loss = DiceLoss(include_background=False)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    #def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #    return self(batch['image'])

    def configure_optimizers(self):
        #set optimizer
        optimizer = torch.optim.??(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def _prepare_batch(self, batch):
        return batch['image'], batch['label']

    def _common_step(self, batch, batch_idx, stage: str):
        inputs, gt_input = self._prepare_batch(batch)
        outputs = self.forward(inputs)
        DSC = self.DSC_Loss(outputs, gt_input)
        train_steps = self.current_epoch + batch_idx

        #logging, should be updated for your preferences (writing losses to graphs, saving test images, etc.)
        #be sure to log training and validation losses
        print('step', float(train_steps))
        print('epoch', float(self.current_epoch))
        print('batch_size', self.hparams.batch_size)
        print(f'{stage}_DiceLoss', DSC.item())

        return DSC


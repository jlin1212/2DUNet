import os
import glob
import numpy as np
import torchvision
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    RandGaussianNoised,
    NormalizeIntensityd,
    #be sure to add any additional transforms here
    #look at other options: https://docs.monai.io/en/stable/transforms.html#dictionary-transforms
)

class ImageData(Dataset):
    def __init__(self):
        self.dataset_path = "/home/jlin1212/covid"

        self.image_path = glob.glob(os.path.join(self.dataset_path, 'frames', '*'))
        self.seg_path = glob.glob(os.path.join(self.dataset_path, 'masks', '*'))

        self.transform = Compose(
            [
                RandGaussianNoised(['image'], std=0.1*255),
                ScaleIntensityd(['image', 'label']),
                NormalizeIntensityd(['image'])
                #additional transforms here
            ]
        )

    def __len__(self):
        return len(self.dataset_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        segmentation_path = self.seg_path[index]

        image_data = torchvision.io.read_image(image_path)
        seg_data = torchvision.io.read_image(segmentation_path)

        image_transformed = self.transform({'image': image_data, 'label': seg_data})
        #additional label adjustments here, if needed
        return image_transformed

import os
import glob

import torch
import torchvision

from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    RandGaussianNoised,
    NormalizeIntensityd,
    RandRotated,
    RandZoomd,
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
                RandRotated(['image', 'label'], prob=1, range_x=0.4),
                RandZoomd(['image', 'label'], prob=1),
                RandGaussianNoised(['image'], std=0.1*255),
                ScaleIntensityd(['image', 'label']),
                NormalizeIntensityd(['image']),
            ]
        )

    def __len__(self):
        return 400

    def __getitem__(self, index):
        image_path = self.image_path[index]
        segmentation_path = self.seg_path[index]

        image_data = torchvision.io.read_image(image_path).type(torch.FloatTensor)
        image_data = torch.mean(image_data, dim=0, keepdims=True)
        seg_data = torchvision.io.read_image(segmentation_path)
        seg_data = seg_data[0:1,:,:]

        image_transformed = self.transform({'image': image_data, 'label': seg_data})

        bg_channel = torch.where(image_transformed['label'] < 0.5, 1., 0.)

        image_transformed['label'] = torch.cat([bg_channel, 1 - bg_channel], dim=0)

        # print(image_transformed['image'].shape, image_transformed['label'].shape)

        return image_transformed

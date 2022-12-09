import os
import glob
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImageD
    #be sure to add any additional transforms here
    #look at other options: https://docs.monai.io/en/stable/transforms.html#dictionary-transforms
)

class ImageData(Dataset):
    def __init__(self):
        self.dataset_path = #"your/image/path/as/string/here"
        self.raw_images_path = glob.glob(os.path.join(self.dataset_path + '/*'))

        self.transform = Compose(
            [
                LoadImageD(keys=["Image", "Label"]),
                #additional transforms here
            ]
        )

    def __len__(self):
        return len(self.dataset_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        segmentation_path = self.seg_path[index]
        image = {"image": image_path, "label": segmentation_path}
        image_transformed = self.transform(image)
        #additional label adjustments here, if needed
        return image_transformed

from Models.Training import UNet_Train
from Data.Data import ImageData

import matplotlib.pyplot as plt
import torch.nn as nn

model = UNet_Train()
model.load_from_checkpoint('/home/jlin1212/2DUNet/saved_models/epoch=3-step=320.ckpt')

dataset = ImageData()
sample = dataset[0]

fig, axes = plt.subplots(1, 3)

axes[0].imshow(sample['image'][0,:,:])
axes[1].imshow(sample['label'][0,:,:])

prediction = model(sample['image'][None,...])

axes[2].imshow(nn.Sigmoid()(prediction[0][1,:,:]).detach().numpy())

print(prediction[0].shape)

plt.savefig('demo.png')

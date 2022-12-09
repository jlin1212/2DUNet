from Data import ImageData
import torch

dataset = ImageData()
print(dataset[0])
print(torch.amax(dataset[0]['label']))
print(len(dataset))

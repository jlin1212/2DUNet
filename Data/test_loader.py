from Data import ImageData
import torch

dataset = ImageData()
# print(dataset[0])
print(torch.sum(dataset[0]['label'][1]))
print(len(dataset))

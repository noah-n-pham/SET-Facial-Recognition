import os
import sys
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim  # optim for adam
import albumentations as A
import cv2
import numpy as np
import shutil

nameList = ["tyler", "ben", "james", "rishab",
            "noah", "joyce", "nate", "janav", "hoek"]

# nameList = ["tyler", "ben"]

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Illumination(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.Resize(224, 224),
    A.GaussNoise(std_range=(0.05, 0.05), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomGamma(p=0.3),
    A.CLAHE(p=0.2),
    # A.Normalize(mean=[0.5, 0.5,0.5],std=[0.5,0.5, 0.5]),
])


# count = 0
# for name in nameList:
#     count = 0
#     for i in range(0, 100, 5):
#         image = cv2.imread("Dataset/"+name+"/"+name+"_"+str(i)+".png")
#         augmented_image = transform(image=image)["image"]
#         augmented_image2 = transform(image=image)["image"]
#         augmented_image3 = transform(image=image)["image"]
#         augmented_image4 = transform(image=image)["image"]
#         augmented_image5 = transform(image=image)["image"]

#         cv2.imwrite(
#             f"AlbumentationAugments/{name}/{name}_augmented_{count}.png", augmented_image)
#         count += 1
#         cv2.imwrite(
#             f"AlbumentationAugments/{name}/{name}_augmented_{count}.png", augmented_image2)
#         count += 1
#         cv2.imwrite(
#             f"AlbumentationAugments/{name}/{name}_augmented_{count}.png", augmented_image3)
#         count += 1
#         cv2.imwrite(
#             f"AlbumentationAugments/{name}/{name}_augmented_{count}.png", augmented_image4)
#         count += 1
#         cv2.imwrite(
#             f"AlbumentationAugments/{name}/{name}_augmented_{count}.png", augmented_image5)
#         count += 1

train_data = datasets.ImageFolder('TrainingData/train', transform=transform)
val_data = datasets.ImageFolder('TrainingData/val', transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)
model = models.resnet18(pretrained=True)


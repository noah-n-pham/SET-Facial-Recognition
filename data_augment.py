import os
import sys
import torch
import albumentations as A
import cv2
import numpy as np

nameList = ["tyler","ben","james","rishab","noah","joyce","nate","janav","hoek"]


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Illumination(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.AdditiveNoise(p=0.5),
    A.Blur(blur_limit=7, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.HueSaturationValue(p=0.5),
])
count = 0
for i in range(0,100,5):
    image = cv2.imread("Dataset/tyler/tyler_"+str(i)+".png") 
    augmented_image = transform(image=image)["image"]
    cv2.imwrite(f"AlbumentationAugments/tyler/tyler_augmented_{count}.png", augmented_image)
    count+=1

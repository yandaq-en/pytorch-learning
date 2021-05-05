import cv2
import albumentations as A
import numpy as np
from utils import plot_examples

image = cv2.cvtColor(cv2.imread("../dataset/IMAGES/cat.jpg"), cv2.COLOR_BGR2RGB)
bboxes = [[13, 170, 224, 410]]

transform = A.Compose([
    A.Resize(width=1920, height=1080),
    A.RandomCrop(width=1280, height=720),
    A.Rotate(limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5),
    ], p=1.)
], bbox_params=A.BboxParams(format="pascal_voc", min_area=2048, min_visibility=0.3, label_fields=[]))

images_list = [image]
saved_bboxes = [bboxes[0]]
for i in range(15):
    augmentations = transform(image=image, bboxes = bboxes)
    if len(augmentations["bboxes"]) == 0:
        continue
    images_list.append(augmentations["image"])
    saved_bboxes.append(augmentations["bboxes"][0])

plot_examples(images_list, saved_bboxes)

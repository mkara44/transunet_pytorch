import cv2
import torch
import random
import numpy as np


def flip_horizontal(img, mask):
    img = np.flip(img, axis=1)
    mask = np.flip(mask, axis=1)
    return img, mask


def rotate(img, mask, angle_abs=5):
    h, w, _ = img.shape
    angle = random.choice([angle_abs, -angle_abs])

    M = cv2.getRotationMatrix2D((h, w), angle, 1.0)
    img = cv2.warpAffine(img, M, (h, w), flags=cv2.INTER_CUBIC)
    mask = cv2.warpAffine(mask, M, (h, w), flags=cv2.INTER_CUBIC)
    mask = np.expand_dims(mask, axis=-1)
    return img, mask


class RandomAugmentation:
    augmentations = [flip_horizontal, rotate]

    def __init__(self, max_augment_count):
        if max_augment_count <= len(self.augmentations):
            self.max_augment_count = max_augment_count
        else:
            self.max_augment_count = len(self.augmentations)

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']

        augmentation_count = random.randint(0, self.max_augment_count)
        selected_augmentations = random.sample(self.augmentations, k=augmentation_count)
        for augmentation in selected_augmentations:
            img, mask = augmentation(img, mask)

        return {'img': img, 'mask': mask}

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


class BGR2RGB:
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {'img': img, 'mask': mask}


class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        img = cv2.resize(img, (self.output_size, self.output_size))
        mask = cv2.resize(mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)
        return {'img': img, 'mask': mask}


class Normalize:
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        img = img / 255.
        mask = mask / 255.
        return {'img': img, 'mask': mask}


class ToTensor:
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.astype('float32'))
        mask = mask.transpose((2, 0, 1))
        mask = torch.from_numpy(mask.astype('float32'))
        return {'img': img, 'mask': mask}

import torch
import numpy as np

from datasets_variables import *
from torchvision import transforms
from logging import getLogger

logger = getLogger('datasets-manager')


def get_loader(dataset_name, batch_size, num_loader_workers=3, train=True):
    if dataset_name not in datasets_dict:
        logger.error('Dataset' + dataset_name + 'is not supported')
    if train:
        return __get_train_loader(dataset_name, batch_size, num_loader_workers)
    return __get_test_loader(dataset_name, batch_size)


def __get_train_loader(dataset_name, batch_size,num_loader_workers):
    train_set = datasets_dict[dataset_name](root='./data', train=True,
                                            download=True, transform=ImageTransforms.basic_transform())
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                       shuffle=True, num_workers=num_loader_workers, pin_memory=True)


def __get_test_loader(dataset_name, batch_size):
    test_set = datasets_dict[dataset_name](root='./data', train=False, download=True, transform=ImageTransforms.basic_transform())
    return torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


class ImageTransforms:
    ROT0, ROT90, ROT180, ROT270 = 0, 1, 2, 3

    # Rotation of single image
    @staticmethod
    def rot0(image):
        return image

    @staticmethod
    def rot90(image):
        return torch.rot90(image, ImageTransforms.ROT90, dims=[1, 2])

    @staticmethod
    def rot180(image):
        return torch.rot90(image, ImageTransforms.ROT180, dims=[1, 2])

    @staticmethod
    def rot270(image):
        return torch.rot90(image, ImageTransforms.ROT270, dims=[1, 2])

    # Rotation of entire batch with the same rotation
    @staticmethod
    def rot0_batch(batch):
        return batch

    @staticmethod
    def rot90_batch(batch):
        return torch.rot90(batch, ImageTransforms.ROT90, dims=[2, 3])

    @staticmethod
    def rot180_batch(batch):
        return torch.rot90(batch, ImageTransforms.ROT180, dims=[2, 3])

    @staticmethod
    def rot270_batch(batch):
        return torch.rot90(batch, ImageTransforms.ROT270, dims=[2, 3])

    @staticmethod
    def random_rotate_batch(batch):
        possible_rotations = [ImageTransforms.rot0, ImageTransforms.rot90, ImageTransforms.rot180,
                              ImageTransforms.rot270]
        rotated_batch = []
        rotation_labels = []
        for image in batch:
            rotation = np.random.randint(0, 4)
            rotated_image = possible_rotations[rotation](image)
            rotated_batch.append(rotated_image)
            rotation_labels.append(rotation)

        return rotated_batch, rotation_labels

    @staticmethod
    def random_rotate_entire_batch(batch):
        possible_rotations = [ImageTransforms.rot0_batch, ImageTransforms.rot90_batch,
                              ImageTransforms.rot180_batch, ImageTransforms.rot270_batch]
        rotation = np.random.randint(0, 4)
        return possible_rotations[rotation](batch), rotation

    @staticmethod
    def basic_transform():
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                               (0.5, 0.5, 0.5))])

import os.path

import torchvision

from torch.utils.data import Dataset
from skimage import io, transform

datasets_dict = {
    'cifar10': {
        'from_file': False,
        'class': torchvision.datasets.CIFAR10,
    },
    'mnist': {
        'from_file': False,
        'class': torchvision.datasets.MNIST,
    },
    'fashion_mnist': {
        'from_file': False,
        'class': torchvision.datasets.FashionMNIST,
    },
    'tiny_imagenet': {
        'from_file': True,
        'path': 'data/tiny-imagenet-200'
    }
}

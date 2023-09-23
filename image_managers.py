from abc import abstractmethod
import math
import random
from typing import List, Optional

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor, Normalize, RandomHorizontalFlip, \
    RandomCrop, Compose, Resize, ColorJitter


class IImageManager:
    @abstractmethod
    def create_train_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    @abstractmethod
    def plot_image_grid(self, images: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def denorm(self, imgs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError


class ImageManagerBase(IImageManager):
    def __init__(self, color_mean: List[float], color_std: List[float]):
        super().__init__()
        self.color_mean = color_mean
        self.color_std = color_std

    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _get_train_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def create_train_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        train_set = self._get_train_dataset()

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=4,
                                                   shuffle=True)
        return train_loader

    def denorm(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = imgs * torch.tensor(self.color_std)[None, :, None, None]
        imgs = imgs + torch.tensor(self.color_mean)[None, :, None, None]
        return torch.clamp(imgs, 0.0, 1.0)

    def plot_image_grid(self, images: torch.Tensor, margin: Optional[int] = 1) -> np.ndarray:
        images = self.denorm(images)
        images = images.permute(0, 2, 3, 1)
        images = images.numpy()

        batch_size, height, width, channels = images.shape

        grid_size = math.isqrt(batch_size)
        assert grid_size**2 == batch_size

        output_shape = (
            (height+2*margin)*grid_size,
            (width+2*margin)*grid_size,
            channels
        )

        result = np.zeros(shape=output_shape, dtype=images.dtype)

        for i in range(grid_size):
            for j in range(grid_size):
                x = i*(height+2*margin) + margin
                y = j*(width+2*margin) + margin
                result[x:x+height, y:y+width, :] = images[i*grid_size+j, ...]

        return result


class CelebAManager(ImageManagerBase):

    class Crop(torch.nn.Module):
        def __init__(self, image_size: int, image_padding: int):
            super().__init__()
            self.image_size = image_size
            self.image_padding = image_padding

        def forward(self, img):
            return torchvision.transforms.functional.crop(img, round(1.125 * self.image_padding), 0,
                                                          self.image_size + self.image_padding,
                                                          self.image_size + self.image_padding)

    def num_classes(self) -> int:
        return -1

    class Noise(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, img):
            return img + (torch.rand_like(img)-0.5) / 255.0

    def __init__(self, data_root: str):
        super().__init__(color_mean=[0.485, 0.456,
                                     0.406], color_std=[0.229, 0.224, 0.225])
        self.data_root = data_root

    def _get_train_dataset(self) -> torch.utils.data.Dataset:
        image_size = 64
        image_padding = image_size // 8
        train_transforms = Compose([
            Resize(image_size + image_padding),
            RandomHorizontalFlip(),
            CelebAManager.Crop(image_size, image_padding),
            RandomCrop(size=image_size),
            ToTensor(),
            CelebAManager.Noise(),
            Normalize(mean=self.color_mean, std=self.color_std)
        ])

        return torchvision.datasets.CelebA(root=self.data_root, split="all",
                                           download=True, transform=train_transforms)


class Cifar10Manager(ImageManagerBase):

    class Noise(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, img):
            return img + (torch.rand_like(img)-0.5) / 255.0

    class RandomResize(torch.nn.Module):
        def __init__(self, min_size: int, max_size: int):
            super().__init__()
            self.min_size = min_size
            self.max_size = max_size

        def forward(self, img):
            size = random.randint(self.min_size, self.max_size)
            return torchvision.transforms.functional.resize(
                img, size, torchvision.transforms.functional.InterpolationMode.BILINEAR,
                None, "warn")

    class OneHot(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, y):
            return torch.nn.functional.one_hot(torch.tensor(y), num_classes=10).type(torch.float32)

    def __init__(self, data_root: Optional[str] = None):
        super().__init__(color_mean=[0.485, 0.456,
                                     0.406], color_std=[0.229, 0.224, 0.225])
        self.data_root = data_root

    def num_classes(self) -> int:
        return 10

    def _get_train_dataset(self) -> torch.utils.data.Dataset:
        image_size = 32
        image_padding = image_size // 8
        train_transforms = Compose([
            RandomHorizontalFlip(),
            Cifar10Manager.RandomResize(image_size, image_size + image_padding),
            RandomCrop(size=image_size),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0),
            ToTensor(),
            Cifar10Manager.Noise(),
            Normalize(mean=self.color_mean, std=self.color_std)
        ])

        target_transforms = Compose([Cifar10Manager.OneHot()])

        return torchvision.datasets.CIFAR10(root=self.data_root, train=True,
                                            download=True, transform=train_transforms,
                                            target_transform=target_transforms)

import os
import pandas as pd
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from torchvision.transforms import functional as TF
import random



class PairedTransform:
    def __init__(self, crop_size=(240, 320), hflip_prob=0.5, rotate_degree=10):
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.rotate_degree = rotate_degree

    def __call__(self, sample):
        if self.hflip_prob == 0:
            image = sample['image']

            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
            image = TF.to_tensor(image)
            return {'image': image, 'file_name': sample['file_name']}
        else:
            image, depth, mask = sample['image'], sample['depth'], sample['mask']

            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                depth = np.fliplr(depth).copy()

            image = TF.to_tensor(image)
            depth = torch.from_numpy(depth).unsqueeze(0).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            return {'image': image, 'depth': depth, 'file_name': sample['file_name'], 'mask': mask}


def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class TestDataset(Dataset):
    def __init__(self, root_dir, data_type="folder", transform=None):
        self.root_dir = root_dir
        self.data_type = data_type
        self.transform = transform

    def __len__(self):
        if self.data_type == "image":
            return 1

        elif self.data_type == "folder":
            return len(os.listdir(self.root_dir))

    def RGB_to_AIS(self, image):
        r, g, b = image.split()
        r, g, b = np.array(r), np.array(g), np.array(b)
        r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        grad_rgb = (gradient_magnitude - np.min(gradient_magnitude)) / (
                np.max(gradient_magnitude) - np.min(gradient_magnitude))
        r2gb = np.sqrt((r - g) ** 2 + (r - b) ** 2)
        gray_c = np.array(ImageOps.grayscale(image))
        combined = np.stack((grad_rgb, r2gb, gray_c), axis=-1)
        return Image.fromarray(combined)

    def __getitem__(self, idx):

        if self.data_type == "image":
            img_fn = self.root_dir
            print(img_fn)

        elif self.data_type == "folder":
            img_fn = os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])

        image = (Image.open(img_fn)).resize((640,480))
        img_fn = img_fn.split("/")[-1]

        sample1 = {'image': image, 'file_name': img_fn}
        if self.transform:
            sample1 = self.transform(sample1)
        return sample1


def RGB_to_AIS_fast(images: torch.Tensor) -> torch.Tensor:

    assert images.ndim == 4 and images.size(1) == 3, "images = [B,3,H,W]"
    device = images.device
    dtype  = images.dtype if images.dtype.is_floating_point else torch.float32

    x = images.to(dtype)
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=dtype).view(1,3,1,1)
    gray = (x * w).sum(dim=1, keepdim=True)

    kx = torch.tensor([[1., 0., -1.],
                       [2., 0., -2.],
                       [1., 0., -1.]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[1.,  2.,  1.],
                       [0.,  0.,  0.],
                       [-1., -2., -1.]], device=device, dtype=dtype).view(1,1,3,3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy + torch.finfo(dtype).eps)

    B = grad_mag.size(0)
    gmin = grad_mag.view(B, -1).min(dim=1).values.view(B,1,1,1)
    gmax = grad_mag.view(B, -1).max(dim=1).values.view(B,1,1,1)
    grad_rgb = (grad_mag - gmin) / (gmax - gmin + torch.finfo(dtype).eps)

    r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
    r2gb = torch.sqrt((r - g) ** 2 + (r - b) ** 2 + torch.finfo(dtype).eps)
    out = torch.cat([grad_rgb, r2gb, gray], dim=1)
    return out


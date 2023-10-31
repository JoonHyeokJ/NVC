from . import augmentations_augmix
import numpy as np
from PIL import Image

from torch import nn, Tensor
import torch
from torchvision.transforms import ToPILImage
from torchvision import transforms

# # CIFAR-10 constants
# MEAN = [0.4914, 0.4822, 0.4465]
# STD = [0.2023, 0.1994, 0.2010]

# ImageNet-1k constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)

def denormalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = image * std[:, None, None] + mean[:, None, None]
  return image.transpose(1, 2, 0)

def apply_op(image, op, severity):
  image = np.clip(image * 255., 0, 255).astype(np.uint8)
  pil_img = Image.fromarray(image)  # Convert to PIL.Image
  pil_img = op(pil_img, severity)
  return np.asarray(pil_img) / 255.


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1., img_size=224):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.

  Returns:
    mixed: Augmented and mixed image.
  """
  augmentations_augmix.IMAGE_SIZE = img_size

  ws = np.float32(
      np.random.dirichlet([alpha] * width))
  m = np.float32(np.random.beta(alpha, alpha))

  mix = np.zeros_like(image)
  for i in range(width):
    image_aug = image.copy()
    d = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(d):
      op = np.random.choice(augmentations_augmix.augmentations)
      image_aug = apply_op(image_aug, op, severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * normalize(image_aug)

  mixed = (1 - m) * normalize(image) + m * mix
  return mixed

from copy import deepcopy

class Augmix(nn.Module):    # augmix for an image batch
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], aug_severity=3, all_ops=False,
                 mixture_width=3, mixture_depth=-1):
        super(Augmix, self).__init__()
        self.mean = mean
        self.std = std
        self.severity = aug_severity
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.aug_list = augmentations_augmix.augmentations_all if all_ops else augmentations_augmix.augmentations
        self.width = mixture_width
        self.depth = mixture_depth

    @torch.no_grad()
    def forward(self, img_batch: Tensor):
        B, C, H, W = img_batch.shape
        augmentations_augmix.IMAGE_SIZE = H
        temp = []
        for img in img_batch:
            img_augmix = self.forward_single_img(img)
            temp.append(img_augmix)
        new_batch = torch.stack(temp)
        return new_batch

    @torch.no_grad()
    def forward_single_img(self, img: Tensor):
        device_img = img.device.type
        # 1. denorm & ToPIL
        mean = torch.Tensor(self.mean).to(device_img)
        std = torch.Tensor(self.std).to(device_img)
        denorm_img = deepcopy(img).mul(std[:,None,None]).add(mean[:,None,None])
        denorm_img = ToPILImage()(denorm_img)
        
        # 2. augmix + ToTensor & norm(renorm)
        ws = np.float32(np.random.dirichlet([1] * self.width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(self.preprocess(denorm_img))
        for i in range(self.width):
            image_aug = denorm_img.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self.preprocess(image_aug)

        mixed = (1 - m) * self.preprocess(denorm_img) + m * mix
        return mixed


if __name__ == '__main__':
    from PIL import Image
    img = Image.open('./test_img/real_033_000211.jpg').resize((224, 224))
    np_float_img = np.asarray(img)/255
    
    augmix_img = augment_and_mix(np_float_img, img_size=np_float_img.shape[0])
    # breakpoint()

    # augmix_img1 = Image.fromarray(((augmix_img-augmix_img.min())/(augmix_img.max()-augmix_img.min())*255).astype(np.uint8))
    augmix_img1 = Image.fromarray(((augmix_img-np.amin(augmix_img, axis=(0,1)))/(np.amax(augmix_img, axis=(0,1))-np.amin(augmix_img, axis=(0,1)))*255).astype(np.uint8))
    # augmix_img2 = Image.fromarray(((augmix_img-augmix_img.min())/(augmix_img.max()-augmix_img.min())*255).astype(np.uint8), 'RGB')
    augmix_img1.save('./runs/result1.jpg')
    # augmix_img2.save('./runs/result2.jpg')
    # breakpoint()

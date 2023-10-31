import os
from glob import glob
import torch
from torch import nn, Tensor
from torchvision import transforms
import copy
from PIL import Image
import copy
import numpy as np
import math
# import cv2
import random

class P_Shuffle_single_img(object):
    def __init__(self, patch_size=16):  # The default values are determined in assumption that the size of input image is 224*224.
        self.patch_size = patch_size

    def __call__(self, img):    # img must be a tensor whose shape has the form of (C, H, W).
                                # img must be a tensor to which all the transforms which can affect the tensor's shape (e.g. Resize, RandomCrop, etc.) are applied.
        temp = torch.zeros_like(img)
        
        patches, n_patch_vertical, n_patch_horizontal, height_patch, width_patch = split_and_arrange_img(img, self.patch_size)
        random.shuffle(patches)
        
        for idx, patch in enumerate(patches):
            patch_idx_vertical = idx // n_patch_horizontal
            patch_idx_horizontal = idx % n_patch_horizontal
            temp[:, patch_idx_vertical*height_patch : (patch_idx_vertical+1)*height_patch, patch_idx_horizontal*width_patch : (patch_idx_horizontal+1)*width_patch] = patch

        resultant_img = temp
        return resultant_img

class P_Selective_Shuffle_single_img(object):
    def __init__(self, patch_size=16, ratio=1.):  # The default values are determined in assumption that the size of input image is 224*224.
        self.patch_size = patch_size
        self.ratio = ratio

    def __call__(self, img):    # img must be a tensor whose shape has the form of (C, H, W).
                                # img must be a tensor to which all the transforms which can affect the tensor's shape (e.g. Resize, RandomCrop, etc.) are applied.
        temp = torch.zeros_like(img)
        
        patches, n_patch_vertical, n_patch_horizontal, height_patch, width_patch = split_and_arrange_img(img, self.patch_size)
        # random.shuffle(patches)
        n_patch = n_patch_vertical*n_patch_horizontal
        n_select = round(n_patch*self.ratio)
        idx_select_rand_order = random.sample(range(n_patch),n_select)
        idx_select_sorted = sorted(idx_select_rand_order)
        
        selected_patch_rand_order = [patches[i] for i in idx_select_rand_order]
        for i, idx_sorted in enumerate(idx_select_sorted):
            patches[idx_sorted] = selected_patch_rand_order[i]
        
        for idx, patch in enumerate(patches):
            patch_idx_vertical = idx // n_patch_horizontal
            patch_idx_horizontal = idx % n_patch_horizontal
            temp[:, patch_idx_vertical*height_patch : (patch_idx_vertical+1)*height_patch, patch_idx_horizontal*width_patch : (patch_idx_horizontal+1)*width_patch] = patch

        resultant_img = temp
        return resultant_img


def split_and_arrange_img(img, patch_size):
    h, w = img.shape[-2:]
    
    if isinstance(patch_size, int) and h % patch_size==0 and w % patch_size==0:
        h_patch = w_patch = patch_size
        num_patch_vertical = h // patch_size
        num_patch_horizontal = w // patch_size
        num_patch = num_patch_vertical * num_patch_horizontal
    elif isinstance(patch_size, (tuple, list)) and len(patch_size)==2 and h % patch_size==0 and w % patch_size==0:
        h_patch = patch_size[0]
        w_patch = patch_size[1]
        num_patch_vertical = h // h_patch
        num_patch_horizontal = w // w_patch
        num_patch = num_patch_vertical * num_patch_horizontal
    else: raise Exception("Please check the size of image or patch_size")
    
    patches = []
    for i in range(num_patch_vertical):
        for j in range(num_patch_horizontal):
            single_patch = img[:, i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch]
            patches.append(single_patch)
    return patches, num_patch_vertical, num_patch_horizontal, h_patch, w_patch
    
class P_Shuffle(nn.Module):
    def __init__(self, patch_size):
        super(P_Shuffle, self).__init__()

        self.patch_size = patch_size
        self.shuffle = P_Shuffle_single_img(self.patch_size)

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        temporal_list = []
        for i, single_img in enumerate(img):
            neg_aug_img = self.shuffle(single_img)
            temporal_list.append(neg_aug_img)
        # breakpoint()
        neg_aug_batch = torch.stack(temporal_list, dim=0)

        return neg_aug_batch

class P_Selective_Shuffle(nn.Module):
    def __init__(self, patch_size, ratio):
        super(P_Selective_Shuffle, self).__init__()

        self.patch_size = patch_size
        self.ratio = ratio
        self.shuffle = P_Selective_Shuffle_single_img(self.patch_size, self.ratio)

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        temporal_list = []
        for i, single_img in enumerate(img):
            neg_aug_img = self.shuffle(single_img)
            temporal_list.append(neg_aug_img)
        # breakpoint()
        neg_aug_batch = torch.stack(temporal_list, dim=0)

        return neg_aug_batch
    
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='testing p-shuffle')
    parser.add_argument('--resize', default=448, type=int)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--img_path', default="./image.jpg")
    parser.add_argument('--save_dir', default='./runs/p-shuffle')
    parser.add_argument('--result_name', default='result.jpg')
    parser.add_argument('--selective', action='store_true')
    parser.add_argument('--ratio', default=0.5, type=float)
    
    args = parser.parse_args()
    print(args)
    
    if not args.selective:
        transform = transforms.Compose([transforms.PILToTensor(),
                                        transforms.Resize((args.resize, args.resize)),
                                        P_Shuffle_single_img(args.patch_size),
                                        transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.PILToTensor(),
                                        transforms.Resize((args.resize, args.resize)),
                                        P_Selective_Shuffle_single_img(args.patch_size, args.ratio),
                                        transforms.ToPILImage()])
    
    img = Image.open(args.img_path)
    img_trans = transform(img)
    
    os.makedirs(args.save_dir, exist_ok=True)
    img_trans.save(os.path.join(args.save_dir, args.result_name))
    
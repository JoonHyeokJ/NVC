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

class P_Infill_single_img(object):
    def __init__(self, replace_rate):  # replace_rate: Now implemented in the case of 0.25 and 0.375.
        self.replace_rate = replace_rate
        if not (self.replace_rate in [0.25, 0.375, '0.25', '0.375']): raise Exception("Sorry, but the case of replace_rate {} is not implemented.".format(self.replace_rate))
        
    def __call__(self, img):    # img must be a tensor whose shape has the form of (C, H, W).
                                # img must be a tensor to which all the transforms which can affect the tensor's shape (e.g. Resize, RandomCrop, etc.) are applied.
        temp = copy.copy(img)
        
        h, w = img.shape[-2:]
        if self.replace_rate == 0.25 or self.replace_rate == '0.25':
            assert h % 4 == 0 and w % 4 == 0, "Invalid h: {} or w: {}".format(h, w)
            height_patch, width_patch = h // 4, w // 4
            n_patch_vertical, n_patch_horizontal = 4, 4
            
            raider1 = temp[:, 0:height_patch*1, 0:width_patch*2]
            raider2 = temp[:, height_patch*(n_patch_vertical-1):height_patch*n_patch_vertical, width_patch*(n_patch_horizontal-2):width_patch*n_patch_horizontal]

            temp[:, height_patch:height_patch*2, width_patch:width_patch*3] = raider1
            temp[:, height_patch*2:height_patch*3, width_patch:width_patch*3] = raider2
            
            instilled_img = temp
            return instilled_img
        elif self.replace_rate == 0.375 or self.replace_rate == '0.375':
            assert h % 4 == 0 and w % 8 == 0, "Invalid h: {} or w: {}".format(h, w)
            height_patch, width_patch = h // 4, w // 8
            n_patch_vertical, n_patch_horizontal = 4, 8
            
            raider1 = temp[:, 0:height_patch*1, 0:width_patch*6]
            raider2 = temp[:, height_patch*(n_patch_vertical-1):height_patch*n_patch_vertical, width_patch*(n_patch_horizontal-6):width_patch*n_patch_horizontal]

            temp[:, height_patch:height_patch*2, width_patch:width_patch*7] = raider1
            temp[:, height_patch*2:height_patch*3, width_patch:width_patch*7] = raider2
            
            instilled_img = temp
            return instilled_img
            
        else:
            raise Exception("Invalid replace_rate: {}".format(self.replace_rate))
            return None
        
        
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

class P_Infill(nn.Module):
    def __init__(self, replace_rate):
        super(P_Infill, self).__init__()

        self.replace_rate = replace_rate
        self.infill = P_Infill_single_img(self.replace_rate)

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        temporal_list = []
        for i, single_img in enumerate(img):
            neg_aug_img = self.infill(single_img)
            temporal_list.append(neg_aug_img)
        # breakpoint()
        neg_aug_batch = torch.stack(temporal_list, dim=0)

        return neg_aug_batch
    
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='testing p-infill')
    parser.add_argument('--resize', default=640, type=int)
    parser.add_argument('--img_path', default='./1538130_3.jpg')
    parser.add_argument('--save_dir', default='./runs/p-infill')
    parser.add_argument('--replace_rate', type=float, default=0.25)
    parser.add_argument('--result_name', default='result.jpg')
    
    args = parser.parse_args()
    print(args)
    
    transform = transforms.Compose([transforms.PILToTensor(),
                                    transforms.Resize((args.resize, args.resize)),
                                    P_Infill(args.replace_rate),
                                    transforms.ToPILImage()])
    
    img = Image.open(args.img_path)
    img_trans = transform(img)
    
    os.makedirs(os.path.join(args.save_dir, 'rep_rate{}'.format(args.replace_rate)), exist_ok=True)
    img_trans.save(os.path.join(args.save_dir, 'rep_rate{}'.format(args.replace_rate), args.result_name))
    
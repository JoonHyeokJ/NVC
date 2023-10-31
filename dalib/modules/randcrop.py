# import random
# import warnings

# import kornia
import numpy as np
import torch
# from einops import repeat
from torch import nn, Tensor
# from torch.nn import functional as F
from torchvision import transforms

# warnings.filterwarnings("ignore", category=DeprecationWarning)

class RandCropForBatch(nn.Module):
    def __init__(self, size=224, denorm_and_toPIL=True, s=1, p_colorjitter=0.8, p_grayscale=0.2, gaussian_blur=True, p_horizontalflip=0,
                 norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
        super(RandCropForBatch, self).__init__()

        # self.block_size = block_size
        # self.ratio = ratio
        
        self.denorm_and_toPIL = denorm_and_toPIL
        self.gaussian_blur = gaussian_blur
        self.color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(p=p_horizontalflip),
                                              transforms.RandomApply([self.color_jitter], p=p_colorjitter),
                                              transforms.RandomGrayscale(p=p_grayscale)])
        self.gaussian = GaussianBlur(kernel_size=int(0.1 * size))
        self.pil2tensor = transforms.ToTensor()
        self.tensor2pil = transforms.ToPILImage()
        self.norm = transforms.Normalize(mean=norm_mean, std=norm_std)
        

    @torch.no_grad()
    def forward(self, image_batch: Tensor):
        device = image_batch.device.type
        img_list = []
        for img in image_batch:
            #####
            # import os
            # os.makedirs('./img_log/', exist_ok=True)
            # tmp = self.tensor2pil(img)
            # tmp.save('./img_log/denorm_and_toPIL_before_denorm.jpg')
            #####
            if self.denorm_and_toPIL:
                img = denorm_single_image(img)
                img = self.tensor2pil(img)
                #####
                # import os
                # os.makedirs('./img_log/', exist_ok=True)
                # img.save('./img_log/denorm_and_toPIL.jpg')
                # breakpoint()
                #####
            
            img = self.data_transforms(img)
            
            if self.gaussian_blur:
                img = self.gaussian(img)
            
            #####
            # import os
            # os.makedirs('./img_log/', exist_ok=True)
            # tmp = img
            # tmp.save('./img_log/after_randcrop1.jpg')
            # breakpoint()
            #####
            
            img = self.pil2tensor(img)
            img = self.norm(img)
            
            #####
            # import os
            # os.makedirs('./img_log/', exist_ok=True)
            # tmp = self.tensor2pil(img)
            # tmp.save('./img_log/after_norm.jpg')
            # breakpoint()
            #####
            
            img_list.append(img)
        
        img_batch = torch.stack(img_list, dim=0)
        img_batch = img_batch.to(device)
        return img_batch
        
        # img = img.clone()
        # B, _, H, W = img.shape

        # if self.augmentation_params is not None:
        #     img = strong_transform(self.augmentation_params, data=img.clone())

        # mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        # input_mask = torch.rand(mshape, device=img.device)
        # input_mask = (input_mask > self.ratio).float()
        # input_mask = resize(input_mask, size=(H, W))
        # masked_img = img * input_mask

        # return masked_img

def denorm_single_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # image: tensor with shape CHW
    from copy import deepcopy
    temp = deepcopy(image)
    device_img = image.device.type
    mean_tensor = torch.Tensor(mean)[:,None,None].to(device_img)
    std_tensor = torch.Tensor(std)[:,None,None].to(device_img)
    return temp.mul(std_tensor).add(mean_tensor)

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

if __name__ == '__main__':
    from PIL import Image
    image_path = './image.jpg'
    img = Image.open(image_path)
    
    random_crop = RandCropForBatch(size=448, denorm_and_toPIL=False)
    tf = random_crop.data_transforms
    blur = random_crop.gaussian

    result = tf(img)
    result = blur(result)

    import os
    os.makedirs('./img_log/', exist_ok=True)
    tmp = result
    tmp.save('./img_log/after_randcrop1.jpg')
    # breakpoint()

# I borrowed a code of the function 'rollout' and 'show_mask_on_image' from https://hongl.tistory.com/234
import cv2
import torch

import numpy as np

from PIL import Image
from torchvision import transforms

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="max",
        discard_ratio=0.9, return_pred=False):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []
        self.return_pred = return_pred

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            # print("We are in the inference step...")
            output = self.model(input_tensor)
        if self.return_pred:
            return rollout(self.attentions, self.discard_ratio, self.head_fusion), output
        else:
            return rollout(self.attentions, self.discard_ratio, self.head_fusion), None

def rollout(attentions, discard_ratio, head_fusion):
    '''
    attentions: a list of 12 tensors, each of which has a shape of (1,3,197,197])
    '''
    result = torch.eye(attentions[0].size(-1))
    
    # breakpoint()
    with torch.no_grad():
        # print("We are in the step of rollout...")
        for i, attention in enumerate(attentions):
            # if not(i == len(attentions)-1): continue
            # print(i)

            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), dim=-1, largest=False)
            indices = indices[indices != 0]
            flat[0, indices] = 0
            attention_heads_fused = flat.view(-1, attention_heads_fused.size(1), attention_heads_fused.size(2))
            
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result) # [1,197,197]

    # print("Making mask...")
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[:, 0 , 1 :] # [1,196]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask # [14,14]   

def show_mask_on_image(img, mask, grayscale=False, mask_resize=True):
    np_img = np.array(img) # pil -> ndarray
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0])) if mask_resize else mask
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    img = np.float32(img) / 255
    img = (img - img.min()) / (img.max() - img.min())
    
    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET) if not grayscale else np.repeat(np.expand_dims(np.uint8(255 * mask), axis=2), repeats=3, axis=2)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET) if not grayscale else cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_BONE)
    heatmap = heatmap[:,:, ::-1] # bgr -> rgb
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    result = np.uint8(255 * cam)
    result = Image.fromarray(result)
    return result

if __name__ == '__main__':
    import os

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),    # ImageNet
        ])
    img_path = "/your/path/to/test_image.jpg"
    img_original = Image.open(img_path)
    input_tensor = transform(img_original).unsqueeze(0)
    # breakpoint()
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    ####### create model ######
    ckpt = "/your/path/to/best.pth"
    import sys
    from torch import nn
    sys.path.append('../')
    import examples.utils as utils
    from dalib.adaptation.cdan import ImageClassifier
    num_classes = 12
    arch = 'vit_base_patch16_224'
    pretrained = True
    backbone = utils.get_model(arch, pretrain=pretrained)
    pool_layer = nn.Identity() if 'vit' in arch else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=256,
                             pool_layer=pool_layer, finetune=pretrained)
    
    classifier.load_state_dict(torch.load(ckpt))
    print("The given ckpt is successfully loaded!!")

    if torch.cuda.is_available():
        classifier = classifier.cuda()
    classifier.eval()
    
    ### attention rollout
    vit_attn_rollout = VITAttentionRollout(classifier, head_fusion="mean", discard_ratio=0.3)
    attn_mask = vit_attn_rollout(input_tensor)
    print(classifier(input_tensor))

    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        ])

    vis_result = show_mask_on_image(t(img_original), attn_mask, grayscale=False)
    img_name = img_path.split('/')[-1]
    os.makedirs('./result', exist_ok=True)
    vis_result.save('./result/{}'.format(img_name))


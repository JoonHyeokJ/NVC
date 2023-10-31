from vit_attention_rollout import VITAttentionRollout, show_mask_on_image
import torch
from torch import nn
import argparse
from tqdm import tqdm
import os
import numpy as np
import random
import torchvision.transforms as T
from torch.utils.data import DataLoader
import timm

import sys
sys.path.append('../')

import tsne.SDAT.examples.utils as utils
import tsne.dataset_benchmark as dataset_benchmark
import tsne.dataset_retail as dataset_retail



def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
          (h, w), output size will be matched to this. If size is an int,
          output size will be (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def create_sdat_feature_encoder(args, num_classes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.method == 'source_only': from tsne.source_only.classifier import ImageClassifier
    else: from tsne.SDAT.dalib.adaptation.cdan import ImageClassifier
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=True)
    # print(backbone)
    pool_layer = nn.Identity() if args.no_pool else None  # args.no_pool is usually True.
    # classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
    #                              pool_layer=pool_layer, finetune=not args.scratch).to(device)
    trained_state_dict = torch.load(args.ckpt_path)
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                             pool_layer=pool_layer, finetune=True)
    classifier.load_state_dict(trained_state_dict)
    print('The trained model is loaded from {}!!'.format(args.ckpt_path))
    classifier = torch.nn.DataParallel(classifier).to(device)
    classifier.eval()
    return classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizing attention of ViT when it comes to a dataset')
    # dataset parameters
    parser.add_argument('--use_ImageNet_model', action='store_true',
                        help='True: using ImageNet model / False: using the model trained by UDA')
    parser.add_argument('--ckpt_path', metavar='DIR_CKPT', default='',
                        help='root path of trained model')
    parser.add_argument('--dataset_name', default='Office-Home', help='What is the name of dataset?')
    # parser.add_argument('-s', '--source', default='Ar', help='source domain(s)')
    parser.add_argument('-t', '--target', default='Cl', help='target domain(s)')
    parser.add_argument('--root', default='data/Office-Home', help='root path of dataset')
    parser.add_argument('--method', default='sdat', help='The name of UDA method which you want to test')
    # parser.add_argument('--gpus', default=0, help='gpu index(indices)')

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')  # 'resnet50', 'resnet101', 'vit_small_patch16_224', 'vit_tiny_patch16_224'

    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--num_class_model', default=None, type=int, help='The number of neurons in 1ast layer of classifier model')

    # parameters
    # parser.add_argument('-b', '--batch-size', default=1, type=int,
    #                     metavar='N',
    #                     help='mini-batch size (default: 1)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')

    parser.add_argument("--log", type=str, default='./logs',
                        help="Where to save logs, checkpoints and debugging images.")
    
    parser.add_argument("--result_path", type=str, required=True,
                        help="The path to save the resultant dataset overlapped by attention visualization.")
    
    parser.add_argument("--clarify_pred", action='store_true', help="Add the prediction result of each image into the name of resultant image.")

    # VITAttentionRollout
    parser.add_argument("--rollout_head_fusion", type=str, default='max',
                        help="The path to save the resultant dataset overlapped by attention visualization.")

    # For SDAT
    parser.add_argument("--no-pool", action='store_true')

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform_wo_norm = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        # T.PILToTensor()
    ])

    if 'retail' in args.dataset_name or 'Retail' in args.dataset_name:
        # source_dataset = dataset_product.Retail(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_retail.Retail(root=args.root, task=args.target, transform=val_transform, ret_img_path=True)
        target_dataset_vis = dataset_retail.Retail(root=args.root, task=args.target, transform=val_transform_wo_norm, ret_img_path=True)
    elif 'Office-Home'==args.dataset_name or 'OfficeHome'==args.dataset_name:
        # source_dataset = dataset_benchmark.OfficeHome(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_benchmark.OfficeHome(root=args.root, task=args.target, transform=val_transform, ret_img_path=True)
        target_dataset_vis = dataset_benchmark.OfficeHome(root=args.root, task=args.target, transform=val_transform_wo_norm, ret_img_path=True)
    elif 'Office-31'==args.dataset_name or 'Office31'==args.dataset_name:
        # source_dataset = dataset_benchmark.Office31(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_benchmark.Office31(root=args.root, task=args.target, transform=val_transform, ret_img_path=True)
        target_dataset_vis = dataset_benchmark.Office31(root=args.root, task=args.target, transform=val_transform_wo_norm, ret_img_path=True)
    elif 'VisDA2017'==args.dataset_name:
        # source_dataset = dataset_benchmark.VisDA2017(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_benchmark.VisDA2017(root=args.root, task=args.target, transform=val_transform, ret_img_path=True)
        target_dataset_vis = dataset_benchmark.VisDA2017(root=args.root, task=args.target, transform=val_transform_wo_norm, ret_img_path=True)
    elif 'ImageCLEF'==args.dataset_name:
        # source_dataset = dataset_benchmark.ImageCLEF(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_benchmark.ImageCLEF(root=args.root, task=args.target, transform=val_transform, ret_img_path=True)
        target_dataset_vis = dataset_benchmark.ImageCLEF(root=args.root, task=args.target, transform=val_transform_wo_norm, ret_img_path=True)
    else:
        print('Please check args.datset_name')
        raise
    # source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    target_loader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_class_model = args.num_class_model if args.num_class_model is not None else target_dataset.num_classes

    if args.use_ImageNet_model:
        print("Let's use the ImageNet model.")
        print("Model: {}".format(args.arch))
        feature_encoder = timm.create_model(args.arch, pretrained=True)
        feature_encoder.reset_classifier(0)
        feature_encoder = feature_encoder.to(device)
        feature_encoder = torch.nn.DataParallel(feature_encoder)
    else:
        assert args.method != '', 'Please check args.method!\nargs.method: {}'.format(args.method)
        assert args.ckpt_path != '', 'Please check args.ckpt_path!\nargs.method: {}'.format(args.ckpt_path)
        print("Let's use the model trained by {}.".format(args.method))
        if 'sdat' in args.method or 'SDAT' in args.method or 'negaug' in args.method or 'source_only' == args.method:
            feature_encoder = create_sdat_feature_encoder(args, num_classes=num_class_model)
        else:
            print('The method {} is not implemented.'.format(args.method))

    feature_encoder.eval()

    ### attention rollout
    vit_attn_rollout = VITAttentionRollout(feature_encoder, head_fusion=args.rollout_head_fusion, discard_ratio=0.9, return_pred=args.clarify_pred)
    for ((img, label, path), (img_vis, _, path2)) in zip(tqdm(target_dataset), target_dataset_vis):
        assert path==path2
        if len(img.shape) < 4:
            img = img[None,:]
        attn_mask, pred = vit_attn_rollout(img)
        vis_result = show_mask_on_image(img_vis, attn_mask, grayscale=False)
        # os.makedirs('./tmp', exist_ok=True)
        # vis_result.save('./tmp/result.jpg')
        original_label_folder_name = path.split('/')[-2]
        original_img_name = path.split('/')[-1]
        # breakpoint()
        if args.clarify_pred and pred is not None:
            img_name = "prediction-{}_".format(torch.argmax(pred).item()) + original_img_name
        else:
            img_name = original_img_name
        
        os.makedirs(os.path.join(args.result_path, original_label_folder_name), exist_ok=True)
        vis_result.save(os.path.join(args.result_path, original_label_folder_name, img_name))
        # breakpoint()

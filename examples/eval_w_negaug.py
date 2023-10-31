# Credits: https://github.com/thuml/Transfer-Learning-Library
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
# import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../')
from dalib.adaptation.cdan import ImageClassifier
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter

from dalib.modules.p_infill import P_Infill
from dalib.modules.p_shuffle import P_Shuffle, P_Selective_Shuffle
from dalib.modules.p_rotate import P_Rotate, P_Selective_Rotate
from dalib.modules.p_shuffle_rotate import P_Shuffle_Rotate

from common.utils.logger import CompleteLogger

sys.path.append('.')
import utils

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase, args)
    # if args.log_results:
    #     wandb.init(project="DA", entity="SDAT", name=args.log_name)
    #     wandb.config.update(args)

    cudnn.benchmark = True
    device = args.device
    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source,
                          args.target, train_transform, val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    print(backbone)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

    if args.neg_aug_type == 'shuffle':
        if args.neg_aug_ratio is None:
            negative_aug = P_Shuffle(args.neg_aug_patch_size)
        else:
            print('\nargs.neg_aug_ratio is {}!!\nP_Selective_Shuffle is used!\n'.format(args.neg_aug_ratio))
            negative_aug = P_Selective_Shuffle(args.neg_aug_patch_size, args.neg_aug_ratio)
    elif args.neg_aug_type == 'rotate':
        if args.neg_aug_ratio is None:
            negative_aug = P_Rotate(args.neg_aug_patch_size)
        else:
            print('\nargs.neg_aug_ratio is {}!!\nP_Selective_Rotate is used!\n'.format(args.neg_aug_ratio))
            negative_aug = P_Selective_Rotate(args.neg_aug_patch_size, args.neg_aug_ratio)
    elif args.neg_aug_type == 'infill':
        negative_aug = P_Infill(args.neg_aug_replace_rate)
    elif args.neg_aug_type == 'shuffle_rotate' or args.neg_aug_type == 'shufrot':
        print('\nP_Shuffle_Rotate is used with args.neg_aug_ratio {}!!\n'.format(args.neg_aug_ratio))
        negative_aug = P_Shuffle_Rotate(args.neg_aug_patch_size, args.neg_aug_ratio)
    else:
        raise Exception("Invalid value of args.neg_aug_type: {}".format(args.neg_aug_type))

    # resume from the best checkpoint
    if args.phase != 'train':
        # path = args.weight_path
        path = logger.get_checkpoint_path('best')
        print(f"[INFORMATION] Using the weights stored at {path}")
        classifier.load_state_dict(torch.load(path, map_location='cpu'))

    # if args.phase == 'test':
    if 'test' in args.phase:
        # acc1 = utils.validate(test_loader, classifier, args, device)
        accuracy = utils.validate_with_negaug(test_loader, classifier, args, device, negative_aug)
        # print(accuracy)
    
    logger.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation with negative-view samples')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    # parser.add_argument('-d', '--data', metavar='DATA', default='OfficeHome', choices=utils.get_dataset_names(),
    #                     help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
    #                          ' (default: OfficeHome)')
    parser.add_argument('-d', '--data', metavar='DATA', default='OfficeHome',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: OfficeHome)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default=None,
                        help="Path to the saved weights")
    parser.add_argument("--phase", type=str, default='test', choices=['train', 'test', 'analysis', 'test_negaug'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--test_difficulty', default='', type=str, choices=['val', '1', '2', '3'], help="Only when using 'Product' dataset")
    # parser.add_argument('--log_results', action='store_true',
    #                     help="To log results in wandb")
    # parser.add_argument('--gpu', type=str, default="1", help="GPU ID")
    # parser.add_argument('--log_name', type=str,
    #                     default="log", help="log name for wandb")

    # negative augmentation
    parser.add_argument('--neg_aug_type', default='shuffle', type=str, help="Choose from ['shuffle', 'rotate', 'infill', 'shuffle_rotate', 'shufrot']") # 'shuffle_rotate' and 'shufrot' are the same
    parser.add_argument('--neg_aug_replace_rate', default=0.25, type=str, help="Choose from [0.25, 0.375]. When --neg_aug_type is 'infill', this flag must be specified.")
    parser.add_argument('--neg_aug_patch_size', default=32, type=int, help="When --neg_aug_type is 'shuffle' or 'rotate', this flag must be specified. Recommendation: multiple of size of a patch of input image.")
    parser.add_argument('--neg_aug_ratio', default=1.0, type=float, help="When --neg_aug_type is 'shuffle', this flag can be used for define how many patches are selected to be shuffled. Recommendation: quotient of the total number of patches.")

    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    main(args)

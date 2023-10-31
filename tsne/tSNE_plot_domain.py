import torchvision.transforms as T

import dataset_retail, dataset_benchmark
import argparse
import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys

from sklearn.manifold import TSNE
from datetime import datetime

# SDAT
import SDAT.examples.utils as utils

cnames = {          # color codes for tSNE plot
# 'aliceblue':            '#F0F8FF',
# 'antiquewhite':         '#FAEBD7',
# 'aqua':                 '#00FFFF',
# 'aquamarine':           '#7FFFD4',
# 'azure':                '#F0FFFF',
# 'beige':                '#F5F5DC',
# 'bisque':               '#FFE4C4',
# 'black':                '#000000',
# 'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
# 'blueviolet':           '#8A2BE2',
# 'brown':                '#A52A2A',
# 'burlywood':            '#DEB887',
# 'cadetblue':            '#5F9EA0',
# 'chartreuse':           '#7FFF00',
# 'chocolate':            '#D2691E',
# 'coral':                '#FF7F50',
# 'cornflowerblue':       '#6495ED',
# 'cornsilk':             '#FFF8DC',
# 'crimson':              '#DC143C',
# 'cyan':                 '#00FFFF',
# 'darkblue':             '#00008B',
# 'darkcyan':             '#008B8B',
# 'darkgoldenrod':        '#B8860B',
# 'darkgray':             '#A9A9A9',
# 'darkgreen':            '#006400',
# 'darkkhaki':            '#BDB76B',
# 'darkmagenta':          '#8B008B',
# 'darkolivegreen':       '#556B2F',
# 'darkorange':           '#FF8C00',
# 'darkorchid':           '#9932CC',
# 'darkred':              '#8B0000',
# 'darksalmon':           '#E9967A',
# 'darkseagreen':         '#8FBC8F',
# 'darkslateblue':        '#483D8B',
# 'darkslategray':        '#2F4F4F',
# 'darkturquoise':        '#00CED1',
# 'darkviolet':           '#9400D3',
# 'deeppink':             '#FF1493',
# 'deepskyblue':          '#00BFFF',
# 'dimgray':              '#696969',
# 'dodgerblue':           '#1E90FF',
# 'firebrick':            '#B22222',
# 'floralwhite':          '#FFFAF0',
# 'forestgreen':          '#228B22',
# 'fuchsia':              '#FF00FF',
# 'gainsboro':            '#DCDCDC',
# 'ghostwhite':           '#F8F8FF',
# 'gold':                 '#FFD700',
# 'goldenrod':            '#DAA520',
# 'gray':                 '#808080',
# 'green':                '#008000',
# 'greenyellow':          '#ADFF2F',
# 'honeydew':             '#F0FFF0',
# 'hotpink':              '#FF69B4',
# 'indianred':            '#CD5C5C',
# 'indigo':               '#4B0082',
# 'ivory':                '#FFFFF0',
# 'khaki':                '#F0E68C',
# 'lavender':             '#E6E6FA',
# 'lavenderblush':        '#FFF0F5',
# 'lawngreen':            '#7CFC00',
# 'lemonchiffon':         '#FFFACD',
# 'lightblue':            '#ADD8E6',
# 'lightcoral':           '#F08080',
# 'lightcyan':            '#E0FFFF',
# 'lightgoldenrodyellow': '#FAFAD2',
# 'lightgreen':           '#90EE90',
# 'lightgray':            '#D3D3D3',
# 'lightpink':            '#FFB6C1',
# 'lightsalmon':          '#FFA07A',
# 'lightseagreen':        '#20B2AA',
# 'lightskyblue':         '#87CEFA',
# 'lightslategray':       '#778899',
# 'lightsteelblue':       '#B0C4DE',
# 'lightyellow':          '#FFFFE0',
# 'lime':                 '#00FF00',
# 'limegreen':            '#32CD32',
# 'linen':                '#FAF0E6',
# 'magenta':              '#FF00FF',
# 'maroon':               '#800000',
# 'mediumaquamarine':     '#66CDAA',
# 'mediumblue':           '#0000CD',
# 'mediumorchid':         '#BA55D3',
# 'mediumpurple':         '#9370DB',
# 'mediumseagreen':       '#3CB371',
# 'mediumslateblue':      '#7B68EE',
# 'mediumspringgreen':    '#00FA9A',
# 'mediumturquoise':      '#48D1CC',
# 'mediumvioletred':      '#C71585',
# 'midnightblue':         '#191970',
# 'mintcream':            '#F5FFFA',
# 'mistyrose':            '#FFE4E1',
# 'moccasin':             '#FFE4B5',
# 'navajowhite':          '#FFDEAD',
# 'navy':                 '#000080',
# 'oldlace':              '#FDF5E6',
# 'olive':                '#808000',
# 'olivedrab':            '#6B8E23',
# 'orange':               '#FFA500',
# 'orangered':            '#FF4500',
# 'orchid':               '#DA70D6',
# 'palegoldenrod':        '#EEE8AA',
# 'palegreen':            '#98FB98',
# 'paleturquoise':        '#AFEEEE',
# 'palevioletred':        '#DB7093',
# 'papayawhip':           '#FFEFD5',
# 'peachpuff':            '#FFDAB9',
# 'peru':                 '#CD853F',
# 'pink':                 '#FFC0CB',
# 'plum':                 '#DDA0DD',
# 'powderblue':           '#B0E0E6',
# 'purple':               '#800080',
'red':                  '#FF0000',
# 'rosybrown':            '#BC8F8F',
# 'royalblue':            '#4169E1',
# 'saddlebrown':          '#8B4513',
# 'salmon':               '#FA8072',
# 'sandybrown':           '#FAA460',
# 'seagreen':             '#2E8B57',
# 'seashell':             '#FFF5EE',
# 'sienna':               '#A0522D',
# 'silver':               '#C0C0C0',
# 'skyblue':              '#87CEEB',
# 'slateblue':            '#6A5ACD',
# 'slategray':            '#708090',
# 'snow':                 '#FFFAFA',
# 'springgreen':          '#00FF7F',
# 'steelblue':            '#4682B4',
# 'tan':                  '#D2B48C',
# 'teal':                 '#008080',
# 'thistle':              '#D8BFD8',
# 'tomato':               '#FF6347',
# 'turquoise':            '#40E0D0',
# 'violet':               '#EE82EE',
# 'wheat':                '#F5DEB3',
# 'white':                '#FFFFFF',
# 'whitesmoke':           '#F5F5F5',
# 'yellow':               '#FFFF00',
# 'yellowgreen':          '#9ACD32'
}

def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x-np.min(x)
    return starts_from_zero / value_range

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
    if args.method == 'source_only': from source_only.classifier import ImageClassifier
    else: from SDAT.dalib.adaptation.cdan import ImageClassifier
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
    parser = argparse.ArgumentParser(description='Plotting tSNE for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--use_ImageNet_model', action='store_true',
                        help='True: plotting tSNE for ImageNet model / False: plotting tSNE for the model trained by UDA')
    parser.add_argument('--ckpt_path', metavar='DIR_CKPT', default='',
                        help='root path of trained model')
    parser.add_argument('--dataset_name', default='Office-Home', help='What is the name of dataset?')
    parser.add_argument('-s', '--source', default='Ar', help='source domain(s)')
    parser.add_argument('-t', '--target', default='Cl', help='target domain(s)')
    parser.add_argument('--root', default='./Office-Home', help='root path of dataset')
    parser.add_argument('--method', default='sdat', help='The name of UDA method which you want to test')
    # parser.add_argument('--gpus', default=0, help='gpu index(indices)')

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')  # 'resnet50', 'resnet101', 'vit_small_patch16_224'

    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--num_class_model', default=None, type=int, help='The number of neurons in 1ast layer of classifier model')

    # parameters
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N',
                        help='mini-batch size (default: 3)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')

    parser.add_argument("--log", type=str, default='./logs',
                        help="Where to save logs, checkpoints and debugging images.")
    
    parser.add_argument("--result_path", type=str, default='near_model',
                        help="Where to save resultant figure. If you want to save the result near the model path, use 'near_model'!")
    
    
    # For SDAT
    parser.add_argument("--no-pool", action='store_true')


    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
    
    if 'retail' in args.dataset_name or 'Retail' in args.dataset_name:
        source_dataset = dataset_retail.Retail(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_retail.Retail(root=args.root, task=args.target, transform=val_transform)
    elif 'Office-Home'==args.dataset_name or 'OfficeHome'==args.dataset_name:
        source_dataset = dataset_benchmark.OfficeHome(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_benchmark.OfficeHome(root=args.root, task=args.target, transform=val_transform)
    elif 'Office-31'==args.dataset_name or 'Office31'==args.dataset_name:
        source_dataset = dataset_benchmark.Office31(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_benchmark.Office31(root=args.root, task=args.target, transform=val_transform)
    elif 'VisDA2017'==args.dataset_name:
        source_dataset = dataset_benchmark.VisDA2017(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_benchmark.VisDA2017(root=args.root, task=args.target, transform=val_transform)
    elif 'ImageCLEF'==args.dataset_name:
        source_dataset = dataset_benchmark.ImageCLEF(root=args.root, task=args.source, transform=val_transform)
        target_dataset = dataset_benchmark.ImageCLEF(root=args.root, task=args.target, transform=val_transform)
    else:
        print('Please check args.datset_name')
        raise
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if torch.cuda.is_available():
        device = 'cuda'
    else: device = 'cpu'
    
    num_class_model = args.num_class_model if args.num_class_model is not None else source_dataset.num_classes
    
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
        # breakpoint()
        if 'sdat' in args.method or 'SDAT' in args.method or 'negaug' in args.method or 'NVC' in args.method or 'source_only' == args.method:
            feature_encoder = create_sdat_feature_encoder(args, num_classes=num_class_model)
        else:
            print('The method {} is not implemented.'.format(args.method))

    feature_encoder.eval()
    
    features=None
    labels=None
    for i, batch_s in enumerate(tqdm(source_loader, desc='Running the model inference in source_loader...')):
        image_s, current_label_s = batch_s
        image_s = image_s.to(device)
        with torch.no_grad():
            output = feature_encoder(image_s)
        # breakpoint()
        current_outputs_s = output.to('cpu').detach().numpy()
        if i==0:
            features = current_outputs_s
            labels = np.array(current_label_s)
        else:
            features = np.concatenate((features, current_outputs_s))
            labels = np.concatenate((labels, np.array(current_label_s)))
    num_source_samples = len(labels)

    for j, batch_t in enumerate(tqdm(target_loader, desc='Running the model inference in target_loader...')):
        image_t, current_label_t = batch_t
        image_t = image_t.to(device)
        with torch.no_grad():
            output = feature_encoder(image_t)
        current_outputs_t = output.to('cpu').detach().numpy()
        features = np.concatenate((features, current_outputs_t))
        labels = np.concatenate((labels, np.array(current_label_t)))
    num_target_samples = len(labels) - num_source_samples

    print('Running TSNE...')
    tsne = TSNE(n_components=2).fit_transform(features)
    print('Running TSNE is ended!!!')
    
    tx = tsne[:,0]
    ty = tsne[:,1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # prepare color codes
    keys = list(cnames.keys())

    # colors_per_class = {i: cnames[key] for i, key in enumerate(keys)}
    colors_per_domain = {'target': cnames['red'], 'source': cnames['blue']}

    # plotting
    source_tx = tx[0:num_source_samples]
    source_ty = ty[0:num_source_samples]
    source_labels = labels[0:num_source_samples]

    target_tx = tx[num_source_samples:]
    target_ty = ty[num_source_samples:]
    target_labels = labels[num_source_samples:]
    # breakpoint()
    
    
    # plotting target domain samples
    color = colors_per_domain['target']
    # ax.scatter(target_tx, target_ty, c=color, label='target', marker='.')
    ax.scatter(target_tx, target_ty, c=color, label='target', marker='.', edgecolors=color, s=3) # default of s: 6
    # ax.scatter(target_tx, target_ty, c=color, label='target', marker=',')

    # plotting source domain samples
    color = colors_per_domain['source']
    # ax.scatter(source_tx, source_ty, c=color, label='source', marker='.')
    ax.scatter(source_tx, source_ty, c=color, label='source', marker='.', edgecolors=color, s=3)
    # ax.scatter(source_tx, source_ty, c=color, label='source', marker=',')
    
    now = datetime.now()
    if now.month<10:
        current_month = '0'+str(now.month)
    else:
        current_month = str(now.month)
    if now.day<10:
        current_day = '0'+str(now.day)
    else:
        current_day = str(now.day)
    if now.hour<10:
        current_hour = '0'+str(now.hour)
    else:
        current_hour = str(now.hour)
    if now.minute<10:
        current_minute = '0'+str(now.minute)
    else:
        current_minute = str(now.minute)
    if now.second<10:
        current_second = '0'+str(now.second)
    else:
        current_second = str(now.second)
    
    current_time = '{}{}{}_{}:{}:{}'.format(now.year, current_month, current_day, current_hour, current_minute, current_second)

    ax.legend(loc='best')
    if args.use_ImageNet_model:
        fig_title = 'ImageNet-pretrained '+args.arch
    elif 'source_only' in args.method or 'Source_Only' in args.method or 'sourceonly' in args.method or 'Sourceonly' in args.method:
        fig_title = args.arch+' trained by source-only'
    else:
        fig_title = args.arch+' trained by '+args.method
    fig_title = fig_title + ', dataset: {}, task: {} → {}'.format(args.dataset_name, args.source, args.target)
    plt.title(fig_title, fontsize=10)
    # plt.savefig('./result1.jpg')
    target_folder = os.path.join(os.path.dirname(args.ckpt_path), 'tsne') if args.result_path == 'near_model' else os.path.dirname(args.result_path)
    target_fig_name = 'plot_domain.jpg' if args.result_path == 'near_model' else os.path.basename(args.result_path)
    target_fig_name, ext = os.path.splitext(target_fig_name)
    target_fig_name = target_fig_name + '_' + current_time + ext

    final_target_path = os.path.join(target_folder, target_fig_name)
    os.makedirs(target_folder, exist_ok=True)
    # plt.savefig(args.result_path, dpi=400)
    plt.savefig(final_target_path, dpi=400)
    # plt.show()
    # breakpoint()

    print('End!\n\n')
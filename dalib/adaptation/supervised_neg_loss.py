from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.modules.classifier import Classifier as ClassifierBase
from ..modules.entropy import entropy


__all__ = ['SupervisedNegLoss', 'ImageClassifier']


class SupervisedNegLoss(nn.Module):
    def __init__(self, neg_loss_type='latent', temperature: Optional[float]=1.0,
                 reduction: Optional[str] = 'mean'):
        super(SupervisedNegLoss, self).__init__()
        self.tau = temperature
        self.reduction = reduction
        self.neg_loss_type = neg_loss_type

    def forward(self, f: torch.Tensor, f_neg: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        # the shape of f: (N, d), caution: It can be N != batch_size(denotated as B)
        # the shape of f_neg: (B, d)
        # neg_loss_per_sample = []
        summation = 0
        for i in range(batch_size):
            # 1. distinguish between positive samples and negative samples for a given feature vector and two batches (f and f_neg)
            single_feat = f[i]  # shape: (d,)
            single_feat = single_feat[None, :]  # shape: (1, d)
            rest_f = delete_one_row(f, index=i)     # shape: (N-1, d)
            label = labels[i]
            rest_label = delete_one_row(labels, index=i)     # shape: (N-1,)
            pos_sample_slicer = (rest_label==label)
            positive_samples = rest_f[pos_sample_slicer]
            negative_samples_part1 = rest_f[~pos_sample_slicer]
            negative_samples_part2 = f_neg
            negative_samples = torch.cat([negative_samples_part1, negative_samples_part2])
            
            # 2. compute neg_loss for the single feature vector
            loss_single_feat = self.compute_neg_loss_for_single_feature(single_f=single_feat, pos=positive_samples, neg=negative_samples)
            # neg_loss_per_sample.append(loss_single_feat)
            summation = summation + loss_single_feat
        
        # neg_loss = torch.Tensor(neg_loss_per_sample)
        
        
        if self.reduction=='mean':
            # neg_loss = torch.mean(neg_loss)
            neg_loss = summation / batch_size
        elif self.reduction=='sum':
            # neg_loss = torch.sum(neg_loss)
            neg_loss = summation
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return neg_loss
    
    def compute_neg_loss_for_single_feature(self, single_f, pos, neg):
        # the shape of single_f: (1, d)
        # the shape of pos and neg: (P, d) and (N, d), respectively
        sim_with_pos = cosine_similarity(single_f, pos) # shape: (P,)
        sim_with_neg = cosine_similarity(single_f, neg) # shape: (N,)
        denominator = torch.sum(torch.exp(sim_with_pos/self.tau)) + torch.sum(torch.exp(sim_with_neg/self.tau))
        numerators = torch.exp(sim_with_pos/self.tau)
        antilogarithms = numerators / denominator # shape: (P,)
        neg_logarithms = -torch.log(antilogarithms) # shape: (P,)
        neg_log_for_one_feat = torch.mean(neg_logarithms)
        return neg_log_for_one_feat

def l2_dist_square(input1, input2):
    # input1, input2: shape (B, d)
    return torch.sum((input1-input2)**2, dim=1)  # shape (B,)

def delete_one_row(input, index):
    output = torch.cat((input[:index], input[index+1:]))
    return output



class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

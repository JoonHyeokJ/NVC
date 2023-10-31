from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.modules.classifier import Classifier as ClassifierBase
from ..modules.entropy import entropy


__all__ = ['SelfSupervisedContrastiveLoss', 'SelfSupervisedContrastiveLoss2', 'ImageClassifier']


class SelfSupervisedContrastiveLoss(nn.Module):
    def __init__(self, cont_loss_type='latent', temperature: Optional[float]=1.0,
                 reduction: Optional[str] = 'mean'):
        super(SelfSupervisedContrastiveLoss, self).__init__()
        self.tau = temperature
        self.reduction = reduction
        self.cont_loss_type = cont_loss_type

    def forward(self, f: torch.Tensor, f_pos: torch.Tensor) -> torch.Tensor:
        # the shape of f and f_pos: (B, d)
        # Two feature vectors sharing same index(i.e. f[i] and f_pos[i]) must be corresponding to each other.
        # cont_loss_per_pair = []
        summation = 0
        batch_size = f.shape[0]
        for i in range(batch_size):
            mate1 = f[i]  # m1
            mate2 = f_pos[i]  # m2
            mate1 = mate1[None, :]
            mate2 = mate2[None, :]
            enemies1 = delete_one_row(f, index=i) # e1
            enemies2 = delete_one_row(f_pos, index=i) # e2
            # When mate1 is a center
            sim_m1_m2 = cosine_similarity(mate1, mate2) # shape: (1,)
            sim_m1_e1 = cosine_similarity(mate1, enemies1) # shape: (B-1,)
            sim_m1_e2 = cosine_similarity(mate1, enemies2) # shape: (B-1,)
            antilogarithm1 = torch.exp(sim_m1_m2/self.tau) / ( torch.exp(sim_m1_m2/self.tau) + torch.sum(torch.exp(sim_m1_e1/self.tau)) + torch.sum(torch.exp(sim_m1_e2/self.tau)) )
            negative_log1 = -torch.log(antilogarithm1)
            # When mate2 is a center
            sim_m2_m1 = sim_m1_m2 # shape: (1,)
            sim_m2_e1 = cosine_similarity(mate2, enemies1) # shape: (B-1,)
            sim_m2_e2 = cosine_similarity(mate2, enemies2) # shape: (B-1,)
            antilogarithm2 = torch.exp(sim_m2_m1/self.tau) / ( torch.exp(sim_m2_m1/self.tau) + torch.sum(torch.exp(sim_m2_e1/self.tau)) + torch.sum(torch.exp(sim_m2_e2/self.tau)) )
            negative_log2 = -torch.log(antilogarithm2)
            # combining above two cases
            negative_log = 0.5*(negative_log1 + negative_log2)
            # cont_loss_per_pair.append(negative_log)
            summation = summation + negative_log
        
        # cont_loss = torch.Tensor(cont_loss_per_pair)
        
        if self.reduction=='mean':
            # cont_loss = torch.mean(cont_loss)
            cont_loss = summation / batch_size
        elif self.reduction=='sum':
            # cont_loss = torch.sum(cont_loss)
            cont_loss = summation
        # elif self.reduction=='none':
        #     cont_loss = cont_loss
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return cont_loss

class SelfSupervisedContrastiveLoss2(nn.Module):
    def __init__(self, cont_loss_type='latent', temperature: Optional[float]=1.0,
                 reduction: Optional[str] = 'mean'):
        super(SelfSupervisedContrastiveLoss2, self).__init__()
        self.tau = temperature
        self.reduction = reduction
        self.cont_loss_type = cont_loss_type

    def forward(self, f: torch.Tensor, f_pos: torch.Tensor, f_neg: torch.Tensor) -> torch.Tensor:
        # the shape of f, f_pos, and f_neg: (B, d)
        # Two feature vectors sharing same index(i.e. f[i] and f_pos[i]) must be corresponding to each other.
        # cont_loss_per_pair = []
        summation = 0
        batch_size = f.shape[0]
        for i in range(batch_size):
            mate1 = f[i]  # m1
            mate2 = f_pos[i]  # m2
            mate1 = mate1[None, :]
            mate2 = mate2[None, :]
            enemies1 = delete_one_row(f, index=i) # e1
            enemies1 = torch.cat([enemies1, f_neg])
            enemies2 = delete_one_row(f_pos, index=i) # e2
            # When mate1 is a center
            sim_m1_m2 = cosine_similarity(mate1, mate2) # shape: (1,)
            sim_m1_e1 = cosine_similarity(mate1, enemies1) # shape: (2B-1,)
            sim_m1_e2 = cosine_similarity(mate1, enemies2) # shape: (B-1,)
            antilogarithm1 = torch.exp(sim_m1_m2/self.tau) / ( torch.exp(sim_m1_m2/self.tau) + torch.sum(torch.exp(sim_m1_e1/self.tau)) + torch.sum(torch.exp(sim_m1_e2/self.tau)) )
            negative_log1 = -torch.log(antilogarithm1)
            # When mate2 is a center
            sim_m2_m1 = sim_m1_m2 # shape: (1,)
            sim_m2_e1 = cosine_similarity(mate2, enemies1) # shape: (2B-1,)
            sim_m2_e2 = cosine_similarity(mate2, enemies2) # shape: (B-1,)
            antilogarithm2 = torch.exp(sim_m2_m1/self.tau) / ( torch.exp(sim_m2_m1/self.tau) + torch.sum(torch.exp(sim_m2_e1/self.tau)) + torch.sum(torch.exp(sim_m2_e2/self.tau)) )
            negative_log2 = -torch.log(antilogarithm2)
            # combining above two cases
            negative_log = 0.5*(negative_log1 + negative_log2)
            # cont_loss_per_pair.append(negative_log)
            summation = summation + negative_log
        
        # cont_loss = torch.Tensor(cont_loss_per_pair)
        
        if self.reduction=='mean':
            # cont_loss = torch.mean(cont_loss)
            cont_loss = summation / batch_size
        elif self.reduction=='sum':
            # cont_loss = torch.sum(cont_loss)
            cont_loss = summation
        # elif self.reduction=='none':
        #     cont_loss = cont_loss
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return cont_loss

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

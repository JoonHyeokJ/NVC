from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from common.modules.classifier import Classifier as ClassifierBase



__all__ = ['NegUniformLoss', 'NegDistLoss', 'NegDistLossv2', 'NegDistLossv3', 'NegCosSimLoss', 'NegCosSimLossv2', 'ImageClassifier']


class NegUniformLoss(nn.Module):
    def __init__(self, temperature: Optional[float]=1.0,      # lower temperature, more sensitive
                 reduction: Optional[str] = 'mean'):
        super(NegUniformLoss, self).__init__()
        self.tau = temperature
        self.reduction = reduction

    def forward(self, logit: torch.Tensor) -> torch.Tensor:
        num_classes = logit.shape[-1]
        # logit -> logit from neg-aug img, shape of logit: (B, c), B is batch size, c is the number of classes
        epsilon = 1e-5

        logit_softmax = F.softmax(logit / self.tau, dim=1)
        loss = (-1/num_classes)*torch.log(logit_softmax + epsilon)
        loss = loss.sum(dim=1)
        
        if self.reduction=='mean':
            loss = torch.mean(loss)
        elif self.reduction=='sum':
            loss = torch.sum(loss)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return loss

class NegDistLoss(nn.Module):  # distance on logit space between logit from original img and logit from neg-aug img
    def __init__(self, beta: Optional[float]=0.5,  temperature: Optional[float]=1.0,      # lower temperature, more sensitive
                 dist: Optional[str]='l2', reduction: Optional[str] = 'mean'):
        super(NegDistLoss, self).__init__()
        self.tau = temperature
        self.beta = beta
        self.reduction = reduction
        self.dist_kinds = ['l2', 'l2_square', 'l1', 'smooth_l1']
        self.dist = dist
        assert self.dist in self.dist_kinds, 'Please check the argument "dist"!'

    def forward(self, logit_original: torch.Tensor, logit_negaug: torch.Tensor) -> torch.Tensor:
        # logit_original, logit_negaug -> shape: (B, c), B is batch size, c is the number of classes
        softmax_original = F.softmax(logit_original / self.tau, dim=1)
        softmax_negaug = F.softmax(logit_negaug / self.tau, dim=1)

        if self.dist=='l2_square':
            neg_pair_dist = l2_dist_square(softmax_original, softmax_negaug)
        elif self.dist=='l2':
            l2_dist = torch.nn.PairwiseDistance()
            neg_pair_dist = l2_dist(softmax_original, softmax_negaug)
        elif self.dist=='l1':
            l1_dist = torch.nn.PairwiseDistance(p=1.0)
            neg_pair_dist = l1_dist(softmax_original, softmax_negaug)
        elif self.dist=='smooth_l1':
            smooth_l1 = torch.nn.SmoothL1Loss(reduction='none', beta=self.beta)
            neg_pair_dist = torch.sum(smooth_l1(softmax_original, softmax_negaug), dim=1)
        else: raise Exception('Invalid self.dist: {}'.format(self.dist))
        
        dist_loss = -neg_pair_dist
        
        if self.reduction=='mean':
            dist_loss = torch.mean(dist_loss)
        elif self.reduction=='sum':
            dist_loss = torch.sum(dist_loss)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return dist_loss


class NegDistLossv2(nn.Module):  # distance on logit space between logit from original img and logit from neg-aug img
    def __init__(self, temperature: Optional[float]=1.0,      # lower temperature, more sensitive
                 dist: Optional[str]='l2', reduction: Optional[str] = 'mean'):
        super(NegDistLossv2, self).__init__()
        self.tau = temperature
        self.reduction = reduction
        self.dist_kinds = ['l2', 'l2_square', 'l1']
        self.dist = dist
        assert self.dist in self.dist_kinds, 'Please check the argument "dist"!'

    def forward(self, logit_original: torch.Tensor, logit_negaug: torch.Tensor) -> torch.Tensor:
        # logit_original, logit_negaug -> shape: (B, c), B is batch size, c is the number of classes
        softmax_original = F.softmax(logit_original / self.tau, dim=1)
        softmax_negaug = F.softmax(logit_negaug / self.tau, dim=1)

        if self.dist=='l2_square':
            neg_pair_dist = torch.cdist(softmax_original, softmax_negaug)
            neg_pair_dist = neg_pair_dist ** 2
        elif self.dist=='l2':
            neg_pair_dist = torch.cdist(softmax_original, softmax_negaug)
        elif self.dist=='l1':
            neg_pair_dist = torch.cdist(softmax_original, softmax_negaug, p=1)
        else: raise Exception('Invalid self.dist: {}'.format(self.dist))
        
        neg_dist_loss = -torch.mean(neg_pair_dist, dim=1)
        
        if self.reduction=='mean':
            neg_dist_loss = torch.mean(neg_dist_loss)
        elif self.reduction=='sum':
            neg_dist_loss = torch.sum(neg_dist_loss)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return neg_dist_loss

class NegDistLossv3(nn.Module):  # distance on logit space between logit from original img and logit from neg-aug img
    def __init__(self, temperature: Optional[float]=1.0,      # lower temperature, more sensitive
                 dist: Optional[str]='l2', 
                 temperature_weight: Optional[float]=1.0,
                 lamb: Optional[float]=1.0, thres_weight = None,
                 reduction: Optional[str] = 'mean'):
        super(NegDistLossv3, self).__init__()
        self.tau = temperature
        self.tau_weight = temperature_weight
        self.reduction = reduction
        self.lamb = lamb
        self.thres = thres_weight
        self.dist_kinds = ['l2', 'l2_square', 'l1']
        self.dist = dist
        assert self.dist in self.dist_kinds, 'Please check the argument "dist"!'

    def forward(self, logit_original: torch.Tensor, logit_negaug: torch.Tensor) -> torch.Tensor:
        # logit_original, logit_negaug -> shape: (B, c), B is batch size, c is the number of classes
        softmax_original = F.softmax(logit_original / self.tau, dim=1)
        softmax_negaug = F.softmax(logit_negaug / self.tau, dim=1)

        if self.dist=='l2_square':
            neg_pair_dist = torch.cdist(softmax_original, softmax_negaug)
            neg_pair_dist = neg_pair_dist ** 2
            pos_pair_dist = torch.cdist(softmax_original, softmax_original)
            pos_pair_dist = pos_pair_dist ** 2
        elif self.dist=='l2':
            neg_pair_dist = torch.cdist(softmax_original, softmax_negaug)
            pos_pair_dist = torch.cdist(softmax_original, softmax_original)
        elif self.dist=='l1':
            neg_pair_dist = torch.cdist(softmax_original, softmax_negaug, p=1)
            pos_pair_dist = torch.cdist(softmax_original, softmax_original, p=1)
        else: raise Exception('Invalid self.dist: {}'.format(self.dist))
        
        logit_original_norm = F.normalize(logit_original, dim=1)
        masked_weight = compute_masked_weight(logit_original_norm, logit_original_norm, self.tau_weight, self.thres).detach()

        neg_dist_mean = -torch.mean(neg_pair_dist, dim=1)
        weighted_pos_pair_dist = pos_pair_dist * masked_weight
        pos_dist_mean = torch.sum(weighted_pos_pair_dist, dim=1)

        neg_dist_loss = (neg_dist_mean + pos_dist_mean * self.lamb) / (1 + self.lamb)
        
        if self.reduction=='mean':
            neg_dist_loss = torch.mean(neg_dist_loss)
        elif self.reduction=='sum':
            neg_dist_loss = torch.sum(neg_dist_loss)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return neg_dist_loss

class NegCosSimLoss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean'):
        super(NegCosSimLoss, self).__init__()
        self.reduction = reduction

    def forward(self, feature_original: torch.Tensor, feature_negaug: torch.Tensor) -> torch.Tensor:
        # feature_original, feature_negaug -> shape: (B, d), B is batch size, d is the dimension of single feature vector
        feature_original_norm = F.normalize(feature_original, dim=1)
        feature_negaug_norm = F.normalize(feature_negaug, dim=1)
        cos_sim_matrix = torch.matmul(feature_original_norm, torch.transpose(feature_negaug_norm, 0, 1)) + 1 # '+ 1' means shifting the values of cosine similarity to be always equal to or above 0.

        cos_sim_means = torch.mean(cos_sim_matrix, dim=1)
        
        if self.reduction=='mean':
            cos_sim_loss = torch.mean(cos_sim_means)
        elif self.reduction=='sum':
            cos_sim_loss = torch.sum(cos_sim_means)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return cos_sim_loss

class NegCosSimLossv2(nn.Module):
    def __init__(self, temperature_weight: Optional[float]=1.0,
                 lamb: Optional[float]=1.0, thres_weight = None,
                 reduction: Optional[str] = 'mean'):
        super(NegCosSimLossv2, self).__init__()
        self.tau_weight = temperature_weight
        self.lamb = lamb
        self.thres = thres_weight
        self.reduction = reduction

    def forward(self, feature_original: torch.Tensor, feature_negaug: torch.Tensor) -> torch.Tensor:
        # feature_original, feature_negaug -> shape: (B, d), B is batch size, d is the dimension of single feature vector
        feature_original_norm = F.normalize(feature_original, dim=1)
        feature_negaug_norm = F.normalize(feature_negaug, dim=1)
        cos_sim_matrix_negpair = torch.matmul(feature_original_norm, torch.transpose(feature_negaug_norm, 0, 1)) + 1 # '+ 1' means shifting the values of cosine similarity to be always equal to or above 0.
        cos_dissim_matrix_pospair = 1 - torch.matmul(feature_original_norm, torch.transpose(feature_original_norm, 0, 1))

        masked_weight = compute_masked_weight(feature_original_norm, feature_original_norm, self.tau_weight, self.thres).detach()

        cos_sim_means_negpair = torch.mean(cos_sim_matrix_negpair, dim=1)
        weighted_cos_dissim_matrix_pospair = cos_dissim_matrix_pospair * masked_weight
        cos_dissim_means_pospair = torch.sum(weighted_cos_dissim_matrix_pospair, dim=1)
        
        cos_sim_means = (cos_sim_means_negpair + cos_dissim_means_pospair * self.lamb) / (1 + self.lamb)
        
        if self.reduction=='mean':
            cos_sim_loss = torch.mean(cos_sim_means)
        elif self.reduction=='sum':
            cos_sim_loss = torch.sum(cos_sim_means)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return cos_sim_loss

def l2_dist_square(input1, input2):
    # input1, input2: shape (B, d)
    return torch.sum((input1-input2)**2, dim=1)  # shape (B,)

def compute_masked_weight(input1: torch.Tensor, input2: torch.Tensor, tau=1.0, thres=None):
    # input1, input2: shape (B, d)
    
    x1 = input1.detach()
    x2 = input2.detach()
    
    temp = torch.exp(torch.matmul(x1, torch.transpose(x2, 0, 1)) / tau)
    i = range(x1.shape[0])
    temp[i, i] = 0
    
    temp2 = torch.sum(temp, dim=1, keepdim=True)
    masked_weight = temp / temp2
    
    if thres is None:
        return masked_weight.detach()
    else:
        if isinstance(thres, str):
            thres = float(thres)
        assert isinstance(thres, float), "thres must be a float-type value."
        mask = (masked_weight >= thres) * 1.0
        masked_weight = masked_weight * mask
        return masked_weight.detach()

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

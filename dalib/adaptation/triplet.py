from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.modules.classifier import Classifier as ClassifierBase
from ..modules.entropy import entropy


__all__ = ['TripletLossInLatentSpaceCosSim', 'TripletLossInLatentSpaceCosSimv2', 'TripletLossInLatentSpaceCosSimv2_plus1', 'TripletLossInLatentSpaceCosSimv2_plus2', 'TripletLossInLatentSpaceCosSimv2_plus3', 'TripletLossInLatentSpaceCosSimv3', 'TripletLossInLogisticDist', 'ImageClassifier']


class TripletLossInLatentSpaceCosSim(nn.Module):
    def __init__(self, entropy_conditioning: Optional[bool] = False, temperature: Optional[float]=1.0,
                 reduction: Optional[str] = 'mean'):
        super(TripletLossInLatentSpaceCosSim, self).__init__()
        self.entropy_conditioning = entropy_conditioning
        self.tau = temperature
        self.reduction = reduction

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, pred_anchor: torch.Tensor = None) -> torch.Tensor:
        if self.entropy_conditioning:
            # if pred_anchor is None: breakpoint()
            assert pred_anchor is not None, "pred_anchor is not specified!"
        # anchor, pos, neg -> shape: (B, d), B is batch size, d is dimension of a feature vector
        # pred_anchor -> shape: (B, c), B is batch size, c is the number of classes
        anchor = anchor.detach()    # gradient flow is blocked w.r.t anchor.

        pos_pair_sim = cosine_similarity(anchor, pos, dim=1)
        neg_pair_sim = cosine_similarity(anchor, neg, dim=1)
        
        numerator = torch.exp(pos_pair_sim / self.tau)
        denominator = torch.exp(pos_pair_sim / self.tau) + torch.exp(neg_pair_sim / self.tau)
        
        triplet = -torch.log(numerator / denominator)

        if self.entropy_conditioning:
            pred_anchor = F.softmax(pred_anchor, dim=1).detach()
            batch_size = anchor.shape[0]
            weight = 1.0 + torch.exp(-entropy(pred_anchor))
            weight = weight / torch.sum(weight) * batch_size
            triplet = triplet * weight
        else:
            triplet = triplet
        
        if self.reduction=='mean':
            triplet = torch.mean(triplet)
        elif self.reduction=='sum':
            triplet = torch.sum(triplet)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return triplet

class TripletLossInLatentSpaceCosSimv2(nn.Module):
    def __init__(self, # entropy_conditioning: Optional[bool] = False, 
                 temperature: Optional[float]=1.0,
                 reduction: Optional[str] = 'mean'):
        super(TripletLossInLatentSpaceCosSimv2, self).__init__()
        # self.entropy_conditioning = entropy_conditioning
        self.tau = temperature
        self.reduction = reduction

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, 
                anchor_detach=True,
                # pred_anchor: torch.Tensor = None
                ) -> torch.Tensor:
        ###################################
        # anchor: A minibatch containing original images
        # pos: A minibatch containing the samples positive to anchor (e.g. positive-augmented samples such as masked modeling, random crop, augmix, etc.)
        # neg: A minibatch containing the samples negative to anchor (e.g. negative-augmented samples) 
        # anchor, pos, neg -> shape: (B, d), B is batch size, d is dimension of a feature vector
        ###################################
        # if self.entropy_conditioning:
        #     # if pred_anchor is None: breakpoint()
        #     assert pred_anchor is not None, "pred_anchor is not specified!"
        
        # breakpoint()
        if anchor_detach:
            anchor = anchor.detach()    # gradient flow is blocked w.r.t anchor.
        else:
            anchor = anchor

        D1 = torch.exp(cosine_similarity(anchor, pos) / self.tau)
        
        anchor_norm = F.normalize(anchor, dim=1)
        neg_norm = F.normalize(neg, dim=1)
        
        D2 = torch.sum(torch.exp(torch.matmul(anchor_norm, torch.transpose(neg_norm, 0, 1)) / self.tau), dim=1)
        
        triplet = -torch.log(D1 / (D1+D2))
        # breakpoint()
        
        # if self.entropy_conditioning:
        #     pred_anchor = F.softmax(pred_anchor, dim=1).detach()
        #     batch_size = anchor.shape[0]
        #     weight = 1.0 + torch.exp(-entropy(pred_anchor))
        #     weight = weight / torch.sum(weight) * batch_size
        #     triplet = triplet * weight
        # else:
        #     triplet = triplet
        
        if self.reduction=='mean':
            triplet = torch.mean(triplet)
        elif self.reduction=='sum':
            triplet = torch.sum(triplet)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return triplet

class TripletLossInLatentSpaceCosSimv2_plus1(nn.Module):  # negative pair: (x_t, all neg_views) + (x_t, all pos_views except for pos_view of x_t)
    def __init__(self, # entropy_conditioning: Optional[bool] = False, 
                 temperature: Optional[float]=1.0,
                 reduction: Optional[str] = 'mean'):
        super(TripletLossInLatentSpaceCosSimv2_plus1, self).__init__()
        # self.entropy_conditioning = entropy_conditioning
        self.tau = temperature
        self.reduction = reduction

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, 
                anchor_detach=True,
                # pred_anchor: torch.Tensor = None
                ) -> torch.Tensor:
        ###################################
        # anchor: A minibatch containing original images
        # pos: A minibatch containing the samples positive to anchor (e.g. positive-augmented samples such as masked modeling, random crop, augmix, etc.)
        # neg: A minibatch containing the samples negative to anchor (e.g. negative-augmented samples) 
        # anchor, pos, neg -> shape: (B, d), B is batch size, d is dimension of a feature vector
        ###################################
        # if self.entropy_conditioning:
        #     # if pred_anchor is None: breakpoint()
        #     assert pred_anchor is not None, "pred_anchor is not specified!"
        
        # breakpoint()
        if anchor_detach:
            anchor = anchor.detach()    # gradient flow is blocked w.r.t anchor.
        else:
            anchor = anchor

        D1 = torch.exp(cosine_similarity(anchor, pos) / self.tau)  # positive pairs
        
        anchor_norm = F.normalize(anchor, dim=1)
        neg_norm = F.normalize(neg, dim=1)
        pos_norm = F.normalize(pos, dim=1)
        
        D2 = torch.sum(torch.exp(torch.matmul(anchor_norm, torch.transpose(neg_norm, 0, 1)) / self.tau), dim=1)  # negative pairs (1)

        cossim_mat_anchor_pos = torch.exp(torch.matmul(anchor_norm, torch.transpose(pos_norm, 0, 1)) / self.tau)  # negative pairs (2)
        #cossim_mat_anchor_pos_wo_diag = cossim_mat_anchor_pos - torch.diag(torch.diag(cossim_mat_anchor_pos))  # negative pairs (2)
        #D3 = torch.sum(cossim_mat_anchor_pos_wo_diag, dim=1)  # negative pairs (2)
        D3 = torch.sum(cossim_mat_anchor_pos, dim=1) - torch.diag(cossim_mat_anchor_pos)
        
        triplet = -torch.log(D1 / (D1+D2+D3))
        # breakpoint()
        
        if self.reduction=='mean':
            triplet = torch.mean(triplet)
        elif self.reduction=='sum':
            triplet = torch.sum(triplet)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return triplet

class TripletLossInLatentSpaceCosSimv2_plus2(nn.Module):  # negative pair: (x_t, all neg_views) + (x_t, all original samples except for x_t)
    def __init__(self, # entropy_conditioning: Optional[bool] = False, 
                 temperature: Optional[float]=1.0,
                 reduction: Optional[str] = 'mean'):
        super(TripletLossInLatentSpaceCosSimv2_plus2, self).__init__()
        # self.entropy_conditioning = entropy_conditioning
        self.tau = temperature
        self.reduction = reduction

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, 
                anchor_detach=True,
                # pred_anchor: torch.Tensor = None
                ) -> torch.Tensor:
        ###################################
        # anchor: A minibatch containing original images
        # pos: A minibatch containing the samples positive to anchor (e.g. positive-augmented samples such as masked modeling, random crop, augmix, etc.)
        # neg: A minibatch containing the samples negative to anchor (e.g. negative-augmented samples) 
        # anchor, pos, neg -> shape: (B, d), B is batch size, d is dimension of a feature vector
        ###################################
        # if self.entropy_conditioning:
        #     # if pred_anchor is None: breakpoint()
        #     assert pred_anchor is not None, "pred_anchor is not specified!"
        
        # breakpoint()
        if anchor_detach:
            anchor = anchor.detach()    # gradient flow is blocked w.r.t anchor.
        else:
            anchor = anchor

        D1 = torch.exp(cosine_similarity(anchor, pos) / self.tau)  # positive pairs
        
        anchor_norm = F.normalize(anchor, dim=1)
        neg_norm = F.normalize(neg, dim=1)
        
        D2 = torch.sum(torch.exp(torch.matmul(anchor_norm, torch.transpose(neg_norm, 0, 1)) / self.tau), dim=1)  # negative pairs (1)

        cossim_mat_anchor_anchor = torch.exp(torch.matmul(anchor_norm, torch.transpose(anchor_norm, 0, 1)) / self.tau)  # negative pairs (2)
        cossim_mat_anchor_anchor_wo_diag = cossim_mat_anchor_anchor - torch.diag(torch.diag(cossim_mat_anchor_anchor))  # negative pairs (2)
        D3 = torch.sum(cossim_mat_anchor_anchor_wo_diag, dim=1)  # negative pairs (2)
        
        triplet = -torch.log(D1 / (D1+D2+D3))
        # breakpoint()
        
        if self.reduction=='mean':
            triplet = torch.mean(triplet)
        elif self.reduction=='sum':
            triplet = torch.sum(triplet)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return triplet

class TripletLossInLatentSpaceCosSimv2_plus3(nn.Module):  # negative pair: (x_t, all neg_views) + (x_t, all original samples except for x_t) + (x_t, all pos_views except for pos_view of x_t)
    def __init__(self, # entropy_conditioning: Optional[bool] = False, 
                 temperature: Optional[float]=1.0,
                 reduction: Optional[str] = 'mean'):
        super(TripletLossInLatentSpaceCosSimv2_plus3, self).__init__()
        # self.entropy_conditioning = entropy_conditioning
        self.tau = temperature
        self.reduction = reduction

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, 
                anchor_detach=True,
                # pred_anchor: torch.Tensor = None
                ) -> torch.Tensor:
        ###################################
        # anchor: A minibatch containing original images
        # pos: A minibatch containing the samples positive to anchor (e.g. positive-augmented samples such as masked modeling, random crop, augmix, etc.)
        # neg: A minibatch containing the samples negative to anchor (e.g. negative-augmented samples) 
        # anchor, pos, neg -> shape: (B, d), B is batch size, d is dimension of a feature vector
        ###################################
        # if self.entropy_conditioning:
        #     # if pred_anchor is None: breakpoint()
        #     assert pred_anchor is not None, "pred_anchor is not specified!"
        
        # breakpoint()
        if anchor_detach:
            anchor = anchor.detach()    # gradient flow is blocked w.r.t anchor.
        else:
            anchor = anchor

        D1 = torch.exp(cosine_similarity(anchor, pos) / self.tau)  # positive pairs
        
        anchor_norm = F.normalize(anchor, dim=1)
        neg_norm = F.normalize(neg, dim=1)
        pos_norm = F.normalize(pos, dim=1)

        D2 = torch.sum(torch.exp(torch.matmul(anchor_norm, torch.transpose(neg_norm, 0, 1)) / self.tau), dim=1)  # negative pairs (1)

        cossim_mat_anchor_anchor = torch.exp(torch.matmul(anchor_norm, torch.transpose(anchor_norm, 0, 1)) / self.tau)  # negative pairs (2)
        cossim_mat_anchor_anchor_wo_diag = cossim_mat_anchor_anchor - torch.diag(torch.diag(cossim_mat_anchor_anchor))  # negative pairs (2)
        D3 = torch.sum(cossim_mat_anchor_anchor_wo_diag, dim=1)  # negative pairs (2)

        cossim_mat_anchor_pos = torch.exp(torch.matmul(anchor_norm, torch.transpose(pos_norm, 0, 1)) / self.tau)  # negative pairs (3)
        cossim_mat_anchor_pos_wo_diag = cossim_mat_anchor_pos - torch.diag(torch.diag(cossim_mat_anchor_pos))  # negative pairs (3)
        D4 = torch.sum(cossim_mat_anchor_pos_wo_diag, dim=1)  # negative pairs (3)
        
        triplet = -torch.log(D1 / (D1+D2+D3+D4))
        # breakpoint()
        
        if self.reduction=='mean':
            triplet = torch.mean(triplet)
        elif self.reduction=='sum':
            triplet = torch.sum(triplet)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return triplet

class TripletLossInLatentSpaceCosSimv3(nn.Module):
    def __init__(self, # entropy_conditioning: Optional[bool] = False, 
                 temperature: Optional[float]=1.0,
                 temperature_weight: Optional[float]=1.0,
                 lamb: Optional[float]=1.0, thres_weight = None,
                 reduction: Optional[str] = 'mean'):
        super(TripletLossInLatentSpaceCosSimv3, self).__init__()
        # self.entropy_conditioning = entropy_conditioning
        self.tau = temperature
        self.tau_weight = temperature_weight
        self.reduction = reduction
        self.lamb = lamb
        self.thres = thres_weight

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, 
                anchor_detach=True,
                # pred_anchor: torch.Tensor = None
                ) -> torch.Tensor:
        ###################################
        # anchor: A minibatch containing original images
        # pos: A minibatch containing the samples positive to anchor (e.g. positive-augmented samples such as masked modeling, random crop, augmix, etc.)
        # neg: A minibatch containing the samples negative to anchor (e.g. negative-augmented samples) 
        # anchor, pos, neg -> shape: (B, d), B is batch size, d is dimension of a feature vector
        ###################################
        # if self.entropy_conditioning:
        #     # if pred_anchor is None: breakpoint()
        #     assert pred_anchor is not None, "pred_anchor is not specified!"
        
        # breakpoint()
        if anchor_detach:
            anchor = anchor.detach()    # gradient flow is blocked w.r.t anchor.
        else:
            anchor = anchor

        anchor_norm = F.normalize(anchor, dim=1)
        pos_norm = F.normalize(pos, dim=1)
        neg_norm = F.normalize(neg, dim=1)
        
        temp1 = torch.exp(torch.matmul(anchor_norm, torch.transpose(neg_norm, 0, 1)) / self.tau)
        temp2 = torch.exp(torch.matmul(anchor_norm, torch.transpose(pos_norm, 0, 1)) / self.tau)
        temp3 = torch.exp(torch.matmul(anchor_norm, torch.transpose(anchor_norm, 0, 1)) / self.tau)
        
        D = torch.sum(temp1, dim=1) + torch.sum(temp2, dim=1) + torch.sum(temp3, dim=1) - temp3.diagonal()
        
        # weight1 = compute_masked_weight(anchor, pos).detach()
        # weight2 = compute_masked_weight(anchor, anchor).detach()
        weight1 = compute_masked_weight(anchor_norm, pos_norm, self.tau_weight, self.thres).detach()
        weight2 = compute_masked_weight(anchor_norm, anchor_norm, self.tau_weight, self.thres).detach()
        
        term1 = -torch.log(temp2.diagonal() / D)
        term2 = torch.sum( ((-torch.log(temp2 / D[:,None])) * weight1), dim=1)
        term3 = torch.sum( ((-torch.log(temp3 / D[:,None])) * weight2), dim=1)
        
        triplet = (term1 + self.lamb * (term2 + term3) * 0.5 ) / (1 + self.lamb)
        
        # triplet = -torch.log(D1 / (D1+D2))
        # breakpoint()
        
        # if self.entropy_conditioning:
        #     pred_anchor = F.softmax(pred_anchor, dim=1).detach()
        #     batch_size = anchor.shape[0]
        #     weight = 1.0 + torch.exp(-entropy(pred_anchor))
        #     weight = weight / torch.sum(weight) * batch_size
        #     triplet = triplet * weight
        # else:
        #     triplet = triplet
        
        if self.reduction=='mean':
            triplet = torch.mean(triplet)
        elif self.reduction=='sum':
            triplet = torch.sum(triplet)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return triplet

class TripletLossInLogisticDist(nn.Module):
    def __init__(self, entropy_conditioning: Optional[bool] = False, beta: Optional[float]=0.5,
                 dist: Optional[str]='l2_square', reduction: Optional[str] = 'mean'):
        super(TripletLossInLogisticDist, self).__init__()
        self.entropy_conditioning = entropy_conditioning
        self.beta = beta
        self.reduction = reduction
        self.dist_kinds = ['l2', 'l2_square', 'l1', 'smooth_l1']
        self.dist = dist
        assert self.dist in self.dist_kinds, 'Please check the argument "dist"!'

    def forward(self, pred_anchor: torch.Tensor, pred_pos: torch.Tensor, pred_neg: torch.Tensor) -> torch.Tensor:
        if self.entropy_conditioning: assert pred_anchor is not None, "pred_anchor is not specified!"
        # anchor, pos, neg -> shape: (B, d), B is batch size, d is dimension of a feature vector
        # pred_anchor -> shape: (B, c), B is batch size, c is the number of classes
        pred_anchor = F.softmax(pred_anchor, dim=1).detach()    # gradient flow is blocked w.r.t anchor.
        pred_pos = F.softmax(pred_pos, dim=1)
        pred_neg = F.softmax(pred_neg, dim=1)

        if self.dist=='l2_square':
            pos_pair_dist = l2_dist_square(pred_anchor, pred_pos)
            neg_pair_dist = l2_dist_square(pred_anchor, pred_neg)
        elif self.dist=='l2':
            l2_dist = torch.nn.PairwiseDistance()
            pos_pair_dist = l2_dist(pred_anchor, pred_pos)
            neg_pair_dist = l2_dist(pred_anchor, pred_neg)
        elif self.dist=='l1':
            l1_dist = torch.nn.PairwiseDistance(p=1.0)
            pos_pair_dist = l1_dist(pred_anchor, pred_pos)
            neg_pair_dist = l1_dist(pred_anchor, pred_neg)
        elif self.dist=='smooth_l1':
            smooth_l1 = torch.nn.SmoothL1Loss(reduction='none', beta=self.beta)
            pos_pair_dist = torch.sum(smooth_l1(pred_anchor, pred_pos), dim=1)
            neg_pair_dist = torch.sum(smooth_l1(pred_anchor, pred_neg), dim=1)
        else: raise Exception('Invalid self.dist: {}'.format(self.dist))
        
        triplet = pos_pair_dist-neg_pair_dist

        if self.entropy_conditioning:
            batch_size = pred_anchor.shape[0]
            weight = 1.0 + torch.exp(-entropy(pred_anchor))
            weight = weight / torch.sum(weight) * batch_size
            triplet = triplet * weight
        else:
            triplet = triplet
        
        if self.reduction=='mean':
            triplet = torch.mean(triplet)
        elif self.reduction=='sum':
            triplet = torch.sum(triplet)
        else:
            raise Exception('Invalid self.reduction: {}'.format(self.reduction))

        return triplet

def l2_dist_square(input1, input2):
    # input1, input2: shape (B, d)
    return torch.sum((input1-input2)**2, dim=1)  # shape (B,)

def delete_one_row(input, index):
    output = torch.cat((input[:index], input[index+1:]))
    return output

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

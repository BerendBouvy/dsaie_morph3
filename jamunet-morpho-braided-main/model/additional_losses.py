# This module stores the additional loss functions for binary classification
# These functions are supposed to improve the training process, allowing the model to improve the predictions
# These functions are recommended for imbalanced datasets but were never tested

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import label

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return bce * self.bce_weight + dice * (1 - self.bce_weight)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        true_pos = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()

        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        
        return 1 - tversky

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - jaccard
    
class BCEConnectivityLoss(nn.Module):
    '''
    Penalize predictions for which water areas are not connected to each other
    (i.e., avoid disconnected reaches/'water' areas)
    '''
    def __init__(self):
        super(BCEConnectivityLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, preds, targets):
        # standard BCE loss
        bce_loss = self.bce_loss(preds, targets)
        
        # Convert tensors to numpy arrays for connected component analysis
        preds_np = preds.cpu().numpy()
        
        # initialize connectivity penalty
        connectivity_penalty = 0.0
        
        # iterate over each prediction in the batch
        for pred in preds_np:
            labeled_array, num_features = label(pred)
            # compute penalty based on the number of connected components
            connectivity_penalty += num_features - 1  

        # normalize the connectivity penalty by the batch size
        connectivity_penalty /= preds.size(0)
        
        # add the connectivity penalty to BCE loss
        total_loss = bce_loss + torch.tensor(connectivity_penalty, dtype=torch.float32, requires_grad=True)
        
        return total_loss
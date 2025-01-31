# This module stores the functions needed for the training and inference steps of JamUNet

import torch

import torch.nn as nn
import numpy as np

from postprocessing.metrics import compute_metrics

# add the following code at the beginning of the notebook or .py file where the model is trained or tested. 
# if only one GPU is present you might need to remove the index "0" 
# torch.device('cuda:0') --> torch.device('cuda') / torch.cuda.get_device_name(0) --> torch.cuda.get_device_name() 

'''
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("CUDA Device Count: ", torch.cuda.device_count())
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    
print(f'Using device: {device}')
'''

def get_predictions(model, input_dataset, device='cuda:0'):
    '''
    Compute the predictions given the deep-learning model class and input dataset

    Inputs:
           model = class, deep-learning model
           input_dataset = TensorDataset, inputs for the model 
                           The input dataset has shape (n_samples, n_years_input=4, height=1000, width=500), 
           device = str, specifies the device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
                    Always remember to correctly set this key before running the simulations to avoid issues
                    (see the default code at the beginning of this module or in one of the training notebooks) 
    
    Output:
           predictions = list, model predictions
    '''    
    predictions = model(input_dataset.to(device))  
    return predictions.squeeze(1) 

def training_unet(model, loader, optimizer, nonwater=0, water=1, pixel_size=60, water_threshold=0.5, 
                  device='cuda:0', loss_f='BCE', physics=False, alpha_er = 1e-2, alpha_dep = 1e-3, loss_er_dep='Huber'):
    '''
    Training loop for the deep-learning model. Allows to choose the loss function for binary classification.
    Enables the inclusion of physics-induced loss terms (regression losses).

    Inputs: 
           model = class, deep-learning model to be trained
           loader = torch.utils.data.DataLoader element, data loader that combines a dataset 
                    and a sampler to feed data to the model in batches
           nonwater = int, 'non-water' class pixel value
                      default: 0 (scaled classification). 
                      If the original classification is used, this key should be set to 1
           water = int, 'water' class pixel value
                   default: 1 (scaled classification). 
                   If the original classification is used, this should be set to 2 
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
           water_threshold = float, generate binary predictions to compute the total areas of erosion and deposition
           device = str, specifies the device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
                    Always remember to correctly set this key before running the simulations to avoid issues
                    (see the default code at the beginning of this module or in one of the training notebooks)
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           physics = bool, sets whether physics-induced loss terms (total areas of erosion
                     and deposition) are included in the loss.
                     default: False, not included
           alpha_er = float, weight of the erosion loss term within the total loss.
                      default: 1e-2. Suggested range [1e-5, 1e-2] 
           alpha_dep = float, weight of the deposition loss term within the total loss.
                       default: 1e-3. Suggested range [1e-5, 1e-2] 
           loss_er_dep = str, regression loss function for erosion and deposition terms
                         default: 'Huber'. Other available options: 'RMSE', 'MAE'
                         If other loss functions are set it raises an Exception.
    
    Output: 
           losses = array of scalars, training losses 
    '''
    model.to(device)
    model.train() # specifies the model is in training mode

    losses = []
    # split in batches
    for batch in loader:
        input = batch[0].to(device)
        target = batch[1].to(device)

        # get predictions
        predictions = get_predictions(model, input, device=device)
        
        # compute binary classification loss
        binary_loss = choose_loss(predictions, target, loss_f)
        
        # physics-induced loss terms
        if physics:
            # need binary predictions
            binary_predictions = (predictions >= water_threshold).float()

            # get real and predicted total areas of erosion and deposition
            real_erosion_deposition = get_erosion_deposition(input[0][-1], target, nonwater, water, pixel_size)
            predicted_erosion_deposition = get_erosion_deposition(input[0][-1], binary_predictions, nonwater, water, pixel_size)
            
            # compute regression losses
            erosion_loss = choose_er_dep_loss(predicted_erosion_deposition[0], real_erosion_deposition[0], loss_er_dep)
            deposition_loss = choose_er_dep_loss(predicted_erosion_deposition[1], real_erosion_deposition[1], loss_er_dep)
            
            # sum loss terms with individual weights
            total_loss = binary_loss + alpha_er * erosion_loss + alpha_dep * deposition_loss 
            losses.append(total_loss.cpu().detach())
        
        else:
            total_loss = binary_loss
            losses.append(total_loss.cpu().detach())

        # backpropagate and update weights
        optimizer.zero_grad(set_to_none=True)   # reset the computed gradients
        total_loss.backward()                   # compute the gradients using backpropagation
        optimizer.step()                        # update the weights with the optimizer
        
    losses = np.array(losses).mean() 

    return losses

def validation_unet(model, loader, nonwater=0, water=1, device='cuda:0', loss_f='BCE', water_threshold=0.5):
    '''
    Validation loop for the deep-learning model. Allows to choose the loss function for binary classification.
    Computes binary validation metrics by setting a threshold for water classification. 
    Physics-induced loss terms are not included.

    Inputs: 
           model = class, deep-learning model to be validated/tested
           loader = torch.utils.data.DataLoader element, data loader that combines a dataset 
                    and a sampler to feed data to the model in batches
           nonwater = int, 'non-water' class pixel value
                      default: 0 (scaled classification). 
                      If the original classification is used, this key should be set to 1
           water = int, 'water' class pixel value
                   default: 1 (scaled classification). 
                   If the original classification is used, this should be set to 2 
           device = str, specifies the device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
                    Always remember to correctly set this key before running the simulations to avoid issues
                    (see the default code at the beginning of this module or in one of the training notebooks)
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           water_threshold = float, generate binary predictions to compute the binary validation metrics
    
    Output: 
           losses, accuracies, precisions, recalls, f1_scores, csi_scores 
                   = array of scalars, validation losses and metrics 
    '''
    model.to(device)
    model.eval() # specifies the model is in evaluation mode = validation/testing

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    csi_scores = []
    
    with torch.no_grad(): # specifies gradient is not computed
        # split in batches
        for batch in loader:
            input = batch[0].to(device)
            target = batch[1].to(device)
    
            # get predictions
            predictions = get_predictions(model, input, device=device)
            # generate binary predictions
            binary_predictions = (predictions >= water_threshold).float()
            # compute loss 
            loss = choose_loss(predictions, target, loss_f)
            # compute metrics
            accuracy, precision, recall, f1_score, csi_score = compute_metrics(binary_predictions, target, nonwater, water)
            
            losses.append(loss.cpu().detach())
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            csi_scores.append(csi_score)

    losses = np.array(losses).mean()             
    accuracies = np.array(accuracies).mean()     
    precisions = np.array(precisions).mean()    
    recalls = np.array(recalls).mean()           
    f1_scores = np.array(f1_scores).mean()      
    csi_scores = np.array(csi_scores).mean()    

    return losses, accuracies, precisions, recalls, f1_scores, csi_scores

def choose_loss(preds, targets, loss_f='BCE'):
    '''
    Choose the binary classification loss function for the training and inference steps.
    Allows to choose among the following options: 
        - Binary Cross Entropy (BCE) loss, measures the difference between binary predictions 
          (by default not activated with a Sigmoid layer) and targets (which should be within the range [0,1]). [*]
        - BCE with Logits, combines the previous one with a Sigmoid layer 
          and it is said to be more stable than the single BCE activated by Sigmoid. [**]
        - Focal loss, updated BCE/BCE with Logits loss recommended for imbalanced datasets. 
          If a Sigmoid layer is not included in the network, the BCE with Logits adaptation should be used. [***]

    If "BCE_Logits" is chosen, the Sigmoid activation is implemented within the loss and should be removed in the network. 
    Despite being said to be more stable, this function generated instabilities during the training process.
    The use of the simple "BCE" function with a Sigmoid activation as final layer of the network is recommended. 
     
    Inputs: 
           preds = torch.tensor, predictions generated by the model
           targets = torch.tensor, targets of the dataset
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception

    Output: 
           loss = scalar, classification loss between predictions and targets 

    [*] from torch.nn.BCELoss (https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss)
    [**] from torch.nn.BCEWithLogitsLoss (https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss)
    [***] adapted from torch.nn.functional.binary_cross_entropy ()
          and torch.nn.functional.binary_cross_entropy_with_logits()
    '''      
    if loss_f == 'Dice':
        smooth = 1e-6
        # intersection = (preds * targets).sum()
        # union = preds.sum() + targets.sum()
        # loss = (2. * intersection + smooth) / (union + smooth)
        # Calculate intersection and union
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
    
        # Compute Dice Coefficient
        dice = (2. * intersection + smooth) / (union + smooth)
    
        # Return Dice Loss
        loss = 1 - dice.mean()
    elif loss_f == 'BCE':
        # requires sigmoid activation
        loss = nn.BCELoss()(preds, targets)
    elif loss_f == 'BCE_Logits':
        # sigmoid activated by default within the function
        loss = nn.BCEWithLogitsLoss()(preds, targets)
    elif loss_f == 'Focal':
        # allows to choose between adapted BCE and adapted BCE with Logits
        loss = FocalLoss()(preds, targets)

    else: 
        raise Exception('The specified loss function is wrong. Check the documentation for the available loss functions.')

    return loss

class FocalLoss(nn.Module):
    '''
    Focal loss for binary classification of the satellite images. Recommended for imbalanced datasets.
    '''
    
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        '''
        Inputs:
               alpha, gamma = float, hyperparameters of the loss function to be fine-tuned
               logits = bool, allows to choose between BCE and BCE with Logits (sigmoid activation within the loss)
                        default: False, loss function is BCE.
               reduce = bool, allows to get the mean value of the loss
                        default: False, full array is returned
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, preds, targets): 
        '''
        Inputs:
               preds = torch.tensor, predictions geenrated by the model
               targets = torch.tensor, targets of the dataset
        
        Output: 
               F_loss = Focal loss between predictions and targets 
        '''
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def get_erosion_deposition(previous_year_img, current_year_img, nonwater=0, water=1, pixel_size=60):
    '''
    Compute the total areas of erosion and deposition. 
    Makes a pixel-wise comparison of the last input year and the target year image. 
    The area of erosion is the sum of 'water' pixels of the target year image 
    that were 'non-water' in the last input year image. 
    The area of deposition is the sum of 'non-water' pixels of the target year image
    that were 'water' in the last input year image.

    The total number of pixels of both the areas of erosion and deposition are multiplied 
    by the pixel area (the square of `pixel_size`).

    Inputs:
           previous_year_img = 2D array or tensor, representing previous year image
           current_year_img = 2D array or tensor, representing current year image
           nonwater = int, 'non-water' class pixel value
                      default: 0 (scaled classification). 
                      If the original classification is used, this key should be set to 1
           water = int, 'water' class pixel value
                   default: 1 (scaled classification). 
                   If the original classification is used, this should be set to 2
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
    
    Output:
           list, contains total areas of erosion and deposition in km^2
    '''  
    # sum erosion and deposition pixels
    erosion = torch.sum((previous_year_img == nonwater) & (current_year_img == water))
    deposition = torch.sum((previous_year_img == water) & (current_year_img == nonwater))
    
    # calculate areas of erosion and deposition
    factor = (pixel_size**2) / (1000**2) # conversion factor to get pixel area in km^2
    erosion_areas = erosion * factor
    deposition_areas = deposition * factor
    # .item() is required to correctly save the float number ### check
    return [erosion_areas.item(), deposition_areas.item()] 

def choose_er_dep_loss(preds, targets, loss_er_dep='Huber'):
    '''
    Choose the regression loss function of the total areas of erosion and deposition areas for the training step.

    It allows to choose among three options: 
        - Huber loss, computed with the pytorch function. It combines MAE and MSE loss, computed based 
          on the value of the error compared to the `delta` parameter (set to default = 1). [*]
        - Root Mean Square Error, computed by adding a square root to the pytorch Mean Square Error (MSE) function. [**]
        - Mean Absolut Error, computed using the pytorch function. [***]
    
    Given the intrinsic adaptability to the difference between target and prediction, the Huber loss is recommended. 
     
    Inputs: 
           preds = torch.tensor, predictions of total areas of erosion and deposition
           targets = torch.tensor, real areas of total erosion and deposition
           loss_er_dep = str, regression loss function for erosion and deposition terms
                         default: 'Huber'. Other available options: 'RMSE', 'MAE'
                         If other loss functions are set it raises an Exception.

    Output: 
           loss = scalar, regression loss with the specified function between predictions and targets 

    [*] from torch.nn.HuberLoss (https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)
    [**] from torch.nn.MSELoss (https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
    [***] from torch.nn.L1Loss (https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)
    '''
    if loss_er_dep == 'Huber':
        loss = nn.HuberLoss()(preds, targets)
    elif loss_er_dep == 'RMSE':
        loss = torch.sqrt(nn.MSELoss()(preds, targets))
    elif loss_er_dep == 'MAE':
        loss = nn.L1Loss()(preds, targets)
    else: 
        raise Exception('The specified loss function is wrong. Check the documentation for the available loss functions')
    
    return loss 
# This module contains the functions used for the analysis related to the no-change benchmark method

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset 
from matplotlib.colors import ListedColormap 

from model.train_eval import choose_loss, get_erosion_deposition
from postprocessing.metrics import compute_metrics

def nochange_preds_dataset(dataset, sample=None, return_targets=False):
    ''' 
    Make the "no-change" scenario predictions by coping the last (4th) input image for each sample and generating the predicted TensorDataest.

    Input: 
          dataset = TensorDataset, contains inputs and targets for the model
          sample = int or None, specifies the input-target combination to make the prediction
                   default: None, gets full dataset
          return_targets = bool, sets whether the function returns the targets dataset too
                           default: False, function does not return target dataset.
                           If set to True it returns the target dataset as well

    Output:
           predictions = torch.Tensor, predicted images for the "no-change" scenario
           targets = torch.Tensor, target images
           or
           TensorDataset(predictions) = TensorDataset, if return_targets = True
    '''
    
    if sample is None:
        # split inputs and targets
        inputs = dataset[:][0]
        targets = dataset[:][1]
        # extract last image for each sample  
        predictions = torch.Tensor(inputs[:, -1, :, :])  
    else:
        inputs = dataset[sample][0]
        targets = dataset[sample][1]
        # extract last image for each sample  
        predictions = torch.Tensor(inputs[-1, :, :])  

    return predictions, targets if return_targets else TensorDataset(predictions)

def validation_nochange(dataset, sample=None, return_targets=True, device='cuda:0', loss_f='BCE', nonwater=0, water=1, batch_size=None):
    '''
    Compute the loss and metrics considering the "no-change" scenario predictions.

    Inputs: 
           dataset = TensorDataset, contains inputs and targets for the model
           sample = int or None, specifies the input-target combination to make the prediction
                    default: None, gets full dataset
           return_targets = bool, sets whether the function returns the targets dataset too
                           default: False, function does not return target dataset.
                           If set to True it returns the target dataset as well
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu'
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           nonwater = int, represents pixel value of non-water class.
                      default: 0, based on the updated pixel classes. 
           water = int, represents pixel value of water class.
                   default: 1, based on the updated pixel classes.
           batch_size = int, number of samples per batch
                        default: None, batch size = predictions length 

    Outputs: 
            avg_loss = array of scalars, contains validation losses between predictions and targets  
            avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_csi_score = floats, average assessment metrics 
    '''

    predictions, targets = nochange_preds_dataset(dataset, sample=sample, return_targets=return_targets)
    combined_dataset = TensorDataset(predictions, targets)

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    csi_scores = []

    if batch_size is None:
        batch_size = len(predictions)  # process all at once if no batch size specified
    
    # create dataloader if batch processing is desired
    data_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in data_loader:
            pred_batch, target_batch = batch
            pred_batch, target_batch = pred_batch.to(device), target_batch.to(device)

            loss = choose_loss(pred_batch, target_batch, loss_f)
            accuracy, precision, recall, f1_score, csi_score = compute_metrics(pred_batch, target_batch, nonwater, water)
            
            losses.append(loss.cpu().detach())
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            csi_scores.append(csi_score)

    avg_loss = np.array(losses).mean()

    avg_accuracy = np.array(accuracies).mean()
    avg_precision = np.array(precisions).mean()
    avg_recall = np.array(recalls).mean()
    avg_f1_score = np.array(f1_scores).mean()
    avg_csi_score = np.array(csi_scores).mean()

    return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_csi_score

def plot_erosion_deposition_nochange(sample_img, dataset, nonwater=0, water=1, pixel_size=60, ax=None):
    '''
    Compute and plot the real and predicted total areas of erosion and deposition

    Inputs:
           sample_img = int, specifies the input-target combination to make the prediction
           dataset = TensorDataset, contains inputs and targets for the model
           nonwater = int, represents pixel value of non-water class.
                      default: 0, based on the updated pixel classes. 
           water = int, represents pixel value of water class.
                   default: 1, based on the updated pixel classes.
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
           ax = matplotlib axis, axis to plot the bar plot on.
                default: None, a new figure will be created. If used in combination with the function `show_evolution`
                         it should be set equal to `ax[1,3]`
    
    Output:
           None, plots the real and predicted erosion and deposition areas.
    '''
    input_img = dataset[sample_img][0].unsqueeze(0)

    prediction, target_img = nochange_preds_dataset(dataset, sample=sample_img, return_targets=True)
    prediction, target_img = prediction.cpu(), target_img.cpu()

    real_erosion_deposition = get_erosion_deposition(input_img[0][-1].cpu(), target_img, nonwater, water, pixel_size)
    pred_erosion_deposition = get_erosion_deposition(input_img[0][-1].cpu(), prediction, nonwater, water, pixel_size)

    categories = ['Erosion', 'Deposition']

    # adjust bar width and positions
    bar_width = 0.3 
    bar_positions = np.arange(len(categories))

    if ax is None:
        fig, ax = plt.subplots()

    ax.bar(bar_positions - bar_width/2, real_erosion_deposition, bar_width, label='Real areas', color='white', edgecolor='k', hatch='///')
    ax.bar(bar_positions + bar_width/2, pred_erosion_deposition, bar_width, label='Predicted areas', color='white', edgecolor='k', hatch='\\\\')

    ax.set_ylabel('Area (km²)')
    ax.set_title('Real and predicted erosion\n and deposition areas')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(categories)
    if real_erosion_deposition[0] > real_erosion_deposition[1]:
        loc = 'upper right'
    else:
        loc = 'upper left'
    ax.legend(loc=loc)

    if ax is None:
        plt.show() 

def show_evolution_nochange(sample_img, dataset, nonwater=0, water=1, pixel_size=60, water_threshold=0.5, 
                            train_val_test='testing', device='cuda:0', save_img=False):
    '''
    Plot input images, target image, predicted image, and misclassification map (prediciton minus target).
    It also includes the bar plot with the real and predicted total areas of erosion and deposition.

    Inputs:
           sample_img = int, specifies the input-target combination to make the prediction
           dataset = TensorDataset, contains inputs and targets for the model
           nonwater = int, represents pixel value of non-water class.
                      default: 0, based on the updated pixel classes. 
           water = int, represents pixel value of water class.
                   default: 1, based on the updated pixel classes.
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)
           train_val_test = str, specifies what the images are used for.
                            default: 'testing'. Other available options: 'training', 'validation'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu' 
           save_img = bool, sets whether the function is used for saving the image or not
                      default: False, image is not being saved
    
    Output:
           None, plots the input, target and predicted images as well as the misclassification map and the total areas of erosion and deposition 
    '''
    input_img = dataset[sample_img][0].unsqueeze(0)

    prediction, target_img = nochange_preds_dataset(dataset, sample=sample_img, return_targets=True)
    prediction, target_img = prediction.cpu(), target_img.cpu()
    
    diff = prediction - target_img

    shp = target_img.shape
    x_ticks = np.arange(0, shp[1], 300)
    y_ticks = np.arange(0, shp[0], 300)  

    # convert x_ticks and y_ticks from pixels to meters
    x_tick_labels = [round(tick * 60/1000, 2) for tick in x_ticks]  
    y_tick_labels = [round(tick * 60/1000, 2) for tick in y_ticks]

    year, year2 = [1988 + i for i in range(2)], [2000 + i  for i in range(17)] # change later if 2021 prediction is made
    year = year + year2

    fig, ax = plt.subplots(2, 4, figsize=(10,10))
    
    # custom colormaps
    binary_cmap = ListedColormap(['palegoldenrod', 'navy']) 
    diff_cmap = ListedColormap(['red', 'white', 'green'])
    binary_diff_cmap = ListedColormap(['black', 'white'])

    for i in range(ax.shape[1]):
        ax[0,i].imshow(input_img[0][i].cpu(), cmap=binary_cmap, vmin=0)
        ax[0,i].set_title(f'Input year {year[sample_img]+i*1}', fontsize=13)
    
    im1 = ax[1,0].imshow(target_img, cmap=binary_cmap, vmin=0)
    ax[1,1].imshow(prediction, cmap=binary_cmap)
    im2 = ax[1,2].imshow(diff, cmap=diff_cmap, vmin=-1, vmax=1)
    ax[1,2].imshow(target_img, cmap=binary_diff_cmap, vmin=0, alpha=0.2)

    # include erosion and deposition areas plot in the bottom right corner by setting `ax=ax[1,3]`
    prediction_binary = (prediction >= water_threshold).float()
    real_erosion_deposition = get_erosion_deposition(input_img[0][-1].cpu(), target_img, nonwater, water, pixel_size)
    pred_erosion_deposition = get_erosion_deposition(input_img[0][-1].cpu(), prediction_binary, nonwater, water, pixel_size)
    
    categories = ['Erosion', 'Deposition']
    bar_width = 0.3
    bar_positions = np.arange(len(categories))

    ax[1,3].bar(bar_positions - bar_width/2, real_erosion_deposition, bar_width, label='Real areas', color='white', edgecolor='k', hatch='///')
    ax[1,3].bar(bar_positions + bar_width/2, pred_erosion_deposition, bar_width, label='Predicted areas', color='white', edgecolor='k', hatch='xxx')

    ax[1,3].set_ylabel('Area (km²)', fontsize=13)
    ax[1,3].set_title('Real and predicted \nerosion and deposition areas', fontsize=13)
    ax[1,3].set_xticks(bar_positions, fontsize=12)
    ax[1,3].set_xticklabels(categories, fontsize=12)
    ax[1,3].yaxis.tick_right()  # move ticks to the right
    ax[1,3].yaxis.set_label_position('right')  # move label to the right
    ax[1,3].tick_params(left=False)

    ax[1,0].set_title(f'Target year {year[sample_img]+4}\n', fontsize=13)
    ax[1,1].set_title(f'Predicted image\n', fontsize=13)
    ax[1,2].set_title(f'Misclassification map\n(prediction - target)', fontsize=13)
    ax[1,3].set_title(f'Erosion and\n deposition areas', fontsize=13)

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            if j == 3 and i == 1:
                continue  # skip ticks and labels for the last subplot (erosion and depsotion areas) 

            ax[i,j].set_xticks(x_ticks)
            ax[i,j].set_yticks(y_ticks)

            if i == 1 and j < (ax.shape[1]-1):
                ax[i,j].set_xlabel('Width (km)', fontsize=13)
                ax[i,j].set_xticklabels(x_tick_labels)
            elif i != 1 and j <= ax.shape[1]:  # don't add x-ticks in the bottom right plot as it shows erosion/deposition areas and i != ax.shape[0]
                ax[i,j].set_xticklabels([])

            if j == 0:
                ax[i,j].set_yticklabels(y_tick_labels)
                ax[i,j].set_ylabel('Length (km)', fontsize=13) 
            else:
                ax[i,j].set_yticklabels([]) 
    
    fig.subplots_adjust(wspace=0.1, hspace=0.2)
    if save_img:
        plt.savefig(rf'images\report\4_results\nochange\{train_val_test}{sample_img}_nochange.png', 
                    bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close(fig)  # close the figure to free memory
    else:
        plt.show()
    return None
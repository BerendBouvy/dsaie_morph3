# This module contains the functions used for postprocessing the model predictions, visualising them, and performing further analysis 

import torch 
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.ticker as ticker

from matplotlib.colors import ListedColormap 
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader

from model.train_eval import validation_unet, get_predictions, choose_loss, get_erosion_deposition
from postprocessing.metrics import *

def load_losses_metrics(path):
    '''
    Load .csv file with training and validation losses and metrics.

    Inputs: 
           path = str, path where the .csv file is stored
    
    Output:
           df = dataframe, contains training and validation losses and metrics separated in different columns
    '''
    df = pd.read_csv(path, header=0)
    return df

def plot_losses(train_losses, validation_losses, model, loss_f='BCE', folder_save=r'postprocessing\losses', save_name=None, combined=False):
    '''
    Plot the training and validation losses. 
    
    Inputs: 
           train_losses = list or array, contains training losses
           validation_losses = list or array, contains validation losses
           model = class, trained deep-learning model to be validated/tested
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           folder_save = str, directory where losses plots are stored
                         default: r'postprocessing\losses'
           save_name = str, name for saving losses plot
                       default: None, in this case a warning message is printed, otherwise a *.png image is exported
           combined = bool, sets whether the function is used in combination with `plot_metrics` to combine both plots.
                      default: False, if set to True it makes sure that the plot is shown when the function is called
    
    Outputs: 
            None, it plots the training and validation losses evolution throughout epochs
            and saves the plot as a *.png file if `save_name` is specified
    '''
    model_who = str(model.__class__.__name__)
    
    if not combined:
        plt.figure() 
    plt.plot(train_losses, color='navy', linewidth=2.5, ls='-', label='training')
    plt.plot(validation_losses, color='crimson', linewidth=2.5, ls='--', label='validation')
    
    # # identify min validation loss
    # min_val_loss = min(validation_losses)
    # min_val_epoch = validation_losses.index(min_val_loss)
    # plt.plot(min_val_epoch, min_val_loss, 'o', color='green', label='min validation Loss', 
    #          markerfacecolor='none', markersize=10, markeredgewidth=2.5)
    # plt.yscale('log')
    
    plt.title(f'{model_who} training and validation with {loss_f} loss', fontsize=18)
    plt.xlabel('Epochs (-)', fontsize=16)
    plt.ylabel(f'{loss_f} loss (-)', fontsize=16)
    plt.legend()
    if not combined:
        plt.show()

    if not combined and save_name==None:
        print(f"ATTENTION: the argument `save_name` is not specified: the plot is not saved.")
    elif not combined and save_name is not None:
        save_path = os.path.join(folder_save, save_name)
        plt.savefig(save_path + '.png')

    return None

def plot_metrics(metrics, model, water_threshold=0.5, folder_save=r'postprocessing\metrics', save_name=None, combined=False):
    '''
    Plot validation metrics (accuracy, precision, recall, F1-score, and CSI).
    
    Inputs: 
           metrics = list of lists, each containing validation metrics (accuracy, precision, recall, f1_score, csi)
           model = class, trained deep-learning model to be validated/tested
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)
           folder_save = str, directory where plot is stored
                         default: r'postprocessing\metrics'
           save_name = str, name for saving metrics plot
                       default: None, in this case a warning message is printed, otherwise a *.png image is exported
           combined = bool, sets whether the function is used in combination with `plot_metrics` to combine both plots.
                      default: False, if set to True it makes sure that the plot is shown when the function is called
        
    Outputs: 
            None, it plots the validation metrics evolution across epochs given a water threshold for binary classification
            and saves the plot as a *.png file if `save_name` is specified
    '''

    model_who = str(model.__class__.__name__)

    accuracy = metrics[0]
    precision = metrics[1]
    recall = metrics[2]
    f1_score = metrics[3]
    csi_score = metrics[4]

    if not combined:
        plt.figure() 
    plt.plot(accuracy, color='mediumblue', label='accuracy', linewidth=2.5, ls=(5, (10, 5)))
    plt.plot(precision, color='crimson', label='precision', linewidth=2.5, ls='-.') 
    plt.plot(recall, color='darkgoldenrod', label='recall', linewidth=2.5)
    plt.plot(f1_score, color='black', label='F1-score', linewidth=2.5, ls=(0, (3, 1, 1, 1, 1, 1)))
    plt.plot(csi_score, color='seagreen', label='CSI', linewidth=2.5, ls=(0, (5, 1)))    
    
    # # identify max validation recall
    # max_recall = np.max(recall)
    # max_recall_epoch = recall.index(max_recall)
    # plt.plot(max_recall_epoch, max_recall,  'o', color='black', label='max recall', 
    #          markerfacecolor='none', markersize=10, markeredgewidth=2.5)
    
    plt.title(f'{model_who} validation metrics with water threshold {water_threshold}', fontsize=18)
    plt.xlabel('Epochs (-)', fontsize=16)
    plt.ylabel(f' Metrics (-)', fontsize=16)
    plt.legend()
    if not combined: 
        plt.show()

    if not combined and save_name==None:
        print(f"ATTENTION: the argument `save_name` is not specified: the plot is not saved.")
    elif not combined and save_name is not None:
        save_path = os.path.join(folder_save, save_name)
        plt.savefig(save_path + '.png')

    return None

def plot_losses_metrics(train_losses, validation_losses, metrics, model, loss_f='BCE', water_threshold=0.5, folder_save=r'images\report\4_results', save_name=None):
    '''
    Plot the training and validation losses and metrics.
    Combine single plots as two subplots (losses on the left, metrics on the right).
    
    Inputs: 
           train_losses = list or array, contains training losses
           validation_losses = list or array, contains validation losses
           metrics = list of lists, each containing validation metrics (accuracy, precision, recall, f1_score, csi)
           model = class, trained deep-learning model to be validated/tested
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)
           folder_save = str, directory where plot is saved
                         default: r'images\report\4_results'
           save_name = str, name for saving losses plot
                       default: None, in this case a warning message is printed, otherwise a *.png image is exported
    
    Outputs: 
            None, it plots the training and validation losses and metrics evolution across epochs 
            and saves the plot as a *.png file if `save_name` is specified
    '''
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # plot losses
    plt.sca(axs[0])
    plot_losses(train_losses, validation_losses, model, loss_f, folder_save, save_name, combined=True)
    axs[0].set_title(f'{loss_f} training and validation losses', fontsize=17)
    axs[0].tick_params(axis='both', which='major', labelsize=14) 
    axs[0].legend(fontsize=13)
    axs[0].set_xlim([-0.5, len(train_losses)+0.5])
    axs[0].set_xticks(range(0, len(train_losses)+10, 10))  
    axs[0].set_xticklabels([str(i) if i % 10 == 0 else '' for i in range(0, len(train_losses)+10, 10)])
    
    # plot metrics
    plt.sca(axs[1])
    plot_metrics(metrics, model, water_threshold, folder_save, save_name, combined=True)
    axs[1].set_title(rf'Validation metrics with binary threshold $w_{{thr}}$={water_threshold}', fontsize=17)
    axs[1].tick_params(axis='both', which='major', labelsize=14) 
    axs[1].legend(fontsize=13) 
    axs[1].set_xlim([-0.5, len(train_losses)+0.5])
    axs[1].set_xticks(range(0, len(train_losses)+10, 10)) 
    axs[1].set_xticklabels([str(i) if i % 10 == 0 else '' for i in range(0, len(train_losses)+10, 10)]) 

    plt.tight_layout()
    
    if save_name is None:
        print("ATTENTION: the argument `save_name` is not specified: the plot is not saved.")
        plt.show()
    else:
        save_path = os.path.join(folder_save, save_name)
        fig.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

    return None

def show_evolution_nolegend(sample_img, dataset, model, nonwater=0, water=1, water_threshold=0.5, pixel_size=60, device='cuda:0', 
                   train_val_test='testing', loss_recall='min loss', spatial_temporal='spatial', save_img=False):
    '''
    Plot input images, target image, predicted image, and misclassification map (prediciton minus target).
    It also includes the bar plot with the real and predicted total areas of erosion and deposition.

    Inputs:
           sample_img = int, specifies the input-target combination
           dataset = TensorDataset, contains inputs and targets for the model
           model = class, trained deep-learning model to be validated/tested
           nonwater = int, represents pixel value of non-water class.
                      default: 0, based on the updated pixel classes. 
           water = int, represents pixel value of water class.
                   default: 1, based on the updated pixel classes.
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu'
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           loss_recall = str, specifies the model type, whether the one with minimum validatoin loss or the one with maximum validation recall.
                         Available options: 'min loss', 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal'
           save_img = bool, sets whether the function is used for saving the image or not
                      default: False, image is not being saved

    Output:
           None, it plots the inputs, target, and predicted images as well as the misclassification map 
                 and barplot of the areas of erosion and deposition 
    '''
    input_img = dataset[sample_img][0].unsqueeze(0)
    target_img = dataset[sample_img][1].cpu()

    prediction = model(input_img).detach().cpu()
    prediction = (prediction >= water_threshold).float()

    diff = prediction - target_img

    shp = target_img.shape
    x_ticks = np.arange(0, shp[1], 300)
    y_ticks = np.arange(0, shp[0], 300)

    # convert x_ticks and y_ticks from pixels to meters
    x_tick_labels = [round(tick * 60/1000, 2) for tick in x_ticks]  
    y_tick_labels = [round(tick * 60/1000, 2) for tick in y_ticks]

    fig, ax = plt.subplots(2, 4, figsize=(10,10))
    
    # custom colormaps
    grey_cmap = ListedColormap(['palegoldenrod', 'navy'])
    diff_cmap = ListedColormap(['red', 'white', 'green'])
    grey_diff_cmap = ListedColormap(['black', 'white'])
    
    # get target year
    year, year2 = [1988 + i for i in range(2)], [2000 + i  for i in range(17)] # update if 2021 prediction is made
    year = year + year2
    
    for i in range(ax.shape[1]):
        ax[0,i].imshow(input_img[0][i].cpu(), cmap=grey_cmap, vmin=0)
        if spatial_temporal == 'spatial':
            ax[0,i].set_title(f'Input year {year[sample_img]+i*1}', fontsize=13)
        else:
            ax[0,i].set_title(f'Input year {2016+i*1}', fontsize=13)

    im1 = ax[1,0].imshow(target_img, cmap=grey_cmap, vmin=0)
    ax[1,1].imshow(prediction[0][0], cmap=grey_cmap)
    im2 = ax[1,2].imshow(diff[0][0], cmap=diff_cmap, vmin=-1, vmax=1)
    ax[1,2].imshow(target_img, cmap=grey_diff_cmap, vmin=0, alpha=0.2)

    # compute locations of erosion and deposition
    prediction_binary = (prediction >= water_threshold).float()
    real_erosion_deposition = get_erosion_deposition(input_img[0][-1].cpu(), target_img, nonwater, water, pixel_size)
    pred_erosion_deposition = get_erosion_deposition(input_img[0][-1].cpu(), prediction_binary, nonwater, water, pixel_size)
    
    categories = ['Erosion', 'Deposition']
    
    # adjust bar width and positions
    bar_width = 0.3
    bar_positions = np.arange(len(categories))

    ax[1,3].bar(bar_positions - bar_width/2, real_erosion_deposition, bar_width, label='Real areas', color='white', edgecolor='k', hatch='///')
    ax[1,3].bar(bar_positions + bar_width/2, pred_erosion_deposition, bar_width, label='Predicted areas', color='white', edgecolor='k', hatch='xxx')
    
    ax[1,3].set_ylabel('Area (km²)', fontsize=13)
    ax[1,3].set_xticks(bar_positions, fontsize=12)
    ax[1,3].set_xticklabels(categories, fontsize=12)
    ax[1,3].yaxis.tick_right()  # move ticks to the right
    ax[1,3].yaxis.set_label_position('right')  # move label to the right
    ax[1,3].tick_params(left=False)

    if spatial_temporal == 'spatial':
        ax[1,0].set_title(f'Target year {year[sample_img]+4}\n', fontsize=13)
    else:
        ax[1,0].set_title(f'Target year 2020\n', fontsize=13)

    ax[1,1].set_title(f'Predicted image\n', fontsize=13)
    ax[1,2].set_title(f'Misclassification map\n(prediction - target)', fontsize=13)
    ax[1,3].set_title(f'Erosion and\n deposition areas', fontsize=13)

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            if j == 3 and i == 1:
                continue  # skip ticks and labels for the last subplot (erosion and deposition areas) 

            ax[i,j].set_xticks(x_ticks, fontsize=12)
            ax[i,j].set_yticks(y_ticks, fontsize=12)

            if i == 1 and j < (ax.shape[1]-1):
                ax[i,j].set_xlabel('Width (km)', fontsize=14)
                ax[i,j].set_xticklabels(x_tick_labels, fontsize=12)
            elif i != 1 and j <= ax.shape[1]:  # don't add x-ticks in the bottom right plot as it shows erosion/deposition areas and i != ax.shape[0]
                ax[i,j].set_xticklabels([])

            if j == 0:
                ax[i,j].set_yticklabels(y_tick_labels, fontsize=12)
                ax[i,j].set_ylabel('Length (km)', fontsize=14) 
            else:
                ax[i,j].set_yticklabels([]) 

    # adjust spacing
    fig.subplots_adjust(wspace=0.1, hspace=0.2) #top=0.85, , bottom=0.15

    if save_img:
        plt.savefig(rf'images\report\4_results\{loss_recall}_{spatial_temporal}\{train_val_test}{sample_img}_{loss_recall}_{spatial_temporal}.png', 
                    bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close(fig)  # close the figure to free memory
    else:
        plt.show()
    return None

def erosion_sites(model, dataset, sample, nonwater=0, water=1, water_threshold=0.5, train_val_test='testing', 
                  model_type='min loss', spatial_temporal='spatial', device='cuda:0', save_img=False):
    '''
    Plot predicted and real locations of erosion and deposition.
    Erosion: 'water' pixels at target year that were 'non-water' at last input year.
    Deposition: 'non-water' pixels at target year that were 'water' at last input year. 

    Inputs:
           model = class, trained deep-learning model to be validated/tested
           dataset = TensorDataset, dataset used for the model
           sample = int, specifies the input-target combination 
           nonwater = int, class value for non-water pixels.
                      default: 0, if classes are not scaled it should be set = 1
           water = int, class value for water pixels.
                   default: 1, if classes are not scaled it should be set = 2 
           water_threshold = float, threshold value for classifying water pixels
                             default: 0.5
           train_val_test = str, specifies for what the images are used for.
                            default: 'testing'. Other available options: 'training', 'validation'
           model_type = str, specifies which model is considered
                        default: 'min loss'. Other available option: 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu' 
           save_img = bool, sets whether the function is used for saving the image or not
                      default: False, image is not being saved 

    Output: 
           None, plots the predicted and real locations of erosion and deposition
    '''
    input  = dataset[sample][0].to(device)
    target = dataset[sample][1].to(device)

    prediction = model(input.unsqueeze(0)).to(device).squeeze(0).squeeze(0)
    binary_prediction = (prediction >= water_threshold).float()
    
    # locations of erosion and deposition are computed considering the last input year conditions
    previous_year = input[-1]

    real_erosion_sites = torch.where((previous_year==nonwater) & (target==water), 1, 0)
    pred_erosion_sites = torch.where((previous_year==nonwater) & (binary_prediction==water), 1, 0)
    real_deposition_sites = torch.where((previous_year==water) & (target==nonwater), 1, 0)
    pred_deposition_sites = torch.where((previous_year==water) & (binary_prediction==nonwater), 1, 0)
    
    # clone last input year image to avoid overwriting 
    previous_year_pred_er, previous_year_real_er = previous_year.clone(), previous_year.clone()
    previous_year_pred_dep, previous_year_real_dep = previous_year.clone(), previous_year.clone()
    
    # get mask of predicted and real locations of erosion 
    real_erosion_mask = (real_erosion_sites == 1).float()
    pred_erosion_mask = (pred_erosion_sites == 1).float()
    # update last input year clone images pixels overlapping with mask for easier postprocessing
    previous_year_real_er[real_erosion_mask==1] = 2
    previous_year_pred_er[pred_erosion_mask==1] = 2

    # repeat same procedure for locations of deposition 
    real_deposition_mask = (real_deposition_sites == 1).float()
    pred__deposition_mask = (pred_deposition_sites == 1).float()
    previous_year_real_dep[real_deposition_mask==1] = 2
    previous_year_pred_dep[pred__deposition_mask==1] = 2

    fig, axes = plt.subplots(1, 4, figsize=(10, 10), sharey=True)
    plt.subplots_adjust(hspace=0.1)
    
    cmap_erosion = ListedColormap(['palegoldenrod', 'navy', 'red']) 
    cmap_deposition = ListedColormap(['palegoldenrod', 'navy', 'green'])

    shp = target.shape
    x_ticks = np.arange(0, shp[1], 300)
    y_ticks = np.arange(0, shp[0], 300)  

    # convert x_ticks and y_ticks from pixels to meters
    x_tick_labels = [round(tick * 60/1000, 2) for tick in x_ticks]  
    y_tick_labels = [round(tick * 60/1000, 2) for tick in y_ticks]
    
    axes[0].imshow(previous_year_pred_er.cpu(), cmap=cmap_erosion, vmin=0, vmax=2)
    axes[1].imshow(previous_year_real_er.cpu(), cmap=cmap_erosion, vmin=0, vmax=2)

    axes[2].imshow(previous_year_pred_dep.cpu(), cmap=cmap_deposition, vmin=0, vmax=2)
    axes[3].imshow(previous_year_real_dep.cpu(), cmap=cmap_deposition, vmin=0, vmax=2)

    axes[0].set_title('Predicted')
    axes[1].set_title('Real')

    axes[2].set_title('Predicted')
    axes[3].set_title('Real')
    fig.subplots_adjust(wspace=0.3) 

    axes[0].set_ylabel('Length (km)', fontsize=14)
    axes[0].set_yticklabels(y_tick_labels, fontsize=12)

    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    
    for ax in axes:
        ax.set_xticks(x_ticks, fontsize=12)
        ax.set_xticklabels(x_tick_labels, fontsize=12)
        ax.set_yticks(y_ticks, fontsize=12)
        ax.set_xlabel('Width (km)', fontsize=14)
    
    fig.text(0.3, 0.74, 'Erosion', ha='center', va='center', fontsize=16)
    fig.text(0.77, 0.74, 'Deposition', ha='center', va='center', fontsize=16)

    plt.tight_layout()

    if save_img:        
        plt.savefig(rf'images\report\4_results\locations\{train_val_test}{sample}_{model_type}_{spatial_temporal}_loc.png', 
                    bbox_inches='tight', dpi=1000) 
        plt.show()
        plt.close(fig)  # close the figure to free memory
    else:
        plt.show()
    plt.show()
    return None

def plot_erosion_deposition(sample_img, dataset, model, nonwater=0, water=1, pixel_size=60, water_threshold=0.5, device='cuda:0', ax=None):
    '''
    Plot the real and predicted areas of erosion and deposition.

    Inputs:
           sample_img = int, specifies the input-target combination
           dataset = TensorDataset, contains inputs and targets for the model
           model = class, trained deep-learning model to be validated/tested
           nonwater = int, represents pixel value of non-water class.
                      default: 0, based on the updated pixel classes. 
           water = int, represents pixel value of water class.
                   default: 1, based on the updated pixel classes.
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu'
           ax = matplotlib axis, axis to plot the bar plot on.
                default: None, a new figure will be created. If used in combination with the function `show_evolution`
                         it should be set equal to `ax[1,3]`
    
    Output:
           None, it plots the real and predicted total areas of erosion and deposition
    '''

    input_img = dataset[sample_img][0].unsqueeze(0).to(device)
    target_img = dataset[sample_img][1].cpu()

    prediction = model(input_img).detach().cpu().squeeze(0).squeeze(0)
    binary_predictions = (prediction >= water_threshold).float()

    real_erosion_deposition = get_erosion_deposition(input_img[0][-1].cpu(), target_img, nonwater, water, pixel_size)
    pred_erosion_deposition = get_erosion_deposition(input_img[0][-1].cpu(), binary_predictions, nonwater, water, pixel_size)
    
    categories = ['Erosion', 'Deposition']

    # adjust bar width and positions
    bar_width = 0.3
    bar_positions = np.arange(len(categories)) * 0.8

    if ax is None:
        fig, ax = plt.subplots(figsize=(2,5))

    ax.bar(bar_positions - bar_width/2, real_erosion_deposition, bar_width, label='Real areas', color='white', edgecolor='k', hatch='///')
    ax.bar(bar_positions + bar_width/2, pred_erosion_deposition, bar_width, label='Predicted areas', color='white', edgecolor='k', hatch='xxx')

    ax.set_ylabel('Area (km²)', fontsize=13)
    ax.set_title('Real and predicted erosion and deposition areas', fontsize=13)
    ax.set_xticks(bar_positions, fontsize=12)
    ax.set_xticklabels(categories, fontsize=12)
    
    if ax is None:
        plt.show()

def averaging_images(dataset, sample_id, water_threshold=None):
    '''
    Compute the average image from the four input images.
    It is used as a benchmark to compare model performances with a simple averaging. The function allows to set a water threshold
    to have a binary classification. In case the threshold is set to 'None', the function returns the predicted average
    (hence considering negative values if no-data pixels are present within the input images).

    Inputs:
           dataset = TensorDataset, contains inputs and targets for the model
           sample_id = int, specifies the input-target combination
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)
    
    Output:
           predicted_average or binary_predictions = tensors, represents the simple average and the binary average, respectively
           binary
    '''
    input = dataset[sample_id][0]
    predicted_average = torch.mean(input, dim=0) # need to set dimension of averaging to avoid size mismatch
    if water_threshold is not None:
        binary_predictions = (predicted_average >= water_threshold).float()
    return binary_predictions if water_threshold is not None else predicted_average

def total_losses_metrics_dataset(model, dataset, loss_f='BCE', nonwater=0, water=1, water_threshold=0.5, overall=True, model_type='min loss', device='cuda:0'):
    '''
    Get loss and metrics for each sample of the given dataset

    Inputs:
           model = class, trained deep-learning model to be validated/tested
           dataset = TensorDataset, contains inputs and targets for the model
           loss_f = str, binary classification loss function
                    default: 'BCE'. Other available options: 'BCE_Logits', 'Focal'
                    If other loss functions are set it raises an Exception
           nonwater = int, represents pixel value of non-water class.
                      default: 0, based on the updated pixel classes. 
           water = int, represents pixel value of water class.
                   default: 1, based on the updated pixel classes.
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)
           overall = bool, sets whether functin returns the dataset average loss and metrics
                     default: True
           model_type = str, specifies the model type, whether the one with minimum validatoin loss or the one with maximum validation recall.
                        default: 'min loss'. Other available option: 'max recall'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu'
    
    Output: 
           None, prints avg_loss, avg_accuracies, avg_precisions, 
           avg_recalls, avg_f1_scores, avg_csi_scores = lists,
                                                        averaged loss and metrics by batch
           or 
           losses, accuracies, precisions, recalls, f1_scores, csi_scores = lists, averaged loss and metrics for all dataset samples
    ''' 
    model.to(device)
    model.eval()

    # set batch_size = 1 to remove its effect on the loss averaging computation 
    batch_size = 1
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    losses = []  

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    csi_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            input = batch[0].to(device)
            target = batch[1].to(device)
    
            predictions = get_predictions(model, input, device=device)
            binary_predictions = (predictions >= water_threshold).float()
            
            # compute loss without scaling by batch size
            loss = choose_loss(predictions, target, loss_f)
            losses.append(loss.item())  # store the loss
            
            # compute metrics
            accuracy, precision, recall, f1_score, csi_score = compute_metrics(binary_predictions, target, nonwater, water)
            
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            csi_scores.append(csi_score)
    
    if overall:
        avg_loss = np.mean(losses) 
        avg_accuracies = np.mean(accuracies) 
        avg_precisions = np.mean(precisions) 
        avg_recalls = np.mean(recalls) 
        avg_f1_scores = np.mean(f1_scores)
        avg_csi_scores = np.mean(csi_scores) 
        
        print(f'Average metrics for test dataset using {model_type} model:\n\n\
{loss_f} loss:          {avg_loss:.3e}\n\
Accuracy:          {avg_accuracies:.3f}\n\
Precision:         {avg_precisions:.3f}\n\
Recall:            {avg_recalls:.3f}\n\
F1 score:          {avg_f1_scores:.3f}\n\
CSI score:         {avg_csi_scores:.3f}')
    
    else:
        return losses, accuracies, precisions, recalls, f1_scores, csi_scores

def total_losses_metrics_dataset(model, dataset, loss_f='BCE', nonwater=0, water=1, water_threshold=0.5, device='cuda:0'):
    '''
    Get loss and metrics for each sample of the given dataset

    Inputs:
           model = class, deep-learning model trained for predicting the morphological changes  
           dataset = TensorDataset, dataset used for the model
           loss_f = str, key that specifies the function for computing the loss,
                    default: 'BCE', the other available option is 'BCE_Logits' but in this case make sure 
                    that the model output is not activated with a sigmoid function.
                    If other loss functions are set it raises an Exception.
           mask = bool, sets whether the input dataset is masked in order to remove no-data pixels and replace these with 0 = non-water
           nonwater = int, class value for non-water pixels.
                      default: 0, if classes are not scaled it should be set = 1
           water = int, class value for water pixels.
                   default: 1, if classes are not scaled it should be set = 2 
           water_threshold = float, threshold value for classifying water pixels
                             default: 0.5
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0', other availble option: 'cpu'
    
    Output: 
           losses, accuracies, precisions, recalls, f1_scores, csi_scores = lists of floats, contain the loss,
                                                                            accuracy, precision, recall, F1-score and CSI-score
                                                                            for all dataset samples
    '''   
#     losses, accuracies, precisions, recalls, f1_scores, csi_scores = validation_unet(model, dataset, nonwater=nonwater, water=water, device=device, loss_f=loss_f, 
#                                                                                      water_threshold=0.5)
    model.to(device)
    model.eval() # specifies the model is in evaluation mode = validation

    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    csi_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            input = batch[0].to(device)
            target = batch[1].to(device)
    
            # get predictions
            predictions = get_predictions(model, input, device=device)
            binary_predictions = (predictions >= water_threshold).float()
            # compute loss 
            loss = choose_loss(predictions, target, loss_f)
            accuracy, precision, recall, f1_score, csi_score = compute_metrics(binary_predictions, target, nonwater, water)
            
            losses.append(loss.cpu().detach())
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            csi_scores.append(csi_score)
    avg = np.mean(losses)
    print(avg)
    return losses, accuracies, precisions, recalls, f1_scores, csi_scores 

def plot_dataset_losses_metrics(model, dataset, loss_f='BCE', train_val_test='testing', nonwater=0, water=1, water_threshold=0.5,  
                                model_type='min loss', spatial_temporal='spatial', device='cuda:0', save_img=False):
    '''
    Plot loss and metrics for each sample of the given dataset.

    Inputs:
           model = class, trained deep-learning model to be validated/tested  
           dataset = TensorDataset, dataset used for the model
           loss_f = str, key that specifies the function for computing the loss,
                    default: 'BCE', the other available option is 'BCE_Logits' but in this case make sure 
                    that the model output is not activated with a sigmoid function.
                    If other loss functions are set it raises an Exception. 
           train_val_test = str, specifies for what the images are used for.
                            default: 'testing'. Other available options: 'training', 'validation'
           nonwater = int, class value for non-water pixels.
                      default: 0, if classes are not scaled it should be set = 1
           water = int, class value for water pixels.
                   default: 1, if classes are not scaled it should be set = 2 
           water_threshold = float, threshold value for classifying water pixels
                             default: 0.5
           model_type = str, specifies which model is considered
                        default: 'min loss'. Other available option: 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu' 
           save_img = bool, sets whether the function is used for saving the image or not
                      default: False, image is not being saved 

    Output: 
           None, plots loss and metrics for each dataset sample
    '''
    
    losses, accuracies, precisions, recalls, f1_scores, csi_scores = total_losses_metrics_dataset(model, dataset, loss_f, nonwater, 
                                                                                                  water, water_threshold, overall=False, device=device) 

    avg_loss = np.mean(losses)
    samples = np.arange(1, len(losses) + 1, 1)

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # losses plot
    axs[0].scatter(samples, losses, s=70, zorder=2, color='navy', label='loss', edgecolor='k')
    axs[0].axhline(avg_loss, linestyle='--', zorder=1, lw=2.25, alpha=0.75, color='navy', label='average loss')
    axs[0].legend(loc='lower left', fontsize=13)
    axs[0].tick_params(axis='x', labelsize=15) 
    axs[0].tick_params(axis='y', labelsize=15)  
    axs[0].set_ylabel(f'Loss (-)', fontsize=18)

    if spatial_temporal=='spatial':
        yticks = np.arange(0.12, np.max(losses)+0.01, 0.01)
    else:
        yticks = np.arange(0.05, 0.2501, 0.025)  
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels([f'{tick:.2f}' for tick in yticks], fontsize=15)
    
    # metrics plot
    markers = ['P', 'X', '^', 's', 'D']
    lines = [(5, (10, 5)), '-.', '-', (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1))]
    size=[70, 90, 60, 30, 70]
    colors = ['mediumblue', 'crimson', 'darkgoldenrod', 'black', 'seagreen']
    metrics = [accuracies, precisions, recalls, f1_scores, csi_scores]
    avg_metrics = [np.mean(metrics[i]) for i in range(len(metrics))]
    metrics_names = ['accuracy', 'precision', 'recall', 'F1-score', 'CSI']

    for i, metric in enumerate(metrics):
        axs[1].scatter(samples, metric, s=size[i], zorder=2, marker=markers[i], facecolor=colors[i], edgecolor='k', label=f'{metrics_names[i]}')
        axs[1].axhline(avg_metrics[i], color=colors[i], zorder=1, lw=2.5, ls=lines[i], alpha=0.75, label=f'average {metrics_names[i]}')

    axs[1].legend(loc='lower left', ncol=len(metrics), fontsize=13)
    axs[1].set_xticks(samples)
    axs[1].set_xlabel(f'{train_val_test} dataset sample', fontsize=16)
    axs[1].set_ylabel(f'Metrics (-)', fontsize=18)
    axs[1].set_ylim(np.min(csi_scores)-0.2, 1)

    axs[1].tick_params(axis='x', labelsize=15) 
    axs[1].tick_params(axis='y', labelsize=15)

    yticks2 = np.arange(0.3, 1.01, 0.1)  
    axs[1].set_yticklabels([f'{tick:.1f}' for tick in yticks2], fontsize=15)

    if save_img:
        plt.savefig(rf'images\report\4_results\loss_metrics_{train_val_test}_{model_type}_{spatial_temporal}.png', bbox_inches='tight', dpi=600)
        plt.show()
        plt.close(fig)  # close the figure to free memory
    else:
        plt.show()

    return None

def box_plots(model, dataset, loss_f='BCE', nonwater=0, water=1, water_threshold=0.5, train_val_test='testing', 
              model_type='min loss', spatial_temporal='spatial', device='cuda:0', save_img=False):
    '''
    Plot loss and metrics box-plots for a given dataset.

    Inputs:
           model = class, trained deep-learning model to be validated/tested 
           dataset = TensorDataset, dataset used for the model
           loss_f = str, key that specifies the function for computing the loss,
                    default: 'BCE', the other available option is 'BCE_Logits' but in this case make sure 
                    that the model output is not activated with a sigmoid function.
                    If other loss functions are set it raises an Exception. 
           nonwater = int, class value for non-water pixels.
                      default: 0, if classes are not scaled it should be set = 1
           water = int, class value for water pixels.
                   default: 1, if classes are not scaled it should be set = 2 
           water_threshold = float, threshold value for classifying water pixels
                             default: 0.5
           train_val_test = str, specifies for what the images are used for.
                            default: 'testing'. Other available options: 'training', 'validation'
           model_type = str, specifies which model is considered
                        default: 'min loss'. Other available option: 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu' 
           save_img = bool, sets whether the function is used for saving the image or not
                      default: False, image is not being saved 

    Output: 
           None, plots the loss and metrics boxplots for the given dataset
    '''
    # compute loss and metrics
    losses, accuracies, precisions, recalls, f1_scores, csi_scores = total_losses_metrics_dataset(model, dataset, loss_f=loss_f, nonwater=nonwater, 
                                                                                                  water=water, water_threshold=water_threshold, 
                                                                                                  overall=False, device=device)
    
    # collect metrics
    metrics = [accuracies, precisions, recalls, f1_scores, csi_scores]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'CSI']

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3.5])

    # left subplot: losses
    ax0 = plt.subplot(gs[0])
    ax0.boxplot(losses, patch_artist=True, 
                boxprops=dict(facecolor='navy', color='navy'),
                whiskerprops=dict(color='navy'),
                capprops=dict(color='navy'),
                medianprops=dict(color='white'),  
                flierprops=dict(marker='o', color='navy', alpha=0.8, markersize=8),
                whis=0.5) 
    
    ax0.set_title('Loss', fontsize=16)
    ax0.set_ylabel('Values (-)', fontsize=16)
    # ax0.set_yscale('log')
    ax0.tick_params(axis='both', labelsize=15)
    ax0.set_xticklabels(['BCE Loss'], rotation=60, ha='right', fontsize=14)

    # right subplot: metrics
    ax1 = plt.subplot(gs[1])
    colors = ['mediumblue', 'crimson', 'darkgoldenrod', 'black', 'seagreen']
    colors_line = ['white', 'black', 'black', 'white', 'black']
    markers = ['P', 'X', '^', 's', 'D']
    for i in range(len(metrics)):
        ax1.boxplot(metrics[i], positions=[i+1], 
                    widths=0.3,
                    patch_artist=True,  
                    boxprops=dict(facecolor=colors[i], color=colors[i]), 
                    whiskerprops=dict(color=colors[i]),
                    capprops=dict(color=colors[i]),
                    medianprops=dict(color=colors_line[i]),  
                    flierprops=dict(marker=markers[i], color=colors[i], alpha=0.7, markersize=8), 
                    whis=1.3)  
    
    ax1.set_title('Metrics', fontsize=16)
    ax1.set_xticklabels(metric_names, rotation=60, ha='right', fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    yticks2 = np.arange(0.5, 1.01, 0.1)  
    ax1.set_yticks(yticks2)
    ax1.set_yticklabels([f'{tick:.1f}' for tick in yticks2], fontsize=15)
    
    plt.tight_layout()

    if save_img:
        plt.savefig(rf'images\report\4_results\boxmetrics_{train_val_test}_{model_type}_{spatial_temporal}.png', bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close(fig)  # close the figure to free memory
    else:
        plt.show()

    return None

def erosion_deposition_distribution(model, dataset, nonwater=0, water=1, water_threshold=0.5, pixel_size=60, train_val_test='testing', 
                                    model_type='min loss', spatial_temporal='spatial', device='cuda:0', save_img=False):
    '''
    Plot distributions of total areas of erosion and deposition
    
    Inputs:
           model = class, trained deep-learning model to be validated/tested 
           dataset = TensorDataset, dataset used for the model
           nonwater = int, class value for non-water pixels.
                      default: 0, if classes are not scaled it should be set = 1
           water = int, class value for water pixels.
                   default: 1, if classes are not scaled it should be set = 2 
           water_threshold = float, threshold value for classifying water pixels
                             default: 0.5
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
           train_val_test = str, specifies for what the images are used for.
                            default: 'testing'. Other available options: 'training', 'validation'
           model_type = str, specifies which model is considered
                        default: 'min loss'. Other available option: 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu' 
           save_img = bool, sets whether the function is used for saving the image or not
                      default: False, image is not being saved 

    Output: 
           None, plots the distributions of predicted and real total areas of erosion (left) and deposition (right)
    '''
    
    pred_erosion, pred_deposition = [], []
    real_erosion, real_deposition = [], []
    
    for sample in range(len(dataset)):
        input = dataset[sample][0].unsqueeze(0).to(device)
        target = dataset[sample][1].cpu()

        pred = model(input).detach().cpu().squeeze(0).squeeze(0)
        binary_pred = (pred >= water_threshold).float()

        predicted_erosion_deposition = get_erosion_deposition(input[0][-1].cpu(), binary_pred, nonwater=nonwater, water=water, pixel_size=pixel_size)
        real_erosion_deposition = get_erosion_deposition(input[0][-1].cpu(), target, nonwater=nonwater, water=water, pixel_size=pixel_size)
        
        pred_erosion.append(predicted_erosion_deposition[0]), pred_deposition.append(predicted_erosion_deposition[1])
        real_erosion.append(real_erosion_deposition[0]), real_deposition.append(real_erosion_deposition[1])       
    
    # generate kde and pdf for predicted and real erosion
    kde_pred_erosion = gaussian_kde(pred_erosion)
    x_pred_erosion = np.linspace(min(pred_erosion), max(pred_erosion), 10000)
    pdf_pred_erosion = kde_pred_erosion(x_pred_erosion)
    
    kde_real_erosion = gaussian_kde(real_erosion)
    x_real_erosion = np.linspace(min(real_erosion), max(real_erosion), 10000)
    pdf_real_erosion = kde_real_erosion(x_real_erosion)
    
    # repeat for deposition
    kde_pred_deposition = gaussian_kde(pred_deposition, bw_method='scott')
    x_pred_deposition= np.linspace(min(pred_deposition), max(pred_deposition), 10000)
    pdf_pred_deposition = kde_pred_deposition(x_pred_deposition)

    kde_real_deposition = gaussian_kde(real_deposition, bw_method='scott')
    x_real_deposition = np.linspace(min(real_deposition), max(real_deposition), 10000)
    pdf_real_deposition = kde_real_deposition(x_real_deposition)
    
    fig, axs = plt.subplots(1,2, figsize=(14, 6)) 
    # erosion
    axs[0].plot(x_pred_erosion, pdf_pred_erosion, lw=3, color='navy', label='predicted distribution')
    axs[0].plot(x_real_erosion, pdf_real_erosion, lw=3, color='royalblue', label='real distribution')
    axs[0].hist(pred_erosion, bins=5, density=True, alpha=0.5, color='navy', edgecolor='navy', linewidth=2, label='predicted histogram')
    axs[0].hist(real_erosion, bins=6, density=True, alpha=0.5, color='royalblue', edgecolor='royalblue', linewidth=2, label='real histogram')
    axs[0].set_xlabel('Erosion area (km²)', fontsize=16)
    axs[0].set_ylabel('Density (-)', fontsize=16)
    axs[0].legend(fontsize=14)
    
    # deposition
    axs[1].plot(x_pred_deposition, pdf_pred_deposition, lw=3, color='mediumblue', label='predicted distribution')
    axs[1].plot(x_real_deposition, pdf_real_deposition, lw=3, color='dodgerblue', label='real distribution')
    axs[1].hist(pred_deposition, bins=6, density=True, alpha=0.5, color='mediumblue', edgecolor='mediumblue', linewidth=2, label='predicted histogram') 
    axs[1].hist(real_deposition, bins=5, density=True, alpha=0.5, color='dodgerblue', edgecolor='dodgerblue', linewidth=2,label='real histogram') 
    axs[1].set_xlabel('Deposition area (km²)', fontsize=16)
    axs[1].set_ylabel('Density (-)', fontsize=16)
    axs[1].legend(fontsize=14)

    for ax in axs:
        ax.tick_params(axis='both', labelsize=15)

    if save_img:
        plt.savefig(rf'images\report\4_results\erdep_{train_val_test}_{model_type}_{spatial_temporal}.png', bbox_inches='tight', dpi=600)
        plt.show()
        plt.close(fig)  # close the figure to free memory
    else:
        plt.show()

    return None

def correlation_metrics(model, dataset, loss_f='BCE', nonwater=0, water=1, water_threshold=0.5, train_val_test='testing', 
                        model_type='min loss', spatial_temporal='spatial', device='cuda:0', save_img=False):
    '''
    Plot correlation matrix of loss and metrics. 
    
    Inputs:
           model = class, trained deep-learning model to be validated/tested
           dataset = TensorDataset, dataset used for the model
           loss_f = str, key that specifies the function for computing the loss,
                    default: 'BCE', the other available option is 'BCE_Logits' but in this case make sure 
                    that the model output is not activated with a sigmoid function.
                    If other loss functions are set it raises an Exception. 
           nonwater = int, class value for non-water pixels.
                      default: 0, if classes are not scaled it should be set = 1
           water = int, class value for water pixels.
                   default: 1, if classes are not scaled it should be set = 2 
           water_threshold = float, threshold value for classifying water pixels
                             default: 0.5
           train_val_test = str, specifies for what the images are used for.
                            default: 'testing'. Other available options: 'training', 'validation'
           model_type = str, specifies which model is considered
                        default: 'min loss'. Other available option: 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu' 
           save_img = bool, sets whether the function is used for saving the image or not
                      default: False, image is not being saved 

    Output: 
           None, plots the correlation matrix of loss and metrics
    '''
    losses, accuracies, precisions, recalls, f1_scores, csi_scores = [], [], [], [], [], []
    
    for sample in range(len(dataset)):
        single_input = dataset[sample][0].unsqueeze(0).to(device)
        single_target = dataset[sample][1].to(device)
        
        prediction = get_predictions(model, single_input, device).squeeze(0)
              
        loss = choose_loss(prediction, single_target, loss_f).detach()
        accuracy, precision, recall, f1_score, csi_score = compute_metrics(prediction.detach(), single_target, nonwater, water, water_threshold) 
        
        losses.append(loss.item()), accuracies.append(accuracy), precisions.append(precision)
        recalls.append(recall), f1_scores.append(f1_score), csi_scores.append(csi_score)
        
    df = pd.DataFrame({
        'Loss': losses,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-score': f1_scores,
        'CSI': csi_scores
    })

    # create pairplot figure
    # by setting corner=True the above-diagonal plots are removed. 
    # however, that induces problems for other parts of the code (color choice and annotations)
    pairplot_fig = sns.pairplot(df, diag_kind='kde', height=1.75, plot_kws={'s': 75, 'color': 'navy', 'edgecolor': 'palegoldenrod'})

    # colors for diagonals plots
    colors = ['navy', 'mediumblue', 'crimson', 'darkgoldenrod', 'black', 'seagreen']

    kde_linewidth = 2.5
    # adjust color on kde diagonal plots for each metric
    for i, ax in enumerate(pairplot_fig.diag_axes):
        color = mcolors.to_rgba(colors[i % len(colors)], alpha=0.5) 
        for kde in ax.collections:
            kde.set_edgecolor(colors[i % len(colors)])  
            kde.set_facecolor(color)  
            kde.set_linewidth(kde_linewidth)

    # adjust scatter plot colors and markers on the off-diagonal
    for i, ax in enumerate(pairplot_fig.axes.flatten()):
        if ax is not None and i % len(df.columns) != i // len(df.columns):
            row = i // len(df.columns)
            col = i % len(df.columns)
            scatter_color = colors[row % len(colors)]  # change color based on row index
            
            # clear existing scatter plots if any
            ax.clear() 
            
            # plot new scatter plot
            if scatter_color != 'black' and scatter_color != 'navy':
                sns.scatterplot(x=df.iloc[:, col], y=df.iloc[:, row], ax=ax, color=scatter_color, edgecolor='black', s=50)
            else:
                sns.scatterplot(x=df.iloc[:, col], y=df.iloc[:, row], ax=ax, color=scatter_color, edgecolor='white', s=50)
            
            # # uncomment the following lines to include the pearson correlation factor 
            # Add correlation value annotation
            # corr_val = df.corr().iloc[row, col]
            # ax.annotate(f'{corr_val:.2f}', xy=(0.7, 0.4), xycoords='axes fraction',
            #             ha='left', va='center', fontsize=13, color='navy',
            #             bbox=dict(boxstyle="round,pad=0.3", edgecolor='navy', facecolor=(1, 1, 1, 0.7)))
    
    for ax in pairplot_fig.axes.flatten():
        if ax is not None:
            ax.tick_params(axis='both', labelsize=14)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))  
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))
            ax.set_xlabel(ax.get_xlabel(), fontsize=16)  
            ax.set_ylabel(ax.get_ylabel(), fontsize=16)

    if save_img:
        plt.savefig(rf'images\report\4_results\correlation_metrics_{train_val_test}_{model_type}_{spatial_temporal}.png', bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()
    else:
        plt.show()

def correlation_erdep(model, dataset, nonwater=0, water=1, water_threshold=0.5, pixel_size=60, train_val_test='testing', 
                      model_type='min loss', spatial_temporal='spatial', device='cuda:0', save_img=False):
    '''
    Plot correlation matrix of total areas of erosion and deposition.
    
    Inputs:
           model = class, trained deep-learning model to be validated/tested
           dataset = TensorDataset, dataset used for the model
           nonwater = int, class value for non-water pixels.
                      default: 0, if classes are not scaled it should be set = 1
           water = int, class value for water pixels.
                   default: 1, if classes are not scaled it should be set = 2 
           water_threshold = float, threshold value for classifying water pixels
                             default: 0.5
           pixel_size = int, image pixel resolution (m). Used for computing the erosion and deposition areas
                        default: 60, exported image resolution from Google Earth Engine
           train_val_test = str, specifies for what the images are used for.
                            default: 'testing'. Other available options: 'training', 'validation'
           model_type = str, specifies which model is considered
                        default: 'min loss'. Other available option: 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu' 
           save_img = bool, sets whether the function is used for saving the image or not
                      default: False, image is not being saved 

    Output: 
           None, plots the correlation matrix of the total areas of erosion and deposition
    '''
    pred_erosion, pred_deposition, real_erosion, real_deposition = [], [], [], []
    
    for sample in range(len(dataset)):
        single_input = dataset[sample][0].unsqueeze(0).to(device)
        single_target = dataset[sample][1].to(device)
        prediction = get_predictions(model, single_input, device).squeeze(0)
        binary_prediction = (prediction >= water_threshold).float()

        predicted_erosion_deposition = get_erosion_deposition(single_input[0][-1].to(device), binary_prediction.to(device), 
                                                              nonwater=nonwater, water=water, pixel_size=pixel_size)
        real_erosion_deposition = get_erosion_deposition(single_input[0][-1].to(device), single_target.to(device), 
                                                         nonwater=nonwater, water=water, pixel_size=pixel_size)
        
        pred_erosion.append(predicted_erosion_deposition[0]), pred_deposition.append(predicted_erosion_deposition[1])
        real_erosion.append(real_erosion_deposition[0]), real_deposition.append(real_erosion_deposition[1])
        
    df = pd.DataFrame({
        'Pred. erosion': pred_erosion,
        'Pred. deposition': pred_deposition,
        'Real erosion': real_erosion,
        'Real deposition': real_deposition
    })
    
    # create pairplot figure
    # by setting corner=True the above-diagonal plots are removed. 
    # however, that induces problems for other parts of the code (color choice and annotations)
    pairplot_fig = sns.pairplot(df, diag_kind='kde', height=2, plot_kws={'s': 50, 'color': 'navy', 'edgecolor': 'palegoldenrod'})
    
    # colors for diagonals plots
    colors = ['navy', 'mediumblue', 'royalblue', 'dodgerblue'] 
    
    kde_linewidth = 2.5
    # adjust color on kde diagonal plots for each value
    for i, ax in enumerate(pairplot_fig.diag_axes):
        color = mcolors.to_rgba(colors[i % len(colors)], alpha=0.5) 
        for kde in ax.collections:
            kde.set_edgecolor(colors[i % len(colors)])  
            kde.set_facecolor(color)  
            kde.set_linewidth(kde_linewidth)
       
    for i, ax in enumerate(pairplot_fig.axes.flatten()):
        if ax is not None and i % len(df.columns) != i // len(df.columns):
            row = i // len(df.columns)
            col = i % len(df.columns)
            scatter_color = colors[row % len(colors)]  # change color based on row index
            
            # clear existing scatter plots if any
            ax.clear()  

            if scatter_color != 'mediumblue' and scatter_color != 'navy':
                sns.scatterplot(x=df.iloc[:, col], y=df.iloc[:, row], ax=ax, color=scatter_color, edgecolor='black', s=50)
            else:
                sns.scatterplot(x=df.iloc[:, col], y=df.iloc[:, row], ax=ax, color=scatter_color, edgecolor='white', s=50)

    for ax in pairplot_fig.axes.flatten():
        if ax is not None:
            ax.tick_params(axis='both', labelsize=13.5)  
            ax.get_yaxis().set_label_coords(-0.3,0.5)
            
            # create ticks based on min and max values
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='upper'))  
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both')) 
            ax.set_xlabel(ax.get_xlabel(), fontsize=15)  
            ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    
    if save_img:
        plt.savefig(rf'images\report\4_results\correlation_erdep_{train_val_test}_{model_type}_{spatial_temporal}.png', bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()
    else:
        plt.show()

def metrics_thresholds(model, data_loader, loss_f='BCE', device='cuda:0', save_img=False):
    '''
    Plot metrics evolution with varying binary threshold.
    
    Inputs:
           model = class, trained deep-learning model to be validated/tested 
           dataloader = loader = torch.utils.data.DataLoader element, data loader that combines a dataset 
                        and a sampler to feed data to the model in batches
           loss_f = str, key that specifies the function for computing the loss,
                    default: 'BCE', the other available option is 'BCE_Logits' but in this case make sure 
                    that the model output is not activated with a sigmoid function.
                    If other loss functions are set it raises an Exception. 
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu' 
           save_img = bool, sets whether the function is used for saving the image or not
                      default: False, image is not being saved 

    Output: 
           None, plots the correlation matrix of the total areas of erosion and deposition
    '''

    accuracies, precisions, recalls, f1_scores, csi_scores = [], [], [], [], []
    thresholds = np.arange(0,1,0.05)

    for threshold in thresholds:
        _, accuracy, precision, recall, f1_score, csi_score = validation_unet(model, data_loader, device=device, 
                                                                              loss_f=loss_f, water_threshold=threshold)
        accuracies.append(accuracy), precisions.append(precision)
        recalls.append(recall), f1_scores.append(f1_score), csi_scores.append(csi_score)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    
    plt.figure(figsize=(10,6))
    
    plt.plot(thresholds, accuracies, color='mediumblue', linewidth=2.5, label='accuracy', ls=(5, (10, 5)))
    plt.plot(thresholds, precisions, color='crimson', linewidth=2.5, label='precision', ls='-.') 
    plt.plot(thresholds, recalls, color='darkgoldenrod', linewidth=2.5, label='recall')
    plt.plot(thresholds, f1_scores, color='black', linewidth=2.5, label='F1-score', ls=(0, (3, 1, 1, 1, 1, 1)))
    plt.plot(thresholds, csi_scores, color='seagreen', linewidth=2.5, label='CSI', ls=(0, (5, 1))) 

    plt.xlabel('Thresholds [-]', fontsize=14)
    plt.ylabel('Metrics [-]', fontsize=14)
    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.xticks(np.arange(0, 1.1, 0.10), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.10), fontsize=12)
    formatted_best_thr = f'{best_thr:.3f}'
    plt.annotate(f'Water threshold for\nmax F1-score: {formatted_best_thr}', xy=(0.505,0.75), fontsize=12, 
                 ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))
    plt.legend(ncol=5, bbox_to_anchor=(0.1, 0.05), loc='lower left')
    if save_img:        
        plt.savefig(rf'images\report\4_results\metrics_with_thresholds.png', 
                    bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()  # close the figure to free memory
    else:
        plt.show()
    return None
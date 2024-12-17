# This module stores the function used to save the model results from a specific notebook and to reproduce these in other notebooks

import os
import torch
import copy

import pandas as pd

def save_losses_metrics(train_losses, val_losses, metrics, spatial_temporal, model, month_dataset, init_hid_dim, 
                        kernel_size, pooling, learning_rate, step_size, gamma, batch_size, num_epochs, 
                        water_threshold, physics=False, alpha_er=None, alpha_dep=None, dir_output=r'model\losses_metrics'):
    '''
    Save training and validation losses and metrics in a .csv file. Could be used for a later visualisation of the losses and metrics evolution. 
    It is assumed that the model performs 4 downsamples.

    Inputs:
           train_losses = list or array, contains training losses
           val_losses = list or array, contains validation losses
           metrics = list of lists, each containing validation metrics (accuracy, precision, recall, f1_score, csi)
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal'
           model = class, trained deep-learning model to be validated/tested
           month_dataset = int, low-flow season month from which images are taken for the dataset.
                           Available options: 1, 2, 3, and 4.
           init_hid_dim = int, initial hidden dimension of the model after first convolution
           kernel_size = int, size of the convolutional kernels
           pooling = str, pooling hyperparameter
                     Available options: 'max', 'avg'.
           learning_rate = float, learning rate
           step_size = int, epoch step after which learning rate is multiplied by `gamma`
           gamma = float, multiplication factor for adapting the learning rate every `step_size` epochs 
           batch_size = int, number of samples per batch
           num_epochs = int, number of training epochs
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)
           physics = bool, sets whether physics-induced loss terms (total areas of erosion
                     and deposition) are included in the loss.
                     default: False, not included
           alpha_er = float, weight of the erosion loss term within the total loss.
                      default: 1e-2. Suggested range [1e-5, 1e-2] 
           alpha_dep = float, weight of the deposition loss term within the total loss.
                       default: 1e-3. Suggested range [1e-5, 1e-2]
           dir_output = str, general path where .csv file is stored
                        default: 'model\losses_metrics'
    
    Output: 
           None, saves a .csv file with training and validation losses and metrics
    '''
    model_name = model.__class__.__name__

    # assumes that 4 downsamples are performed (4dwns) 
    file_name = f'{model_name}_{spatial_temporal}_losses&metrics_month{month_dataset}_4dwns_{init_hid_dim}ihiddim_{kernel_size}ker_{pooling}pool_{learning_rate}ilr_'
    if (step_size and gamma) is not None:
        file_name = file_name + f'{step_size}step_{gamma}gamma_'
        file_name = file_name + f'{batch_size}batch_{num_epochs}epochs_{water_threshold}wthr'
    else:
        file_name = file_name + f'{batch_size}batch_{num_epochs}epochs_{water_threshold}wthr'

    if physics:
        if alpha_er == None or alpha_dep == None:
            raise Exception(f'Attention! you set `physics` = {physics} but `alpha_er`= {alpha_er} and `alpha_dep`= {alpha_dep}.\n\
You must set both `alpha_er` and `alpha_dep`.')
        else:
            file_name = file_name + f'_{alpha_er}aer_{alpha_dep}adep'
    
    save_path = os.path.join(dir_output, file_name + '.csv')
    losses_metrics = {'Training loss': train_losses, 'Validation loss': val_losses, 'Accuracy': metrics[0], 
                      'Precision': metrics[1], 'Recall': metrics[2], 'F1-score': metrics[3], 'CSI-score': metrics[4]}
    
    df = pd.DataFrame(losses_metrics)
    df.to_csv(save_path, header=True, index=False)
    print(save_path)
    return None

def save_model_path(model, spatial_temporal, loss_recall, month_dataset, init_hid_dim, kernel_size, pooling, learning_rate, step_size, gamma, batch_size, num_epochs,
                    water_threshold, physics=False, alpha_er=None, alpha_dep=None, dir_output=r'model\models_trained'):
    '''
    Save the model .pth file path from the training notebook. It is then loaded in a different notebook for testing the model.
    It is assumed that the model performs 4 downsamples.

    Inputs:
           model = class, trained deep-learning model to be validated/tested
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal'
           loss_recall = str, specifies the model type, whether the one with minimum validatoin loss or the one with maximum validation recall.
                         Available options: 'min loss', 'max recall'
           month_dataset = int, low-flow season month from which images are taken for the dataset.
                           Available options: 1, 2, 3, and 4.
           init_hid_dim = int, initial hidden dimension of the model after first convolution
           kernel_size = int, size of the convolutional kernels
           pooling = str, pooling hyperparameter
                     Available options: 'max', 'avg'.
           learning_rate = float, learning rate
           step_size = int, epoch step after which learning rate is multiplied by `gamma`
           gamma = float, multiplication factor for adapting the learning rate every `step_size` epochs 
           batch_size = int, number of samples per batch
           num_epochs = int, number of training epochs
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)
           physics = bool, sets whether physics-induced loss terms (total areas of erosion
                     and deposition) are included in the loss.
                     default: False, not included
           alpha_er = float, weight of the erosion loss term within the total loss.
                      default: 1e-2. Suggested range [1e-5, 1e-2] 
           alpha_dep = float, weight of the deposition loss term within the total loss.
                       default: 1e-3. Suggested range [1e-5, 1e-2]
           dir_output = str, general path where .pth file is stored
                        default: 'model\models_trained'
    
    Output:
           None, saves the model .pth path
    '''
    best_model = copy.deepcopy(model)

    model_name = model.__class__.__name__
    file_name = f'{model_name}_b{loss_recall}_{spatial_temporal}_month{month_dataset}_4dwns_{init_hid_dim}ihiddim_{kernel_size}ker_{pooling}pool_{learning_rate}ilr_'
    if (step_size and gamma) is not None:
        file_name = file_name + f'{step_size}step_{gamma}gamma_'
        file_name = file_name + f'{batch_size}batch_{num_epochs}epochs_{water_threshold}wthr'
    else:
        file_name = file_name + f'{batch_size}batch_{num_epochs}epochs_{water_threshold}wthr'

    if physics:
        if alpha_er == None or alpha_dep == None:
            raise Exception(f'Attention! you set `physics` = {physics} but `alpha_er`= {alpha_er} and `alpha_dep`= {alpha_dep}.\n\
You must set both `alpha_er` and `alpha_dep`.')
        else:
            file_name = file_name + f'_{alpha_er}aer_{alpha_dep}adep'
    
    save_path = os.path.join(dir_output, file_name + '.pth')
    print(save_path)
    
    torch.save(best_model.state_dict(), save_path)
    return None

def load_model(model, save_path, device):
    '''
    Load the trained model in a different notebook for the testing phase.

    Inputs:
           model = class, trained deep-learning model to be validated/tested
           save_path = str, path where the model .pth file is stored
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu'
    '''
    # load model 
    return model.load_state_dict(torch.load(save_path, map_location = torch.device(device)))
# This module contains the functions used for computing the metrics and generating the classification curves of JamuNet model

import torch 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                            average_precision_score, balanced_accuracy_score)

def compute_metrics(pred, target, nonwater_value=0, water_value=1, water_threshold=0.5):
    '''
    Compute the metrics (accuracy, recall, precision, F1 score and CSI score) based on assigned water threshold.
    Values larger than the threshold are set equal to 1, whereas lower values are set to 0.
    Positive = water (1). Negative = non-water (0)

    Inputs:
           pred = 2D array or tensor, model predictions
           target =  2D array or tensor, real targets
           nonwater_value = int, represents pixel value of non-water class.
                         default: 0, based on the updated pixel classes. 
           water_value = int, represents pixel value of water class.
                         default: 1, based on the updated pixel classes. 
           water_threshold = float, threshold for binary classification.
                             default: 0.5, accepted range 0-1 (excluded)

    Output:
           accuracy, precision, recall, f1_score, csi = floats, classification metrics 
    '''
    # generate binary predictions
    binary_predictions = (pred >= water_threshold).float()
    
    # false negative is predicted non-water but actually water and the other way around for false positive
    tp = torch.sum((binary_predictions == water_value) & (target == water_value)).item()
    fp = torch.sum((binary_predictions == water_value) & (target == nonwater_value)).item()
    tn = torch.sum((binary_predictions == nonwater_value) & (target == nonwater_value)).item()
    fn = torch.sum((binary_predictions == nonwater_value) & (target == water_value)).item()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return accuracy, precision, recall, f1_score, csi 

def single_roc_curve(model, dataset, sample, train_val_test='testing', device='cuda:0', get_avg=False):
    '''
    Plot or return the FPR, TPR and ROC AUC of a single sample.

    Inputs:
           model = class, deep-learning model to be validated/tested
           dataset = TensorDataset, contains inputs and targets for the model
           sample = int, specifies the input-target combination
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda' (GPU), other availble option: 'cpu'
           get_avg = bool, sets whether the function plots ROC curve or only returns fpr, tpr and roc_auc 
                     default: False, plots the curve
    Output:
           None or fpr, tpr, roc_auc, depending on get_avg key
    '''
    # need internal import statement to avoid circular imports
    from model.train_eval import get_predictions

    single_input = dataset[sample][0]
    single_target = dataset[sample][1].cpu()
    
    target_flat = single_target.flatten()

    prediction = get_predictions(model, single_input.unsqueeze(0), device)
    prediction_flat = prediction.detach().cpu().flatten() # unsqueeze needed to match dimensions
    
    # compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(target_flat, prediction_flat)
    roc_auc = auc(fpr, tpr)

    if not get_avg:
        plt.figure()
        plt.plot(fpr, tpr, color='navy', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.fill_between(fpr, tpr,  color='palegoldenrod')
        plt.plot([0, 1], [0, 1], color='red', lw=2.5, linestyle='--', label=f'Random classifier = 0.5')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate [-]', fontsize=14)
        plt.ylabel('True Positive Rate [-]', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(f'Receiver Operating Characteristic curve\n for {train_val_test} sample {sample}', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()

    return fpr, tpr, roc_auc if get_avg else None

def get_total_roc_curve(model, dataset, train_val_test='testing', model_type='min loss', spatial_temporal='spatial', device='cuda:0', save_img=False):
    '''
    Plot the average ROC curve for a given dataset by averaging all samples. 
    To ensure that FPR and TPR have the same dimension across all samples, these metrics are resampled and interpolated to a fixed amount of elements.

    Inputs:
           model = class, deep-learning model to be validated/tested
           dataset = TensorDataset, contains inputs and targets for the model
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           model_type = str, specifies which model is considered
                        default: 'min loss'. Other available option: 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal' 
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda' (GPU), other availble option: 'cpu'
           save_img = bool, sets whether image is saved in the repository.
                      default: False 
    
    Output:
           None, plots the average ROC curve
    '''
    fprs = []
    tprs = []
    roc_aucs = []

    # loop through all dataset samples
    for sample in range(len(dataset)):
        fpr, tpr, roc_auc = single_roc_curve(model, dataset, sample, train_val_test, device, get_avg=True)
        
        # resample lists to have same dimension and be able to average them 
        fpr_resam = np.interp(np.linspace(0, 120000, 120000), np.arange(len(fpr)), fpr)
        tpr_resam = np.interp(np.linspace(0, 120000, 120000), np.arange(len(tpr)), tpr)
        
        fprs.append(fpr_resam), tprs.append(tpr_resam), roc_aucs.append(roc_auc)
    
    # average all arrays 
    avg_fpr = np.mean(fprs, axis=0)
    avg_tpr = np.mean(tprs, axis=0)
    avg_roc_auc = np.mean(roc_aucs)

    plt.figure()
    plt.plot(avg_fpr, avg_tpr, color='navy', lw=2.5, label=f'ROC curve (AUC = {avg_roc_auc:.3f})')
    plt.fill_between(avg_fpr, avg_tpr, color='palegoldenrod')
    plt.plot([0, 1], [0, 1], color='red', lw=2.5, linestyle='--', label=f'Random classifier = 0.5')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate [-]', fontsize=18)
    plt.ylabel('True Positive Rate [-]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title(f'Average Receiver Operating Characteristic curve\n for {train_val_test} dataset with {model_type} model', fontsize=16)
    plt.legend(bbox_to_anchor=(1.01,0.25), fontsize=16)
    plt.tight_layout()
    
    if save_img:
        plt.savefig(rf'images\report\4_results\roc_{train_val_test}_{model_type}_{spatial_temporal}.png', bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()  # close the figure to free memory
    else:
        plt.show()
    return None

def single_pr_curve(model, dataset, sample, train_val_test='testing', device='cuda:0', get_avg=False):
    '''
    Plot or return the Precision, Recall and Average Precision area of a single sample.

    Inputs:
           model = class, deep-learning model to be validated/tested
           dataset = TensorDataset, contains inputs and targets for the model
           sample = int, specifies the input-target combination
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda' (GPU), other availble option: 'cpu'
           get_avg = bool, sets whether the function plots PR curve or only returns precision, recall, 
                     average precision score, positive ratio, and best threshold 
                     default: False, plots the curve
    Output:
           None or precision, recall, ap, positive_ratio, and best_thr depending on get_avg key
    '''
    # need internal import statement to avoid circular imports
    from model.train_eval import get_predictions

    single_input = dataset[sample][0].to(device)
    single_target = dataset[sample][1]
    
    target_flat = single_target.flatten().to(device)

    prediction = get_predictions(model, single_input.unsqueeze(0), device)
    prediction_flat = prediction.detach().flatten() # unsqueeze needed to match dimensions
    
    # compute PR curve and average score
    precision, recall, thresholds = precision_recall_curve(target_flat.cpu(), prediction_flat.cpu())
    ap = average_precision_score(target_flat.cpu(), prediction_flat.cpu())
    
    # compute optimal threshold by maximising F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    
    # random classifier
    positive_ratio = torch.sum(target_flat)/len(target_flat)
    positive_ratio = positive_ratio.cpu()

    if not get_avg:
        plt.figure()
        plt.plot(recall, precision, color='navy', lw=2.5, label=f'PR curve (AUC = {ap:.3f})')
        plt.fill_between(recall, precision,  color='palegoldenrod')
        plt.axhline(positive_ratio, color='red', lw=2.5, linestyle='--', label=f'Random classifier (AP = {positive_ratio:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall [-]', fontsize=14)
        plt.ylabel('Precision [-]', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(f'Precision Recall curve\n for {train_val_test} sample {sample}', fontsize=16)
        plt.legend(loc="upper right", fontsize=12)
        plt.show()
 
    return precision, recall, ap, positive_ratio, best_thr if get_avg else None

def get_total_pr_curve(model, dataset, train_val_test='testing', model_type= 'min loss', spatial_temporal='spatial', device='cuda:0', save_img=False):
    '''
    Plot the average PR curve for a given dataset by averaging all samples. 
    To ensure that Precision and Recall have the same dimension across all samples, these metrics are resampled and interpolated to a fixed amount of elements.

    Inputs:
           model = class, deep-learning model to be validated/tested
           dataset = TensorDataset, contains inputs and targets for the model
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           model_type = str, specifies which model is considered
                        default: 'min loss'. Other available option: 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal' 
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda' (GPU), other availble option: 'cpu'
           save_img = bool, sets whether image is saved in the repository.
                      default: False 
    
    Output:
           None, plots the average PR curve
    '''
    precisions = []
    recalls = []
    aps = []
    positive_ratios = []
    best_thrs = []
    
    # loop through all samples
    for sample in range(len(dataset)):
        precision, recall, ap, positive_ratio, best_thr = single_pr_curve(model, dataset, sample, train_val_test, device, get_avg=True)
        
        # resample lists to have same dimensions and be able to average them 
        prec_resam = np.interp(np.linspace(0, 1, 500000), np.linspace(0, 1, len(precision)), precision)
        rec_resam = np.interp(np.linspace(0, 1, 500000), np.linspace(0, 1, len(recall)), recall)
        
        precisions.append(prec_resam), recalls.append(rec_resam), aps.append(ap), positive_ratios.append(positive_ratio), best_thrs.append(best_thr)
    
    # average all arrays 
    avg_prec = np.mean(precisions, axis=0)
    avg_rec = np.mean(recalls, axis=0)
    avg_ap = np.mean(aps)
    avg_pos_ratio = np.mean(positive_ratios)
    avg_best_thr = np.mean(best_thr)

    plt.figure()
    plt.plot(avg_rec, avg_prec, color='navy', lw=2.5, label=f'PR curve (AUC = {avg_ap:.3f})')
    plt.fill_between(avg_rec, avg_prec,  color='palegoldenrod')
    plt.axhline(avg_pos_ratio, color='red', lw=2.5, linestyle='--', label=f'Random classifier (AP = {avg_pos_ratio:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall [-]', fontsize=18)
    plt.ylabel('Precision [-]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title(f'Average Precision Recall curve for {train_val_test} dataset\n' + 
    #           rf'with {model_type} model and $w_{{thr}}$ = {avg_best_thr:.3f}', fontsize=16)
    # # add note on optimal threshold
    # formatted_best_thr = f'{avg_best_thr:.3f}'
    # plt.annotate(f'Water threshold for\nmax F1-score: {formatted_best_thr}', xy=(0.505,0.75), fontsize=16, 
    #              ha='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))
    plt.legend(bbox_to_anchor=(0.9,0.145), fontsize=16)
    
    if save_img:
        plt.savefig(rf'images\report\4_results\pr_{train_val_test}_{model_type}_{spatial_temporal}.png', bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()  # close the figure to free memory
    else:
        plt.show()
    return None

def balanced_accuracy(model, dataset, sample, train_val_test='testing', device='cuda:0', get_avg=False):
    '''
    Plot or return the Balanced Accuracy and BA AUC of a single sample.

    Inputs:
           model = class, deep-learning model to be validated/tested
           dataset = TensorDataset, contains inputs and targets for the model
           sample = int, specifies the input-target combination
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda' (GPU), other availble option: 'cpu'
           get_avg = bool, sets whether the function plots BA curve or only returns balanced accuracy and auc
                     default: False, plots the curve
    Output:
           None or balanced accuracy and auc depending on get_avg key
    '''
    # need internal import statement to avoid circular imports
    from model.train_eval import get_predictions
    
    single_input = dataset[sample][0].to(device)
    single_target = dataset[sample][1]
    
    target_flat = single_target.flatten().to(device)

    prediction = get_predictions(model, single_input.unsqueeze(0), device)
    prediction_flat = prediction.detach().flatten() 

    # define thresholds
    thresholds = np.linspace(0, 1, 20)

    # calculate balanced accuracy for each threshold
    balanced_accuracies = []
    for threshold in thresholds:
        predictions_binary = (prediction_flat > threshold).to(torch.int)
        balanced_acc = balanced_accuracy_score(target_flat.cpu(), predictions_binary.cpu())
        balanced_accuracies.append(balanced_acc)

    auc_balanced_accuracy = np.trapz(balanced_accuracies, thresholds)
    
    max_acc = max(balanced_accuracies)
    thr_max = thresholds[np.argmax(balanced_accuracies)]

    if not get_avg:
        plt.figure()
        plt.plot(thresholds, balanced_accuracies, color='navy', lw=2.5, label=f'BA curve (area = {auc_balanced_accuracy:.3f})')
        plt.fill_between(thresholds, balanced_accuracies,  color='palegoldenrod')
        plt.plot(thr_max, max_acc, 'o', color='red', label='max balanced accuracy', 
                 markerfacecolor='none', markersize=10, markeredgewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Thresholds [-]', fontsize=14)
        plt.ylabel('Balanced accuracy [-]', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12) 
        plt.title(f'Balanced Accuracy curve\n for {train_val_test} sample {sample}', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()

    return balanced_accuracies, auc_balanced_accuracy if get_avg else None

def get_total_balanced_accuracy(model, dataset, train_val_test='testing', model_type= 'min loss', spatial_temporal='spatial', device='cuda:0', save_img=False):
    '''
    Plot the average BA curve for a given dataset by averaging all samples. 
    To ensure that Precision and Recall have the same dimension across all samples, these metrics are resampled and interpolated to a fixed amount of elements.

    Inputs:
           model = class, deep-learning model to be validated/tested
           dataset = TensorDataset, contains inputs and targets for the model
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           model_type = str, specifies which model is considered
                        default: 'min loss'. Other available option: 'max recall'
           spatial_temporal = str, specifies if model is trained with spatial or temporal dataset
                              default: 'spatial'. Other available option: 'temporal' 
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda' (GPU), other availble option: 'cpu'
           save_img = bool, sets whether image is saved in the repository.
                      default: False 
    
    Output:
           None, plots the average BA curve
    '''
    b_accuracies, aucs = [], []
    
    # loop through all samples
    for sample in range(len(dataset)):
        b_accuracy, auc = balanced_accuracy(model, dataset, sample, train_val_test, device=device, get_avg=True)
        b_accuracies.append(b_accuracy), aucs.append(auc)
    
    # define thresholds - same dimensions of threshold array for single samples
    thresholds = np.linspace(0, 1, 20)
    
    # average all arrays 
    avg_bacc = np.mean(b_accuracies, axis=0)
    avg_auc = np.mean(aucs)

    plt.figure()
    plt.plot(thresholds, avg_bacc, color='navy', lw=2.5, label=f'Average BA curve (area = {avg_auc:.3f})')
    plt.fill_between(thresholds, avg_bacc, color='palegoldenrod')
    # plt.plot(thr_max, max_acc, 'o', color='red', label='max balanced accuracy', markerfacecolor='none', markersize=10, markeredgewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall [-]', fontsize=14)
    plt.ylabel('Precision [-]', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(f'Average Balanced Accuracy curve\n for {train_val_test} dataset', fontsize=16)
    plt.legend(loc="center left", fontsize=12)
    
    if save_img:
        plt.savefig(rf'images\report\4_results\ba_{train_val_test}_{model_type}_{spatial_temporal}.png', bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()  # close the figure to free memory
    else:
        plt.show()
    return None
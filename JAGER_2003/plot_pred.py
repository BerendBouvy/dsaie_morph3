from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def plot_pred(file):
    grey_cmap = ListedColormap(['palegoldenrod', 'navy'])

    plt.subplot(1,3,1)
    year = 2021
    area = "average_testing_r1"
    path = f"data/satellite/averages/{area}/average_{year}_testing_r1.csv"
    path2 = f"data/satellite/averages/{area}/average_{year-1}_testing_r1.csv"
    path3 = f"data/prediction.npy"
    im_cnn = np.load(path3)[0]
    prediction = pd.read_csv(file)
    image = np.genfromtxt(path, delimiter=',')
    image2 = np.genfromtxt(path2, delimiter=',')
    # Create a color map where 1s are blue and 0s are light brown/orange
    color_map = np.zeros((image.shape[0], image.shape[1], 3))
    color_map[image2 == 1] = [0, 0, 1]  # Blue
    color_map[image2 == 0] = [.9, .9, .9] 
    image_rgb = color_map
    # image_rgb = np.stack((image2,)*3, axis=-1)
    pred_map = np.zeros((image.shape[0], image.shape[1], 3))

    x_coords = prediction['index_x'].values
    y_coords = prediction['index_y'].values
    preds = np.round(prediction['Predictions'])
    targets = prediction['Targets'].astype(int).values
    # Define the zoomed-in region (e.g., center 100x100 pixels)
    center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
    zoom_sizey = 200
    zoom_sizex = zoom_sizey*2
    x_start, x_end = center_x - zoom_sizex // 2, center_x + zoom_sizex // 2
    y_start, y_end = center_y - zoom_sizey // 2, center_y + zoom_sizey // 2

    # Update pred_map for the zoomed-in region
    pred_map[x_coords, y_coords] = np.array([[0.56, 0.93, 0.56] if target == 0 and pred == 0 else  # lightgreen
                                             [0.39, 0.58, 0.93] if target == 1 and pred == 1 else  # cornflowerblue
                                             [1, 0.55, 0.41] if target == 0 and pred == 1 else     # salmon
                                             [1, 0.65, 0.0] for pred, target in zip(preds, targets)])  # orange
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='True Negative', markerfacecolor='lightgreen', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='True Positive', markerfacecolor='cornflowerblue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='False Positive', markerfacecolor='salmon', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='False Negative', markerfacecolor='orange', markersize=10)
    ]
    pred_map[np.all(pred_map == [0, 0, 0], axis=-1)] = [0.93, 0.91, 0.67]  # palegoldenrod

    plt.legend(handles=legend_elements, loc='upper right')

    # Plot the zoomed-in region
    plt.imshow(pred_map[x_start:x_end, y_start:y_end])
    plt.title(f'Prediction ANN (Zoomed In)')
    plt.axis('off')

    plt.subplot(1,3,2)
    # cnn plot
    pred_map = np.zeros((image.shape[0], image.shape[1], 3))
    pred_map[x_coords, y_coords] = np.array([[0.56, 0.93, 0.56] if target == 0 and pred == 0 else  # lightgreen
                                             [0.39, 0.58, 0.93] if target == 1 and pred == 1 else  # cornflowerblue
                                             [1, 0.55, 0.41] if target == 0 and pred == 1 else     # salmon
                                             [1, 0.65, 0.0] for pred, target in zip(im_cnn[x_coords, y_coords].flatten(), targets)])  # orange
    pred_map[np.all(pred_map == [0, 0, 0], axis=-1)] = [0.93, 0.91, 0.67]  # palegoldenrod

    # Plot the zoomed-in region for CNN
    plt.imshow(pred_map[x_start:x_end, y_start:y_end])
    plt.title(f'Prediction CNN (Zoomed In)')
    plt.axis('off')
    # plt.imshow(pred_map)
    # plt.title(f'Prediction CNN')
    # plt.axis('off')
    plt.legend(handles=legend_elements, loc='upper right')    
    
    plt.subplot(1,3,3)
    # plot the difference
    diff = preds - im_cnn[x_coords, y_coords].flatten()
    diff_map = np.zeros_like(image)
    diff_map[x_coords, y_coords] = diff

    plt.imshow(np.abs(diff_map), cmap=grey_cmap)
    # Add a red square to indicate the zoomed-in region
    rect = plt.Rectangle((y_start, x_start), zoom_sizey, zoom_sizex, linewidth=4, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.title(f'Prediction Difference')
    plt.show()
    
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    print("Confusion Matrix: ")
    csi = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0])
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Critical Success Index (CSI): {csi:.3f}")
    
    im2_pred = image2.copy()
    im2_pred[x_coords, y_coords] = preds
    accuracy = accuracy_score(image.flatten(), im2_pred.flatten())
    precision = precision_score(image.flatten(), im2_pred.flatten())
    recall = recall_score(image.flatten(), im2_pred.flatten())
    f1 = f1_score(image.flatten(), im2_pred.flatten())
    # Calculate confusion matrix
    cm = confusion_matrix(image.flatten(), im2_pred.flatten())
    print("Confusion Matrix:")
    print(cm)
    # Calculate Critical Success Index (CSI)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    csi = TP / (TP + FP + FN)
    print(f"Critical Success Index (CSI): {csi:.3f}")
    

    print(f"Updated Accuracy: {accuracy:.3f}")
    print(f"Updated Precision: {precision:.3f}")
    print(f"Updated Recall: {recall:.3f}")
    print(f"Updated F1 Score: {f1:.3f}")
    
    plt.subplot(1,2,1)
    plt.imshow(im2_pred, cmap=grey_cmap)
    plt.title(f'Prediction ANN')
    plt.axis('off')
    plt.subplot(1,2,2, )
    plt.imshow(im_cnn, cmap=grey_cmap)
    plt.title(f'Prediction CNN')
    plt.axis('off')
    plt.show()
    
    # cnn scores
    accuracy = accuracy_score(image.flatten(), im_cnn.flatten())
    precision = precision_score(image.flatten(), im_cnn.flatten())
    recall = recall_score(image.flatten(), im_cnn.flatten())
    f1 = f1_score(image.flatten(), im_cnn.flatten())
    print("CNN Scores:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(image.flatten(), im_cnn.flatten())
    print("Confusion Matrix:")
    print(cm)
    # Calculate Critical Success Index (CSI)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    csi = TP / (TP + FP + FN)
    print(f"Critical Success Index (CSI): {csi:.3f}")
    
    # only select the pixels that are used in ann
    im_cnn_small = im_cnn.copy()
    im_cnn_small = im_cnn_small[x_coords, y_coords]
    accuracy = accuracy_score(targets, im_cnn_small)
    precision = precision_score(targets, im_cnn_small)
    recall = recall_score(targets, im_cnn_small)
    f1 = f1_score(targets, im_cnn_small)
    print("CNN Scores:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, im_cnn_small)
    print("Confusion Matrix:")
    print(cm)
    # Calculate Critical Success Index (CSI)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    csi = TP / (TP + FP + FN)
    print(f"Critical Success Index (CSI): {csi:.3f}")
    
    #naive model of no change
    im_naive = image2.copy()
    accuracy = accuracy_score(image.flatten(), im_naive.flatten())
    precision = precision_score(image.flatten(), im_naive.flatten())
    recall = recall_score(image.flatten(), im_naive.flatten())
    f1 = f1_score(image.flatten(), im_naive.flatten())
    print("Naive Scores:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(image.flatten(), im_naive.flatten())
    print("Confusion Matrix:")
    print(cm)
    # Calculate Critical Success Index (CSI)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    csi = TP / (TP + FP + FN)
    print(f"Critical Success Index (CSI): {csi:.3f}")
    # Plot ANN prediction with correct and incorrect classifications (Zoomed Out)
    plt.subplot(1, 2, 1)
    ann_map = np.zeros((image.shape[0], image.shape[1], 3))
    ann_map[x_coords, y_coords] = np.array([[0.56, 0.93, 0.56] if pred == 0 and target == 0 else  # lightgreen for true negative
                                            [0.39, 0.58, 0.93] if pred == 1 and target == 1 else  # cornflowerblue for true positive
                                            [1.0, 0.55, 0.41] if pred == 1 and target == 0 else  # salmon for false positive
                                            [1.0, 0.65, 0.0] for pred, target in zip(preds, targets)])  # orange for false negative
    ann_map[np.all(ann_map == [0, 0, 0], axis=-1)] = [0.93, 0.91, 0.67]  # palegoldenrod background
    plt.imshow(ann_map)
    plt.title('ANN Prediction (Zoomed Out)')
    plt.axis('off')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='True Negative', markerfacecolor='lightgreen', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='True Positive', markerfacecolor='cornflowerblue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='False Positive', markerfacecolor='salmon', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='False Negative', markerfacecolor='orange', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Plot Naive prediction with correct and incorrect classifications (Zoomed Out)
    plt.subplot(1, 2, 2)
    naive_map = np.zeros((image.shape[0], image.shape[1], 3))
    naive_map[x_coords, y_coords] = np.array([[0.56, 0.93, 0.56] if naive == 0 and target == 0 else  # lightgreen for true negative
                                              [0.39, 0.58, 0.93] if naive == 1 and target == 1 else  # cornflowerblue for true positive
                                              [1.0, 0.55, 0.41] if naive == 1 and target == 0 else  # salmon for false positive
                                              [1.0, 0.65, 0.0] for naive, target in zip(im_naive[x_coords, y_coords], targets)])  # orange for false negative
    naive_map[np.all(naive_map == [0, 0, 0], axis=-1)] = [0.93, 0.91, 0.67]  # palegoldenrod background
    plt.imshow(naive_map)
    plt.title('Naive Prediction (Zoomed Out)')
    plt.axis('off')

    plt.legend(handles=legend_elements, loc='upper right')

    plt.show()
    
if __name__ == "__main__":
    plot_pred("models/predictions_1_5_0.001.csv")
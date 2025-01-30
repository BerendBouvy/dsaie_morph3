import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def plot_pred(file):
    year = 2021
    area = "average_testing_r1"
    path = f"data/satellite/averages/{area}/average_{year}_testing_r1.csv"
    path2 = f"data/satellite/averages/{area}/average_{year-1}_testing_r1.csv"
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
    # pred_map[prediction['index_x'], prediction['index_y']][0] = [float(prediction['Predictions'][i][1:-1]) for i in range(len(prediction['Predictions']))]

    x_coords = prediction['index_x'].values
    y_coords = prediction['index_y'].values
    preds = np.round(prediction['Predictions'])
    targets = prediction['Targets'].astype(int).values
    pred_map[x_coords, y_coords] = np.array([[0, 1, 0] if target == 0 and pred == 0 else
                                             [0, 0, 1] if target == 1 and pred == 1 else
                                             [1, 0, 0] if target == 0 and pred == 1 else
                                             [1, 1, 0] for pred, target in zip(preds, targets)])
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='True Negative', markerfacecolor='g', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='True Positive', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='False Positive', markerfacecolor='r', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='False Negative', markerfacecolor='y', markersize=10)
    ]

    plt.legend(handles=legend_elements, loc='upper right')
    

    
    
    plt.imshow(pred_map, alpha=0.5)
    # plt.imshow(image_rgb, alpha=0.5)
    plt.title(f'Image {year} - {area}')
    plt.axis('off')
    plt.show()

    # plt.scatter(prediction['x'], prediction['y'], c='red', s=10)
    plt.show()
    
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    print("Confusion Matrix: ")
    csi = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0])
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Critical Success Index (CSI): {csi:.2f}")
    
    # naive model of no change
    # naive_preds = np.zeros_like(targets)
    # accuracy = accuracy_score(targets, naive_preds)
    # precision = precision_score(targets, naive_preds)
    # recall = recall_score(targets, naive_preds)
    # f1 = f1_score(targets, naive_preds)
    
    # print(f"Naive Accuracy: {accuracy:.2f}")
    # print(f"Naive Precision: {precision:.2f}")
    # print(f"Naive Recall: {recall:.2f}")
    # print(f"Naive F1 Score: {f1:.2f}")
    
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
    print(f"Critical Success Index (CSI): {csi:.2f}")
    

    print(f"Updated Accuracy: {accuracy:.2f}")
    print(f"Updated Precision: {precision:.2f}")
    print(f"Updated Recall: {recall:.2f}")
    print(f"Updated F1 Score: {f1:.2f}")
    
if __name__ == "__main__":
    plot_pred("models/predictions_1_5_0.001.csv")
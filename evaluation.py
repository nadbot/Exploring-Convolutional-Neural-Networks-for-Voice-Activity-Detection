from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow
from keras.utils import vis_utils
from sklearn.metrics import confusion_matrix


def predict(model: tensorflow.keras.models, test_data: np.ndarray, softmax: bool = False) -> np.ndarray:
    """
    Predict the results of the model for the given test_data
    Args:
        model: Model that is used to predict the results
        test_data: Data for which the labels are predicted
        softmax: Boolean whether the model returns the softmax predictions or argmax
    Returns:
        predictions: np array containing the predictions
    """
    predictions = model.predict(test_data)
    if not softmax:
        # softmax means that the percentages for 0 and 1 are returned
        predictions = np.argmax(predictions, axis=1)
    return predictions


def average_of_neighbors(predicted_labels: np.ndarray, neighbors: int) -> np.ndarray:
    """
    Postprocessing method used in the original paper.
    Args:
        predicted_labels: Argmax labels of the predicted labels
        neighbors: int which is the number of neighbors of which the average is taken. Is split equally on both sides.
                   Default value for this in the paper is 5, which means 2 on each side.
    Returns:
        predicted_post_processing: Updated predicted labels
    """
    neighbors_site = int(neighbors/2)  # get the neighbors on each site, if uneven number round to lower value
    print(f"Postprocessing using {neighbors_site} neighbors")
    predicted_post_processing = []
    for pred_idx in range(len(predicted_labels)):
        start = max(pred_idx - neighbors_site, 0)
        end = min(len(predicted_labels), pred_idx + neighbors_site + 1)  # +1 as last element is not included in slice
        avg = np.average(predicted_labels[start:end])
        prediction = int(np.rint(avg))  # round to either 0 or 1
        predicted_post_processing.append(prediction)
    predicted_post_processing = np.array(predicted_post_processing)
    return predicted_post_processing


def plot_confusion_matrix(predicted_labels: np.ndarray, y_test: np.ndarray,
                          percentage: bool = False) -> confusion_matrix:
    """
    Function to plot a confusion matrix. If percentage is set, the values are given as percentages
    instead of total numbers.
    Args:
        predicted_labels: List of predicted labels
        y_test: Array containing the true labels
        percentage: If True, prints confusion matrix with percentages, otherwise with whole numbers
    Returns:
        cf_matrix: Confusion matrix
    """
    correct_labels = np.argmax(y_test, axis=1)
    # predictions_reshaped = predictions.reshape(-1, 2)
    # correct_reshaped = y_test.reshape(-1, 2)

    cf_matrix = confusion_matrix(correct_labels, predicted_labels)
    if percentage:
        conf_matrix = np.array(cf_matrix)
        cf_matrix = conf_matrix / (conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1])

    print(cf_matrix)
    plt.figure(figsize=(16, 9))
    sns.heatmap(cf_matrix, annot=True, xticklabels=['no speech', 'speech'],
                yticklabels=['no speech', 'speech'], fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return cf_matrix


def calculate_hter(cf_matrix: confusion_matrix) -> Tuple[float, float, float]:
    """
    Calculates HTER from confusion matrix
    Args:
        cf_matrix: Confusion matrix of predicted and true labels
    Returns:
        hter: Half-Total Error Rate
        mr: Miss Rate
        far: False Alarm Rate
    """
    mr = cf_matrix[1][0]/(sum(cf_matrix[1]))
    far = cf_matrix[0][1]/(sum(cf_matrix[0]))
    hter = (mr+far)/2*100
    return hter, mr, far


def visualize_model(model: tensorflow.keras.models) -> None:
    """
    Function to visualize the model using vis_utils from keras.
    Args:
        model: Model that will be visualized
    """
    vis_utils.plot_model(model, rankdir='TB',
                         to_file='test.png',
                         show_shapes=True,
                         show_layer_names=True)

from sklearn.metrics import confusion_matrix

def get_metrics(y_true, y_pred) -> tuple[float, float, float, float]:
    """Creates a confusion matrix and calculates the accuracy, precision, recall and f1 score

    Args:
        y_true (list): 1D list of true labels
        y_pred (list): 1D list of predicted labels

    Returns:
        tuple[float, float, float, float]: A tuple containing the accuracy, precision, recall and f1 score
    """    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1

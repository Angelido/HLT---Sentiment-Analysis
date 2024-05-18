from sklearn import metrics
from sklearn.metrics import classification_report, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scikitplot as skplt

from matplotlib import pyplot as plt


def plot_c_matrix(test_label, test_pred, classifier_name):
    """
    Plots a confusion matrix for the given test labels and predictions.
    
    Args:
        test_label (array-like): True labels of the test data.
        test_pred (array-like): Predicted labels of the test data.
        classifier_name (str): Name of the classifier (used for the plot title).
    
    Returns:
        None
    """
    # Compute the confusion matrix and create a ConfusionMatrixDisplay object
    cm = confusion_matrix(test_label, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    disp.plot()
    plt.title('Confusion matrix of ' + classifier_name)
    # Display the plot
    plt.show()


def report_scores(test_label, test_pred, labels=None):
    """
    Prints the classification report for the given test labels and predictions.
    
    Args:
        test_label (array-like): True labels of the test data.
        test_pred (array-like): Predicted labels of the test data.
        labels (array-like, optional): List of label indices to include in the report.
    
    Returns:
        None
    """
    # Print the classification report using sklearn's classification_report function
    print(classification_report(test_label, test_pred, labels=labels))

    

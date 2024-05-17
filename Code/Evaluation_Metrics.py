from sklearn import metrics
from sklearn.metrics import classification_report, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt


def plot_c_matrix(test_label, test_pred, classifier_name):
    cm = confusion_matrix(test_label, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix of ' + classifier_name)
    plt.show() 
    
def report_scores(test_label, test_pred):
    print(classification_report(test_label, test_pred, zero_division="warn"))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from use_augument_data.hyper_tunnning import fault_types


def confu_matrix(y_test, y_pred):
    test_confu_matrix = confusion_matrix(y_test, y_pred)
    import numpy as np
    row_sum = np.array(np.sum(test_confu_matrix, axis=1)).T
    test_confu_matrix = test_confu_matrix / row_sum

    sns.heatmap(test_confu_matrix, annot=True,
                xticklabels=fault_types, yticklabels=fault_types, cmap="Greens", cbar=False, fmt='.4f')
    plt.ylabel('True')
    plt.show()


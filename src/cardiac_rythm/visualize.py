import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix


def visualize_test_result(correct: NDArray, prediction: NDArray, filename: str) -> None:
    """Plots a confusion matrix for the test result."""
    conf_matrix = confusion_matrix(correct, prediction)
    norm = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]

    f, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(norm, annot=True, linewidths=0.01, cmap="Oranges", linecolor="gray")
    ax.xaxis.set_ticklabels(["AS", "PEA", "PR", "VF", "VT"])
    ax.yaxis.set_ticklabels(["AS", "PEA", "PR", "VF", "VT"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(filename)

    accuracy = accuracy_score(correct, prediction)
    print(f"Test Accuracy: {accuracy}")


def visualize_history(history: dict, filename: str) -> None:
    """Plots everytning in a dictionary using plot(val, label=key)"""
    n = len(history.keys())
    cols = int(np.ceil(np.sqrt(n)))
    f, ax = plt.subplots(cols, cols)
    for i, (key, val) in enumerate(history.items()):
        row = int(np.floor(i / cols))
        col = i % cols
        ax[row, col].plot(val, label=key)
        ax[row, col].legend(loc="upper left")
    plt.savefig(filename)

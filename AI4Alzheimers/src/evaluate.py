import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_confusion_matrix(all_labels, all_preds, classes, acc):
    # A. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Acc: {acc:.2f}%)')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

def print_classification_report(all_labels, all_preds, classes):
    # B. Classification Report
    print("\n📑 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=[str(c) for c in classes]))

def plot_roc_curve(all_labels, all_probs, classes):
    # C. ROC Curve & AUC
    print("📈 Generating ROC Curves...")
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]
    all_probs = np.array(all_probs)

    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {classes[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

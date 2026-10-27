"""
eval_utils.py
Shared evaluation helpers: metrics report, confusion matrix, training curves.
Used by both part_a_traditional_cnn.py and part_b_customized_cnn.py so the
reporting format is identical (needed for the Task 1 comparison deliverable).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data_utils import CLASS_NAMES


def count_params(model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


def evaluate_model(model, x_test, y_test, model_name: str, out_dir: str = "."):
    """
    Runs predictions on the test set and prints/saves everything the
    deliverable table asks for: accuracy, precision, recall, F1, confusion
    matrix (saved as a PNG).
    """
    y_true = np.argmax(y_test, axis=1)
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    print(f"\n===== {model_name}: Test Set Results =====")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Precision(macro): {precision:.4f}")
    print(f"Recall(macro)   : {recall:.4f}")
    print(f"F1-score(macro) : {f1:.4f}")
    print("\nFull classification report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, model_name, out_dir)

    return {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def _plot_confusion_matrix(cm, model_name: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fname = f"{out_dir}/confusion_matrix_{model_name.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {fname}")


def plot_training_curves(history, model_name: str, out_dir: str = "."):
    """Saves accuracy and loss curves (training vs validation) as a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title(f"{model_name}: Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title(f"{model_name}: Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fname = f"{out_dir}/training_curves_{model_name.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {fname}")


class Timer:
    """Simple context manager to measure training time."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

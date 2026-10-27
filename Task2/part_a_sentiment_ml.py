"""
part_a_sentiment_ml.py
Task 2 - Part A: Machine Learning for Text Classification (TF-IDF + LR + SVM)

Run:
    python part_a_sentiment_ml.py

Outputs (in the Task2 folder):
    - part_a_results.json
    - confusion_matrix_LogisticRegression.png
    - confusion_matrix_SVM.png
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

from sentiment_data_utils import load_sentiment_data, class_distribution

CLASS_NAMES = ["negative", "positive"]


def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def evaluate(name, model, x_test_vec, y_test):
    y_pred = model.predict(x_test_vec)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    plot_confusion_matrix(cm, f"{name} - Confusion Matrix",
                           f"confusion_matrix_{name}.png")
    print(f"\n===== {name}: Test Set Results =====")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision(macro): {precision:.4f}")
    print(f"Recall(macro)   : {recall:.4f}")
    print(f"F1-score(macro) : {f1:.4f}")
    print(report)
    return {
        "accuracy": acc, "precision_macro": precision,
        "recall_macro": recall, "f1_macro": f1,
    }


def main():
    (x_train, y_train), (x_test, y_test) = load_sentiment_data()

    print("Class distribution (train):", class_distribution(y_train))
    print("Class distribution (test) :", class_distribution(y_test))
    print("(Roughly balanced -> accuracy is a fair metric here, "
          "but we still report F1 for robustness.)")

    # TF-IDF vectorization (fit on train only, to avoid leakage)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    print(f"\nTF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    # Logistic Regression
    t0 = time.time()
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(x_train_vec, y_train)
    lr_time = time.time() - t0
    lr_metrics = evaluate("LogisticRegression", log_reg, x_test_vec, y_test)
    lr_metrics["training_time_sec"] = lr_time

    # SVM
    t0 = time.time()
    svm = LinearSVC(random_state=42)
    svm.fit(x_train_vec, y_train)
    svm_time = time.time() - t0
    svm_metrics = evaluate("SVM", svm, x_test_vec, y_test)
    svm_metrics["training_time_sec"] = svm_time

    report = {
        "vocab_size": len(vectorizer.vocabulary_),
        "train_class_distribution": class_distribution(y_train),
        "test_class_distribution": class_distribution(y_test),
        "logistic_regression": lr_metrics,
        "svm": svm_metrics,
    }

    with open("part_a_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n===== SUMMARY (save this for your report) =====")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

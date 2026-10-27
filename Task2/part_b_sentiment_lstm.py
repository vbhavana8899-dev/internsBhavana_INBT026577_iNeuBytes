"""
part_b_sentiment_lstm.py
Task 2 - Part B: Deep Learning for Sentiment Analysis (LSTM)

Run this AFTER part_a_sentiment_ml.py (it reads part_a_results.json
for the final comparison table).

Run:
    python part_b_sentiment_lstm.py

Outputs (in the Task2 folder):
    - part_b_results.json
    - lstm_training_curves.png
    - confusion_matrix_LSTM.png
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

from sentiment_data_utils import load_sentiment_data

SEED = 42
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 64
EPOCHS = 15
BATCH_SIZE = 32
CLASS_NAMES = ["negative", "positive"]


def set_seed():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("LSTM - Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("LSTM - Loss"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_title("LSTM - Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
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


def main():
    set_seed()
    # SAME split as Part A (same seed, same function)
    (x_train_text, y_train), (x_test_text, y_test) = load_sentiment_data()
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Tokenize -> sequences -> pad
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train_text)
    x_train_seq = tokenizer.texts_to_sequences(x_train_text)
    x_test_seq = tokenizer.texts_to_sequences(x_test_text)
    x_train_pad = pad_sequences(x_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test_pad = pad_sequences(x_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    model = Sequential([
       Input(shape=(MAX_LEN,)),
       Embedding(VOCAB_SIZE, EMBEDDING_DIM),
       LSTM(64, dropout=0.3, recurrent_dropout=0.3),
       Dense(32, activation="relu"),
       Dropout(0.4),
       Dense(1, activation="sigmoid"),
       ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    total_params = model.count_params()

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    t0 = time.time()
    history = model.fit(
        x_train_pad, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )
    training_time = time.time() - t0

    plot_training_curves(history, "lstm_training_curves.png")

    y_pred_probs = model.predict(x_test_pad, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "confusion_matrix_LSTM.png")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print(f"\n===== LSTM: Test Set Results =====")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision(macro): {precision:.4f}")
    print(f"Recall(macro)   : {recall:.4f}")
    print(f"F1-score(macro) : {f1:.4f}")
    print(report)

    lstm_metrics = {
        "total_params": int(total_params),
        "training_time_sec": training_time,
        "epochs_run": len(history.history["accuracy"]),
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

    with open("part_b_results.json", "w", encoding="utf-8") as f:
        json.dump(lstm_metrics, f, indent=2)

    # Compare against Part A's best classical model
    try:
        with open("part_a_results.json", encoding="utf-8") as f:
            part_a = json.load(f)
        lr_acc = part_a["logistic_regression"]["accuracy"]
        svm_acc = part_a["svm"]["accuracy"]
        best_classical = max(lr_acc, svm_acc)
        best_name = "Logistic Regression" if lr_acc >= svm_acc else "SVM"

        print("\n===== COMPARISON: Logistic Regression vs SVM vs LSTM =====")
        print(f"{'Model':<22}{'Accuracy':>12}{'F1 (macro)':>14}")
        print(f"{'Logistic Regression':<22}{lr_acc:>12.4f}{part_a['logistic_regression']['f1_macro']:>14.4f}")
        print(f"{'SVM':<22}{svm_acc:>12.4f}{part_a['svm']['f1_macro']:>14.4f}")
        print(f"{'LSTM':<22}{acc:>12.4f}{f1:>14.4f}")

        if acc >= best_classical:
            print(f"\n✅ LSTM matched/exceeded the best classical model ({best_name}: {best_classical:.4f}).")
        else:
            gap = (best_classical - acc) * 100
            print(f"\n⚠️ LSTM did NOT beat the best classical model ({best_name}: {best_classical:.4f}), "
                  f"gap of {gap:.2f} points.")
            print("Likely reason: only 2000 training documents is a small dataset for deep learning; "
                  "LSTMs typically need much more data to outperform strong TF-IDF + linear baselines.")
    except FileNotFoundError:
        print("\npart_a_results.json not found — run part_a_sentiment_ml.py first for the comparison.")


if __name__ == "__main__":
    main()

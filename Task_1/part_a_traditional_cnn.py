"""
part_a_traditional_cnn.py
Task 1, Part A: Traditional CNN Model (clean baseline, no augmentation)

Run:
    python part_a_traditional_cnn.py
"""

import json
import io
import contextlib

from tensorflow.keras import layers, models, optimizers

from data_utils import load_data, set_global_seed
from eval_utils import evaluate_model, plot_training_curves, count_params, Timer

EPOCHS = 25          # fixed training budget — Part B must reuse this (or justify using more)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MODEL_NAME = "Traditional_CNN"


def build_traditional_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    AlexNet-style CNN adapted for 32x32 CIFAR-10 images.
    The original AlexNet assumes 227x227 inputs and much larger filter
    counts (96/256/384) — those don't fit 32x32 images without collapsing
    spatial dimensions to nothing, so this is a scaled-down adaptation:
    3 conv blocks with increasing filters (64 -> 128 -> 256), ReLU
    activations, and max pooling, followed by fully-connected layers.
    No batch norm / dropout / augmentation here — those are reserved for
    Part B so their individual effect can be measured against this baseline.
    """
    model = models.Sequential(name=MODEL_NAME)
    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def get_model_summary_str(model) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.summary()
    return buf.getvalue()


def main():
    set_global_seed()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    model = build_traditional_cnn()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    summary_str = get_model_summary_str(model)
    print(summary_str)
    total_params = count_params(model)
    print(f"Total trainable parameters: {total_params:,}")

    with Timer() as t:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
    training_time_sec = t.elapsed
    print(f"Training time: {training_time_sec:.1f} seconds")

    plot_training_curves(history, MODEL_NAME)
    metrics = evaluate_model(model, x_test, y_test, MODEL_NAME)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    report = {
        "model_name": MODEL_NAME,
        "total_params": total_params,
        "epochs": EPOCHS,
        "training_time_sec": training_time_sec,
        "final_train_accuracy": final_train_acc,
        "final_val_accuracy": final_val_acc,
        **metrics,
    }

    with open("part_a_results.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("part_a_model_summary.txt", "w",encoding="utf-8") as f:
        f.write(summary_str)

    model.save("part_a_traditional_cnn.keras")

    print("\n===== SUMMARY (save this for your report) =====")
    print(json.dumps(report, indent=2))

    if metrics["test_accuracy"] >= 0.70:
        print("\n✅ Success threshold met (>=70% test accuracy).")
    else:
        print("\n⚠️  Below the 70% threshold — consider more epochs or tuning LR.")


if __name__ == "__main__":
    main()

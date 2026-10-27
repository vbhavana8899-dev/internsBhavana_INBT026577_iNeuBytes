"""
part_b_customized_cnn.py
Task 1, Part B: Customized CNN Model — must measurably beat Part A.

Changes vs. Part A (each chosen to address a specific weakness of the
baseline, not just "more layers"):
  1. Batch Normalization after every conv layer
     -> stabilizes/speeds up training, lets us use a deeper network
        without vanishing/exploding activations.
  2. Dropout (0.25 in conv blocks, 0.5 before output)
     -> the baseline has no regularization at all and is prone to
        overfitting on only 45k training images; dropout directly
        targets that.
  3. An extra (4th) convolutional block (64->128->256->512 filters)
     -> gives the network more representational capacity, now that
        BatchNorm keeps it trainable.
  4. Data augmentation (random flip / rotation / zoom) applied as
     preprocessing layers inside the model
     -> baseline never sees augmented data and overfits faster;
        augmentation is the single biggest lever for CIFAR-10 CNNs.
  5. Learning-rate scheduling (ReduceLROnPlateau) instead of a fixed LR
     -> lets the optimizer take large steps early and fine-tune later.

Fairness controls (do not change these):
  - Same random seed and same train/val/test split as Part A (via data_utils).
  - Epochs: see EPOCHS below — if different from Part A, the reason is
    stated in the comment next to it, as required by the task rules.

Run:
    python part_b_customized_cnn.py
"""

import json
import io
import contextlib

from tensorflow.keras import layers, models, optimizers, callbacks

from data_utils import load_data, set_global_seed
from eval_utils import evaluate_model, plot_training_curves, count_params, Timer

# NOTE ON EPOCH BUDGET (per task rules, any deviation from Part A must be
# stated and justified): Part A used 25 epochs. Data augmentation makes the
# effective training distribution larger/harder, so the model needs more
# passes to converge under augmentation. Epochs are increased to 35 here for
# that reason — everything else (seed, split, architecture family, dataset)
# is held constant so the *test-accuracy* comparison is still fair; the
# report explicitly calls out this difference as required.
EPOCHS = 35
BATCH_SIZE = 64
INITIAL_LEARNING_RATE = 1e-3
MODEL_NAME = "Customized_CNN"


def build_customized_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential(name=MODEL_NAME)
    model.add(layers.Input(shape=input_shape))

    # Data augmentation (active only during training; identity at inference)
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.08))
    model.add(layers.RandomZoom(0.1))

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Block 4 (new vs. Part A)
    model.add(layers.Conv2D(512, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.3))

    # Classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
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

    model = build_customized_cnn()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    summary_str = get_model_summary_str(model)
    print(summary_str)
    total_params = count_params(model)
    print(f"Total trainable parameters: {total_params:,}")

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
    )

    with Timer() as t:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[lr_scheduler],
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

    with open("part_b_results.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("part_b_model_summary.txt", "w",encoding="utf-8") as f:
        f.write(summary_str)

    model.save("part_b_customized_cnn.keras")

    print("\n===== SUMMARY (save this for your report) =====")
    print(json.dumps(report, indent=2))

    # Compare against Part A if its results file exists
    try:
        with open("part_a_results.json") as f:
            part_a = json.load(f)
        gain = (metrics["test_accuracy"] - part_a["test_accuracy"]) * 100
        print(f"\nImprovement over Part A: {gain:+.2f} percentage points")
        if gain >= 3.0:
            print("✅ Beats Part A by the required >=3 point margin.")
        else:
            print("⚠️  Did not reach the +3 point margin — consider tuning further.")
    except FileNotFoundError:
        print("\n(Run part_a_traditional_cnn.py first to get an automatic comparison.)")


if __name__ == "__main__":
    main()

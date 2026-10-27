"""
Task 1: AlexNet-style CNN baseline, technique experiments, final customized CNN.
Dataset: EuroSAT RGB (27,000 images, 10 classes, 64x64) - satellite land-use classification.
Run: python task1_cnn.py   (results are cached per run in ./results; delete a json to re-run it)
"""
import os, json, time, random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- fixed budget (frozen for the whole task) ----------------
SEED = 42
EPOCHS = 20
BATCH = 64
LR = 1e-3
IMG = 64
HELPFUL_MARGIN = 0.003      # technique is "helpful" if val accuracy beats baseline by >= 0.3 pp
OUT = "results"
os.makedirs(OUT, exist_ok=True)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled:", e)


def set_seed():
    tf.keras.backend.clear_session()
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


# ---------------- data ----------------
def load_data():
    cache = os.path.join(OUT, "eurosat_cache.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], list(d["names"])
    ds, info = tfds.load("eurosat/rgb", split="train", with_info=True, as_supervised=True)
    X, y = [], []
    for img, lab in tfds.as_numpy(ds):
        X.append(img)
        y.append(lab)
    X, y = np.stack(X).astype("uint8"), np.array(y).astype("int64")
    names = info.features["label"].names
    np.savez_compressed(cache, X=X, y=y, names=np.array(names))
    return X, y, names


def get_split(y):
    path = os.path.join(OUT, f"split_seed{SEED}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["train"], d["val"], d["test"]
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
    np.savez(path, train=tr, val=va, test=te)   # frozen split
    return tr, va, te


def make_ds(X, y, train=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        ds = ds.shuffle(len(X), seed=SEED)
    ds = ds.batch(BATCH).map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b))  # normalize pixels
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------- model ----------------
def build_model(aug=False, bn=False, dropout=0.0, l2=0.0):
    reg = regularizers.l2(l2) if l2 > 0 else None
    inp = layers.Input((IMG, IMG, 3))
    x = inp
    if aug:  # active only during training
        x = layers.RandomFlip("horizontal_and_vertical", seed=SEED)(x)
        x = layers.RandomRotation(0.08, fill_mode="reflect", seed=SEED)(x)

    def conv(x, f, k):
        x = layers.Conv2D(f, k, padding="same", kernel_regularizer=reg, use_bias=not bn)(x)
        if bn:
            x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    x = conv(x, 64, 5);  x = layers.MaxPooling2D(2)(x)    # 64 -> 32
    x = conv(x, 128, 3); x = layers.MaxPooling2D(2)(x)    # 32 -> 16
    x = conv(x, 256, 3)
    x = conv(x, 256, 3); x = layers.MaxPooling2D(2)(x)    # 16 -> 8
    x = layers.Flatten()(x)
    for units in (512, 256):
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        if dropout > 0:
            x = layers.Dropout(dropout, seed=SEED)(x)
    out = layers.Dense(10, activation="softmax", kernel_regularizer=reg)(x)
    return models.Model(inp, out)


# ---------------- train / evaluate ----------------
def run(name, data, **cfg):
    path = os.path.join(OUT, f"{name}.json")
    if os.path.exists(path):
        print(f"[{name}] cached")
        return json.load(open(path))
    print(f"\n===== {name}: {cfg} =====")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    set_seed()
    model_cfg = {k: v for k, v in cfg.items() if k in ("aug", "bn", "dropout", "l2")}
    model = build_model(**model_cfg)
    steps = int(np.ceil(len(Xtr) / BATCH))
    lr = tf.keras.optimizers.schedules.CosineDecay(LR, EPOCHS * steps) if cfg.get("cosine") else LR
    model.compile(tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    lines = []
    model.summary(print_fn=lambda s: lines.append(s))

    t0 = time.time()
    hist = model.fit(make_ds(Xtr, ytr, True), validation_data=make_ds(Xva, yva),
                     epochs=EPOCHS, verbose=2)
    train_time = time.time() - t0

    # final-epoch weights are used for every model (no checkpoint picking) -> fair comparison
    train_acc = model.evaluate(make_ds(Xtr, ytr), verbose=0)[1]
    val_acc = model.evaluate(make_ds(Xva, yva), verbose=0)[1]
    test_acc = model.evaluate(make_ds(Xte, yte), verbose=0)[1]
    pred = model.predict(make_ds(Xte, yte), verbose=0).argmax(1)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    pc, rc, fc, _ = precision_recall_fscore_support(yte, pred, average=None, zero_division=0)
    cm = confusion_matrix(yte, pred)

    res = dict(name=name, cfg=cfg, params=int(model.count_params()), train_time_s=round(train_time, 1),
               epochs=EPOCHS, train_acc=float(train_acc), val_acc=float(val_acc), test_acc=float(test_acc),
               precision=float(p), recall=float(r), f1=float(f),
               per_class=dict(precision=pc.tolist(), recall=rc.tolist(), f1=fc.tolist()),
               cm=cm.tolist(), history={k: [float(v) for v in vals] for k, vals in hist.history.items()},
               summary="\n".join(lines))
    json.dump(res, open(path, "w"))
    return res


# ---------------- analysis helpers ----------------
def top_pairs(cm, k=3):
    n = len(cm)
    s = [(int(cm[i][j] + cm[j][i]), i, j) for i in range(n) for j in range(i + 1, n)]
    return sorted(s, reverse=True)[:k]


def plot_cm(cm, names, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right"); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_curves(res_list, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in res_list:
        axes[0].plot(r["history"]["accuracy"], label=r["name"] + " train")
        axes[0].plot(r["history"]["val_accuracy"], "--", label=r["name"] + " val")
        axes[1].plot(r["history"]["loss"], label=r["name"] + " train")
        axes[1].plot(r["history"]["val_loss"], "--", label=r["name"] + " val")
    axes[0].set_title("Accuracy"); axes[1].set_title("Loss")
    for a in axes:
        a.set_xlabel("epoch"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


HYP = {
    frozenset(["AnnualCrop", "PermanentCrop"]): "Both are farmland with similar green/brown tones and field textures at 10 m resolution; the row pattern that separates orchards/vineyards from annual crops is subtle at 64x64.",
    frozenset(["Pasture", "HerbaceousVegetation"]): "Both are low, continuous green vegetation with almost identical colour and texture; the difference is land management, not appearance.",
    frozenset(["Highway", "River"]): "Both are thin, elongated, curving structures crossing the image; at low resolution their shape and surrounding context look alike.",
    frozenset(["Industrial", "Residential"]): "Both are dense built-up areas with similar roof colours and small-scale texture; building size and spacing differ only slightly.",
    frozenset(["Forest", "HerbaceousVegetation"]): "Both are dark-to-medium green vegetation; canopy texture is the main cue and it is weak at this resolution.",
    frozenset(["SeaLake", "River"]): "Both are water bodies with similar spectral colour; shape/extent is the main distinguishing cue.",
    frozenset(["Forest", "Pasture"]): "Both are vegetated with similar green tones; tree-canopy texture is the only strong cue.",
    frozenset(["Highway", "Industrial"]): "Industrial areas contain large roads and paved surfaces that resemble highway segments.",
}


def hypothesis(a, b):
    return HYP.get(frozenset([a, b]), "Visually similar colour/texture at 64x64 resolution (verify by viewing misclassified samples).")


# ---------------- main ----------------
def main():
    X, y, names = load_data()
    tr, va, te = get_split(y)
    data = ((X[tr], y[tr]), (X[va], y[va]), (X[te], y[te]))
    counts = np.bincount(y, minlength=len(names)).tolist()

    base = run("baseline", data)
    exps = {"augmentation": dict(aug=True), "batchnorm": dict(bn=True), "dropout": dict(dropout=0.5),
            "l2_weight_decay": dict(l2=1e-4), "cosine_lr": dict(cosine=True)}
    exp_res = {k: run(k, data, **v) for k, v in exps.items()}

    helpful = {k: v for k, v in exps.items() if exp_res[k]["val_acc"] - base["val_acc"] >= HELPFUL_MARGIN}
    final_cfg = {}
    for v in helpful.values():
        final_cfg.update(v)
    print("Helpful techniques:", list(helpful))
    final = run("final_custom", data, **final_cfg) if final_cfg else base

    # error analysis
    bcm = np.array(base["cm"]); fcm = np.array(final["cm"])
    pairs = top_pairs(bcm)
    plot_cm(bcm, names, "Baseline confusion matrix (test)", f"{OUT}/cm_baseline.png")
    plot_cm(fcm, names, "Final customized CNN confusion matrix (test)", f"{OUT}/cm_final.png")
    plot_curves([base, final], f"{OUT}/curves_baseline_vs_final.png")

    d_acc_pp = (final["test_acc"] - base["test_acc"]) * 100
    d_params_m = (final["params"] - base["params"]) / 1e6
    d_time = final["train_time_s"] - base["train_time_s"]
    per_m = f"{d_acc_pp / d_params_m:.2f} pp per extra million parameters" if d_params_m > 1e-6 else \
        "n/a (final model has no more parameters than the baseline, so the gain is effectively free in size)"
    met = d_acc_pp >= 3.0

    # ---------- report ----------
    L = []
    w = L.append
    w("# Task 1: Image Classification on EuroSAT - Baseline AlexNet-style CNN vs Customized CNN\n")
    w("## 1. Dataset & Experimental Setup")
    w("Dataset: EuroSAT (RGB), Sentinel-2 satellite land-use/land-cover images (Helber et al.), loaded via tensorflow_datasets 'eurosat/rgb'.")
    w("Application domain: remote sensing / land-use classification.")
    w(f"Total images: {len(y)} | Classes: {len(names)} | Image size: {IMG}x{IMG}x3 | Preprocessing: pixel values scaled to [0,1].")
    w("Class distribution: " + ", ".join(f"{n}={c}" for n, c in zip(names, counts)))
    w(f"Fixed random seed: {SEED} | Frozen stratified split: train {len(tr)} / val {len(va)} / test {len(te)} (70/15/15), saved to results/split_seed{SEED}.npz.")
    w(f"Fixed budget for every experiment: {EPOCHS} epochs, batch size {BATCH}, Adam lr {LR}, final-epoch weights used (no checkpoint selection).\n")

    w("## 2. Part A - Traditional CNN Baseline")
    w("Architecture: Conv(64,5x5)-ReLU-MaxPool, Conv(128,3x3)-ReLU-MaxPool, Conv(256,3x3)-ReLU, Conv(256,3x3)-ReLU-MaxPool, Flatten, Dense(512)-ReLU, Dense(256)-ReLU, Dense(10)-Softmax. No augmentation, batch norm or dropout.")
    w(f"Parameters: {base['params']:,} | Epochs: {base['epochs']} | Training time: {base['train_time_s']} s")
    w(f"Train acc {base['train_acc']:.4f} | Val acc {base['val_acc']:.4f} | Test acc {base['test_acc']:.4f}")
    w(f"Test Precision {base['precision']:.4f} | Recall {base['recall']:.4f} | F1 {base['f1']:.4f} (macro averaged)")
    w("Confusion matrix: see cm_baseline.png\n")

    w("## 3. Part A - Baseline Error Analysis")
    for c, i, j in pairs:
        w(f"- {names[i]} <-> {names[j]}: {c} test errors (both directions). Hypothesis: {hypothesis(names[i], names[j])}")
    w("(Verify these hypotheses against cm_baseline.png and some misclassified images, and rewrite in your own words.)\n")

    w("## 4. Part B - Technique Experiments (one change at a time, same split/seed/epochs)")
    w("| Technique | Params | Train time (s) | Val acc | Test acc | Val change vs baseline (pp) | Helpful? |")
    w("|---|---|---|---|---|---|---|")
    w(f"| baseline | {base['params']:,} | {base['train_time_s']} | {base['val_acc']:.4f} | {base['test_acc']:.4f} | - | - |")
    for k, r in exp_res.items():
        d = (r["val_acc"] - base["val_acc"]) * 100
        w(f"| {k} | {r['params']:,} | {r['train_time_s']} | {r['val_acc']:.4f} | {r['test_acc']:.4f} | {d:+.2f} | {'YES' if k in helpful else 'no'} |")
    w(f"Decision rule: a technique is kept only if it improves VALIDATION accuracy by >= {HELPFUL_MARGIN*100:.1f} pp over the baseline (the test set is never used for selection).")
    w("Note: each experiment uses a single seed, so differences of a few tenths of a pp can be within random variation.\n")

    w("## 5. Part B - Final Customized CNN")
    w("Techniques included: " + (", ".join(helpful) if helpful else "none (no technique passed the rule)"))
    w(f"Parameters: {final['params']:,} | Training time: {final['train_time_s']} s | Epochs: {final['epochs']}")
    w(f"Train acc {final['train_acc']:.4f} | Val acc {final['val_acc']:.4f} | Test acc {final['test_acc']:.4f}")
    w(f"Test Precision {final['precision']:.4f} | Recall {final['recall']:.4f} | F1 {final['f1']:.4f}")
    w("Confusion matrix: see cm_final.png\n")

    w("## 6. Part A vs Part B Comparison")
    w("| Metric | Baseline | Final |")
    w("|---|---|---|")
    w(f"| Test accuracy | {base['test_acc']:.4f} | {final['test_acc']:.4f} |")
    w(f"| Parameters | {base['params']:,} | {final['params']:,} |")
    w(f"| Training time (s) | {base['train_time_s']} | {final['train_time_s']} |")
    w(f"Improvement: {d_acc_pp:+.2f} percentage points ({'meets' if met else 'does NOT meet'} the >= 3 pp target).")
    w(f"Extra parameters: {d_params_m:+.3f} M | Extra training time: {d_time:+.1f} s | Accuracy gained: {per_m}.")
    w("Confused-class re-test (baseline's most confused pairs, test errors baseline -> final):")
    for c, i, j in pairs:
        fc = int(fcm[i][j] + fcm[j][i])
        chg = (fc - c) / c * 100 if c else 0
        w(f"- {names[i]} <-> {names[j]}: {c} -> {fc} ({chg:+.0f}%)")
    w("Final verdict: write your own conclusion here - is the accuracy gain worth the added size/training time, and which technique contributed most?\n")

    w("## 7. Observations / Problems faced / My understanding")
    w("(Write this section yourself: what surprised you, overfitting seen in curves_baseline_vs_final.png, what failed, what you learned.)")
    open(f"{OUT}/report_draft.md", "w").write("\n".join(L))
    json.dump(dict(helpful=list(helpful), delta_pp=d_acc_pp, target_met=met),
              open(f"{OUT}/summary.json", "w"), indent=2)
    with open(f"{OUT}/baseline_architecture.txt", "w") as fh:
        fh.write(base["summary"])
    print("\n".join(L))


if __name__ == "__main__":
    main()

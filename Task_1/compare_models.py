"""
compare_models.py
Run this AFTER both part_a_traditional_cnn.py and part_b_customized_cnn.py
have finished. Produces the side-by-side performance table required in the
Task 1 deliverables (accuracy, precision, recall, F1, params, training time).
"""

import json

with open("part_a_results.json") as f:
    a = json.load(f)
with open("part_b_results.json") as f:
    b = json.load(f)

rows = [
    ("Test Accuracy", a["test_accuracy"], b["test_accuracy"]),
    ("Precision (macro)", a["precision_macro"], b["precision_macro"]),
    ("Recall (macro)", a["recall_macro"], b["recall_macro"]),
    ("F1-score (macro)", a["f1_macro"], b["f1_macro"]),
    ("Total Parameters", a["total_params"], b["total_params"]),
    ("Epochs Trained", a["epochs"], b["epochs"]),
    ("Training Time (sec)", a["training_time_sec"], b["training_time_sec"]),
]

print(f"{'Metric':<22}{'Traditional CNN':<20}{'Customized CNN':<20}")
print("-" * 62)
for name, va, vb in rows:
    if isinstance(va, float):
        print(f"{name:<22}{va:<20.4f}{vb:<20.4f}")
    else:
        print(f"{name:<22}{va:<20,}{vb:<20,}")

gain = (b["test_accuracy"] - a["test_accuracy"]) * 100
extra_params = b["total_params"] - a["total_params"]
extra_time = b["training_time_sec"] - a["training_time_sec"]

print("\nAccuracy gain: {:+.2f} percentage points".format(gain))
print("Extra parameters: {:+,}".format(extra_params))
print("Extra training time: {:+.1f} sec".format(extra_time))
print("\nUse this table + the gain/cost numbers directly in your Google Doc report,")
print("under 'Comparison between the traditional CNN and the customized CNN'.")

import pickle
from pathlib import Path

# Load the regression data to see raw scores
data_path = Path("data") / "train_data_regression_data.pkl"

if not data_path.exists():
    print(f"❌ Regression data not found at {data_path}")
    print("Checking binning data instead...")
    data_path = Path("data") / "train_data_binning_data.pkl"

with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Collect all valid labels (not -1)
all_labels = []
for graph in data:
    labels = graph.y
    valid_labels = labels[labels != -1].numpy()
    all_labels.extend(valid_labels.tolist())

import numpy as np
labels_array = np.array(all_labels)

print(f"{'='*60}")
print("ACTUAL DATA ANALYSIS")
print(f"{'='*60}")

if 'regression' in str(data_path):
    print("Data type: REGRESSION (raw scores)")
    print(f"Min score: {labels_array.min()}")
    print(f"Max score: {labels_array.max()}")
    print(f"Score range that could exist: 0-999+ (depends on game)")
    print(f"\nWith bin_size=10:")
    print(f"  Min possible bin: {int(labels_array.min()) // 10}")
    print(f"  Max possible bin: {int(labels_array.max()) // 10}")
    print(f"  Bins that COULD exist: 0-99 (if scores go to 999)")
else:
    print("Data type: BINNING (already binned)")
    print(f"Bins present: {sorted(np.unique(labels_array).tolist())}")
    print(f"Min bin: {labels_array.min()}")
    print(f"Max bin: {labels_array.max()}")

    # Reverse engineer what scores created these bins
    print(f"\nReverse-engineering scores from bins (bin_size=10):")
    for bin_val in sorted(np.unique(labels_array)):
        score_min = int(bin_val) * 10
        score_max = int(bin_val) * 10 + 9
        print(f"  Bin {int(bin_val)}: scores {score_min}-{score_max}")

print(f"\n{'='*60}")
print("THE REAL QUESTION:")
print(f"{'='*60}")
print("During inference, could the model receive NEW scores it hasn't")
print("seen in training that would map to bins 5-98?")
print(f"\nIf YES → num_bins = 100 (to handle all possible bins)")
print(f"If NO  → num_bins = 6 (only for the bins in your data)")

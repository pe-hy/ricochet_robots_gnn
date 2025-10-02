import pickle
import numpy as np
from pathlib import Path

# Path to binning data
data_path = Path("data") / "train_data_binning_data.pkl"

if not data_path.exists():
    print(f"❌ Error: Data file not found at {data_path}")
    exit(1)

with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Collect all valid bin labels (not -1)
all_bins = []
for graph in data:
    labels = graph.y
    valid_bins = labels[labels != -1].numpy()
    all_bins.extend(valid_bins.tolist())

bins = np.array(all_bins)

max_bin = int(bins.max())
min_bin = int(bins.min())
unique_bins = np.unique(bins)

print(f"{'='*60}")
print("CORRECT NUM_BINS FOR CONFIG")
print(f"{'='*60}")
print(f"Minimum bin value: {min_bin}")
print(f"Maximum bin value: {max_bin}")
print(f"Unique bins present: {sorted(unique_bins.tolist())}")
print(f"Number of unique bins: {len(unique_bins)}")

from collections import Counter
bin_counts = Counter(bins)
print(f"\nBin distribution:")
for bin_val in sorted(unique_bins):
    count = bin_counts[bin_val]
    pct = 100 * count / len(bins)
    print(f"  Bin {bin_val}: {count} ({pct:.2f}%)")

print(f"\n{'='*60}")
print("ANSWER:")
print(f"{'='*60}")
print(f"Option 1 (WASTEFUL): Set num_bins = {max_bin + 1}")
print(f"  - Creates {max_bin + 1} output classes")
print(f"  - BUT only {len(unique_bins)} are actually used")
print(f"  - Wastes {max_bin + 1 - len(unique_bins)} unused output neurons")
print(f"\nOption 2 (EFFICIENT): Set num_bins = {len(unique_bins)}")
print(f"  - Creates {len(unique_bins)} output classes (one per actual bin)")
print(f"  - Requires remapping bins to 0-{len(unique_bins)-1} in data generation")
print(f"\n✅ RECOMMENDATION: Use Option 2 and fix your binning strategy")

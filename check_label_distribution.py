import pickle
import torch
from pathlib import Path

data_dir = Path("data")

tasks = {
    'basic': 'train_data_basic_data.pkl',
    'threshold': 'train_data_threshold_data.pkl',
    'unreachable': 'train_data_unreachable_components_data.pkl',
    'regression': 'train_data_regression_data.pkl',
    'binning': 'train_data_binning_data.pkl',
    'best_component': 'train_data_best_component_data.pkl'
}

for task_name, filename in tasks.items():
    filepath = data_dir / filename

    if not filepath.exists():
        print(f"\n{task_name}: FILE NOT FOUND")
        continue

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    print(f"Total graphs: {len(data)}")

    # Analyze labels
    all_labels = []
    graphs_with_positive = 0

    for graph in data:
        labels = graph.y
        all_labels.extend(labels.tolist())

        # Check if graph has any positive labels (excluding -1 for regression/binning)
        if task_name in ['regression', 'binning']:
            has_positive = (labels[labels != -1] > 0).any() if (labels != -1).any() else False
        else:
            has_positive = (labels > 0).any()

        if has_positive:
            graphs_with_positive += 1

    all_labels = torch.tensor(all_labels)

    # Task-specific analysis
    if task_name in ['regression']:
        valid_labels = all_labels[all_labels != -1]
        print(f"Valid nodes: {len(valid_labels)} / {len(all_labels)}")
        print(f"Label range: [{valid_labels.min().item():.2f}, {valid_labels.max().item():.2f}]")
        print(f"Label mean: {valid_labels.float().mean().item():.2f}")
        print(f"Label std: {valid_labels.float().std().item():.2f}")

    elif task_name in ['binning']:
        valid_labels = all_labels[all_labels != -1]
        print(f"Valid nodes: {len(valid_labels)} / {len(all_labels)}")
        print(f"Unique bins: {valid_labels.unique().tolist()}")
        print(f"Max bin: {valid_labels.max().item()}")
        print(f"Min bin: {valid_labels.min().item()}")
        print(f"Bin distribution:")
        for bin_val in sorted(valid_labels.unique().tolist()):
            count = (valid_labels == bin_val).sum().item()
            pct = 100 * count / len(valid_labels)
            print(f"  Bin {bin_val}: {count} ({pct:.2f}%)")

    else:  # Binary classification tasks
        num_zeros = (all_labels == 0).sum().item()
        num_ones = (all_labels == 1).sum().item()
        total = len(all_labels)

        print(f"Label 0: {num_zeros} ({100*num_zeros/total:.2f}%)")
        print(f"Label 1: {num_ones} ({100*num_ones/total:.2f}%)")
        print(f"Graphs with at least one positive label: {graphs_with_positive} / {len(data)}")
        print(f"Class imbalance ratio: {num_zeros/num_ones:.2f}:1" if num_ones > 0 else "Class imbalance ratio: inf:1 (no positive labels!)")

print(f"\n{'='*60}")

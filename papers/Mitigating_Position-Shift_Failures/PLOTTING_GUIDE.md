# Plotting Training Curves

The `training_curves.json` file contains all training data for plotting.

## Data Structure

```json
{
  "experiment_name": {
    "seeds": {
      "seed_42": {
        "train": {
          "steps": [0, 20, 40, ...],
          "accuracy": [0.01, 0.05, 0.12, ...],
          "loss": [4.5, 3.2, 2.1, ...]
        },
        "eval_A": [
          {"step": 200, "accuracy": 0.15},
          {"step": 400, "accuracy": 0.45},
          ...
        ],
        "eval_B": [
          {"step": 200, "accuracy": 0.12},
          ...
        ],
        "eval_C0": [
          {"step": 200, "accuracy": 0.05},
          ...
        ]
      },
      "seed_43": { ... },
      "seed_44": { ... }
    }
  }
}
```

## Quick Plotting in Python

```python
import json
import matplotlib.pyplot as plt

# Load data
with open('training_curves.json') as f:
    curves = json.load(f)

# Plot Eval-A for all experiments
fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    'baseline_001': 'blue',
    'i1_001_1': 'green',
    'i1_002_alibi': 'red',
    'i1_002a': 'purple'
}

for exp_name, exp_data in curves.items():
    color = colors.get(exp_name, 'gray')

    for seed_name, seed_data in exp_data['seeds'].items():
        eval_a = seed_data['eval_A']
        steps = [d['step'] for d in eval_a if d['accuracy'] is not None]
        accs = [d['accuracy'] for d in eval_a if d['accuracy'] is not None]

        label = exp_name if seed_name == 'seed_42' else None
        ax.plot(steps, accs, 'o-', color=color, alpha=0.6, label=label)

ax.set_xlabel('Training Steps')
ax.set_ylabel('Eval-A Accuracy')
ax.set_title('Eval-A Accuracy vs Training Steps')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Plotting in Google Colab

Upload `training_curves.json` to your Colab session and run:

```python
# Install if needed
# !pip install matplotlib seaborn

import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Load curves
with open('training_curves.json') as f:
    curves = json.load(f)

# Create 2x2 subplot for all metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {
    'baseline_001': 'C0',
    'i1_001_1': 'C1',
    'i1_002_alibi': 'C2',
    'i1_002a': 'C3'
}

labels = {
    'baseline_001': 'Baseline',
    'i1_001_1': 'I1_001_1 (Position)',
    'i1_002_alibi': 'I1_002_ALIBI',
    'i1_002a': 'I1_002a (Main)'
}

# Training accuracy
ax = axes[0, 0]
for exp_name, exp_data in curves.items():
    color = colors[exp_name]
    label = labels[exp_name]

    for i, (seed_name, seed_data) in enumerate(exp_data['seeds'].items()):
        steps = seed_data['train']['steps']
        accs = seed_data['train']['accuracy']
        lbl = label if i == 0 else None
        ax.plot(steps, accs, color=color, alpha=0.5, linewidth=1, label=lbl)

ax.set_xlabel('Steps')
ax.set_ylabel('Accuracy')
ax.set_title('Training Accuracy')
ax.legend()
ax.set_ylim(0, 1)

# Eval-A
ax = axes[0, 1]
for exp_name, exp_data in curves.items():
    color = colors[exp_name]
    label = labels[exp_name]

    for i, (seed_name, seed_data) in enumerate(exp_data['seeds'].items()):
        eval_data = seed_data['eval_A']
        steps = [d['step'] for d in eval_data if d['accuracy'] is not None]
        accs = [d['accuracy'] for d in eval_data if d['accuracy'] is not None]
        lbl = label if i == 0 else None
        ax.plot(steps, accs, 'o-', color=color, alpha=0.6, markersize=3, label=lbl)

ax.set_xlabel('Steps')
ax.set_ylabel('Accuracy')
ax.set_title('Eval-A (In-Distribution)')
ax.legend()
ax.set_ylim(0, 1)

# Eval-B
ax = axes[1, 0]
for exp_name, exp_data in curves.items():
    color = colors[exp_name]
    label = labels[exp_name]

    for i, (seed_name, seed_data) in enumerate(exp_data['seeds'].items()):
        eval_data = seed_data['eval_B']
        steps = [d['step'] for d in eval_data if d['accuracy'] is not None]
        accs = [d['accuracy'] for d in eval_data if d['accuracy'] is not None]
        lbl = label if i == 0 else None
        ax.plot(steps, accs, 'o-', color=color, alpha=0.6, markersize=3, label=lbl)

ax.set_xlabel('Steps')
ax.set_ylabel('Accuracy')
ax.set_title('Eval-B (Position Shift)')
ax.legend()
ax.set_ylim(0, 1)

# Eval-C0
ax = axes[1, 1]
for exp_name, exp_data in curves.items():
    color = colors[exp_name]
    label = labels[exp_name]

    for i, (seed_name, seed_data) in enumerate(exp_data['seeds'].items()):
        eval_data = seed_data['eval_C0']
        steps = [d['step'] for d in eval_data if d['accuracy'] is not None]
        accs = [d['accuracy'] for d in eval_data if d['accuracy'] is not None]
        lbl = label if i == 0 else None
        ax.plot(steps, accs, 'o-', color=color, alpha=0.6, markersize=3, label=lbl)

ax.set_xlabel('Steps')
ax.set_ylabel('Accuracy')
ax.set_title('Eval-C0 (Template OOD)')
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Key Observations to Highlight in Plots

### Baseline (blue):
- Trains to ~99% quickly
- **Catastrophic cliff**: Eval-B drops from 99% (pos 0) to 1% (pos 64)
- Eval-C0 stays near 0%

### I1_002a (purple, main result):
- Trains to ~96% (slightly slower)
- **No catastrophic cliff**: Eval-B reaches ~75% and stays stable
- Eval-C0 reaches ~80%
- Eval-C1 reaches ~95% (with anchors)

### I1_002_ALIBI (red, negative result):
- **Fails completely**: stays at ~20% on all metrics
- Important negative result showing ALiBi incompatibility

### I1_001_1 (green, ablation):
- Similar to I1_002a but without template diversity
- Slightly worse: Eval-B ~72%, Eval-C0 ~60%
- Shows template diversity helps

## Reproducibility Note

All curves show 3 seeds (42, 43, 44) for each experiment. The variance is very low (std < 5%) for successful experiments, showing results are robust.

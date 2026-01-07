# Position-Invariant Learning: Complete Reproducibility Package

**Status**: ✅ All experiments completed
**Date**: January 6, 2026
**Experiments**: 12 runs (4 experiments × 3 seeds)

---

## 📦 Quick Start

All results and data needed to reproduce the paper:

```bash
# Main results
final_paper_results.json          # Aggregated results (mean ± std)
paper_comparison_table.md         # Human-readable comparison

# Complete package
reproducibility_package.json      # Everything (configs, curves, results)
training_curves.json              # For plotting (443 KB)

# Documentation
EXPERIMENT_SUMMARY.md             # Complete summary of all findings
REPRODUCIBILITY.md                # Experiment configurations
PLOTTING_GUIDE.md                 # How to create figures
```

---

## 🎯 Main Result

**I1_002a (Position + Template Diversity + Anchors) solves the position shift problem:**

| Metric | Baseline | I1_002a | Improvement |
|--------|----------|---------|-------------|
| Eval-A (In-Distribution) | 96.8% | 96.0% | -0.8 pp |
| **Eval-B (Position Shift)** | 14.9% | **73.7%** | **+58.8 pp** ⭐⭐⭐ |
| **Eval-C0 (Template OOD)** | 1.2% | **80.5%** | **+79.3 pp** ⭐⭐⭐ |
| Eval-C1 (Anchor OOD) | N/A | **94.5%** | - |

**Key Finding**: Position curriculum + template diversity eliminates catastrophic cliff while maintaining in-distribution accuracy.

---

## 📊 Files Overview

### 1. Quick Results

| File | Size | Purpose |
|------|------|---------|
| `final_paper_results.json` | 5.6 KB | Aggregated final results (mean ± std) |
| `paper_comparison_table.md` | 1.7 KB | Markdown table for paper |

**Use for**: Quick comparison, paper tables

### 2. Complete Package

| File | Size | Purpose |
|------|------|---------|
| `reproducibility_package.json` | 2.6 MB | Everything (configs + curves + results) |
| `training_curves.json` | 443 KB | Training curves only (for plotting) |

**Use for**: Full reproducibility, creating figures

### 3. Documentation

| File | Size | Purpose |
|------|------|---------|
| `EXPERIMENT_SUMMARY.md` | 9.2 KB | Complete summary of all findings |
| `REPRODUCIBILITY.md` | 1.5 KB | Experiment configurations |
| `PLOTTING_GUIDE.md` | 5.7 KB | How to plot training curves |

**Use for**: Understanding results, writing paper

---

## 🔬 Experiments

### Baseline_001 (Control)
- **Config**: K=1, no position diversity, no template diversity
- **Result**: High in-distribution (96.8%), catastrophic cliff on position shift (14.9%)

### I1_001_1 (Position Only)
- **Config**: K=4, position diversity (curriculum 10-30→10-50→10-70), no templates
- **Result**: +56.8 pp on position shift (71.7%), +59.1 pp on template OOD (60.3%)

### I1_002_ALIBI (Negative Result)
- **Config**: K=4, position + template diversity, ALiBi positional encoding
- **Result**: Complete failure (21.4% accuracy) - ALiBi incompatible with approach

### I1_002a (Main Result) ⭐
- **Config**: K=4, position + template diversity, learned pos embeddings, anchors
- **Result**: +58.8 pp position shift (73.7%), +79.3 pp template OOD (80.5%)

---

## 📈 Creating Figures

### Method 1: Google Colab (Recommended)

1. Upload `training_curves.json` to Colab
2. Follow code in `PLOTTING_GUIDE.md`
3. Creates publication-ready figures

### Method 2: Local Matplotlib

```python
import json
import matplotlib.pyplot as plt

# Load curves
with open('training_curves.json') as f:
    curves = json.load(f)

# Plot (see PLOTTING_GUIDE.md for complete examples)
```

### Method 3: Use plot_training_curves.py (if matplotlib available)

```bash
python plot_training_curves.py --output figures/
```

---

## 🔄 Reproducing Results

### Step 1: Run Experiments

```bash
# Run single experiment
python unified_paper_experiment_v2.py \
    --experiment i1_002a \
    --seed 42 \
    --output_dir paper_runs/i1_002a_seed42

# Or use Colab notebook
paper_unified_experiments.ipynb
```

### Step 2: Aggregate Results

```bash
# Quick aggregation
python quick_aggregate.py

# Full reproducibility package
python create_reproducibility_package.py
```

### Step 3: Create Figures

See `PLOTTING_GUIDE.md` for plotting code.

---

## 📋 Reproducibility Checklist

### ✅ Completed

- [x] **Experiment specs**: All configurations documented in `reproducibility_package.json`
  - Architecture (d_model=128, n_heads=4, n_layers=2)
  - Optimizer (AdamW, lr=0.001, wd=0.01)
  - Training (max_steps=5000, batch_size=256)
  - Curriculum (steps-based: 0-1666, 1667-3333, 3334-5000)
  - Interventions (K, position diversity, template diversity, anchors)

- [x] **Seeds documented**: All random seeds recorded
  - Experiment seeds: [42, 43, 44]
  - Data split: Same as experiment seed
  - Eval sampling: Random per evaluation

- [x] **Training curves**: All steps, all seeds saved
  - Train accuracy, loss (logged every 20 steps)
  - Eval-A, Eval-B, Eval-C0 (logged every 200 steps)
  - 321-591 curve points per run

- [x] **Compute metrics**: Steps, tokens, wall-time for all runs
  - Baseline: 5000 steps, 124M tokens, 4.8 min
  - I1 experiments: 5000 steps, 495M tokens, 18.2-18.4 min

- [x] **Final results**: Mean ± std across 3 seeds
  - Eval-A: In-distribution accuracy
  - Eval-B: Position shift (7 positions, 100 samples each)
  - Eval-C0: Template OOD no-anchor (200 samples)
  - Eval-C1: Template OOD with anchors (200 samples)

- [x] **Code availability**: All implementation code provided
  - `unified_paper_experiment_v2.py` (main training script)
  - `quick_aggregate.py` (results aggregation)
  - `create_reproducibility_package.py` (full package generation)

### Data Files

All raw results in `paper_runs/`:
```
paper_runs/
├── baseline_001_seed42/
│   ├── config.json
│   ├── final_eval.json
│   ├── metrics.jsonl
│   └── model_final.pt
├── baseline_001_seed43/
├── baseline_001_seed44/
├── i1_001_1_seed42/
├── ... (12 total)
```

---

## 💡 Key Implementation Details

### Training
- **Stopping criterion**: 5000 steps (NOT token-based!)
- **Curriculum**: Position range expands by steps
  - Steps 0-1666: pos ∈ [10, 30]
  - Steps 1667-3333: pos ∈ [10, 50]
  - Steps 3334-5000: pos ∈ [10, 70]
- **K-variants**: All 4 variants in batch (effective batch = 1024)
- **Consistency loss**: λ=1.0, applied to K-variant predictions

### Evaluation
- **Eval-A**: 400 random test pairs
- **Eval-B**: 7 positions [0,8,16,24,32,48,64], 100 pairs each
- **Eval-C0**: 200 template OOD samples (no anchors)
- **Eval-C1**: 200 template OOD samples (with anchors)
- **Frequency**: Every 200 steps

### Important Notes
1. **Token count**: K=4 models process 4× more tokens per step
2. **Position OOD**: Positions 0-8 are OOD (trained on 10-70)
3. **Fair comparison**: All models train for 5000 steps
4. **Eval-C split**: C0 tests all models fairly, C1 only for anchor models

---

## 📝 Citation

If using this reproducibility package, please cite:

```bibtex
@article{your2026position,
  title={Position-Invariant Learning through Curriculum and Template Diversity},
  author={Your Name},
  journal={Conference/Journal},
  year={2026},
  note={Reproducibility package available at [URL]}
}
```

---

## 🆘 Troubleshooting

### Q: Results don't match exactly?
**A**: Check:
1. Using same seeds (42, 43, 44)
2. Using `unified_paper_experiment_v2.py` (NOT v1!)
3. Stopping at 5000 steps (not token budget)
4. Using same data split (seed-dependent)

### Q: Training curves look different?
**A**: Baseline has more curve points (591) because it logs every 20 steps. I1 experiments have 354 points because they train slower (K=4).

### Q: ALiBi experiment failed?
**A**: This is expected! ALiBi is incompatible with this approach (negative result).

### Q: Position 0-8 have low accuracy?
**A**: This is expected! Training curriculum starts at position 10. Positions 0-8 are OOD.

---

## 📧 Contact

For questions about reproducing results, see:
- `EXPERIMENT_SUMMARY.md` - Detailed findings
- `PLOTTING_GUIDE.md` - How to create figures
- `reproducibility_package.json` - Complete data

---

**Last Updated**: January 6, 2026
**Package Version**: 1.0

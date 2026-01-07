# Reproducibility Package: Mitigating Position-Shift Failures

**Paper:** "Mitigating Position-Shift Failures in Text-Based Modular Arithmetic via Position Curriculum and Template Diversity"
**Author:** Nikolay Yudin
**Date:** January 6, 2026

---

## 📦 Package Contents

This directory contains all artifacts needed to reproduce the paper results.

### 1. **Data Files**

| File | Size | Description |
|------|------|-------------|
| `final_paper_results.json` | 5.6 KB | Aggregated results (mean ± std across 3 seeds) |
| `training_curves.json` | 443 KB | Training curves for all experiments (all steps, all seeds) |
| `reproducibility_package.json` | 2.6 MB | Complete package: configs + curves + compute metrics |

### 2. **Code**

| File | Description |
|------|-------------|
| `unified_paper_experiment_v2.py` | Main training script (steps-based, 5000 steps) |
| `quick_aggregate.py` | Quick aggregation of results from `paper_runs/` |
| `create_reproducibility_package.py` | Generates complete reproducibility package |
| `plot_paper_figures.ipynb` | **Standalone Jupyter notebook** for generating all figures |

### 3. **Documentation**

| File | Description |
|------|-------------|
| `README_REPRODUCIBILITY.md` | **START HERE** - main reproducibility guide |
| `EXPERIMENT_SUMMARY.md` | Complete summary of all findings and results |
| `REPRODUCIBILITY.md` | Experiment configurations summary |
| `PLOTTING_GUIDE.md` | Manual plotting instructions |

---

## 🚀 Quick Start

### Option 1: View Results Only

1. Open `final_paper_results.json` to see aggregated metrics
2. Read `EXPERIMENT_SUMMARY.md` for detailed findings
3. See `paper_comparison_table.md` for quick comparison

### Option 2: Generate Figures

**Using Google Colab (Recommended):**

1. Upload `plot_paper_figures.ipynb` to Google Colab
2. Upload `training_curves.json` and `final_paper_results.json` to Colab
3. Run all cells
4. Download generated figures (or `paper_figures.zip`)

**Locally:**

```bash
pip install matplotlib seaborn numpy
jupyter notebook plot_paper_figures.ipynb
# Run all cells
```

**Output:**
- `figure1_training_curves_all.png` - Training dynamics
- `figure2_comparison_i1_002a_vs_baseline.png` - Main comparison
- `figure3_position_breakdown.png` - Position-shift robustness
- `figure4_final_performance_summary.png` - Final performance summary

### Option 3: Reproduce Full Experiments

```bash
# Install dependencies
pip install torch numpy tqdm

# Run single experiment
python unified_paper_experiment_v2.py \
    --experiment i1_002a \
    --seed 42 \
    --output_dir my_runs/i1_002a_seed42

# Run all 12 experiments (4 experiments × 3 seeds)
for exp in baseline_001 i1_001_1 i1_002_alibi i1_002a; do
    for seed in 42 43 44; do
        python unified_paper_experiment_v2.py \
            --experiment $exp \
            --seed $seed \
            --output_dir my_runs/${exp}_seed${seed}
    done
done

# Aggregate results
python quick_aggregate.py --runs_dir my_runs
```

---

## 📊 Main Results

**Table 1: Main Metrics (mean ± std over 3 seeds)**

| Experiment | Eval-A | Eval-B | Eval-C0 | Eval-C1 |
|------------|--------|--------|---------|---------|
| Baseline-001 | 96.8±4.2 | 14.9±0.5 | 1.2±0.8 | — |
| I1_001_1 (Position) | 96.5±0.9 | 71.7±0.6 | 60.3±6.3 | — |
| I1_002a (Full) | **96.0±0.5** | **73.7±0.7** | **80.5±3.0** | **94.5±2.2** |
| I1_002-ALiBi | 21.4±1.0 | 34.3±3.0 | 15.5±2.2 | 34.5±3.5 |

**Key Findings:**
- **Position curriculum** eliminates catastrophic cliff (Eval-B: 14.9% → 73.7%, +58.8 pp)
- **Template diversity** improves OOD robustness (Eval-C0: 1.2% → 80.5%, +79.3 pp)
- **ALiBi fails** in character-level parsing regime (Eval-A: 21.4%)

---

## 🔬 Experiment Configurations

### Architecture
- Model: 2-layer Transformer encoder
- d_model: 128
- n_heads: 4
- Positional embeddings: Learned absolute (except ALiBi ablation)

### Training
- Optimizer: AdamW (lr=0.001, wd=0.01)
- Steps: 5000 (fixed budget)
- Batch size: 256
- Curriculum: Steps-based position range expansion
  - Steps 0-1666: positions [10, 30]
  - Steps 1667-3333: positions [10, 50]
  - Steps 3334-5000: positions [10, 70]

### Interventions
- **K-variants**: 4 (except baseline K=1)
- **Consistency loss**: λ=1.0 (MSE on pre-softmax logits)
- **Template diversity**: 40% padding / 40% NL / 20% mixed
- **Anchors**: `<EXPR>...</EXPR>` (I1_002a only)

### Evaluation
- **Eval-A**: 400 random test pairs (in-distribution)
- **Eval-B**: 7 positions [0,8,16,24,32,48,64], 100 pairs each
- **Eval-C0**: 200 OOD templates without anchors
- **Eval-C1**: 200 OOD templates with anchors

---

## 📁 File Structure

```
Mitigating_Position-Shift_Failures/
├── README.md                              # This file
├── README_REPRODUCIBILITY.md              # Detailed reproducibility guide
├── EXPERIMENT_SUMMARY.md                  # Complete findings summary
├── REPRODUCIBILITY.md                     # Config specifications
├── PLOTTING_GUIDE.md                      # Manual plotting instructions
│
├── final_paper_results.json               # Aggregated results (5.6 KB)
├── training_curves.json                   # All training curves (443 KB)
├── reproducibility_package.json           # Complete package (2.6 MB)
│
├── unified_paper_experiment_v2.py         # Main training script
├── quick_aggregate.py                     # Results aggregation
├── create_reproducibility_package.py      # Package generation
└── plot_paper_figures.ipynb               # Standalone plotting notebook
```

---

## 🎯 Usage Examples

### Example 1: Quick Comparison

```python
import json

with open('final_paper_results.json') as f:
    results = json.load(f)

baseline = results['baseline_001']
i1_002a = results['i1_002a']

print(f"Baseline Eval-B: {baseline['eval_B']['mean']*100:.1f}%")
print(f"I1_002a Eval-B: {i1_002a['eval_B']['mean']*100:.1f}%")
print(f"Improvement: +{(i1_002a['eval_B']['mean'] - baseline['eval_B']['mean'])*100:.1f} pp")
```

### Example 2: Plot Single Metric

```python
import json
import matplotlib.pyplot as plt

with open('training_curves.json') as f:
    curves = json.load(f)

# Plot Eval-B for baseline and I1_002a
for exp in ['baseline_001', 'i1_002a']:
    for seed_name, seed_data in curves[exp]['seeds'].items():
        eval_b = seed_data['eval_B']
        steps = [d['step'] for d in eval_b if d['accuracy'] is not None]
        accs = [d['accuracy'] for d in eval_b if d['accuracy'] is not None]
        plt.plot(steps, accs, label=f"{exp}_{seed_name}")

plt.xlabel('Steps')
plt.ylabel('Eval-B Accuracy')
plt.legend()
plt.show()
```

### Example 3: Reproduce Specific Configuration

```bash
# Reproduce I1_002a with seed 42
python unified_paper_experiment_v2.py \
    --experiment i1_002a \
    --seed 42 \
    --output_dir my_runs/i1_002a_seed42 \
    --max_steps 5000

# Should produce results matching:
# paper_runs/i1_002a_seed42/final_eval.json
```

---

## ⚙️ System Requirements

**Minimal (for plotting only):**
- Python 3.8+
- matplotlib, seaborn, numpy

**Full reproduction:**
- Python 3.8+
- PyTorch 1.10+ (CPU or GPU)
- numpy, tqdm
- ~2 GB RAM for training
- ~20 minutes per full run (4 experiments × 3 seeds × ~18 min each)

---

## 📖 Citation

If using this reproducibility package, please cite:

```bibtex
@article{yudin2026position,
  title={Mitigating Position-Shift Failures in Text-Based Modular Arithmetic via Position Curriculum and Template Diversity},
  author={Yudin, Nikolay},
  year={2026},
  note={Reproducibility package available at https://github.com/nick-yudin/Generalization}
}
```

---

## 🆘 Troubleshooting

### Issue: Figures don't match paper exactly
**Solution:** Ensure you're using `training_curves.json` and `final_paper_results.json` from this package, not from a different run.

### Issue: Results don't reproduce exactly
**Check:**
1. Using correct script (`unified_paper_experiment_v2.py`, NOT v1)
2. Same seeds (42, 43, 44)
3. Steps-based stopping (5000 steps, not token budget)
4. Correct experiment name (e.g., `i1_002a`, not `i1_002`)

### Issue: ALiBi experiment fails
**Expected!** ALiBi is incompatible with this setup (negative result). It achieves only ~21% Eval-A.

### Issue: Position 0-8 have low accuracy
**Expected!** Training curriculum starts at position 10. Positions 0-8 are intentional OOD stress tests.

---

## 📧 Contact

For questions or issues:
- GitHub: https://github.com/nick-yudin/Generalization
- Email: n.yudin@gmail.com

---

**Last Updated:** January 7, 2026
**Package Version:** 1.0
**Status:** ✅ Complete and tested

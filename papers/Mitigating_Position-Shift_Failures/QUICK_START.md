# Quick Start Guide

**Paper:** Mitigating Position-Shift Failures in Text-Based Modular Arithmetic

---

## 🚀 Three Ways to Use This Package

### 1️⃣ Just Want to See Results? (30 seconds)

```bash
# View aggregated results
cat final_paper_results.json | python -m json.tool

# Or read the summary
cat EXPERIMENT_SUMMARY.md
```

**Key takeaway:** I1_002a improves position-shift robustness by **+58.8 pp** (14.9% → 73.7%) and template OOD by **+79.3 pp** (1.2% → 80.5%).

---

### 2️⃣ Want to Generate Figures? (5 minutes)

**Google Colab (easiest):**

1. Go to https://colab.research.google.com/
2. Upload `plot_paper_figures.ipynb`
3. Upload `training_curves.json` and `final_paper_results.json` to Files panel
4. Click Runtime → Run all
5. Download the 4 generated figures (or `paper_figures.zip`)

**Local (if you have matplotlib):**

```bash
pip install matplotlib seaborn numpy
jupyter notebook plot_paper_figures.ipynb
# Run all cells
```

**Output:**
- `figure1_training_curves_all.png` - Training dynamics (2x2 grid)
- `figure2_comparison_i1_002a_vs_baseline.png` - Direct comparison (1x3 grid)
- `figure3_position_breakdown.png` - Position-shift breakdown (bar chart)
- `figure4_final_performance_summary.png` - Final performance (bar charts)

---

### 3️⃣ Want to Reproduce Experiments? (20 minutes per run)

**Run single experiment:**

```bash
pip install torch numpy tqdm

python unified_paper_experiment_v2.py \
    --experiment i1_002a \
    --seed 42 \
    --output_dir my_runs/i1_002a_seed42
```

**Run all 12 experiments (4 × 3 seeds):**

```bash
for exp in baseline_001 i1_001_1 i1_002_alibi i1_002a; do
    for seed in 42 43 44; do
        python unified_paper_experiment_v2.py \
            --experiment $exp \
            --seed $seed \
            --output_dir my_runs/${exp}_seed${seed}
    done
done
```

**Aggregate results:**

```bash
python quick_aggregate.py --runs_dir my_runs
# Creates final_paper_results.json
```

---

## 📊 What You'll Get

### Main Results

| Experiment | Eval-A | Eval-B | Eval-C0 |
|------------|--------|--------|---------|
| Baseline | 96.8% | **14.9%** ⚠️ | 1.2% |
| I1_002a | 96.0% | **73.7%** ✅ | 80.5% |
| Improvement | -0.8 pp | **+58.8 pp** 🎯 | +79.3 pp |

### Position Breakdown (Eval-B)

Baseline shows **catastrophic cliff**:
- Position 0: 99.0%
- Position 8: 0.7% ⬇️ (98% drop!)
- Position 16-64: ~1%

I1_002a shows **position invariance**:
- Position 0-8: 5-22% (OOD, trained on 10-70)
- Position 16-64: **97-99%** ✅ (stable!)

---

## 🎯 Evaluation Suite

**Eval-A** (in-distribution)
- 400 random test pairs
- Same distribution as training

**Eval-B** (position shift)
- 7 positions: [0, 8, 16, 24, 32, 48, 64]
- 100 pairs per position
- Tests robustness to absolute position changes

**Eval-C0** (template OOD, no anchors)
- 200 samples with novel templates
- Questions and commands
- Fair test for all models

**Eval-C1** (template OOD, with anchors)
- 200 samples with `<EXPR>...</EXPR>` markers
- Only for anchor-trained models

---

## 🔧 Experiment Configurations

### Baseline-001
```
K = 1
Position diversity: ✗
Template diversity: ✗
Anchors: ✗
```

### I1_001_1 (Position Only)
```
K = 4
Position diversity: ✓ (curriculum 10-30 → 10-50 → 10-70)
Template diversity: ✗
Anchors: ✗
Consistency loss: ✓
```

### I1_002a (Full Intervention)
```
K = 4
Position diversity: ✓ (curriculum)
Template diversity: ✓ (40% pad / 40% NL / 20% mixed)
Anchors: ✓ (<EXPR>...</EXPR>)
Consistency loss: ✓
```

### I1_002-ALiBi (Negative Result)
```
Same as I1_002a but:
Positional embeddings: ALiBi (instead of learned)
Result: FAILURE (Eval-A only 21.4%)
```

---

## 📖 Next Steps

**For detailed information:**
- Read `README_REPRODUCIBILITY.md` for full reproducibility guide
- Read `EXPERIMENT_SUMMARY.md` for complete findings analysis
- Read `REPRODUCIBILITY.md` for exact configurations

**For plotting:**
- Read `PLOTTING_GUIDE.md` for manual plotting examples
- Use `plot_paper_figures.ipynb` for automated figure generation

**For data exploration:**
- `final_paper_results.json` - aggregated metrics
- `training_curves.json` - all training curves
- `reproducibility_package.json` - complete package with configs

---

## ⚠️ Common Issues

**Q: ALiBi experiment has very low accuracy?**
A: This is expected! ALiBi is incompatible with character-level parsing in our setup (negative result).

**Q: Position 0-8 have low accuracy in I1 models?**
A: This is expected! Training curriculum starts at position 10. Positions 0-8 are intentional OOD stress tests.

**Q: My reproduced results don't match exactly?**
A: Check:
1. Using correct script (`unified_paper_experiment_v2.py`)
2. Same seeds (42, 43, 44)
3. Steps-based stopping (5000 steps)

---

## 📧 Questions?

- GitHub: https://github.com/nick-yudin/Generalization
- Email: n.yudin@gmail.com

---

**Last Updated:** January 7, 2026

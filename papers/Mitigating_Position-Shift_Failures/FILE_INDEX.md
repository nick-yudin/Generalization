# File Index - Reproducibility Package

Quick reference guide to all files in this package.

---

## 📖 START HERE

**First time?** Read these in order:

1. [`QUICK_START.md`](QUICK_START.md) - 5-minute quickstart guide
2. [`README.md`](README.md) - Main package documentation
3. [`PACKAGE_INFO.txt`](PACKAGE_INFO.txt) - One-page package summary

---

## 📁 All Files (16 total, 3.2 MB)

### 🗂️ Documentation (7 files)

| File | Purpose | Read When |
|------|---------|-----------|
| **QUICK_START.md** | 5-minute quickstart | You want to get started fast |
| **README.md** | Main documentation | You want full package overview |
| **PACKAGE_INFO.txt** | One-page summary | You want a quick reference |
| **README_REPRODUCIBILITY.md** | Detailed reproducibility guide | You want to reproduce experiments |
| **EXPERIMENT_SUMMARY.md** | Complete findings | You want scientific details |
| **REPRODUCIBILITY.md** | Config specifications | You need exact hyperparameters |
| **PLOTTING_GUIDE.md** | Manual plotting examples | You want to customize plots |

### 📊 Data Files (3 files, 3.05 MB)

| File | Size | Contents | Use For |
|------|------|----------|---------|
| **final_paper_results.json** | 5.6 KB | Aggregated results (mean±std) | Quick metrics lookup |
| **training_curves.json** | 443 KB | All training curves | Generating figures |
| **reproducibility_package.json** | 2.6 MB | Complete package | Full reproducibility |

### 💻 Code (4 files)

| File | Lines | Purpose |
|------|-------|---------|
| **plot_paper_figures.ipynb** | ~400 | **STANDALONE** figure generator (Colab-ready) |
| **unified_paper_experiment_v2.py** | ~800 | Main training script |
| **quick_aggregate.py** | ~250 | Results aggregation |
| **create_reproducibility_package.py** | ~450 | Package generation |

### 🔍 Metadata (2 files)

| File | Purpose |
|------|---------|
| **MANIFEST.txt** | Complete file inventory with descriptions |
| **CHECKSUMS.md5** | Data integrity verification |

---

## 🎯 Common Tasks

### Task: View Paper Results
```bash
cat final_paper_results.json | python -m json.tool
# OR
cat EXPERIMENT_SUMMARY.md  # Human-readable summary
```

**Files needed:** `final_paper_results.json` OR `EXPERIMENT_SUMMARY.md`

---

### Task: Generate All Figures
```bash
# Option 1: Google Colab (recommended)
# Upload to Colab: plot_paper_figures.ipynb, training_curves.json, final_paper_results.json
# Run all cells

# Option 2: Local
jupyter notebook plot_paper_figures.ipynb
```

**Files needed:** `plot_paper_figures.ipynb`, `training_curves.json`, `final_paper_results.json`

---

### Task: Reproduce Single Experiment
```bash
python unified_paper_experiment_v2.py \
    --experiment i1_002a \
    --seed 42 \
    --output_dir my_runs/i1_002a_seed42
```

**Files needed:** `unified_paper_experiment_v2.py`

---

### Task: Reproduce All Experiments
```bash
# See QUICK_START.md for full script
for exp in baseline_001 i1_001_1 i1_002_alibi i1_002a; do
    for seed in 42 43 44; do
        python unified_paper_experiment_v2.py \
            --experiment $exp --seed $seed \
            --output_dir my_runs/${exp}_seed${seed}
    done
done

# Then aggregate
python quick_aggregate.py --runs_dir my_runs
```

**Files needed:** `unified_paper_experiment_v2.py`, `quick_aggregate.py`

---

### Task: Create Custom Reproducibility Package
```bash
python create_reproducibility_package.py \
    --runs_dir my_runs \
    --output reproducibility_package_custom.json
```

**Files needed:** `create_reproducibility_package.py`, your run data

---

### Task: Verify Data Integrity
```bash
md5sum -c CHECKSUMS.md5
```

**Files needed:** `CHECKSUMS.md5`, all JSON data files

---

## 📋 Dependencies by Task

### Just Viewing Results
- **No dependencies** (plain text/JSON)

### Generating Figures
- Python 3.8+
- matplotlib
- seaborn
- numpy

### Reproducing Experiments
- Python 3.8+
- PyTorch 1.10+
- numpy
- tqdm

---

## 🔗 File Relationships

```
QUICK_START.md
    ├─→ README.md (for details)
    ├─→ plot_paper_figures.ipynb (for figures)
    │   ├─→ training_curves.json
    │   └─→ final_paper_results.json
    └─→ unified_paper_experiment_v2.py (for reproduction)

README_REPRODUCIBILITY.md
    ├─→ REPRODUCIBILITY.md (for configs)
    ├─→ EXPERIMENT_SUMMARY.md (for findings)
    └─→ reproducibility_package.json (for full data)

PLOTTING_GUIDE.md
    ├─→ training_curves.json (for manual plotting)
    └─→ final_paper_results.json (for metrics)
```

---

## 📊 Data File Contents

### final_paper_results.json
```json
{
  "experiment_name": {
    "eval_A": {"mean": 0.96, "std": 0.005, "values": [...]},
    "eval_B": {"mean": 0.737, "std": 0.007, "per_position": {...}},
    "eval_C0": {"mean": 0.805, "std": 0.03, "values": [...]},
    "eval_C1": {"mean": 0.945, "std": 0.022, "values": [...]}
  }
}
```

### training_curves.json
```json
{
  "experiment_name": {
    "seeds": {
      "seed_42": {
        "train": {"steps": [...], "accuracy": [...], "loss": [...]},
        "eval_A": [{"step": 200, "accuracy": 0.5}, ...],
        "eval_B": [{"step": 200, "accuracy": 0.3}, ...],
        "eval_C0": [{"step": 200, "accuracy": 0.1}, ...]
      }
    }
  }
}
```

### reproducibility_package.json
- Full experiment specifications
- All training curves
- Compute metrics (tokens, wall-time)
- Seeds documentation
- Complete configurations

---

## 🎓 For Different Audiences

### Reviewers
1. Start: `EXPERIMENT_SUMMARY.md`
2. Check: `final_paper_results.json`
3. Verify: `REPRODUCIBILITY.md`

### Researchers Wanting to Build On This
1. Start: `README_REPRODUCIBILITY.md`
2. Code: `unified_paper_experiment_v2.py`
3. Data: `reproducibility_package.json`

### Readers Wanting Figures
1. Start: `QUICK_START.md`
2. Use: `plot_paper_figures.ipynb`
3. Data: `training_curves.json` + `final_paper_results.json`

---

**Last Updated:** January 7, 2026
**Package Version:** 1.0

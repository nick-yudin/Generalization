# Latent Control (Paper 2)

**Title:** *Beyond Majority Voting: Selecting LLM Answers via Hidden State Trajectory Probes*

Author: Nikolay Yudin (n.yudin@gmail.com)  
Repo: https://github.com/nick-yudin/Generalization

## Contents

- `beyond_majority_voting.pdf` — the paper PDF (preprint).
- `paper2_release/` — reproducible release bundle:
  - `paper2_00_reproduce_figures.ipynb` (CPU) — regenerates all tables/figures from the canonical JSON.
  - `paper2_01_end_to_end.ipynb` (GPU) — end-to-end pipeline (requires HF access for Llama weights).
  - `paper2_02_ablations.ipynb` (CPU) — key diagnostics/ablations.
  - `paper2_utils.py` — shared utilities used by the notebooks.
  - `data/` — canonical results/ckpts used for reproduction.

## Notes

LaTeX sources and internal build scripts are intentionally not included in this public folder.

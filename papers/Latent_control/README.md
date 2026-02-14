# Latent Control (Paper 2)

**Title:** *The Model Already Knows Which Answer Is Better: Pairwise Probing of Hidden-State Trajectories for Best-of-K Selection*

Author: **Nikolay Yudin** (`n.yudin@gmail.com`)

Repository folder (public): `https://github.com/nick-yudin/Generalization/tree/main/papers/Latent_control`

This directory is reserved for the Paper 2 (Latent Control) release bundle:
- paper PDF/LaTeX sources
- standalone reproducibility notebooks
- small, non-sensitive aggregated artifacts (CSV/JSON) for figure regeneration

**Abstract**

Practitioners often sample multiple answers from a large language model (LLM) and select one via majority voting or an expensive external judge. We show that a frozen model’s own hidden-state trajectories provide a stronger, nearly free signal for best-of-K answer selection. We extract projected trajectory features (mean, standard deviation, and final-state summaries) from eight evenly spaced transformer layers over the generation, and train a single linear probe (logistic regression; ~6k parameters) with a pairwise ranking (RankNet) objective to score candidates. On TriviaQA with Llama-3.1-8B-Instruct (n=500, K=6), the probe improves lenient accuracy from 56.2% (majority vote) to 61.6% (+5.4 pp), recovering 55% of the oracle gap and achieving PickAcc 92.8%. On MATH (n=500, K=4), the probe consistently beats majority vote by ~1–2 pp (PickAcc ≈85%). We find that layer-wise informativeness forms a domain fingerprint (distributed for factual recall, late-layer concentrated for mathematical reasoning), yet a single probe trained on mixed-domain pairs matches domain-specific specialists. Finally, the probe remains effective after removing answer length as a feature, confirming it captures solution quality beyond superficial correlates. These results suggest that “the model already knows”: reading hidden states can substantially improve best-of-K selection at negligible cost.

**Paper Outline (short)**

1. Introduction
2. Method (trajectory features; pairwise probe; metrics)
3. Experiments (MATH; TriviaQA; baselines)
4. Analysis (layer signal; K-sweep; length confound; transfer/merge)
5. Related Work
6. Limitations
7. Conclusion

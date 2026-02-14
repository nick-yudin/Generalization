# Latent Control (Paper 2)

**Title:** *Beyond Majority Voting: Selecting LLM Answers via Hidden State Trajectory Probes*

Author: **Nikolay Yudin** (`n.yudin@gmail.com`)

Repository folder (public): `https://github.com/nick-yudin/Generalization/tree/main/papers/Latent_control`

This directory is reserved for the Paper 2 (Latent Control) release bundle:
- paper PDF/LaTeX sources
- standalone reproducibility notebooks
- small, non-sensitive aggregated artifacts (CSV/JSON) for figure regeneration

**Abstract**

When a language model generates multiple candidate answers, how should we pick the best one? The default strategy—majority voting—treats the model as a black box, discarding everything except final answer strings. We show that the model’s internal computations already contain a usable signal for answer quality, and that a remarkably simple method can extract it.

We propose *trajectory probes*: lightweight linear models trained on hidden-state features aggregated across the generation process. From each candidate answer, we extract mean, standard deviation, and final-token activations at eight evenly spaced layers, projected to 256 dimensions—a 6,144-dimensional trajectory fingerprint. A logistic regression probe trained with a pairwise ranking objective (RankNet) learns to prefer correct answers over incorrect ones from the same question.

On TriviaQA (Llama-3.1-8B-Instruct, $t{=}0.3$, $n{=}500$, $K{=}6$), the probe reaches 61.6\% accuracy versus 56.2\% for majority voting, recovering 55\% of the gap to the oracle upper bound, with PickAcc 92.8\% among questions that have at least one correct answer. On MATH ($n{=}500$, $K{=}4$), the probe consistently beats majority voting by $\sim$1–2 pp (PickAcc $\approx$85\%), reflecting the harder discrimination problem in mathematical reasoning. The probe itself trains in under 60 seconds on CPU given precomputed features, adds negligible overhead at inference, and requires no additional LLM calls.

Two findings surprised us. First, the choice of training objective matters more than classification quality: a binary classifier with higher cross-validated AUC can underperform a pairwise probe with lower AUC, because ranking among candidates is a fundamentally different task than classifying correctness in isolation. Second, the per-layer signal distribution acts as a domain fingerprint—factual recall spreads information across layers while mathematical reasoning concentrates it late—yet a single probe trained on mixed-domain data can match domain-specific specialists with no interference.

Our results suggest that the “verifier” for best-of-$K$ selection need not be a separate model. It can be a linear function of what the model already computes.

**Paper Outline (short)**

1. Introduction
2. Method (trajectory features; pairwise probe; metrics)
3. Experiments (MATH; TriviaQA; baselines)
4. Analysis (layer signal; K-sweep; length confound; transfer/merge)
5. Related Work
6. Limitations
7. Conclusion

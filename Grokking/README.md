# Connection to Grokking Phenomenon

Recent work on "grokking" (Power et al., 2022; Nanda et al., 2023) shows that procedural solutions can emerge **spontaneously** after prolonged training, but this:
- Requires massive compute (10⁵-10⁶ steps)
- Is unpredictable (no early indicators)
- Occurs only under specific conditions (e.g., high weight decay)
- Works only on clean algorithmic data

**Our work asks**: Can we **steer** models toward procedural solutions proactively, rather than waiting for spontaneous emergence?

**Key Insight**: We leverage grokking insights (role of weight decay, rank compression, three-phase dynamics) but transform them into **actionable training interventions** that work under **realistic noise**.

**Bridge**:
- Grokking = **passive observation** of delayed generalization
- This work = **active steering** toward procedural solutions
- Grokking = clean data, long wait
- This work = noisy data, early intervention

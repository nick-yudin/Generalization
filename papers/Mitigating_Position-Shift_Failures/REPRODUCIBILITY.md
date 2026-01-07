# Paper Reproducibility Package

Complete configuration and results for all experiments.

## Experiments Overview

### baseline_001

**Configuration:**
- K-variants: 1
- Position diversity: False
- Template diversity: False
- Anchors: False
- ALiBi: False
- Consistency loss: False

**Results (Mean ± Std):**
- Eval-A: 96.8 ± 4.2%
- Eval-B: 14.9 ± 0.5%
- Eval-C0: 1.2 ± 0.8%

**Compute:**
- Steps: 5000
- Tokens: 123,792,000
- Wall time: 4.8 ± 0.0 min

### i1_001_1

**Configuration:**
- K-variants: 4
- Position diversity: True
- Template diversity: False
- Anchors: False
- ALiBi: False
- Consistency loss: True

**Results (Mean ± Std):**
- Eval-A: 96.5 ± 0.9%
- Eval-B: 71.7 ± 0.6%
- Eval-C0: 60.3 ± 6.3%

**Compute:**
- Steps: 5000
- Tokens: 495,168,000
- Wall time: 18.2 ± 0.0 min

### i1_002_alibi

**Configuration:**
- K-variants: 4
- Position diversity: True
- Template diversity: True
- Anchors: True
- ALiBi: True
- Consistency loss: True

**Results (Mean ± Std):**
- Eval-A: 21.4 ± 1.0%
- Eval-B: 34.3 ± 3.0%
- Eval-C0: 15.5 ± 2.2%
- Eval-C1: 34.5 ± 3.5%

**Compute:**
- Steps: 5000
- Tokens: 495,168,000
- Wall time: 18.2 ± 0.0 min

### i1_002a

**Configuration:**
- K-variants: 4
- Position diversity: True
- Template diversity: True
- Anchors: True
- ALiBi: False
- Consistency loss: True

**Results (Mean ± Std):**
- Eval-A: 96.0 ± 0.5%
- Eval-B: 73.7 ± 0.7%
- Eval-C0: 80.5 ± 3.0%
- Eval-C1: 94.5 ± 2.2%

**Compute:**
- Steps: 5000
- Tokens: 495,168,000
- Wall time: 18.4 ± 0.0 min

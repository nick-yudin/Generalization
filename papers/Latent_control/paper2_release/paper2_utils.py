"""
Shared utilities for:
  "Beyond Majority Voting: Selecting LLM Answers via Hidden State Trajectory Probes"
  Nikolay Yudin, 2025

Functions used by paper2_00, paper2_01, and paper2_02 notebooks.
"""

import re
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ═══════════════════════════════════════════════════════════════════════
#  Random Projection & Layer Selection
# ═══════════════════════════════════════════════════════════════════════

def stable_random_projection(d_in, d_out, seed=0):
    """Deterministic Gaussian random projection matrix (d_in → d_out)."""
    rng = np.random.RandomState(seed)
    return (rng.randn(d_in, d_out) / np.sqrt(d_out)).astype(np.float32)


def resolve_layer_ids(name, n_layers):
    """Resolve named layer subset to integer indices.

    'spread8' — 8 evenly spaced layers (default for the paper).
    'top8'    — last 8 layers.
    'all'     — all layers.
    """
    if name == 'spread8':
        return [int(round(i * (n_layers - 1) / 7)) for i in range(8)]
    elif name == 'top8':
        return list(range(n_layers - 8, n_layers))
    elif name == 'all':
        return list(range(n_layers))
    else:
        raise ValueError(f"Unknown layer subset: {name}")


def get_layer_indices(layer_ids, n_layers, proj_dim):
    """Feature indices for selected layers from the full [mean|std|last] vector.

    The full vector is laid out as:
      [mean_layer0, mean_layer1, ..., std_layer0, ..., last_layer0, ...]
    Each block has n_layers × proj_dim entries.
    """
    idx = []
    for block_start in [0, n_layers * proj_dim, 2 * n_layers * proj_dim]:
        for l in layer_ids:
            s = block_start + l * proj_dim
            idx.extend(range(s, s + proj_dim))
    return np.array(idx)


# ═══════════════════════════════════════════════════════════════════════
#  Trajectory Features
# ═══════════════════════════════════════════════════════════════════════

def trajectory_features(hs, min_tokens=3):
    """Aggregate per-token hidden states into mean ⊕ std ⊕ last vector.

    Args:
        hs: array of shape (n_tokens, feat_dim_all) — projected hidden states.
        min_tokens: minimum tokens for a valid feature vector.

    Returns:
        1-D array of shape (3 × feat_dim_all,) or None if too few tokens.
    """
    if hs.shape[0] < min_tokens:
        return None
    return np.concatenate([hs.mean(0), hs.std(0), hs[-1]])


# ═══════════════════════════════════════════════════════════════════════
#  TriviaQA Helpers
# ═══════════════════════════════════════════════════════════════════════

def normalize_answer_text(s):
    """Lowercase, strip articles and punctuation (TriviaQA standard)."""
    s = (s or '').strip().lower()
    s = re.sub(r'[^0-9a-z]+', ' ', s)
    s = ' '.join(s.split())
    s = re.sub(r'^(a|an|the)\s+', '', s)
    return s


def trivia_f1(pred, aliases):
    """Token-level F1 between prediction and best alias."""
    p_toks = normalize_answer_text(pred).split()
    if not p_toks:
        return 0.0
    best = 0.0
    for a in aliases:
        g_toks = normalize_answer_text(a).split()
        if not g_toks:
            continue
        common = sum(min(p_toks.count(t), g_toks.count(t)) for t in set(p_toks))
        if common == 0:
            continue
        prec, rec = common / len(p_toks), common / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec + 1e-12))
    return best


def trivia_is_correct(pred, aliases):
    """TriviaQA correctness: exact match OR token F1 ≥ 0.8."""
    if not pred or not aliases:
        return False
    p = normalize_answer_text(pred)
    if any(p == normalize_answer_text(a) for a in aliases):
        return True
    return trivia_f1(pred, aliases) >= 0.8


def extract_final_answer(text):
    """Extract final answer from TriviaQA generation (FINAL: pattern)."""
    ms = re.findall(r'FINAL:\s*([^\n\r]*)', text, flags=re.IGNORECASE)
    if ms:
        s = (ms[-1] or '').strip()
        for sep in ['Human:', 'Assistant:', 'FINAL:', 'Question:']:
            if sep in s:
                s = s.split(sep, 1)[0].strip()
        if s:
            return s
    m = re.search(r'[Tt]he answer (?:is|was)\s+(.+?)\.?\s*$', text, re.MULTILINE)
    if m:
        return m.group(1).strip().strip('\'\".')
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1].strip('\'\".')[:200] if lines else ''


def get_trivia_aliases(answer_dict):
    """Extract all answer aliases from TriviaQA answer dict."""
    aliases = []
    if answer_dict.get('value'):
        aliases.append(answer_dict['value'])
    aliases.extend(answer_dict.get('aliases', []))
    if answer_dict.get('normalized_value'):
        aliases.append(answer_dict['normalized_value'])
    aliases.extend(answer_dict.get('normalized_aliases', []))
    return [a for a in aliases if a]


# ═══════════════════════════════════════════════════════════════════════
#  MATH Helpers
# ═══════════════════════════════════════════════════════════════════════

def extract_boxed(text):
    """Extract content from \\boxed{} in MATH answers (last occurrence)."""
    results = []
    i = 0
    while i < len(text):
        if text[i:i+7] == '\\boxed{':
            start = i + 7
            depth = 1
            j = start
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            results.append(text[start:j-1])
            i = j
        else:
            i += 1
    return results[-1].strip() if results else None


def normalize_math(s):
    """Normalize MATH answer string for comparison."""
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r'\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}', r'\1', s)
    s = s.replace('\\,', '').replace('\\!', '').replace('\\;', '')
    s = s.replace(' ', '').replace('$', '').rstrip('.')
    return s


def check_correct_math(pred, gold):
    """MATH correctness: normalized exact match."""
    if pred is None or gold is None:
        return False
    return normalize_math(pred) == normalize_math(gold)


# ═══════════════════════════════════════════════════════════════════════
#  Probe Training
# ═══════════════════════════════════════════════════════════════════════

def build_pairwise_pairs(train_data, feat_idx, k_train):
    """Build (x_correct − x_wrong, 1) and (x_wrong − x_correct, 0) pairs."""
    X_pairs, y_pairs = [], []
    for q in train_data:
        atts = q['attempts'][:k_train]
        correct = [a['features'][feat_idx] for a in atts
                   if a['correct'] and a.get('feat_valid', True)]
        wrong = [a['features'][feat_idx] for a in atts
                 if not a['correct'] and a.get('feat_valid', True)]
        for xc in correct:
            for xw in wrong:
                X_pairs.append(xc - xw)
                y_pairs.append(1)
                X_pairs.append(xw - xc)
                y_pairs.append(0)
    return np.array(X_pairs, dtype=np.float32), np.array(y_pairs)


def train_pairwise_probe(train_data, feat_idx, k_train, seed=0,
                         C_candidates=(0.001, 0.01, 0.1, 1.0)):
    """Train pairwise ranking probe with cross-validated C selection.

    Returns: (probe, best_C, best_cv_auc, n_pairs)
    """
    X_pairs, y_pairs = build_pairwise_pairs(train_data, feat_idx, k_train)
    best_c, best_auc = C_candidates[0], 0
    for C in C_candidates:
        cv = cross_val_score(
            LogisticRegression(C=C, solver='lbfgs', max_iter=3000,
                               random_state=seed),
            X_pairs, y_pairs, cv=5, scoring='roc_auc').mean()
        if cv > best_auc:
            best_auc, best_c = cv, C

    probe = LogisticRegression(C=best_c, solver='lbfgs', max_iter=3000,
                               random_state=seed)
    probe.fit(X_pairs, y_pairs)
    return probe, best_c, best_auc, len(X_pairs)


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════

def majority_vote(attempts, domain='trivia'):
    """Determine MV correctness for a single question's attempts."""
    if domain == 'trivia':
        preds = [normalize_answer_text(a.get('pred', '')) for a in attempts]
        preds = [p for p in preds if p]
    else:  # math
        preds = [a.get('pred', '') or '' for a in attempts]
        preds = [p for p in preds if p]

    if not preds:
        return attempts[0]['correct']

    vote = Counter(preds).most_common(1)[0][0]
    if domain == 'trivia':
        return any(a['correct'] for a in attempts
                   if normalize_answer_text(a.get('pred', '')) == vote)
    else:
        return any(a['correct'] for a in attempts
                   if (a.get('pred', '') or '') == vote)


def evaluate_at_k(test_data, probe, feat_idx, k_eval, domain='trivia'):
    """Evaluate probe vs MV at given K_eval.

    Returns dict with base/mv/probe/oracle accuracy, PickAcc, Recovery,
    head-to-head counts, and per-question test_ckpt.
    """
    N = len(test_data)
    n_base = n_oracle = n_mv = n_probe = 0
    n_oracle_total = n_picked_correct = 0
    test_ckpt = []

    for q in test_data:
        atts = q['attempts'][:k_eval]
        n_base += int(atts[0]['correct'])
        has_correct = any(a['correct'] for a in atts)
        n_oracle += int(has_correct)

        mv_corr = majority_vote(atts, domain)
        n_mv += int(mv_corr)

        # Probe scoring
        scores = []
        for a in atts:
            if a.get('feat_valid', True) and a.get('features') is not None:
                s = float(probe.decision_function(
                    a['features'][feat_idx].reshape(1, -1))[0])
            else:
                s = -999.0
            scores.append(s)
        best_idx = int(np.argmax(scores))
        probe_corr = atts[best_idx]['correct']
        n_probe += int(probe_corr)
        if has_correct:
            n_oracle_total += 1
            n_picked_correct += int(probe_corr)

        test_ckpt.append({
            'attempts': [{'correct': a['correct'], 'pred': a.get('pred', ''),
                          'n_tok': a.get('n_tok', 0),
                          'score_pairwise': scores[i]}
                         for i, a in enumerate(atts)],
            'mv_correct': bool(mv_corr),
            'probe_correct': bool(probe_corr),
        })

    base_acc = n_base / N
    oracle_acc = n_oracle / N
    mv_acc = n_mv / N
    probe_acc = n_probe / N
    pickacc = n_picked_correct / max(n_oracle_total, 1) * 100
    recovery = ((probe_acc - base_acc) / (oracle_acc - base_acc) * 100
                if oracle_acc > base_acc else 0)

    both_r = sum(1 for t in test_ckpt if t['probe_correct'] and t['mv_correct'])
    p_only = sum(1 for t in test_ckpt
                 if t['probe_correct'] and not t['mv_correct'])
    m_only = sum(1 for t in test_ckpt
                 if not t['probe_correct'] and t['mv_correct'])
    both_w = sum(1 for t in test_ckpt
                 if not t['probe_correct'] and not t['mv_correct'])

    return {
        'k_eval': k_eval,
        'base_acc': base_acc, 'mv_acc': mv_acc,
        'probe_acc': probe_acc, 'oracle_acc': oracle_acc,
        'pickacc': pickacc, 'recovery': recovery,
        'h2h': {'both_right': both_r, 'probe_only': p_only,
                'mv_only': m_only, 'both_wrong': both_w},
        'test_ckpt': test_ckpt,
    }


def print_eval_table(results_by_k, domain, seed, k_gen, k_train):
    """Pretty-print multi-K evaluation table."""
    print(f'\n{"="*70}')
    print(f'SEED={seed} {domain}  (K_gen={k_gen}, K_train={k_train})')
    print(f'{"="*70}')
    hdr = (f'{"K":>4} {"Base":>6} {"MV":>6} {"Probe":>6} '
           f'{"D(P-MV)":>8} {"Oracle":>7} {"PickAcc":>8} {"Recov":>6}  '
           f'{"H2H(P/M)":>9}')
    print(hdr)
    print('-' * 70)
    for ke, r in sorted(results_by_k.items()):
        delta = (r['probe_acc'] - r['mv_acc']) * 100
        print(f"{ke:>4} {r['base_acc']:>6.1%} {r['mv_acc']:>6.1%} "
              f"{r['probe_acc']:>6.1%} {delta:>+7.1f}pp "
              f"{r['oracle_acc']:>6.1%} {r['pickacc']:>7.1f}% "
              f"{r['recovery']:>5.1f}%  "
              f"{r['h2h']['probe_only']:>3}/{r['h2h']['mv_only']:<3}")


# ═══════════════════════════════════════════════════════════════════════
#  Statistical Tests
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_delta(test_ckpt, n_boot=10000, seed=42):
    """Bootstrap CI for Δ(Probe − MV) accuracy.

    Returns dict with delta, 95% CI bounds, and P(Δ > 0).
    """
    rng = np.random.RandomState(seed)
    n = len(test_ckpt)
    probe = np.array([t['probe_correct'] for t in test_ckpt])
    mv = np.array([t['mv_correct'] for t in test_ckpt])

    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        deltas.append(probe[idx].mean() - mv[idx].mean())
    deltas = np.array(deltas)

    return {
        'delta': float(probe.mean() - mv.mean()),
        'ci_lo': float(np.percentile(deltas, 2.5)),
        'ci_hi': float(np.percentile(deltas, 97.5)),
        'p_positive': float((deltas > 0).mean()),
        'n_boot': n_boot,
    }


def mcnemar_test(test_ckpt):
    """McNemar test (two-sided binomial) on Probe vs MV discordant pairs."""
    from scipy.stats import binomtest

    probe_only = sum(1 for t in test_ckpt
                     if t['probe_correct'] and not t['mv_correct'])
    mv_only = sum(1 for t in test_ckpt
                  if not t['probe_correct'] and t['mv_correct'])
    n_discordant = probe_only + mv_only

    if n_discordant == 0:
        return {'p_value': 1.0, 'probe_only': 0, 'mv_only': 0,
                'n_discordant': 0}

    result = binomtest(probe_only, n_discordant, 0.5)
    return {
        'p_value': float(result.pvalue),
        'probe_only': probe_only,
        'mv_only': mv_only,
        'n_discordant': n_discordant,
    }


def print_stats(test_ckpt, label=''):
    """Print bootstrap CI + McNemar for a test checkpoint."""
    bs = bootstrap_delta(test_ckpt)
    mc = mcnemar_test(test_ckpt)
    print(f'\n--- Statistics{" (" + label + ")" if label else ""} ---')
    print(f'  Δ(Probe−MV) = {bs["delta"]*100:+.1f}pp  '
          f'95% CI [{bs["ci_lo"]*100:+.1f}, {bs["ci_hi"]*100:+.1f}]pp  '
          f'P(Δ>0) = {bs["p_positive"]:.1%}')
    print(f'  McNemar: {mc["probe_only"]} probe wins vs {mc["mv_only"]} MV wins  '
          f'(p = {mc["p_value"]:.2e})')
    return bs, mc

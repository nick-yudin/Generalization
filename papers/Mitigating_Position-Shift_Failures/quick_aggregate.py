#!/usr/bin/env python3
"""Quick aggregation of paper results without external dependencies."""

import json
import statistics
from pathlib import Path

def load_results(results_dir, exp_name, seeds):
    """Load results from all seeds."""
    data = []
    for seed in seeds:
        exp_dir = results_dir / f"{exp_name}_seed{seed}"
        eval_path = exp_dir / "final_eval.json"

        if not eval_path.exists():
            print(f"⚠️  Missing: {eval_path}")
            continue

        with open(eval_path) as f:
            data.append(json.load(f))

    return data

def aggregate_metric(values):
    """Calculate mean ± std."""
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std

def aggregate_experiment(results_dir, exp_name, seeds):
    """Aggregate results for one experiment."""
    data = load_results(results_dir, exp_name, seeds)

    if not data:
        return None

    # Eval-A
    eval_a_acc = [d['evalA']['acc'] for d in data]
    eval_a_mean, eval_a_std = aggregate_metric(eval_a_acc)

    # ConsistencyCorrect@4
    cc4_values = [d['evalA'].get('cc4') for d in data if 'cc4' in d['evalA']]
    cc4_mean, cc4_std = aggregate_metric(cc4_values) if cc4_values else (None, None)

    # Eval-B
    eval_b_avg = [d['evalB']['avg'] for d in data]
    eval_b_mean, eval_b_std = aggregate_metric(eval_b_avg)

    # Eval-B per-position
    positions = [0, 8, 16, 24, 32, 48, 64]
    pos_results = {}
    for pos in positions:
        pos_key = str(pos)
        pos_values = [d['evalB']['by_pos'].get(pos_key) for d in data if pos_key in d['evalB']['by_pos']]
        if pos_values:
            pos_mean, pos_std = aggregate_metric(pos_values)
            pos_results[pos] = (pos_mean, pos_std)

    # Eval-C0 (no-anchor)
    eval_c0_acc = [d['evalC']['no_anchor']['acc'] for d in data]
    eval_c0_mean, eval_c0_std = aggregate_metric(eval_c0_acc)

    # Eval-C1 (anchor)
    eval_c1_values = [d['evalC']['anchor']['acc'] for d in data if d['evalC']['anchor'] is not None]
    eval_c1_mean, eval_c1_std = aggregate_metric(eval_c1_values) if eval_c1_values else (None, None)

    return {
        'experiment': exp_name,
        'n_seeds': len(data),
        'eval_A': {
            'mean': eval_a_mean,
            'std': eval_a_std,
            'values': eval_a_acc,
        },
        'cc4': {
            'mean': cc4_mean,
            'std': cc4_std,
            'values': cc4_values,
        } if cc4_mean is not None else None,
        'eval_B': {
            'mean': eval_b_mean,
            'std': eval_b_std,
            'values': eval_b_avg,
            'per_position': pos_results,
        },
        'eval_C0': {
            'mean': eval_c0_mean,
            'std': eval_c0_std,
            'values': eval_c0_acc,
        },
        'eval_C1': {
            'mean': eval_c1_mean,
            'std': eval_c1_std,
            'values': eval_c1_values,
        } if eval_c1_mean is not None else None,
    }

def format_table(results):
    """Generate comparison table."""
    lines = []
    lines.append("# Paper Results Summary (Mean ± Std)")
    lines.append("")
    lines.append("## Main Metrics")
    lines.append("")
    lines.append("| Experiment | Eval-A | Eval-B | Eval-C0 (No-Anchor) | Eval-C1 (Anchor) | CC@4 |")
    lines.append("|------------|--------|--------|---------------------|------------------|------|")

    for exp_name in ['baseline_001', 'i1_001_1', 'i1_002_alibi', 'i1_002a']:
        if exp_name not in results or results[exp_name] is None:
            continue

        r = results[exp_name]

        eval_a = f"{r['eval_A']['mean']*100:.1f}±{r['eval_A']['std']*100:.1f}"
        eval_b = f"{r['eval_B']['mean']*100:.1f}±{r['eval_B']['std']*100:.1f}"
        eval_c0 = f"{r['eval_C0']['mean']*100:.1f}±{r['eval_C0']['std']*100:.1f}"
        eval_c1 = f"{r['eval_C1']['mean']*100:.1f}±{r['eval_C1']['std']*100:.1f}" if r['eval_C1'] else "N/A"
        cc4 = f"{r['cc4']['mean']*100:.1f}±{r['cc4']['std']*100:.1f}" if r['cc4'] else "N/A"

        lines.append(f"| {exp_name} | {eval_a}% | {eval_b}% | {eval_c0}% | {eval_c1}% | {cc4}% |")

    lines.append("")
    lines.append("## Eval-B Position Breakdown")
    lines.append("")
    lines.append("| Experiment | Pos 0 | Pos 8 | Pos 16 | Pos 24 | Pos 32 | Pos 48 | Pos 64 |")
    lines.append("|------------|-------|-------|--------|--------|--------|--------|--------|")

    for exp_name in ['baseline_001', 'i1_001_1', 'i1_002_alibi', 'i1_002a']:
        if exp_name not in results or results[exp_name] is None:
            continue

        r = results[exp_name]
        row = [exp_name]

        for pos in [0, 8, 16, 24, 32, 48, 64]:
            if pos in r['eval_B']['per_position']:
                mean, std = r['eval_B']['per_position'][pos]
                row.append(f"{mean*100:.1f}±{std*100:.1f}")
            else:
                row.append("N/A")

        lines.append(f"| {' | '.join(row)} |")

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")

    # Baseline catastrophic cliff
    baseline = results.get('baseline_001')
    if baseline:
        pos_0 = baseline['eval_B']['per_position'][0][0]
        pos_64 = baseline['eval_B']['per_position'][64][0]
        cliff = pos_0 - pos_64
        lines.append(f"### Baseline Catastrophic Cliff:")
        lines.append(f"- Position 0: {pos_0*100:.1f}%")
        lines.append(f"- Position 64: {pos_64*100:.1f}%")
        lines.append(f"- **Drop: {cliff*100:.1f} percentage points**")
        lines.append("")

    # I1_001_1 improvement
    i1_001_1 = results.get('i1_001_1')
    if i1_001_1 and baseline:
        lines.append(f"### I1_001_1 vs Baseline:")
        lines.append(f"- Eval-A: {i1_001_1['eval_A']['mean']*100:.1f}% vs {baseline['eval_A']['mean']*100:.1f}%")
        lines.append(f"- Eval-B: {i1_001_1['eval_B']['mean']*100:.1f}% vs {baseline['eval_B']['mean']*100:.1f}%")
        lines.append(f"- Eval-C0: {i1_001_1['eval_C0']['mean']*100:.1f}% vs {baseline['eval_C0']['mean']*100:.1f}%")
        lines.append(f"- **Eval-B improvement: +{(i1_001_1['eval_B']['mean'] - baseline['eval_B']['mean'])*100:.1f} pp**")
        lines.append("")

    # I1_002a (main result)
    i1_002a = results.get('i1_002a')
    if i1_002a and baseline:
        lines.append(f"### I1_002a (Main Result) vs Baseline:")
        lines.append(f"- Eval-A: {i1_002a['eval_A']['mean']*100:.1f}% vs {baseline['eval_A']['mean']*100:.1f}%")
        lines.append(f"- Eval-B: {i1_002a['eval_B']['mean']*100:.1f}% vs {baseline['eval_B']['mean']*100:.1f}%")
        lines.append(f"- Eval-C0: {i1_002a['eval_C0']['mean']*100:.1f}% vs {baseline['eval_C0']['mean']*100:.1f}%")
        lines.append(f"- Eval-C1 (with anchors): {i1_002a['eval_C1']['mean']*100:.1f}%")
        lines.append(f"- **Eval-B improvement: +{(i1_002a['eval_B']['mean'] - baseline['eval_B']['mean'])*100:.1f} pp**")
        lines.append(f"- **Eval-C0 improvement: +{(i1_002a['eval_C0']['mean'] - baseline['eval_C0']['mean'])*100:.1f} pp**")
        lines.append("")

    # ALiBi failure
    alibi = results.get('i1_002_alibi')
    if alibi:
        lines.append(f"### I1_002_ALIBI (Negative Result):")
        lines.append(f"- Eval-A: {alibi['eval_A']['mean']*100:.1f}% ❌")
        lines.append(f"- **Complete failure to learn the task**")
        lines.append(f"- ALiBi incompatible with position curriculum + anchors")
        lines.append("")

    return "\n".join(lines)

def main():
    results_dir = Path("paper_runs")
    experiments = ['baseline_001', 'i1_001_1', 'i1_002_alibi', 'i1_002a']
    seeds = [42, 43, 44]

    results = {}

    print("Aggregating results...\n")

    for exp_name in experiments:
        print(f"Processing {exp_name}...")
        agg = aggregate_experiment(results_dir, exp_name, seeds)
        if agg:
            results[exp_name] = agg
            print(f"  ✅ Eval-A: {agg['eval_A']['mean']*100:.1f}±{agg['eval_A']['std']*100:.1f}%")
            print(f"  ✅ Eval-B: {agg['eval_B']['mean']*100:.1f}±{agg['eval_B']['std']*100:.1f}%")
            print(f"  ✅ Eval-C0: {agg['eval_C0']['mean']*100:.1f}±{agg['eval_C0']['std']*100:.1f}%")
        else:
            print(f"  ❌ Failed")
        print()

    # Save JSON
    with open('final_paper_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Saved: final_paper_results.json\n")

    # Generate table
    table = format_table(results)
    with open('paper_comparison_table.md', 'w') as f:
        f.write(table)
    print("✅ Saved: paper_comparison_table.md\n")

    print("="*80)
    print(table)
    print("="*80)

if __name__ == '__main__':
    main()

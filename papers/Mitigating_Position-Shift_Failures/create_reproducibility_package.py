#!/usr/bin/env python3
"""
Create complete reproducibility package for the paper.

Includes:
1. Experiment configs (architecture, optimizer, curriculum)
2. Split/eval seeds
3. Training curves (all steps, all seeds)
4. Compute metrics (steps, wall-clock, tokens)
5. Final aggregated results
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any


def load_experiment_config(exp_dir: Path) -> Dict[str, Any]:
    """Load config.json from experiment directory."""
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        return json.load(f)


def load_training_curves(exp_dir: Path) -> List[Dict[str, Any]]:
    """Load full training curves from metrics.jsonl."""
    metrics_path = exp_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []

    curves = []
    with open(metrics_path) as f:
        for line in f:
            if line.strip():
                curves.append(json.loads(line))

    return curves


def extract_experiment_spec(exp_name: str) -> Dict[str, Any]:
    """
    Extract canonical experiment specification.

    This defines what makes each experiment unique and reproducible.
    """
    # Common base config
    base = {
        "model": {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "vocab_size": 80,
            "max_length": 100,
            "pooling": "cls",
        },
        "training": {
            "max_steps": 5000,
            "batch_size": 256,
            "lr": 0.001,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
        },
        "task": {
            "p": 97,
            "train_fraction": 0.5,
        },
        "curriculum": {
            "type": "steps_based",
            "early": {"steps": "0-1666", "pos_range": [10, 30]},
            "mid": {"steps": "1667-3333", "pos_range": [10, 50]},
            "late": {"steps": "3334-5000", "pos_range": [10, 70]},
        },
    }

    # Experiment-specific modifications
    if exp_name == "baseline_001":
        spec = {
            **base,
            "intervention": {
                "K": 1,
                "position_diversity": False,
                "template_diversity": False,
                "use_anchors": False,
                "use_alibi": False,
                "consistency_loss": False,
            },
            "curriculum": None,  # No curriculum for baseline
        }

    elif exp_name == "i1_001_1":
        spec = {
            **base,
            "intervention": {
                "K": 4,
                "position_diversity": True,
                "template_diversity": False,
                "use_anchors": False,
                "use_alibi": False,
                "consistency_loss": True,
                "consistency_lambda": 1.0,
            },
        }

    elif exp_name == "i1_002_alibi":
        spec = {
            **base,
            "intervention": {
                "K": 4,
                "position_diversity": True,
                "template_diversity": True,
                "use_anchors": True,
                "use_alibi": True,
                "consistency_loss": True,
                "consistency_lambda": 1.0,
            },
        }
        # ALiBi uses different positional encoding
        spec["model"]["positional_encoding"] = "alibi"

    elif exp_name == "i1_002a":
        spec = {
            **base,
            "intervention": {
                "K": 4,
                "position_diversity": True,
                "template_diversity": True,
                "use_anchors": True,
                "use_alibi": False,
                "consistency_loss": True,
                "consistency_lambda": 1.0,
            },
        }
        spec["model"]["positional_encoding"] = "learned"

    return spec


def extract_seeds_and_sampling(seeds: List[int]) -> Dict[str, Any]:
    """
    Document all seeds used for reproducibility.
    """
    return {
        "experiment_seeds": seeds,
        "data_split": {
            "method": "disjoint_50_50",
            "seed": "experiment_seed",  # Same as experiment seed
            "note": "All pairs shuffled with seed, then split at 50%",
        },
        "eval_sampling": {
            "eval_A": {
                "n": 400,
                "method": "random_sample_from_test_set",
                "seed": "fixed_42",
                "note": "Same 400 pairs for all experiments",
            },
            "eval_B": {
                "positions": [0, 8, 16, 24, 32, 48, 64],
                "n_per_position": 100,
                "method": "random_generation",
                "seed": "random_per_call",
                "note": "100 random (a,b) pairs generated per position",
            },
            "eval_C": {
                "C0_no_anchor": {
                    "n": 200,
                    "method": "random_generation",
                    "categories": ["questions", "commands"],
                    "seed": "random_per_call",
                },
                "C1_anchor": {
                    "n": 200,
                    "method": "random_generation",
                    "categories": ["questions", "commands"],
                    "seed": "random_per_call",
                    "note": "Only for experiments with use_anchors=True",
                },
            },
        },
    }


def aggregate_metric(values: List[float]) -> Dict[str, float]:
    """Calculate mean ± std."""
    if not values:
        return {"mean": None, "std": None, "values": []}

    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def process_experiment(results_dir: Path, exp_name: str, seeds: List[int]) -> Dict[str, Any]:
    """
    Process all runs for one experiment and create complete package.
    """
    print(f"\nProcessing {exp_name}...")

    # 1. Experiment specification
    spec = extract_experiment_spec(exp_name)

    # 2. Collect data from all seeds
    runs = []

    for seed in seeds:
        exp_dir = results_dir / f"{exp_name}_seed{seed}"

        if not exp_dir.exists():
            print(f"  ⚠️  Missing: {exp_dir}")
            continue

        # Load config
        config = load_experiment_config(exp_dir)
        if not config:
            print(f"  ⚠️  No config: {exp_dir}")
            continue

        # Load final eval
        final_eval_path = exp_dir / "final_eval.json"
        if not final_eval_path.exists():
            print(f"  ⚠️  No final_eval: {exp_dir}")
            continue

        with open(final_eval_path) as f:
            final_eval = json.load(f)

        # Load training curves
        curves = load_training_curves(exp_dir)

        # Extract compute metrics
        compute = {
            "seed": seed,
            "final_step": final_eval.get("final_step"),
            "tokens_processed": final_eval.get("tokens_processed"),
            "wall_time_seconds": final_eval.get("wall_time_seconds"),
            "wall_time_minutes": final_eval.get("wall_time_seconds", 0) / 60.0,
        }

        run_data = {
            "seed": seed,
            "config": config,
            "final_eval": final_eval,
            "compute": compute,
            "training_curves": curves,
        }

        runs.append(run_data)
        print(f"  ✅ Seed {seed}: {len(curves)} curve points")

    if not runs:
        return None

    # 3. Aggregate final results
    aggregated_results = aggregate_final_results(runs)

    # 4. Aggregate compute metrics
    compute_stats = {
        "steps": aggregate_metric([r["compute"]["final_step"] for r in runs]),
        "tokens_processed": aggregate_metric([r["compute"]["tokens_processed"] for r in runs]),
        "wall_time_seconds": aggregate_metric([r["compute"]["wall_time_seconds"] for r in runs]),
        "wall_time_minutes": aggregate_metric([r["compute"]["wall_time_minutes"] for r in runs]),
    }

    return {
        "experiment": exp_name,
        "specification": spec,
        "seeds": seeds,
        "n_runs": len(runs),
        "runs": runs,
        "aggregated_results": aggregated_results,
        "compute_stats": compute_stats,
    }


def aggregate_final_results(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate final evaluation results across seeds."""

    # Eval-A
    eval_a_acc = [r["final_eval"]["evalA"]["acc"] for r in runs]
    eval_a_agg = aggregate_metric(eval_a_acc)

    cc4_values = [r["final_eval"]["evalA"].get("cc4") for r in runs if "cc4" in r["final_eval"]["evalA"]]
    cc4_agg = aggregate_metric(cc4_values) if cc4_values else None

    # Eval-B
    eval_b_avg = [r["final_eval"]["evalB"]["avg"] for r in runs]
    eval_b_agg = aggregate_metric(eval_b_avg)

    # Eval-B per-position
    positions = [0, 8, 16, 24, 32, 48, 64]
    per_pos = {}
    for pos in positions:
        pos_key = str(pos)
        pos_values = [r["final_eval"]["evalB"]["by_pos"].get(pos_key)
                      for r in runs if pos_key in r["final_eval"]["evalB"]["by_pos"]]
        if pos_values:
            per_pos[f"pos_{pos}"] = aggregate_metric(pos_values)

    # Eval-C0 (no-anchor)
    eval_c0_acc = [r["final_eval"]["evalC"]["no_anchor"]["acc"] for r in runs]
    eval_c0_agg = aggregate_metric(eval_c0_acc)

    # Eval-C1 (anchor, if available)
    eval_c1_values = [r["final_eval"]["evalC"]["anchor"]["acc"]
                      for r in runs if r["final_eval"]["evalC"]["anchor"] is not None]
    eval_c1_agg = aggregate_metric(eval_c1_values) if eval_c1_values else None

    return {
        "eval_A": {
            "accuracy": eval_a_agg,
            "consistency_correct_4": cc4_agg,
        },
        "eval_B": {
            "average": eval_b_agg,
            "per_position": per_pos,
        },
        "eval_C": {
            "C0_no_anchor": eval_c0_agg,
            "C1_anchor": eval_c1_agg,
        },
    }


def create_training_curves_summary(all_experiments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training curves for plotting.

    Returns curves in format ready for matplotlib/seaborn.
    """
    curves_summary = {}

    for exp_name, exp_data in all_experiments.items():
        if exp_data is None:
            continue

        curves_summary[exp_name] = {
            "seeds": {},
        }

        for run in exp_data["runs"]:
            seed = run["seed"]
            curves = run["training_curves"]

            # Extract key metrics over time
            train_steps = []
            train_acc = []
            train_loss = []
            eval_a_acc = []
            eval_b_avg = []
            eval_c0_acc = []

            for entry in curves:
                step = entry.get("step")

                if entry.get("type") == "train":
                    train_steps.append(step)
                    train_acc.append(entry.get("train_acc"))
                    train_loss.append(entry.get("train_loss"))

                elif entry.get("type") == "eval":
                    eval_a_acc.append({
                        "step": step,
                        "accuracy": entry.get("evalA", {}).get("acc"),
                    })
                    eval_b_avg.append({
                        "step": step,
                        "accuracy": entry.get("evalB", {}).get("avg"),
                    })
                    eval_c0_acc.append({
                        "step": step,
                        "accuracy": entry.get("evalC", {}).get("no_anchor", {}).get("acc"),
                    })

            curves_summary[exp_name]["seeds"][f"seed_{seed}"] = {
                "train": {
                    "steps": train_steps,
                    "accuracy": train_acc,
                    "loss": train_loss,
                },
                "eval_A": eval_a_acc,
                "eval_B": eval_b_avg,
                "eval_C0": eval_c0_acc,
            }

    return curves_summary


def format_markdown_summary(all_experiments: Dict[str, Any]) -> str:
    """Generate markdown summary document."""
    lines = []

    lines.append("# Paper Reproducibility Package")
    lines.append("")
    lines.append("Complete configuration and results for all experiments.")
    lines.append("")
    lines.append("## Experiments Overview")
    lines.append("")

    for exp_name in ["baseline_001", "i1_001_1", "i1_002_alibi", "i1_002a"]:
        exp_data = all_experiments.get(exp_name)
        if not exp_data:
            continue

        lines.append(f"### {exp_name}")
        lines.append("")

        # Specification
        spec = exp_data["specification"]
        lines.append("**Configuration:**")
        lines.append(f"- K-variants: {spec['intervention']['K']}")
        lines.append(f"- Position diversity: {spec['intervention']['position_diversity']}")
        lines.append(f"- Template diversity: {spec['intervention']['template_diversity']}")
        lines.append(f"- Anchors: {spec['intervention']['use_anchors']}")
        lines.append(f"- ALiBi: {spec['intervention']['use_alibi']}")
        lines.append(f"- Consistency loss: {spec['intervention']['consistency_loss']}")
        lines.append("")

        # Results
        agg = exp_data["aggregated_results"]
        lines.append("**Results (Mean ± Std):**")

        eval_a = agg["eval_A"]["accuracy"]
        lines.append(f"- Eval-A: {eval_a['mean']*100:.1f} ± {eval_a['std']*100:.1f}%")

        eval_b = agg["eval_B"]["average"]
        lines.append(f"- Eval-B: {eval_b['mean']*100:.1f} ± {eval_b['std']*100:.1f}%")

        eval_c0 = agg["eval_C"]["C0_no_anchor"]
        lines.append(f"- Eval-C0: {eval_c0['mean']*100:.1f} ± {eval_c0['std']*100:.1f}%")

        if agg["eval_C"]["C1_anchor"]:
            eval_c1 = agg["eval_C"]["C1_anchor"]
            lines.append(f"- Eval-C1: {eval_c1['mean']*100:.1f} ± {eval_c1['std']*100:.1f}%")
        lines.append("")

        # Compute
        compute = exp_data["compute_stats"]
        lines.append("**Compute:**")
        lines.append(f"- Steps: {compute['steps']['mean']:.0f}")
        lines.append(f"- Tokens: {compute['tokens_processed']['mean']:,.0f}")
        lines.append(f"- Wall time: {compute['wall_time_minutes']['mean']:.1f} ± {compute['wall_time_minutes']['std']:.1f} min")
        lines.append("")

    return "\n".join(lines)


def main():
    results_dir = Path("paper_runs")
    experiments = ["baseline_001", "i1_001_1", "i1_002_alibi", "i1_002a"]
    seeds = [42, 43, 44]

    print("Creating reproducibility package...")
    print("=" * 80)

    # Process all experiments
    all_experiments = {}
    for exp_name in experiments:
        exp_data = process_experiment(results_dir, exp_name, seeds)
        if exp_data:
            all_experiments[exp_name] = exp_data

    # Add seeds and sampling documentation
    seeds_doc = extract_seeds_and_sampling(seeds)

    # Extract training curves for plotting
    print("\nExtracting training curves...")
    training_curves = create_training_curves_summary(all_experiments)

    # Create complete package
    package = {
        "metadata": {
            "version": "1.0",
            "description": "Complete reproducibility package for position-invariant learning paper",
            "experiments": list(all_experiments.keys()),
            "seeds": seeds,
        },
        "seeds_and_sampling": seeds_doc,
        "experiments": all_experiments,
        "training_curves": training_curves,
    }

    # Save complete package
    output_path = Path("reproducibility_package.json")
    with open(output_path, "w") as f:
        json.dump(package, f, indent=2)
    print(f"\n✅ Saved complete package: {output_path}")

    # Save training curves separately (for easier plotting)
    curves_path = Path("training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(training_curves, f, indent=2)
    print(f"✅ Saved training curves: {curves_path}")

    # Generate markdown summary
    markdown = format_markdown_summary(all_experiments)
    markdown_path = Path("REPRODUCIBILITY.md")
    with open(markdown_path, "w") as f:
        f.write(markdown)
    print(f"✅ Saved summary: {markdown_path}")

    print("\n" + "=" * 80)
    print("\nFiles created:")
    print(f"  1. {output_path} - Complete package (all data)")
    print(f"  2. {curves_path} - Training curves (for plotting)")
    print(f"  3. {markdown_path} - Human-readable summary")
    print("\nPackage includes:")
    print("  ✅ Experiment specifications (architecture, optimizer, curriculum)")
    print("  ✅ Seeds and sampling documentation")
    print("  ✅ Training curves (all steps, all seeds)")
    print("  ✅ Compute metrics (steps, wall-clock, tokens)")
    print("  ✅ Final aggregated results")


if __name__ == "__main__":
    main()

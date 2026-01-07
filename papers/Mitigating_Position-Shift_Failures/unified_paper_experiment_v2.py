#!/usr/bin/env python3
"""
Unified Paper Experiment Script v2 - Steps-Based Training

FIXED: Now stops at MAX_STEPS (5000) instead of token budget.
       Tokens are logged but not used for stopping.
       Eval-C split into C0 (no-anchor) and C1 (anchor).

Runs all 4 experiments with equal training steps:
- baseline_001: Standard training (K=1, no interventions)
- i1_001_1: Position diversity only (K=4, no template diversity)
- i1_002_alibi: Position + template diversity + ALiBi (K=4)
- i1_002a: Position + template diversity + learned pos emb (K=4)

Usage:
    python unified_paper_experiment_v2.py --experiment baseline_001 --seed 42
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from pathlib import Path
import time
import random
import numpy as np
from typing import List, Tuple, Dict
import sys

# Constants (same for all experiments)
MAX_STEPS = 5000  # All experiments run for exactly 5000 steps
P = 97
TRAIN_FRACTION = 0.5
MAX_LENGTH = 100
VOCAB_SIZE = 80
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
BATCH_SIZE = 256
LR = 0.001
WEIGHT_DECAY = 0.01
LOG_EVERY_STEPS = 20
EVAL_EVERY_STEPS = 200

# Curriculum boundaries (steps-based)
CURRICULUM_EARLY = 1666   # steps 0-1666
CURRICULUM_MID = 3333     # steps 1667-3333
# steps 3334-5000 = late


def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocab():
    """Character-level vocab (same for all experiments)"""
    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!+-=×÷()%")
    vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<EXPR>': 3, '</EXPR>': 4}
    for i, char in enumerate(chars, start=5):
        vocab[char] = i
    return vocab


def create_disjoint_split(p: int, train_fraction: float, seed: int):
    """Create disjoint 50/50 split (same for all experiments)"""
    random.seed(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * train_fraction)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]

    return train_pairs, test_pairs


class SimpleTransformer(nn.Module):
    """Transformer with optional ALiBi"""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 num_classes: int, max_length: int, pooling: str = 'cls', use_alibi: bool = False):
        super().__init__()
        self.pooling = pooling
        self.use_alibi = use_alibi

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if not use_alibi:
            self.position_embedding = nn.Embedding(max_length, d_model)

        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        B, L = input_ids.shape

        x = self.token_embedding(input_ids)

        if not self.use_alibi:
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
            x = x + self.position_embedding(positions)

        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.transformer(x)

        if self.pooling == 'cls':
            pooled = x[:, 0]
        elif self.pooling == 'mean':
            pooled = x.mean(dim=1)
        else:
            pooled = x[:, -1]

        return self.output(pooled)


class BaselineGenerator:
    """Simple generator for baseline (no position diversity, no K-variants)"""

    def __init__(self, vocab, max_length):
        self.vocab = vocab
        self.max_length = max_length
        self.use_anchors = False

    def generate(self, a: int, b: int, **kwargs) -> List[str]:
        """Generate single template: 'a + b'"""
        return [f"{a} + {b}"]

    def tokenize(self, text: str) -> List[int]:
        tokens = [self.vocab.get(c, self.vocab['<UNK>']) for c in text]
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return tokens


class PositionDiverseGeneratorNoDiversity:
    """Position diversity WITHOUT template diversity (for i1_001_1)"""

    def __init__(self, vocab, max_length, K=4, use_anchors=False, seed=42):
        self.vocab = vocab
        self.max_length = max_length
        self.K = K
        self.use_anchors = use_anchors
        self.rng = random.Random(seed)
        self.filler_chunks = ["um ", "uh ", "so ", "like ", "well "]

    def build_padding(self, target_pos):
        padding = ""
        while len(padding) < target_pos:
            padding += self.rng.choice(self.filler_chunks)
        return padding[:target_pos]

    def generate(self, a: int, b: int, step: int = 0) -> List[str]:
        """Generate K variants with position diversity (curriculum based on steps)"""

        # Curriculum based on step count
        if step <= CURRICULUM_EARLY:
            pos_range = (10, 30)
        elif step <= CURRICULUM_MID:
            pos_range = (10, 50)
        else:
            pos_range = (10, 70)

        variants = []
        for _ in range(self.K):
            pos = self.rng.randint(*pos_range)
            padding = self.build_padding(pos)

            if self.use_anchors:
                template = f"{padding}<EXPR>{a} + {b}</EXPR>"
            else:
                template = f"{padding}{a} + {b}"

            variants.append(template)

        return variants

    def tokenize(self, text: str) -> List[int]:
        tokens = [self.vocab.get(c, self.vocab['<UNK>']) for c in text]
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return tokens


class PositionDiverseGeneratorV3:
    """Full position + template diversity (for i1_002a/alibi)"""

    def __init__(self, vocab, max_length, K=4, use_anchors=True, seed=42):
        self.vocab = vocab
        self.max_length = max_length
        self.K = K
        self.use_anchors = use_anchors
        self.rng = random.Random(seed)
        self.filler_chunks = ["um ", "uh ", "so ", "like ", "well "]

        # Template categories (40% padding / 40% NL / 20% mixed)
        self.padding_templates = [
            "<EXPR>{a} + {b}</EXPR>",
            "<EXPR>{a} plus {b}</EXPR>",
        ]
        self.nl_templates = [
            "What is <EXPR>{a} + {b}</EXPR>?",
            "Calculate <EXPR>{a} plus {b}</EXPR>",
            "Compute <EXPR>{a} + {b}</EXPR>",
        ]
        self.mixed_templates = [
            "<EXPR>{a} + {b}</EXPR> equals?",
            "Result of <EXPR>{a} + {b}</EXPR>",
        ]

    def build_padding(self, target_pos):
        padding = ""
        while len(padding) < target_pos:
            padding += self.rng.choice(self.filler_chunks)
        return padding[:target_pos]

    def generate(self, a: int, b: int, step: int = 0) -> List[str]:
        """Generate K variants with full diversity (curriculum based on steps)"""

        # Curriculum based on step count
        if step <= CURRICULUM_EARLY:
            pos_range = (10, 30)
        elif step <= CURRICULUM_MID:
            pos_range = (10, 50)
        else:
            pos_range = (10, 70)

        variants = []
        for _ in range(self.K):
            pos = self.rng.randint(*pos_range)
            padding = self.build_padding(pos)

            # Template mix: 40% padding, 40% NL, 20% mixed
            roll = self.rng.random()
            if roll < 0.4:
                template = self.rng.choice(self.padding_templates)
            elif roll < 0.8:
                template = self.rng.choice(self.nl_templates)
            else:
                template = self.rng.choice(self.mixed_templates)

            # Fill template
            template = template.format(a=a, b=b)
            full_template = padding + template

            variants.append(full_template)

        return variants

    def tokenize(self, text: str) -> List[int]:
        tokens = [self.vocab.get(c, self.vocab['<UNK>']) for c in text]
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return tokens


class ArithmeticDataset(Dataset):
    """Dataset supporting variable K (works for K=1 and K=4)"""

    def __init__(self, pairs, generator, step):
        self.pairs = pairs
        self.generator = generator
        self.step = step

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        label = (a + b) % P

        # Generate K variants (K=1 for baseline, K=4 for interventions)
        templates = self.generator.generate(a, b, step=self.step)

        # Tokenize all variants
        input_ids = [self.generator.tokenize(t) for t in templates]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),  # [K, max_length]
            'label': torch.tensor(label, dtype=torch.long),
        }


def compute_consistency_loss(logits_variants, labels):
    """Penalize when K variants produce different predictions"""
    K = logits_variants.size(1)
    if K == 1:
        return torch.tensor(0.0, device=logits_variants.device)

    preds = logits_variants.argmax(dim=-1)  # [batch, K]

    # Consistency = all K predictions are identical
    # For each instance, check if std of predictions is 0
    consistency_loss = 0.0
    for i in range(preds.size(0)):
        unique_preds = preds[i].unique()
        if len(unique_preds) > 1:
            # Penalize inconsistency
            consistency_loss += 1.0

    return consistency_loss / preds.size(0)


def train_step(model, batch, optimizer, device, lambda_consistency=0.5):
    """Single training step"""
    model.train()

    input_ids = batch['input_ids'].to(device)  # [batch, K, max_length]
    labels = batch['label'].to(device)  # [batch]

    batch_size, K, seq_len = input_ids.shape

    # Flatten K variants
    input_ids_flat = input_ids.view(batch_size * K, seq_len)
    labels_expanded = labels.unsqueeze(1).expand(-1, K).reshape(-1)

    # Forward
    logits = model(input_ids_flat)  # [batch*K, num_classes]
    logits_variants = logits.view(batch_size, K, -1)  # [batch, K, num_classes]

    # CE loss (average over K variants)
    ce_loss = F.cross_entropy(logits, labels_expanded)

    # Consistency loss (only if K>1)
    if K > 1:
        consistency_loss = compute_consistency_loss(logits_variants, labels)
        total_loss = ce_loss + lambda_consistency * consistency_loss
    else:
        consistency_loss = torch.tensor(0.0)
        total_loss = ce_loss

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Accuracy (average over K variants)
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        acc = (preds == labels_expanded).float().mean()

    # Count tokens processed
    tokens_in_batch = input_ids.numel()

    return {
        'loss': total_loss.item(),
        'ce_loss': ce_loss.item(),
        'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
        'acc': acc.item(),
        'tokens': tokens_in_batch,
    }


def evaluate_model(model, generator, test_pairs, vocab, device, max_length):
    """Eval-A: In-distribution accuracy on 400 test pairs"""
    model.eval()

    n_samples = min(400, len(test_pairs))
    eval_pairs = test_pairs[:n_samples]

    correct = 0
    consistent_and_correct = 0

    with torch.no_grad():
        for a, b in eval_pairs:
            expected = (a + b) % P

            # Generate variants (use late-stage curriculum)
            templates = generator.generate(a, b, step=MAX_STEPS)

            # Tokenize
            input_ids_list = [generator.tokenize(t) for t in templates]
            input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)  # [K, max_length]

            # Predict
            logits = model(input_ids)  # [K, num_classes]
            preds = logits.argmax(dim=-1).cpu().tolist()

            # Accuracy (majority vote or first variant)
            pred = preds[0]
            if pred == expected:
                correct += 1

            # Consistency@K: all same AND correct
            if len(set(preds)) == 1 and preds[0] == expected:
                consistent_and_correct += 1

    acc = correct / n_samples
    cc_k = consistent_and_correct / n_samples

    return {'acc': acc, 'cc4': cc_k, 'n': n_samples}


def evaluate_position_shift(model, generator, vocab, device, max_length):
    """Eval-B: Position-shift test at fixed positions"""
    model.eval()

    positions = [0, 8, 16, 24, 32, 48, 64]
    n_per_pos = 100

    results = {}

    with torch.no_grad():
        for pos in positions:
            correct = 0

            for _ in range(n_per_pos):
                a = random.randint(0, P-1)
                b = random.randint(0, P-1)
                expected = (a + b) % P

                # Fixed position template
                padding = ""
                for _ in range(pos):
                    padding += random.choice(["um ", "uh ", "so ", "like ", "well "])
                padding = padding[:pos]

                # Use anchors if generator supports them
                if hasattr(generator, 'use_anchors') and generator.use_anchors:
                    template = f"{padding}<EXPR>{a} + {b}</EXPR>"
                elif isinstance(generator, PositionDiverseGeneratorV3):
                    template = f"{padding}<EXPR>{a} + {b}</EXPR>"
                else:
                    template = f"{padding}{a} + {b}"

                # Tokenize and predict
                input_ids = torch.tensor([generator.tokenize(template)], dtype=torch.long).to(device)
                logits = model(input_ids)
                pred = logits.argmax(dim=-1).item()

                if pred == expected:
                    correct += 1

            results[str(pos)] = correct / n_per_pos

    # Summary stats
    early_avg = np.mean([results[str(p)] for p in [0, 8]])
    mid_avg = np.mean([results[str(p)] for p in [16, 24]])
    far_avg = np.mean([results[str(p)] for p in [32, 48, 64]])

    return {
        'by_pos': results,
        'avg': np.mean(list(results.values())),
        'min': min(results.values()),
        'early_avg': early_avg,
        'mid_avg': mid_avg,
        'far_avg': far_avg,
    }


def evaluate_template_ood(model, vocab, device, max_length, use_anchors: bool):
    """
    Eval-C: Template OOD test

    Split into two parts:
    - C0 (no-anchor): For ALL models (fair comparison)
    - C1 (anchor): Only for models with use_anchors=True
    """
    model.eval()

    # OOD templates WITHOUT anchors (Eval-C0 - for all models)
    ood_no_anchor_questions = [
        "please compute {a} + {b}",
        "what is {a} plus {b}",
        "how much is {a} + {b}",
        "tell me {a} + {b}",
    ]

    ood_no_anchor_commands = [
        "solve {a} + {b}",
        "calculate {a} + {b}",
        "evaluate {a} plus {b}",
    ]

    # OOD templates WITH anchors (Eval-C1 - only for anchor models)
    ood_anchor_questions = [
        "please compute <EXPR>{a} + {b}</EXPR>",
        "what is <EXPR>{a} plus {b}</EXPR>",
        "how much is <EXPR>{a} + {b}</EXPR>",
        "tell me <EXPR>{a} + {b}</EXPR>",
    ]

    ood_anchor_commands = [
        "solve <EXPR>{a} + {b}</EXPR>",
        "calculate <EXPR>{a} + {b}</EXPR>",
        "evaluate <EXPR>{a} plus {b}</EXPR>",
    ]

    def tokenize(text):
        tokens = [vocab.get(c, vocab['<UNK>']) for c in text]
        if len(tokens) < max_length:
            tokens = tokens + [vocab['<PAD>']] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return tokens

    def test_category(templates, n_samples):
        correct = 0
        for _ in range(n_samples):
            a = random.randint(0, P-1)
            b = random.randint(0, P-1)
            expected = (a + b) % P

            # Random template + random position
            template = random.choice(templates).format(a=a, b=b)
            pos = random.randint(0, 70)

            padding = ""
            for _ in range(pos):
                padding += random.choice(["um ", "uh ", "so ", "like ", "well "])
            padding = padding[:pos]

            full_template = padding + template

            # Predict
            input_ids = torch.tensor([tokenize(full_template)], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(input_ids)
                pred = logits.argmax(dim=-1).item()

            if pred == expected:
                correct += 1

        return correct / n_samples

    # Eval-C0: No-anchor OOD (for ALL models)
    no_anchor_questions_acc = test_category(ood_no_anchor_questions, 100)
    no_anchor_commands_acc = test_category(ood_no_anchor_commands, 100)
    no_anchor_acc = (no_anchor_questions_acc + no_anchor_commands_acc) / 2

    results = {
        'no_anchor': {
            'acc': no_anchor_acc,
            'n': 200,
        }
    }

    # Eval-C1: Anchor OOD (only for anchor models)
    if use_anchors:
        anchor_questions_acc = test_category(ood_anchor_questions, 100)
        anchor_commands_acc = test_category(ood_anchor_commands, 100)
        anchor_acc = (anchor_questions_acc + anchor_commands_acc) / 2

        results['anchor'] = {
            'acc': anchor_acc,
            'n': 200,
        }
    else:
        results['anchor'] = None

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, choices=['baseline_001', 'i1_001_1', 'i1_002_alibi', 'i1_002a'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"paper_runs/{args.experiment}_seed{args.seed}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    set_seed(args.seed)

    # Build vocab and data
    vocab = build_vocab()
    train_pairs, test_pairs = create_disjoint_split(P, TRAIN_FRACTION, args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Experiment: {args.experiment}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"Max steps: {MAX_STEPS}")

    # Experiment-specific setup
    if args.experiment == 'baseline_001':
        K = 1
        use_alibi = False
        use_template_diversity = False
        use_anchors = False
        generator = BaselineGenerator(vocab, MAX_LENGTH)

    elif args.experiment == 'i1_001_1':
        K = 4
        use_alibi = False
        use_template_diversity = False
        use_anchors = False
        generator = PositionDiverseGeneratorNoDiversity(vocab, MAX_LENGTH, K=K, use_anchors=use_anchors, seed=args.seed)

    elif args.experiment == 'i1_002_alibi':
        K = 4
        use_alibi = True
        use_template_diversity = True
        use_anchors = True
        generator = PositionDiverseGeneratorV3(vocab, MAX_LENGTH, K=K, use_anchors=use_anchors, seed=args.seed)

    else:  # i1_002a
        K = 4
        use_alibi = False
        use_template_diversity = True
        use_anchors = True
        generator = PositionDiverseGeneratorV3(vocab, MAX_LENGTH, K=K, use_anchors=use_anchors, seed=args.seed)

    # Create model
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        num_classes=P,
        max_length=MAX_LENGTH,
        pooling='cls',
        use_alibi=use_alibi,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Save config
    config = {
        'experiment': args.experiment,
        'seed': args.seed,
        'max_steps': MAX_STEPS,
        'p': P,
        'train_fraction': TRAIN_FRACTION,
        'train_pairs': len(train_pairs),
        'test_pairs': len(test_pairs),
        'max_length': MAX_LENGTH,
        'vocab_size': VOCAB_SIZE,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'use_alibi': use_alibi,
        'K': K,
        'use_template_diversity': use_template_diversity,
        'use_anchors': use_anchors,
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Starting training for {MAX_STEPS} steps...")

    # Training loop
    step = 0
    tokens_processed = 0
    start_time = time.time()
    best_eval_a_acc = 0

    metrics_file = output_dir / 'metrics.jsonl'

    while step < MAX_STEPS:
        # Update dataset with current step for curriculum
        dataset = ArithmeticDataset(train_pairs, generator, step)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for batch in dataloader:
            if step >= MAX_STEPS:
                break

            # Train step
            metrics = train_step(model, batch, optimizer, device, lambda_consistency=0.5 if K > 1 else 0.0)

            tokens_processed += metrics['tokens']
            step += 1

            # Log
            if step % LOG_EVERY_STEPS == 0:
                wall_time = time.time() - start_time
                log_entry = {
                    'type': 'train',
                    'step': step,
                    'tokens_processed': tokens_processed,
                    'wall_time_seconds': wall_time,
                    'lr': LR,
                    'train_loss': metrics['loss'],
                    'train_acc': metrics['acc'],
                    'ce_loss': metrics['ce_loss'],
                    'consistency_loss': metrics['consistency_loss'],
                }

                with open(metrics_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

                print(f"Step {step}/{MAX_STEPS} | Tokens {tokens_processed:,} | "
                      f"Loss {metrics['loss']:.4f} | Acc {metrics['acc']:.1%}")

            # Eval
            if step % EVAL_EVERY_STEPS == 0 or step >= MAX_STEPS:
                wall_time = time.time() - start_time

                eval_a = evaluate_model(model, generator, test_pairs, vocab, device, MAX_LENGTH)
                eval_b = evaluate_position_shift(model, generator, vocab, device, MAX_LENGTH)
                eval_c = evaluate_template_ood(model, vocab, device, MAX_LENGTH, use_anchors)

                eval_entry = {
                    'type': 'eval',
                    'step': step,
                    'tokens_processed': tokens_processed,
                    'wall_time_seconds': wall_time,
                    'evalA': eval_a,
                    'evalB': eval_b,
                    'evalC': eval_c,
                }

                with open(metrics_file, 'a') as f:
                    f.write(json.dumps(eval_entry) + '\n')

                print(f"\nEval @ step {step}:")
                print(f"  Eval-A: {eval_a['acc']:.1%} (CC@4: {eval_a['cc4']:.1%})")
                print(f"  Eval-B: avg {eval_b['avg']:.1%}, far {eval_b['far_avg']:.1%}")
                print(f"  Eval-C0 (no-anchor): {eval_c['no_anchor']['acc']:.1%}")
                if eval_c['anchor'] is not None:
                    print(f"  Eval-C1 (anchor): {eval_c['anchor']['acc']:.1%}")
                print()

                # Save best model
                if eval_a['acc'] > best_eval_a_acc:
                    best_eval_a_acc = eval_a['acc']
                    torch.save(model.state_dict(), output_dir / 'best_evalA.pt')

    # Save final checkpoint
    torch.save(model.state_dict(), output_dir / 'checkpoint_final.pt')

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    eval_a = evaluate_model(model, generator, test_pairs, vocab, device, MAX_LENGTH)
    eval_b = evaluate_position_shift(model, generator, vocab, device, MAX_LENGTH)
    eval_c = evaluate_template_ood(model, vocab, device, MAX_LENGTH, use_anchors)

    final_results = {
        'tokens_processed': tokens_processed,
        'final_step': step,
        'wall_time_seconds': time.time() - start_time,
        'evalA': eval_a,
        'evalB': eval_b,
        'evalC': eval_c,
    }

    with open(output_dir / 'final_eval.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"Total steps: {step}")
    print(f"Total tokens: {tokens_processed:,}")
    print(f"Wall time: {final_results['wall_time_seconds']/60:.1f} minutes")
    print(f"\nFinal Results:")
    print(f"  Eval-A: {eval_a['acc']:.1%} (CC@4: {eval_a['cc4']:.1%})")
    print(f"  Eval-B: {eval_b['avg']:.1%}")
    print(f"  Eval-C0 (no-anchor): {eval_c['no_anchor']['acc']:.1%}")
    if eval_c['anchor'] is not None:
        print(f"  Eval-C1 (anchor): {eval_c['anchor']['acc']:.1%}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

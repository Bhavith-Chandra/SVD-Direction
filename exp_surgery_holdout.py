#!/usr/bin/env python3
"""
Held-Out Validation of Surgical Editing

Addresses reviewer concern that surgical editing may be "task-specific tuning"
since direction classification uses the same IOI metric on the same prompts.

Fix: Split prompts into train (classify directions) and test (evaluate surgery).
Also test on entirely new name pairs for generalization.
"""

import torch
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Loading TransformerLens...")
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

with open("results/all_QK.pkl", "rb") as f:
    all_QK = pickle.load(f)

# ── Three IOI prompt sets ───────────────────────────────────────────────

def create_ioi_split():
    """Create train/test/generalization splits."""
    templates = [
        "When {A} and {B} went to the store, {B} gave a drink to",
        "When {A} and {B} went to the park, {B} gave a flower to",
        "When {A} and {B} walked in, {B} said hello to",
        "After {A} and {B} arrived at the party, {B} handed a gift to",
        "When {A} and {B} entered the room, {B} waved at",
    ]
    # Original names for train/test
    names_A = ["Mary", "Alice", "Sarah", "Emma", "Lisa"]
    names_B = ["John", "Bob", "James", "Tom", "Mike"]
    # Held-out names for generalization
    names_A_new = ["Rachel", "Kate", "Diana", "Laura", "Claire"]
    names_B_new = ["David", "Chris", "Peter", "George", "Mark"]

    def make_prompts(n, nA, nB, seed=0):
        np.random.seed(seed)
        prompts, correct, incorrect = [], [], []
        for i in range(n):
            t = templates[i % len(templates)]
            a = nA[i % len(nA)]
            b = nB[i % len(nB)]
            prompts.append(t.format(A=a, B=b))
            correct.append(a)
            incorrect.append(b)
        return prompts, correct, incorrect

    train = make_prompts(50, names_A, names_B, seed=0)
    test = make_prompts(50, names_A, names_B, seed=42)
    gen = make_prompts(50, names_A_new, names_B_new, seed=99)
    return train, test, gen

(train_prompts, train_correct, train_incorrect), \
(test_prompts, test_correct, test_incorrect), \
(gen_prompts, gen_correct, gen_incorrect) = create_ioi_split()

def tokenize_and_ids(prompts, correct, incorrect):
    tokens = model.to_tokens(prompts, prepend_bos=True)
    c_ids = [model.to_single_token(" " + n) for n in correct]
    i_ids = [model.to_single_token(" " + n) for n in incorrect]
    return tokens, c_ids, i_ids

train_tokens, train_c, train_i = tokenize_and_ids(train_prompts, train_correct, train_incorrect)
test_tokens, test_c, test_i = tokenize_and_ids(test_prompts, test_correct, test_incorrect)
gen_tokens, gen_c, gen_i = tokenize_and_ids(gen_prompts, gen_correct, gen_incorrect)

def compute_logit_diff(logits, c_ids, i_ids):
    last = logits[:, -1, :]
    diffs = [last[j, c_ids[j]].item() - last[j, i_ids[j]].item() for j in range(len(c_ids))]
    return np.mean(diffs)

# ── Baselines ───────────────────────────────────────────────────────────
print("Computing baselines...")
with torch.no_grad():
    train_base = compute_logit_diff(model(train_tokens), train_c, train_i)
    test_base = compute_logit_diff(model(test_tokens), test_c, test_i)
    gen_base = compute_logit_diff(model(gen_tokens), gen_c, gen_i)

print(f"Baselines: train={train_base:.4f}  test={test_base:.4f}  gen={gen_base:.4f}")

# ── Classify directions using TRAIN set only ────────────────────────────
heads = [
    ("S-Inhibition L9H6", 9, 6),
    ("Name Mover L9H9", 9, 9),
    ("Name Mover L10H0", 10, 0),
    ("Low-rank L2H2", 2, 2),
]

n_sv = 10
threshold_frac = 0.02
threshold = threshold_frac * train_base

results = {"baselines": {"train": train_base, "test": test_base, "gen": gen_base}}

all_harmful_hooks = []

for head_name, layer, head_idx in heads:
    print(f"\nClassifying directions for {head_name} using TRAIN set...")
    key = (layer, head_idx)
    entry = all_QK[key]
    if isinstance(entry, dict):
        U = entry["U"]
    else:
        U = entry[0]  # tuple: (U, S, Vt)

    helpful, harmful, neutral = [], [], []
    effects = []

    for sv_idx in range(n_sv):
        u_k = torch.tensor(U[:, sv_idx], dtype=torch.float32)

        def hook_fn(resid, hook, u=u_k):
            proj = torch.einsum('bsd,d->bs', resid, u)
            return resid - torch.einsum('bs,d->bsd', proj, u)

        with torch.no_grad():
            logits = model.run_with_hooks(
                train_tokens,
                fwd_hooks=[(get_act_name("resid_pre", layer), hook_fn)]
            )
        ld = compute_logit_diff(logits, train_c, train_i)
        delta = ld - train_base
        effects.append(delta)

        if delta > threshold:
            helpful.append(sv_idx)
        elif delta < -threshold:
            harmful.append(sv_idx)
        else:
            neutral.append(sv_idx)

    print(f"  Helpful: {helpful}, Harmful: {harmful}, Neutral: {neutral}")
    print(f"  Effects: {[f'{e:+.4f}' for e in effects]}")

    results[head_name] = {
        "helpful": helpful, "harmful": harmful, "neutral": neutral,
        "train_effects": effects,
    }

    # Build hooks for harmful direction removal
    if harmful:
        u_vecs = torch.tensor(U[:, harmful], dtype=torch.float32)
        def make_hook(uv=u_vecs, l=layer):
            def hook_fn(resid, hook):
                proj = torch.einsum('bsd,dk->bsk', resid, uv)
                return resid - torch.einsum('bsk,dk->bsd', proj, uv)
            return (get_act_name("resid_pre", l), hook_fn)
        all_harmful_hooks.append(make_hook())

# ── Evaluate on TEST and GENERALIZATION sets ────────────────────────────
print("\nEvaluating surgical editing on held-out sets...")

for split_name, toks, c_ids, i_ids, base_ld in [
    ("train", train_tokens, train_c, train_i, train_base),
    ("test", test_tokens, test_c, test_i, test_base),
    ("generalization", gen_tokens, gen_c, gen_i, gen_base),
]:
    with torch.no_grad():
        logits = model.run_with_hooks(toks, fwd_hooks=all_harmful_hooks)
    edited_ld = compute_logit_diff(logits, c_ids, i_ids)
    improvement = edited_ld - base_ld
    pct = 100 * improvement / abs(base_ld) if base_ld != 0 else 0

    results[f"{split_name}_surgery"] = {
        "baseline": base_ld,
        "edited": edited_ld,
        "improvement": improvement,
        "improvement_pct": pct,
    }
    print(f"  {split_name:15s}: baseline={base_ld:.4f}  edited={edited_ld:.4f}  "
          f"Δ={improvement:+.4f} ({pct:+.1f}%)")

# ── Save ────────────────────────────────────────────────────────────────
with open("results/surgery_holdout.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/surgery_holdout.json")

# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Surgical Editing: Train vs Test vs Generalization\n"
             "Directions classified on train set; surgery evaluated on held-out sets",
             fontsize=12, fontweight='bold')

splits = ["train", "test", "generalization"]
colors = ["#4A90D9", "#E8636F", "#82C991"]

for idx, (split, color) in enumerate(zip(splits, colors)):
    ax = axes[idx]
    r = results[f"{split}_surgery"]
    bars = ax.bar(["Baseline", "After Surgery"],
                  [r["baseline"], r["edited"]],
                  color=[color, color], alpha=[0.4, 0.9], edgecolor='white', linewidth=2)
    ax.set_ylabel("IOI Logit Difference")
    ax.set_title(f"{split.capitalize()} Set\nΔ = {r['improvement']:+.3f} ({r['improvement_pct']:+.1f}%)")
    ax.axhline(y=r["baseline"], color='gray', linestyle='--', alpha=0.3)

    # Annotate improvement
    mid = (r["baseline"] + r["edited"]) / 2
    ax.annotate(f"+{r['improvement']:.3f}", xy=(1, r["edited"]),
                fontsize=11, fontweight='bold', ha='center', va='bottom', color='darkgreen')

plt.tight_layout()
plt.savefig("results/surgery_holdout.png", dpi=150, bbox_inches='tight')
print("Plot saved to results/surgery_holdout.png")
print("\nDone!")

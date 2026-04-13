# False negative rate: what fraction of heads hide significant computation behind near-zero net effects?
# Uses IOI ablation data from GPT-2 + extends to all heads

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from transformer_lens import HookedTransformer

OUT_DIR = "/Users/srimanarayana/Research Project I/results"

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

with open(os.path.join(OUT_DIR, "all_QK.pkl"), "rb") as f:
    all_QK = pickle.load(f)

def create_ioi_dataset(n=100):
    templates = [
        "When {A} and {B} went to the store, {B} gave a drink to",
        "When {A} and {B} went to the park, {B} gave a flower to",
        "When {A} and {B} walked in, {B} said hello to",
        "After {A} and {B} arrived at the party, {B} handed a gift to",
        "When {A} and {B} entered the room, {B} waved at",
    ]
    names_A = [" Mary", " Alice", " Sarah", " Emma", " Lisa"]
    names_B = [" John", " Bob", " James", " Tom", " Mike"]
    prompts = []
    for i in range(n):
        A = names_A[i % len(names_A)]
        B = names_B[i % len(names_B)]
        prompts.append({'prompt': templates[i % len(templates)].format(A=A, B=B),
                        'correct': A, 'incorrect': B})
    return prompts

ioi_data = create_ioi_dataset(100)

def measure_ioi(hooks=None):
    logit_diffs = []
    for item in ioi_data:
        tokens = model.to_tokens(item['prompt'])
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        correct_id = model.to_single_token(item['correct'])
        incorrect_id = model.to_single_token(item['incorrect'])
        logit_diffs.append((logits[0, -1, correct_id] - logits[0, -1, incorrect_id]).item())
    return np.mean(logit_diffs)

baseline = measure_ioi()
print(f"Baseline IOI: {baseline:.3f}")

# Scan ALL 144 heads: individual SV effects + full head effects
print("\nScanning all 144 heads (top-5 SVs each)...")
all_head_data = {}
max_k = 5

for layer in range(12):
    for head in range(12):
        U, S, Vt = all_QK[(layer, head)]

        individual_effects = []
        for k in range(max_k):
            u_k = torch.tensor(U[:, k], dtype=torch.float32)

            def make_hook(u_vec):
                u = u_vec.clone()
                def hook_fn(resid, hook):
                    proj = torch.einsum('bsd,d->bs', resid, u)
                    return resid - torch.einsum('bs,d->bsd', proj, u)
                return hook_fn

            hook = make_hook(u_k)
            score = measure_ioi([(f'blocks.{layer}.hook_resid_pre', hook)])
            individual_effects.append(baseline - score)

        # full head ablation
        def make_head_hook(h):
            def hook_fn(q, hook):
                q[:, :, h, :] = 0
                return q
            return hook_fn

        full_score = measure_ioi([(f'blocks.{layer}.attn.hook_q', make_head_hook(head))])
        full_effect = baseline - full_score

        max_indiv = max(abs(e) for e in individual_effects)
        ratio = max_indiv / abs(full_effect) if abs(full_effect) > 0.01 else float('inf')
        has_opposing = any(e > 0 for e in individual_effects) and any(e < 0 for e in individual_effects)

        all_head_data[f"L{layer}H{head}"] = {
            "layer": layer, "head": head,
            "individual_effects": [float(e) for e in individual_effects],
            "full_head_effect": float(full_effect),
            "max_individual": float(max_indiv),
            "opposition_ratio": float(ratio) if ratio != float('inf') else 999.0,
            "has_opposing": has_opposing,
            "singular_values": [float(S[k]) for k in range(max_k)],
        }

    print(f"  Layer {layer} done")

# Compute statistics
ratios = []
false_negatives = []
significant_threshold = 0.1 * abs(baseline)  # 10% of baseline as "significant"

for key, data in all_head_data.items():
    r = data["opposition_ratio"]
    if r < 999:
        ratios.append(r)

    # false negative: net effect is small but individual effects are large
    if abs(data["full_head_effect"]) < significant_threshold and data["max_individual"] > significant_threshold:
        false_negatives.append(key)

n_opposing = sum(1 for d in all_head_data.values() if d["has_opposing"])
n_3x = sum(1 for d in all_head_data.values() if d["opposition_ratio"] > 3)
n_5x = sum(1 for d in all_head_data.values() if d["opposition_ratio"] > 5)
n_10x = sum(1 for d in all_head_data.values() if d["opposition_ratio"] > 10)

print(f"\n{'='*60}")
print(f"FALSE NEGATIVE ANALYSIS — GPT-2 Small (144 heads)")
print(f"{'='*60}")
print(f"Heads with opposing SVD directions:     {n_opposing}/144 ({n_opposing/144*100:.1f}%)")
print(f"Heads with >3x opposition ratio:        {n_3x}/144 ({n_3x/144*100:.1f}%)")
print(f"Heads with >5x opposition ratio:        {n_5x}/144 ({n_5x/144*100:.1f}%)")
print(f"Heads with >10x opposition ratio:       {n_10x}/144 ({n_10x/144*100:.1f}%)")
print(f"False negatives (small net, large SV):   {len(false_negatives)}/144 ({len(false_negatives)/144*100:.1f}%)")
print(f"False negative heads: {false_negatives}")

stats = {
    "model": "gpt2",
    "n_heads": 144,
    "n_opposing": n_opposing,
    "pct_opposing": round(n_opposing / 144 * 100, 1),
    "n_3x": n_3x,
    "pct_3x": round(n_3x / 144 * 100, 1),
    "n_5x": n_5x,
    "pct_5x": round(n_5x / 144 * 100, 1),
    "n_10x": n_10x,
    "pct_10x": round(n_10x / 144 * 100, 1),
    "n_false_negatives": len(false_negatives),
    "pct_false_negatives": round(len(false_negatives) / 144 * 100, 1),
    "false_negative_heads": false_negatives,
    "mean_ratio": float(np.mean(ratios)),
    "median_ratio": float(np.median(ratios)),
    "baseline_ioi": float(baseline),
    "significant_threshold": float(significant_threshold),
}

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. The histogram — the money shot
ax = axes[0, 0]
finite_ratios = [r for r in ratios if r < 50]
ax.hist(finite_ratios, bins=40, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axvline(1, color='grey', linestyle='-', linewidth=2, label='1x (no opposition)')
ax.axvline(3, color='orange', linestyle='--', linewidth=2, label=f'>3x: {n_3x} heads ({n_3x/144*100:.0f}%)')
ax.axvline(5, color='red', linestyle='--', linewidth=2, label=f'>5x: {n_5x} heads ({n_5x/144*100:.0f}%)')
ax.axvline(10, color='darkred', linestyle='--', linewidth=2, label=f'>10x: {n_10x} heads ({n_10x/144*100:.0f}%)')
ax.set_xlabel('Opposition Ratio (max |individual SV effect| / |full head effect|)', fontsize=11)
ax.set_ylabel('Number of Heads', fontsize=11)
ax.set_title('Distribution of Sub-Head Opposition Ratios\n'
             'GPT-2 Small, IOI Task, All 144 Heads', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# 2. Opposition ratio heatmap
ax = axes[0, 1]
ratio_grid = np.zeros((12, 12))
for l in range(12):
    for h in range(12):
        r = all_head_data[f"L{l}H{h}"]["opposition_ratio"]
        ratio_grid[l, h] = min(r, 20)  # cap for visualization

sns.heatmap(ratio_grid, ax=ax, cmap='YlOrRd',
            xticklabels=[f'H{h}' for h in range(12)],
            yticklabels=[f'L{l}' for l in range(12)],
            annot=True, fmt='.1f', annot_kws={'fontsize': 6},
            vmin=0, vmax=15)
ax.set_title('Opposition Ratio by Head\n(capped at 20 for display)', fontsize=12)

# 3. Full head effect vs max individual effect — scatter
ax = axes[1, 0]
full_effects = [all_head_data[k]["full_head_effect"] for k in all_head_data]
max_indivs = [all_head_data[k]["max_individual"] for k in all_head_data]

ax.scatter(np.abs(full_effects), max_indivs, alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.3)

# highlight false negatives
for fn in false_negatives:
    d = all_head_data[fn]
    ax.scatter(abs(d["full_head_effect"]), d["max_individual"], c='red', s=80,
               marker='*', zorder=5, label='False negative' if fn == false_negatives[0] else '')

ax.plot([0, max(np.abs(full_effects))], [0, max(np.abs(full_effects))], 'k--', alpha=0.3, label='1:1 line')
ax.set_xlabel('|Full Head Effect| (standard ablation)', fontsize=11)
ax.set_ylabel('Max |Individual SV Effect|', fontsize=11)
ax.set_title('Head-Level vs Sub-Head Analysis\n'
             'Points above the line = hidden computation', fontsize=12)
ax.legend(fontsize=9)

# 4. Per-layer statistics
ax = axes[1, 1]
layer_stats = []
for l in range(12):
    n_opp = sum(1 for h in range(12) if all_head_data[f"L{l}H{h}"]["has_opposing"])
    n_high = sum(1 for h in range(12) if all_head_data[f"L{l}H{h}"]["opposition_ratio"] > 3)
    layer_stats.append((n_opp, n_high))

x = np.arange(12)
w = 0.35
ax.bar(x - w/2, [s[0] for s in layer_stats], w, label='Has opposing dirs', color='steelblue', alpha=0.8)
ax.bar(x + w/2, [s[1] for s in layer_stats], w, label='Opposition ratio > 3x', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'L{l}' for l in range(12)])
ax.set_xlabel('Layer')
ax.set_ylabel('Number of Heads')
ax.set_title('Sub-Head Opposition by Layer', fontsize=12)
ax.legend()

plt.suptitle(f'The False Negative Problem in Head-Level Interpretability\n'
             f'{len(false_negatives)} heads ({len(false_negatives)/144*100:.0f}%) have significant '
             f'hidden computation masked by internal cancellation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'false_negative_histogram.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {path}")

# Save
with open(os.path.join(OUT_DIR, "opposition_stats.json"), "w") as f:
    json.dump({"stats": stats, "all_heads": all_head_data}, f, indent=2)

print("False negative analysis complete.")

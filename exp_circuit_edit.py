# Circuit editing: surgically remove harmful SVD directions to improve task performance
# The key claim: you can make a head BETTER at its job by removing its internal opposition

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import json
from transformer_lens import HookedTransformer

OUT_DIR = "/Users/srimanarayana/Research Project I/results"

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

with open(os.path.join(OUT_DIR, "all_QK.pkl"), "rb") as f:
    all_QK = pickle.load(f)

with open(os.path.join(OUT_DIR, "ablation_results_fixed.json"), "r") as f:
    ablation_data = json.load(f)

baseline_ld = ablation_data["baseline_ioi_ld"]

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
        tmpl = templates[i % len(templates)]
        prompts.append({'prompt': tmpl.format(A=A, B=B), 'correct': A, 'incorrect': B})
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
        ld = (logits[0, -1, correct_id] - logits[0, -1, incorrect_id]).item()
        logit_diffs.append(ld)
    return np.mean(logit_diffs), np.std(logit_diffs)

print(f"Baseline IOI logit diff: {baseline_ld:.3f}")

# Step 1: Classify directions as helpful vs harmful for each head
def classify_directions(head_name, layer, head, max_k=10):
    abl = ablation_data["ioi_ablation"][head_name]
    individual = abl["individual"]
    full_effect = abl["full_head_effect"]

    helpful = []   # removing them hurts IOI (positive effect = helps task)
    harmful = []   # removing them helps IOI (negative effect = hurts task)
    neutral = []

    threshold = 0.02 * abs(baseline_ld)  # 2% of baseline as significance threshold
    # effect = baseline - ablated: positive means direction helped (removing it hurts)
    # negative means direction was harmful (removing it helps)
    for k, eff in enumerate(individual[:max_k]):
        if eff > threshold:
            helpful.append(k)   # removing hurts IOI -> direction is helpful
        elif eff < -threshold:
            harmful.append(k)   # removing helps IOI -> direction was harmful
        else:
            neutral.append(k)

    return helpful, harmful, neutral

# Step 2: Surgical editing — remove only harmful directions
def make_selective_ablation_hook(U, directions_to_remove):
    u_vecs = torch.tensor(U[:, directions_to_remove], dtype=torch.float32)
    def hook_fn(resid, hook):
        proj = torch.einsum('bsd,dk->bsk', resid, u_vecs)
        correction = torch.einsum('bsk,dk->bsd', proj, u_vecs)
        return resid - correction
    return hook_fn

def make_head_ablation_hook(head_idx):
    def hook_fn(q, hook):
        q[:, :, head_idx, :] = 0
        return q
    return hook_fn

# Run the circuit editing experiment
heads = {
    'S-Inhibition L9H6': (9, 6),
    'Name Mover L9H9': (9, 9),
    'Name Mover L10H0': (10, 0),
    'Induction L1H4': (1, 4),
    'Low-rank L2H2': (2, 2),
}

results = {}

for head_name, (layer, head) in heads.items():
    print(f"\n{'='*50}")
    print(f"Circuit editing: {head_name} (L{layer}H{head})")
    print(f"{'='*50}")

    U, S, Vt = all_QK[(layer, head)]
    helpful, harmful, neutral = classify_directions(head_name, layer, head)

    print(f"  Helpful directions (removing hurts IOI): {helpful}")
    print(f"  Harmful directions (removing helps IOI): {harmful}")
    print(f"  Neutral: {neutral}")

    hook_name = f'blocks.{layer}.hook_resid_pre'
    q_hook_name = f'blocks.{layer}.attn.hook_q'

    # Condition A: remove only harmful directions (the surgical fix)
    if harmful:
        hook_harmful = make_selective_ablation_hook(U, harmful)
        ld_remove_harmful, std_h = measure_ioi([(hook_name, hook_harmful)])
    else:
        ld_remove_harmful, std_h = baseline_ld, 0

    # Condition B: remove only helpful directions
    if helpful:
        hook_helpful = make_selective_ablation_hook(U, helpful)
        ld_remove_helpful, std_hp = measure_ioi([(hook_name, hook_helpful)])
    else:
        ld_remove_helpful, std_hp = baseline_ld, 0

    # Condition C: remove entire head
    hook_full = make_head_ablation_hook(head)
    ld_full_ablation, std_f = measure_ioi([(q_hook_name, hook_full)])

    # Condition D: remove both harmful AND helpful (what's left = neutral only)
    all_non_neutral = helpful + harmful
    if all_non_neutral:
        hook_both = make_selective_ablation_hook(U, all_non_neutral)
        ld_remove_both, std_b = measure_ioi([(hook_name, hook_both)])
    else:
        ld_remove_both, std_b = baseline_ld, 0

    improvement_over_baseline = ld_remove_harmful - baseline_ld
    improvement_over_full_ablation = ld_remove_harmful - ld_full_ablation

    print(f"\n  Results:")
    print(f"    Baseline:                    {baseline_ld:.3f}")
    print(f"    Remove harmful only:         {ld_remove_harmful:.3f} (delta: {improvement_over_baseline:+.3f})")
    print(f"    Remove helpful only:         {ld_remove_helpful:.3f}")
    print(f"    Remove full head:            {ld_full_ablation:.3f}")
    print(f"    Remove both (neutral only):  {ld_remove_both:.3f}")
    print(f"    Improvement over baseline:   {improvement_over_baseline:+.4f}")

    results[head_name] = {
        "layer": layer, "head": head,
        "helpful_directions": helpful,
        "harmful_directions": harmful,
        "neutral_directions": neutral,
        "singular_values": [float(S[k]) for k in range(min(10, len(S)))],
        "baseline": float(baseline_ld),
        "remove_harmful": float(ld_remove_harmful),
        "remove_helpful": float(ld_remove_helpful),
        "full_ablation": float(ld_full_ablation),
        "remove_both": float(ld_remove_both),
        "improvement_over_baseline": float(improvement_over_baseline),
        "improvement_over_full_ablation": float(improvement_over_full_ablation),
    }

# Multi-head surgical editing: apply surgical fixes to MULTIPLE heads simultaneously
print(f"\n{'='*60}")
print("Multi-head surgical editing")
print(f"{'='*60}")

multi_hooks = []
heads_edited = []
for head_name, (layer, head) in heads.items():
    harmful = results[head_name]["harmful_directions"]
    if harmful:
        U, S, Vt = all_QK[(layer, head)]
        hook = make_selective_ablation_hook(U, harmful)
        hook_name = f'blocks.{layer}.hook_resid_pre'
        multi_hooks.append((hook_name, hook))
        heads_edited.append(head_name)

if multi_hooks:
    ld_multi, std_multi = measure_ioi(multi_hooks)
    multi_improvement = ld_multi - baseline_ld
    print(f"  Edited heads: {heads_edited}")
    print(f"  Multi-head surgical result: {ld_multi:.3f} (delta: {multi_improvement:+.4f})")
    results["multi_head_surgical"] = {
        "heads_edited": heads_edited,
        "result": float(ld_multi),
        "improvement": float(multi_improvement),
    }

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes_flat = axes.flatten()

for idx, (head_name, res) in enumerate(results.items()):
    if head_name == "multi_head_surgical" or idx >= 5:
        continue
    ax = axes_flat[idx]

    conditions = ['Baseline', 'Remove\nHarmful\n(surgical)', 'Remove\nHelpful', 'Full Head\nAblation', 'Remove\nBoth']
    values = [res['baseline'], res['remove_harmful'], res['remove_helpful'],
              res['full_ablation'], res['remove_both']]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#f44336', '#9C27B0']

    bars = ax.bar(conditions, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # highlight the surgical fix
    if res['improvement_over_baseline'] > 0:
        bars[1].set_edgecolor('#00C853')
        bars[1].set_linewidth(3)

    ax.axhline(res['baseline'], color='blue', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_ylabel('IOI Logit Difference')
    ax.set_title(f'{head_name}\n'
                 f'Harmful dirs: {res["harmful_directions"]}, '
                 f'Helpful dirs: {res["helpful_directions"]}',
                 fontsize=9)
    ax.tick_params(axis='x', labelsize=7)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# multi-head result in last panel
ax = axes_flat[5]
if "multi_head_surgical" in results:
    multi = results["multi_head_surgical"]
    bars = ax.bar(['Baseline', 'Multi-Head\nSurgical Edit'],
                  [baseline_ld, multi["result"]],
                  color=['#2196F3', '#00C853'], alpha=0.85, edgecolor='black')
    ax.set_ylabel('IOI Logit Difference')
    ax.set_title(f'Simultaneous surgical editing\n{len(multi["heads_edited"])} heads edited', fontsize=10)
    for bar, val in zip(bars, [baseline_ld, multi["result"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
else:
    ax.set_visible(False)

plt.suptitle('Circuit Editing via SVD Direction Surgery\n'
             'Green border = surgical improvement over baseline',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'circuit_editing_results.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {path}")

# Dose-response curve: gradually scale down harmful directions instead of full removal
print(f"\n{'='*60}")
print("Dose-response: scaling harmful directions in L9H6")
print(f"{'='*60}")

layer, head = 9, 6
U, S, Vt = all_QK[(layer, head)]
harmful = results['S-Inhibition L9H6']['harmful_directions']

if harmful:
    scales = np.linspace(0, 1.5, 16)  # 0 = full removal, 1 = normal, >1 = amplification
    dose_results = []

    for scale in scales:
        u_vecs = torch.tensor(U[:, harmful], dtype=torch.float32)

        def make_scale_hook(u_matrix, s):
            U_mat = u_matrix.clone()
            scale_factor = float(s)
            def hook_fn(resid, hook):
                proj = torch.einsum('bsd,dk->bsk', resid, U_mat)
                correction = torch.einsum('bsk,dk->bsd', proj, U_mat)
                # instead of removing entirely, scale the projection
                return resid - (1 - scale_factor) * correction
            return hook_fn

        hook = make_scale_hook(u_vecs, scale)
        ld, _ = measure_ioi([(f'blocks.{layer}.hook_resid_pre', hook)])
        dose_results.append(float(ld))
        print(f"  Scale={scale:.2f}: IOI LD = {ld:.3f}")

    results['dose_response_L9H6'] = {
        'scales': scales.tolist(),
        'logit_diffs': dose_results,
        'harmful_directions': harmful,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(scales, dose_results, 'o-', color='#E91E63', linewidth=2, markersize=6)
    ax.axhline(baseline_ld, color='blue', linestyle='--', alpha=0.5, label='Baseline (no intervention)')
    ax.axvline(1.0, color='grey', linestyle=':', alpha=0.5, label='Scale=1.0 (normal)')
    ax.axvline(0.0, color='green', linestyle=':', alpha=0.5, label='Scale=0.0 (full removal)')

    best_idx = np.argmax(dose_results)
    best_scale = scales[best_idx]
    best_ld = dose_results[best_idx]
    ax.annotate(f'Best: scale={best_scale:.2f}\nLD={best_ld:.3f}',
                xy=(best_scale, best_ld), xytext=(best_scale + 0.2, best_ld + 0.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, fontweight='bold')

    ax.set_xlabel('Scale factor for harmful directions\n(0=removed, 1=normal, >1=amplified)')
    ax.set_ylabel('IOI Logit Difference')
    ax.set_title('L9H6 Dose-Response: Scaling Harmful SVD Directions\n'
                 'Higher = better IOI performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'dose_response_L9H6.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

with open(os.path.join(OUT_DIR, "circuit_editing_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nCircuit editing experiment complete.")

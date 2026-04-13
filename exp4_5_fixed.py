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

print("Loading model and decompositions")
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
        tmpl = templates[i % len(templates)]
        prompt = tmpl.format(A=A, B=B)
        prompts.append({'prompt': prompt, 'correct': A, 'incorrect': B})
    return prompts

ioi_data = create_ioi_dataset(100)

def measure_ioi_with_hooks(model, ioi_data, fwd_hooks=None):
    logit_diffs = []
    for item in ioi_data:
        tokens = model.to_tokens(item['prompt'])
        with torch.no_grad():
            if fwd_hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            else:
                logits = model(tokens)
        correct_id = model.to_single_token(item['correct'])
        incorrect_id = model.to_single_token(item['incorrect'])
        ld = (logits[0, -1, correct_id] - logits[0, -1, incorrect_id]).item()
        logit_diffs.append(ld)
    return np.mean(logit_diffs), np.std(logit_diffs)

baseline_ld, baseline_std = measure_ioi_with_hooks(model, ioi_data)
print(f"Baseline IOI logit diff: {baseline_ld:.3f} +/- {baseline_std:.3f}")

def ablate_sv_hook_based(model, layer, head, ioi_data, max_k=10):
    U, S, Vt = all_QK[(layer, head)]

    W_Q = model.W_Q[layer, head].detach().cpu()
    W_K = model.W_K[layer, head].detach().cpu()

    individual_results = []
    cumulative_results = []

    for k_target in range(max_k):
        u_k = torch.tensor(U[:, k_target], dtype=torch.float32)
        v_k = torch.tensor(Vt[k_target, :], dtype=torch.float32)
        sigma_k = float(S[k_target])

        q_dir = u_k @ W_Q
        k_dir = v_k @ W_K

        def make_q_projection_hook(u_vec, head_idx):
            u_t = u_vec.clone()
            def hook_fn(resid, hook):
                proj = torch.einsum('bsd,d->bs', resid, u_t)
                resid = resid - torch.einsum('bs,d->bsd', proj, u_t)
                return resid
            return hook_fn

        hook_fn_indiv = make_q_projection_hook(u_k, head)
        hook_name = f'blocks.{layer}.hook_resid_pre'

        ld_indiv, _ = measure_ioi_with_hooks(model, ioi_data,
                                               fwd_hooks=[(hook_name, hook_fn_indiv)])
        individual_results.append(baseline_ld - ld_indiv)

        u_vecs = torch.tensor(U[:, :k_target+1], dtype=torch.float32)

        def make_cumulative_hook(u_matrix):
            U_mat = u_matrix.clone()
            # U columns are orthonormal from SVD, so P = I - U @ U^T
            def hook_fn(resid, hook):
                proj = torch.einsum('bsd,dk->bsk', resid, U_mat)
                correction = torch.einsum('bsk,dk->bsd', proj, U_mat)
                return resid - correction
            return hook_fn

        hook_fn_cum = make_cumulative_hook(u_vecs)
        ld_cum, _ = measure_ioi_with_hooks(model, ioi_data,
                                             fwd_hooks=[(hook_name, hook_fn_cum)])
        cumulative_results.append(baseline_ld - ld_cum)

    def hook_zero_q(q, hook, head_idx=head):
        q[:, :, head_idx, :] = 0
        return q

    q_hook_name = f'blocks.{layer}.attn.hook_q'
    ld_full, _ = measure_ioi_with_hooks(model, ioi_data,
                                          fwd_hooks=[(q_hook_name, hook_zero_q)])
    full_effect = baseline_ld - ld_full

    return {
        'individual': [float(x) for x in individual_results],
        'cumulative': [float(x) for x in cumulative_results],
        'full_head_effect': float(full_effect),
        'singular_values': [float(x) for x in S[:max_k]],
        'baseline': float(baseline_ld)
    }

# Run ablation on key heads
heads_to_ablate = {
    'Name Mover L9H9': (9, 9),
    'Name Mover L10H0': (10, 0),
    'S-Inhibition L9H6': (9, 6),
    'Induction L1H4': (1, 4),
    'Low-rank L2H2': (2, 2),
}

print("\nHook-based IOI causal ablation")
ioi_results = {}

for name, (l, h) in heads_to_ablate.items():
    print(f"\nAblating {name} (L{l}H{h})...")
    res = ablate_sv_hook_based(model, l, h, ioi_data, max_k=10)
    ioi_results[name] = res
    print(f"  Baseline: {res['baseline']:.3f}")
    print(f"  Full head ablation effect: {res['full_head_effect']:.4f}")
    print(f"  Individual SV effects: {[f'{v:.4f}' for v in res['individual'][:5]]}")
    print(f"  Cumulative top-3 effect: {res['cumulative'][2]:.4f}")
    if abs(res['full_head_effect']) > 1e-6:
        pct = res['cumulative'][2] / res['full_head_effect'] * 100
        print(f"  Top-3 captures {pct:.1f}% of full head effect")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(ioi_results)))

for (name, res), color in zip(ioi_results.items(), colors):
    ranks = list(range(len(res['individual'])))
    axes[0].plot(ranks, res['individual'], marker='o', label=name, color=color, ms=4, linewidth=2)
    axes[1].plot(ranks, res['cumulative'], marker='s', label=name, color=color, ms=4, linewidth=2)
    axes[1].axhline(res['full_head_effect'], color=color, linestyle='--', alpha=0.4, linewidth=1)

axes[0].set_xlabel('Singular direction k')
axes[0].set_ylabel('IOI logit diff drop (individual)')
axes[0].set_title('Individual SV Ablation\n(effect of removing each direction alone)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='black', linewidth=0.8)

axes[1].set_xlabel('Top-k directions ablated (cumulative)')
axes[1].set_ylabel('IOI logit diff drop (cumulative)')
axes[1].set_title('Cumulative SV Ablation\n(dashed lines = full head ablation)')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='black', linewidth=0.8)

plt.suptitle('QK Singular Direction Ablation — IOI Task\n'
             'Positive = direction contributes to correct IOI behavior',
             fontsize=13, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'ablation_IOI_QK_fixed.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"\nSaved {path}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, (name, (l, h)) in enumerate(heads_to_ablate.items()):
    if idx >= len(axes_flat):
        break
    ax = axes_flat[idx]
    res = ioi_results[name]
    imp = res['individual']
    S = res['singular_values']
    ranks = list(range(len(imp)))

    colors_bar = ['#4C72B0' if v > 0 else '#DD8452' for v in imp]
    ax.bar(ranks, imp, color=colors_bar, alpha=0.8)

    ax2 = ax.twinx()
    s_norm = [s / S[0] for s in S[:len(imp)]]
    ax2.plot(ranks, s_norm, 'k--', marker='s', ms=3, alpha=0.6, label='Norm SV')
    ax2.set_ylabel('Normalized SV', fontsize=8)
    ax2.legend(fontsize=7, loc='upper right')

    ax.set_xlabel('Singular direction k')
    ax.set_ylabel('IOI logit diff drop')
    ax.set_title(f'{name}\nFull head: {res["full_head_effect"]:.3f}', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(ranks)

if len(heads_to_ablate) < len(axes_flat):
    axes_flat[-1].set_visible(False)

plt.suptitle('Per-Direction Causal Importance on IOI Task (Hook-Based Ablation)\n'
             'Blue = direction helps IOI | Orange = direction hurts IOI',
             fontsize=13, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'sv_causal_importance_ioi_fixed.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

with open(os.path.join(OUT_DIR, "ablation_results_fixed.json"), "w") as f:
    json.dump({
        'baseline_ioi_ld': float(baseline_ld),
        'baseline_ioi_std': float(baseline_std),
        'ioi_ablation': ioi_results,
    }, f, indent=2)

print("\nDone.")

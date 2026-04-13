#!/usr/bin/env python3
"""
Attention-Score-Level Ablation: QK-Isolated Causal Intervention

Addresses the primary reviewer critique: residual-stream ablation at hook_resid_pre
affects ALL downstream computation, not just the target head's QK circuit.

This experiment intervenes at hook_attn_scores to subtract individual rank-1
SVD components from the attention logits of ONLY the target head, isolating
the QK-specific causal effect.

Compares direction-by-direction results against the existing residual-stream method.
"""

import torch
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

print("Loading TransformerLens...")
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

# ── Load model and precomputed SVD ──────────────────────────────────────
print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

with open("results/all_QK.pkl", "rb") as f:
    all_QK = pickle.load(f)

# ── IOI dataset ─────────────────────────────────────────────────────────
def create_ioi_dataset(n=100):
    templates = [
        "When {A} and {B} went to the store, {B} gave a drink to",
        "When {A} and {B} went to the park, {B} gave a flower to",
        "When {A} and {B} walked in, {B} said hello to",
        "After {A} and {B} arrived at the party, {B} handed a gift to",
        "When {A} and {B} entered the room, {B} waved at",
    ]
    names_A = ["Mary", "Alice", "Sarah", "Emma", "Lisa"]
    names_B = ["John", "Bob", "James", "Tom", "Mike"]
    prompts, correct, incorrect = [], [], []
    for i in range(n):
        t = templates[i % len(templates)]
        a = names_A[i % len(names_A)]
        b = names_B[i % len(names_B)]
        prompts.append(t.format(A=a, B=b))
        correct.append(a)  # indirect object
        incorrect.append(b)  # subject (wrong answer)
    return prompts, correct, incorrect

prompts, correct_names, incorrect_names = create_ioi_dataset(100)

# Tokenize
tokens = model.to_tokens(prompts, prepend_bos=True)
correct_ids = [model.to_single_token(" " + n) for n in correct_names]
incorrect_ids = [model.to_single_token(" " + n) for n in incorrect_names]

def compute_logit_diff(logits):
    """IOI logit difference: logit(correct) - logit(incorrect)"""
    last_logits = logits[:, -1, :]
    diffs = []
    for i in range(len(correct_ids)):
        d = last_logits[i, correct_ids[i]] - last_logits[i, incorrect_ids[i]]
        diffs.append(d.item())
    return np.mean(diffs)

# ── Baseline ────────────────────────────────────────────────────────────
print("Computing baseline...")
with torch.no_grad():
    baseline_logits = model(tokens)
baseline_ld = compute_logit_diff(baseline_logits)
print(f"Baseline IOI logit diff: {baseline_ld:.4f}")

# ── Heads to analyze ────────────────────────────────────────────────────
heads = [
    ("S-Inhibition L9H6", 9, 6),
    ("Name Mover L9H9", 9, 9),
    ("Name Mover L10H0", 10, 0),
    ("Induction L1H4", 1, 4),
    ("Low-rank L2H2", 2, 2),
]

d_model = model.cfg.d_model
d_head = model.cfg.d_head
n_sv = 10  # top 10 singular directions

results = {}

for head_name, layer, head_idx in heads:
    print(f"\n{'='*60}")
    print(f"Processing {head_name} (Layer {layer}, Head {head_idx})")
    print(f"{'='*60}")

    key = (layer, head_idx)
    entry = all_QK[key]
    if isinstance(entry, dict):
        U, S, Vt = entry["U"], entry["S"], entry["Vt"]
    else:  # tuple
        U, S, Vt = entry

    resid_effects = []
    attn_score_effects = []

    for sv_idx in range(n_sv):
        u_k = torch.tensor(U[:, sv_idx], dtype=torch.float32)  # query direction
        v_k = torch.tensor(Vt[sv_idx, :], dtype=torch.float32)  # key direction
        sigma_k = S[sv_idx]

        # ── Method 1: Residual-stream ablation (existing method) ────────
        def resid_hook(resid, hook, u=u_k):
            proj = torch.einsum('bsd,d->bs', resid, u)
            correction = torch.einsum('bs,d->bsd', proj, u)
            return resid - correction

        hook_name_resid = get_act_name("resid_pre", layer)
        with torch.no_grad():
            logits_resid = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name_resid, resid_hook)]
            )
        ld_resid = compute_logit_diff(logits_resid)
        delta_resid = ld_resid - baseline_ld
        resid_effects.append(delta_resid)

        # ── Method 2: Attention-score ablation (QK-isolated) ────────────
        # Subtract σ_k * (x_i^T u_k)(v_k^T x_j) / sqrt(d_head) from
        # attention scores of ONLY the target head
        def attn_score_hook(attn_scores, hook, u=u_k, v=v_k, s=sigma_k, h=head_idx):
            # attn_scores shape: [batch, n_heads, seq_q, seq_k]
            # We need residual stream to compute the rank-1 contribution
            # But we're hooking attn_scores which is AFTER W_Q @ W_K^T
            # The rank-1 contribution to attention scores is:
            # σ_k * (x_i^T u_k)(v_k^T x_j) / sqrt(d_head)
            # We get this from the cache
            return attn_scores  # placeholder, we need a different approach

        # Better approach: cache residual stream, compute rank-1 contribution
        # Use two hooks: one to cache resid, one to modify attn_scores
        cached_resid = {}

        def cache_resid_hook(resid, hook):
            cached_resid['resid'] = resid.detach()
            return resid  # don't modify

        def subtract_rank1_hook(attn_scores, hook, u=u_k, v=v_k, s=sigma_k, h=head_idx):
            resid = cached_resid['resid']  # [batch, seq, d_model]
            # Query projections: x_i^T u_k -> [batch, seq]
            q_proj = torch.einsum('bsd,d->bs', resid, u)
            # Key projections: v_k^T x_j -> [batch, seq]
            k_proj = torch.einsum('bsd,d->bs', resid, v)
            # Rank-1 attention score contribution: [batch, seq_q, seq_k]
            rank1 = s * torch.einsum('bi,bj->bij', q_proj, k_proj) / np.sqrt(d_head)
            # Subtract from ONLY the target head
            modified = attn_scores.clone()
            modified[:, h, :, :] = modified[:, h, :, :] - rank1
            return modified

        hook_name_resid_cache = get_act_name("resid_pre", layer)
        hook_name_attn = get_act_name("attn_scores", layer)

        with torch.no_grad():
            cached_resid.clear()
            logits_attn = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (hook_name_resid_cache, cache_resid_hook),
                    (hook_name_attn, subtract_rank1_hook),
                ]
            )
        ld_attn = compute_logit_diff(logits_attn)
        delta_attn = ld_attn - baseline_ld
        attn_score_effects.append(delta_attn)

        print(f"  SV{sv_idx}: resid={delta_resid:+.4f}  attn_score={delta_attn:+.4f}  "
              f"σ={sigma_k:.3f}")

    # Compute correlation
    r_pearson, p_pearson = pearsonr(resid_effects, attn_score_effects)
    r_spearman, p_spearman = spearmanr(resid_effects, attn_score_effects)

    print(f"\n  Pearson r  = {r_pearson:.4f} (p={p_pearson:.2e})")
    print(f"  Spearman ρ = {r_spearman:.4f} (p={p_spearman:.2e})")

    results[head_name] = {
        "layer": layer,
        "head": head_idx,
        "resid_effects": resid_effects,
        "attn_score_effects": attn_score_effects,
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
        "singular_values": S[:n_sv].tolist(),
    }

# ── Save results ────────────────────────────────────────────────────────
with open("results/attn_score_ablation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/attn_score_ablation.json")

# ── Plot comparison ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Residual-Stream vs Attention-Score Ablation\n"
             "Comparing QK-isolated effects with global downstream effects",
             fontsize=13, fontweight='bold')

for idx, (head_name, layer, head_idx) in enumerate(heads):
    ax = axes[idx // 3][idx % 3]
    r = results[head_name]
    x = np.arange(n_sv)

    ax.bar(x - 0.2, r["resid_effects"], 0.35, label="Residual stream", color="#4A90D9", alpha=0.8)
    ax.bar(x + 0.2, r["attn_score_effects"], 0.35, label="Attn score (isolated)", color="#E8636F", alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel("Singular Direction Index")
    ax.set_ylabel("Δ Logit Diff")
    ax.set_title(f"{head_name}\nr={r['pearson_r']:.3f}, ρ={r['spearman_r']:.3f}")
    ax.set_xticks(x)
    ax.legend(fontsize=7)

# Scatter plot in the 6th panel
ax = axes[1][2]
all_resid = []
all_attn = []
colors = ['#4A90D9', '#E8636F', '#82C991', '#F5A623', '#9B59B6']
for idx, (head_name, _, _) in enumerate(heads):
    r = results[head_name]
    ax.scatter(r["resid_effects"], r["attn_score_effects"],
               c=colors[idx], label=head_name.split()[0][:8], alpha=0.7, s=40)
    all_resid.extend(r["resid_effects"])
    all_attn.extend(r["attn_score_effects"])

# Overall correlation
r_all, p_all = pearsonr(all_resid, all_attn)
lims = [min(min(all_resid), min(all_attn)) - 0.1, max(max(all_resid), max(all_attn)) + 0.1]
ax.plot(lims, lims, 'k--', alpha=0.3, label=f"y=x (r={r_all:.3f})")
ax.set_xlabel("Residual-stream Δ")
ax.set_ylabel("Attn-score Δ (isolated)")
ax.set_title(f"Overall correlation\nr={r_all:.3f} (p={p_all:.2e})")
ax.legend(fontsize=6)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("results/attn_score_ablation.png", dpi=150, bbox_inches='tight')
print("Plot saved to results/attn_score_ablation.png")
print("\nDone!")

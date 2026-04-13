# SVD of W_QK and W_OV for all GPT-2 small heads
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from transformer_lens import HookedTransformer
import pickle
import os

OUT_DIR = "/Users/srimanarayana/Research Project I/results"
os.makedirs(OUT_DIR, exist_ok=True)

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

n_layers = model.cfg.n_layers
n_heads  = model.cfg.n_heads
d_model  = model.cfg.d_model
d_head   = model.cfg.d_head
d_vocab  = model.cfg.d_vocab

W_E = model.W_E.detach().cpu().numpy()
W_U = model.W_U.detach().cpu().numpy()

def get_composite_matrices(model, layer, head):
    W_Q = model.W_Q[layer, head].detach().cpu().numpy().astype(np.float64)
    W_K = model.W_K[layer, head].detach().cpu().numpy().astype(np.float64)
    W_V = model.W_V[layer, head].detach().cpu().numpy().astype(np.float64)
    W_O = model.W_O[layer, head].detach().cpu().numpy().astype(np.float64)
    W_QK = W_Q @ W_K.T
    W_OV = W_V @ W_O
    return W_QK, W_OV

def decompose(W):
    U, S, Vt = svd(W, full_matrices=False)
    return U, S, Vt

all_QK = {}
all_OV = {}

for layer in range(n_layers):
    for head in range(n_heads):
        W_QK, W_OV = get_composite_matrices(model, layer, head)
        all_QK[(layer, head)] = decompose(W_QK)
        all_OV[(layer, head)] = decompose(W_OV)

with open(os.path.join(OUT_DIR, "all_QK.pkl"), "wb") as f:
    pickle.dump(all_QK, f)
with open(os.path.join(OUT_DIR, "all_OV.pkl"), "wb") as f:
    pickle.dump(all_OV, f)
np.save(os.path.join(OUT_DIR, "W_E.npy"), W_E)
np.save(os.path.join(OUT_DIR, "W_U.npy"), W_U)

def plot_sv_heatmap(all_decomps, circuit_name, top_k=20):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    data_top1 = np.zeros((n_layers, n_heads))
    for (l, h), (U, S, Vt) in all_decomps.items():
        data_top1[l, h] = S[0]

    sns.heatmap(data_top1, ax=axes[0], cmap='viridis',
                xticklabels=[f'H{h}' for h in range(n_heads)],
                yticklabels=[f'L{l}' for l in range(n_layers)],
                annot=True, fmt='.1f', annot_kws={'fontsize': 7})
    axes[0].set_title(f'{circuit_name}: Largest Singular Value per Head', fontsize=12)
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')

    # effective rank = number of SVs for 90% of Frobenius norm
    data_rank = np.zeros((n_layers, n_heads))
    for (l, h), (U, S, Vt) in all_decomps.items():
        cumulative = np.cumsum(S**2) / np.sum(S**2)
        data_rank[l, h] = np.searchsorted(cumulative, 0.9) + 1

    selected_heads = [(0,0), (1,4), (3,0), (5,5), (9,9), (9,6), (10,0), (10,7), (11,10)]
    for (l, h) in selected_heads:
        _, S, _ = all_decomps[(l, h)]
        axes[1].plot(S[:top_k] / S[0], label=f'L{l}H{h}', marker='o', ms=3)
    axes[1].set_xlabel('Singular value rank')
    axes[1].set_ylabel('Normalized magnitude (S_k / S_0)')
    axes[1].set_title(f'{circuit_name}: Singular Value Spectrum (selected heads)')
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{circuit_name}_singular_values.png')
    plt.savefig(path, dpi=150)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(data_rank, ax=ax2, cmap='YlOrRd',
                xticklabels=[f'H{h}' for h in range(n_heads)],
                yticklabels=[f'L{l}' for l in range(n_layers)],
                annot=True, fmt='.0f', annot_kws={'fontsize': 7})
    ax2.set_title(f'{circuit_name}: Effective Rank (90% Frobenius norm)', fontsize=12)
    ax2.set_xlabel('Head')
    ax2.set_ylabel('Layer')
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, f'{circuit_name}_effective_rank.png')
    plt.savefig(path2, dpi=150)
    plt.close()

plot_sv_heatmap(all_QK, 'W_QK')
plot_sv_heatmap(all_OV, 'W_OV')

print()
for circuit_name, decomps in [('W_QK', all_QK), ('W_OV', all_OV)]:
    top1_vals = [decomps[(l,h)][1][0] for l in range(n_layers) for h in range(n_heads)]
    eff_ranks = []
    for l in range(n_layers):
        for h in range(n_heads):
            S = decomps[(l,h)][1]
            cum = np.cumsum(S**2) / np.sum(S**2)
            eff_ranks.append(np.searchsorted(cum, 0.9) + 1)
    print(f"\n{circuit_name}:")
    print(f"  Top-1 SV: mean={np.mean(top1_vals):.2f}, max={np.max(top1_vals):.2f}, min={np.min(top1_vals):.2f}")
    print(f"  Effective rank (90%): mean={np.mean(eff_ranks):.1f}, max={np.max(eff_ranks):.0f}, min={np.min(eff_ranks):.0f}")

    ranked = sorted([(l,h,r) for (l,h),r in zip([(l,h) for l in range(n_layers) for h in range(n_heads)], eff_ranks)], key=lambda x: x[2])
    print(f"  Most low-rank (top 10):")
    for l, h, r in ranked[:10]:
        print(f"    L{l}H{h}: eff_rank={r:.0f}, top_SV={decomps[(l,h)][1][0]:.2f}")

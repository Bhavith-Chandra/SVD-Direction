import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from scipy.spatial.distance import jensenshannon
from transformer_lens import HookedTransformer

OUT_DIR = "/Users/srimanarayana/Research Project I/results"

print("Loading model and decompositions")
model = HookedTransformer.from_pretrained("gpt2")
model.eval()

with open(os.path.join(OUT_DIR, "all_QK.pkl"), "rb") as f:
    all_QK = pickle.load(f)

W_E = np.load(os.path.join(OUT_DIR, "W_E.npy"))
n_layers = 12
n_heads = 12

def measure_qk_semantic_alignment(layer, head, top_k_sv=10):
    U, S, Vt = all_QK[(layer, head)]
    alignments = []

    for k in range(min(top_k_sv, len(S))):
        u_k = U[:, k]
        v_k = Vt[k, :]

        q_scores = W_E @ u_k
        k_scores = W_E @ v_k

        cos_sim = np.dot(q_scores, k_scores) / (
            np.linalg.norm(q_scores) * np.linalg.norm(k_scores) + 1e-10
        )

        direct_cos = np.dot(u_k, v_k) / (
            np.linalg.norm(u_k) * np.linalg.norm(v_k) + 1e-10
        )

        top_n = 50
        q_top = set(np.argsort(np.abs(q_scores))[-top_n:])
        k_top = set(np.argsort(np.abs(k_scores))[-top_n:])
        overlap = len(q_top & k_top) / top_n

        q_soft = np.exp(q_scores / (q_scores.std() + 1e-10))
        q_soft = q_soft / q_soft.sum()
        k_soft = np.exp(k_scores / (k_scores.std() + 1e-10))
        k_soft = k_soft / k_soft.sum()

        try:
            js = float(jensenshannon(q_soft, k_soft))
        except:
            js = float('nan')

        alignments.append({
            'k': k, 'sigma': float(S[k]), 'sigma_norm': float(S[k] / S[0]),
            'token_cosine': float(cos_sim),
            'direct_cosine': float(direct_cos),
            'top50_overlap': float(overlap),
            'js_divergence': js
        })

    return alignments

print("Computing Q-K alignment for all heads")
all_alignments = {}
for l in range(n_layers):
    for h in range(n_heads):
        all_alignments[(l, h)] = measure_qk_semantic_alignment(l, h, top_k_sv=10)
    print(f"  Layer {l} done.")

alignment_cos_top3 = np.zeros((n_layers, n_heads))
for (l, h), aligns in all_alignments.items():
    alignment_cos_top3[l, h] = np.mean([a['token_cosine'] for a in aligns[:3]])

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(alignment_cos_top3, ax=ax, cmap='coolwarm', center=0,
            xticklabels=[f'H{h}' for h in range(n_heads)],
            yticklabels=[f'L{l}' for l in range(n_layers)],
            annot=True, fmt='.2f', linewidths=0.5, annot_kws={'fontsize': 7})
ax.set_title('Q-K Semantic Alignment: Token-Space Cosine Similarity (mean of top-3 SVs)\n'
             'Positive = semantic matching | Near-zero = orthogonal | Negative = contrastive',
             fontsize=11)
ax.set_xlabel('Head')
ax.set_ylabel('Layer')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'qk_semantic_alignment_cosine_heatmap.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

direct_cos_map = np.zeros((n_layers, n_heads))
for (l, h), aligns in all_alignments.items():
    direct_cos_map[l, h] = np.mean([a['direct_cosine'] for a in aligns[:3]])

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(direct_cos_map, ax=ax, cmap='coolwarm', center=0,
            xticklabels=[f'H{h}' for h in range(n_heads)],
            yticklabels=[f'L{l}' for l in range(n_layers)],
            annot=True, fmt='.2f', linewidths=0.5, annot_kws={'fontsize': 7})
ax.set_title('Q-K Direct Cosine Similarity in d_model Space (mean of top-3 SVs)\n'
             'How geometrically close are query and key singular vectors?',
             fontsize=11)
ax.set_xlabel('Head')
ax.set_ylabel('Layer')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'qk_direct_cosine_heatmap.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

overlap_map = np.zeros((n_layers, n_heads))
for (l, h), aligns in all_alignments.items():
    overlap_map[l, h] = np.mean([a['top50_overlap'] for a in aligns[:3]])

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(overlap_map, ax=ax, cmap='YlOrRd',
            xticklabels=[f'H{h}' for h in range(n_heads)],
            yticklabels=[f'L{l}' for l in range(n_layers)],
            annot=True, fmt='.2f', linewidths=0.5, annot_kws={'fontsize': 7})
ax.set_title('Q-K Top-50 Token Overlap (mean of top-3 SVs)\n'
             'Fraction of top-50 most-activated tokens shared between Q and K directions',
             fontsize=11)
ax.set_xlabel('Head')
ax.set_ylabel('Layer')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'qk_token_overlap_heatmap.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

key_heads = [(9, 9), (9, 6), (10, 0), (1, 4), (2, 2), (11, 10), (11, 8)]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes.flatten()

for idx, (l, h) in enumerate(key_heads):
    if idx >= len(axes_flat):
        break
    ax = axes_flat[idx]
    aligns = all_alignments[(l, h)]

    ks = [a['k'] for a in aligns]
    cos_vals = [a['token_cosine'] for a in aligns]
    overlap_vals = [a['top50_overlap'] for a in aligns]
    sv_norms = [a['sigma_norm'] for a in aligns]

    ax.bar(ks, cos_vals, alpha=0.7, color='steelblue', label='Token cosine')
    ax.plot(ks, sv_norms, 'k--', marker='s', ms=3, alpha=0.6, label='Norm SV')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Singular direction k')
    ax.set_ylabel('Cosine similarity')
    ax.set_title(f'L{l}H{h}', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.2)

if len(key_heads) < len(axes_flat):
    axes_flat[-1].set_visible(False)

plt.suptitle('Per-SV Q-K Semantic Alignment for Key Heads\n'
             'Positive cosine = Q and K activate similar tokens (semantic match)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'qk_alignment_per_sv.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

print("\nComputing random baseline")
np.random.seed(42)
n_random = 1000
random_cosines = []
for _ in range(n_random):
    u_rand = np.random.randn(768)
    v_rand = np.random.randn(768)
    u_rand /= np.linalg.norm(u_rand)
    v_rand /= np.linalg.norm(v_rand)
    q_scores = W_E @ u_rand
    k_scores = W_E @ v_rand
    cos = np.dot(q_scores, k_scores) / (np.linalg.norm(q_scores) * np.linalg.norm(k_scores) + 1e-10)
    random_cosines.append(cos)

random_mean = np.mean(random_cosines)
random_std = np.std(random_cosines)
print(f"Random baseline: mean={random_mean:.4f}, std={random_std:.4f}")

sig_threshold = random_mean + 2 * random_std
sig_neg_threshold = random_mean - 2 * random_std

above_random = []
below_random = []
for (l, h), aligns in all_alignments.items():
    mean_cos = np.mean([a['token_cosine'] for a in aligns[:3]])
    if mean_cos > sig_threshold:
        above_random.append((l, h, mean_cos))
    if mean_cos < sig_neg_threshold:
        below_random.append((l, h, mean_cos))

above_random.sort(key=lambda x: -x[2])
below_random.sort(key=lambda x: x[2])

print(f"\nHeads with POSITIVE semantic alignment (>{sig_threshold:.3f}):")
for l, h, c in above_random[:15]:
    print(f"  L{l}H{h}: {c:.3f}")

print(f"\nHeads with NEGATIVE/contrastive alignment (<{sig_neg_threshold:.3f}):")
for l, h, c in below_random[:15]:
    print(f"  L{l}H{h}: {c:.3f}")

all_cosines = [np.mean([a['token_cosine'] for a in all_alignments[(l, h)][:3]])
               for l in range(n_layers) for h in range(n_heads)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(random_cosines, bins=50, alpha=0.5, density=True, label='Random directions', color='gray')
ax.hist(all_cosines, bins=30, alpha=0.7, density=True, label='Actual Q-K pairs', color='steelblue')
ax.axvline(random_mean, color='gray', linestyle='--', label=f'Random mean ({random_mean:.3f})')
ax.axvline(sig_threshold, color='red', linestyle='--', alpha=0.5, label=f'2σ threshold ({sig_threshold:.3f})')
ax.set_xlabel('Token-space cosine similarity')
ax.set_ylabel('Density')
ax.set_title('Distribution of Q-K Semantic Alignment\n'
             'Actual SVD directions vs. random baseline')
ax.legend()
plt.tight_layout()
path = os.path.join(OUT_DIR, 'qk_alignment_distribution.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

alignment_data = {}
for (l, h), aligns in all_alignments.items():
    alignment_data[f'L{l}H{h}'] = aligns

with open(os.path.join(OUT_DIR, "alignment_results.json"), "w") as f:
    json.dump({
        'alignments': alignment_data,
        'random_baseline': {'mean': random_mean, 'std': random_std},
        'above_random': [(l, h, c) for l, h, c in above_random],
        'below_random': [(l, h, c) for l, h, c in below_random],
    }, f, indent=2)

print("\nDone.")

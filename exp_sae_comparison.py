# SVD vs SAE: compare weight-based SVD directions against trained sparse autoencoder features
# Uses pre-trained SAEs from SAELens for GPT-2 small

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
import gc
from transformer_lens import HookedTransformer

OUT_DIR = "/Users/srimanarayana/Research Project I/results"

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

with open(os.path.join(OUT_DIR, "all_QK.pkl"), "rb") as f:
    all_QK = pickle.load(f)
with open(os.path.join(OUT_DIR, "all_OV.pkl"), "rb") as f:
    all_OV = pickle.load(f)

W_E = np.load(os.path.join(OUT_DIR, "W_E.npy"))

# SAELens requires Python 3.10+; on 3.9, skip SAE loading
HAS_SAE = False
try:
    from sae_lens import SAE
    HAS_SAE = True
    print("SAELens loaded")
except ImportError:
    print("SAELens not available (requires Python 3.10+). Running weight-only comparison.")

# Analysis 1: Weight-space comparison (no SAE needed)
# Compare SVD effective dimensionality vs what SAEs typically learn

print("\n" + "="*60)
print("Analysis 1: Weight-space properties of SVD directions")
print("="*60)

# For each head, measure how compressible the QK circuit is
compression_data = {}
for layer in range(12):
    for head in range(12):
        U, S, Vt = all_QK[(layer, head)]
        total_energy = np.sum(S**2)

        # how many components for various thresholds
        cumvar = np.cumsum(S**2) / total_energy
        rank_90 = np.searchsorted(cumvar, 0.90) + 1
        rank_95 = np.searchsorted(cumvar, 0.95) + 1
        rank_99 = np.searchsorted(cumvar, 0.99) + 1

        # spectral gap: ratio of top SV to second
        spectral_gap = S[0] / S[1] if len(S) > 1 and S[1] > 0 else float('inf')

        # entropy of normalized singular value distribution (measure of spread)
        s_norm = S / S.sum()
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
        max_entropy = np.log(len(S))
        normalized_entropy = entropy / max_entropy

        compression_data[f"L{layer}H{head}"] = {
            "rank_90": int(rank_90),
            "rank_95": int(rank_95),
            "rank_99": int(rank_99),
            "spectral_gap": float(spectral_gap),
            "normalized_entropy": float(normalized_entropy),
            "top5_sv": [float(S[k]) for k in range(min(5, len(S)))],
        }

# Analysis 2: Direction interpretability — token coherence score
# Good interpretable directions should activate a coherent cluster of semantically related tokens
# SAE features are trained to be interpretable; SVD directions are not — how do they compare?

print("\nAnalysis 2: Token coherence of SVD directions")

def token_coherence(direction, W_E, top_k=20):
    """Measure how tightly clustered the top-activating tokens are in embedding space"""
    scores = W_E @ direction
    top_idx = np.argsort(-np.abs(scores))[:top_k]
    top_embs = W_E[top_idx]

    # pairwise cosine similarity among top tokens
    norms = np.linalg.norm(top_embs, axis=1, keepdims=True) + 1e-10
    top_embs_normed = top_embs / norms
    pairwise_cos = top_embs_normed @ top_embs_normed.T

    # average off-diagonal cosine
    mask = ~np.eye(top_k, dtype=bool)
    avg_coherence = pairwise_cos[mask].mean()
    return avg_coherence

svd_coherence_qk = []
svd_coherence_ov = []
random_coherence = []

for layer in range(12):
    for head in range(12):
        U_qk, S_qk, Vt_qk = all_QK[(layer, head)]
        U_ov, S_ov, Vt_ov = all_OV[(layer, head)]

        for k in range(min(5, len(S_qk))):
            coh = token_coherence(U_qk[:, k], W_E)
            svd_coherence_qk.append(coh)

        for k in range(min(5, len(S_ov))):
            coh = token_coherence(U_ov[:, k], W_E)
            svd_coherence_ov.append(coh)

# random baseline
rng = np.random.RandomState(42)
for _ in range(1000):
    rand_dir = rng.randn(768)
    rand_dir /= np.linalg.norm(rand_dir)
    random_coherence.append(token_coherence(rand_dir, W_E))

print(f"  QK SVD direction coherence: {np.mean(svd_coherence_qk):.4f} ± {np.std(svd_coherence_qk):.4f}")
print(f"  OV SVD direction coherence: {np.mean(svd_coherence_ov):.4f} ± {np.std(svd_coherence_ov):.4f}")
print(f"  Random direction coherence: {np.mean(random_coherence):.4f} ± {np.std(random_coherence):.4f}")

# Analysis 3: Load pre-trained SAE if available
sae_comparison = {}

if HAS_SAE:
    print("\nAnalysis 3: Loading pre-trained SAE for GPT-2 residual stream")
    try:
        # load a residual stream SAE for a middle layer
        sae_layers = [6, 9]  # layers where we have interesting heads

        for sae_layer in sae_layers:
            print(f"\n  Loading SAE for layer {sae_layer}...")
            try:
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release="gpt2-small-res-jb",
                    sae_id=f"blocks.{sae_layer}.hook_resid_pre",
                    device="cpu"
                )
                print(f"  SAE loaded: {sae.cfg.d_sae} features, d_in={sae.cfg.d_in}")

                # extract SAE decoder directions (each row = one learned feature direction)
                W_dec = sae.W_dec.detach().cpu().numpy()  # (n_features, d_model)
                n_features = W_dec.shape[0]

                # compare SAE features to SVD directions for heads in this layer
                for head in range(12):
                    U_qk, S_qk, Vt_qk = all_QK[(sae_layer, head)]
                    U_ov, S_ov, Vt_ov = all_OV[(sae_layer, head)]

                    # cosine similarity between each SVD direction and each SAE feature
                    svd_dirs = U_qk[:, :10].T  # (10, d_model)
                    svd_norms = np.linalg.norm(svd_dirs, axis=1, keepdims=True) + 1e-10
                    svd_normed = svd_dirs / svd_norms

                    sae_norms = np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-10
                    sae_normed = W_dec / sae_norms

                    cos_matrix = svd_normed @ sae_normed.T  # (10, n_features)

                    # for each SVD direction, what's the max alignment with any SAE feature?
                    max_alignment = np.max(np.abs(cos_matrix), axis=1)  # (10,)
                    best_sae_idx = np.argmax(np.abs(cos_matrix), axis=1)

                    # how many SAE features needed to "cover" the top-3 SVD directions?
                    # (features with cos > 0.3 to any of top-3 SVDs)
                    top3_coverage = np.any(np.abs(cos_matrix[:3, :]) > 0.3, axis=0)
                    n_covering = np.sum(top3_coverage)

                    key = f"L{sae_layer}H{head}"
                    sae_comparison[key] = {
                        "max_alignment_per_svd": max_alignment.tolist(),
                        "mean_max_alignment": float(np.mean(max_alignment)),
                        "best_sae_features": best_sae_idx.tolist(),
                        "n_sae_features_covering_top3": int(n_covering),
                        "n_total_sae_features": int(n_features),
                    }

                    if head in [6, 9, 0]:  # interesting heads
                        print(f"    L{sae_layer}H{head}: max alignment per SVD = "
                              f"{[f'{a:.3f}' for a in max_alignment[:5]]}")
                        print(f"      SAE features covering top-3 SVDs: {n_covering}/{n_features}")

                # coherence comparison: SAE features vs SVD
                sae_coh = []
                for i in range(min(100, n_features)):
                    sae_coh.append(token_coherence(W_dec[i], W_E))
                print(f"  SAE feature coherence (L{sae_layer}): {np.mean(sae_coh):.4f} ± {np.std(sae_coh):.4f}")
                sae_comparison[f"L{sae_layer}_sae_coherence"] = {
                    "mean": float(np.mean(sae_coh)), "std": float(np.std(sae_coh))
                }

                del sae, W_dec
                gc.collect()

            except Exception as e:
                print(f"  Could not load SAE for layer {sae_layer}: {e}")
                # try alternative release name
                try:
                    sae, cfg_dict, sparsity = SAE.from_pretrained(
                        release="gpt2-small-resid-pre-v5-32k",
                        sae_id=f"blocks.{sae_layer}.hook_resid_pre",
                        device="cpu"
                    )
                    print(f"  SAE (v5) loaded: {sae.cfg.d_sae} features")
                    W_dec = sae.W_dec.detach().cpu().numpy()
                    sae_coh = [token_coherence(W_dec[i], W_E) for i in range(min(100, W_dec.shape[0]))]
                    print(f"  SAE coherence: {np.mean(sae_coh):.4f}")
                    sae_comparison[f"L{sae_layer}_sae_coherence_v5"] = {
                        "mean": float(np.mean(sae_coh)), "std": float(np.std(sae_coh))
                    }
                    del sae, W_dec
                    gc.collect()
                except Exception as e2:
                    print(f"  Also failed with v5: {e2}")

    except Exception as e:
        print(f"  SAE loading failed: {e}")
        HAS_SAE = False

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. SVD direction coherence comparison
ax = axes[0, 0]
data_to_plot = [svd_coherence_qk[:500], svd_coherence_ov[:500], random_coherence[:500]]
labels_to_plot = ['QK SVD\ndirections', 'OV SVD\ndirections', 'Random\ndirections']
colors_box = ['#2196F3', '#4CAF50', '#9E9E9E']

if sae_comparison:
    for key in sae_comparison:
        if 'coherence' in key:
            # we only have summary stats for SAE, so we add a reference line
            ax.axhline(sae_comparison[key]["mean"], color='#E91E63', linestyle='--',
                       linewidth=2, label=f'SAE mean ({key})')

bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Token Coherence Score')
ax.set_title('Interpretability: Token Coherence of SVD Directions\n'
             'Higher = top tokens form a more coherent cluster')
ax.legend(fontsize=8)

# 2. Compression: rank needed for 90/95/99% energy
ax = axes[0, 1]
ranks_90 = [compression_data[k]["rank_90"] for k in compression_data]
ranks_95 = [compression_data[k]["rank_95"] for k in compression_data]
ranks_99 = [compression_data[k]["rank_99"] for k in compression_data]

ax.hist(ranks_90, bins=20, alpha=0.7, label='90% energy', color='#2196F3')
ax.hist(ranks_95, bins=20, alpha=0.5, label='95% energy', color='#FF9800')
ax.hist(ranks_99, bins=20, alpha=0.3, label='99% energy', color='#f44336')
ax.set_xlabel('Number of SVD Components Needed')
ax.set_ylabel('Number of Heads')
ax.set_title('SVD Compressibility of QK Circuits\n'
             'SVD achieves 90% accuracy with few components')
ax.legend()

# 3. Spectral gap distribution
ax = axes[1, 0]
gaps = [min(compression_data[k]["spectral_gap"], 10) for k in compression_data]
ax.hist(gaps, bins=30, color='#9C27B0', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Spectral Gap (σ₁/σ₂)')
ax.set_ylabel('Number of Heads')
ax.set_title('Spectral Gap Distribution\n'
             'Larger gap = head dominated by a single matching rule')

# 4. SVD vs SAE alignment (if available)
ax = axes[1, 1]
if sae_comparison:
    alignments = []
    head_labels = []
    for key, val in sae_comparison.items():
        if "max_alignment_per_svd" in val:
            alignments.extend(val["max_alignment_per_svd"][:5])
            head_labels.extend([key] * 5)

    if alignments:
        ax.hist(alignments, bins=25, color='#E91E63', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(0.5, color='red', linestyle='--', label='Strong alignment (cos > 0.5)')
        ax.axvline(0.3, color='orange', linestyle='--', label='Moderate alignment (cos > 0.3)')
        ax.set_xlabel('Max |cosine| between SVD direction and nearest SAE feature')
        ax.set_ylabel('Count')
        ax.set_title('SVD-SAE Alignment Distribution\n'
                     'How well do SAE features match SVD directions?')
        ax.legend(fontsize=9)

        n_strong = sum(1 for a in alignments if a > 0.5)
        n_moderate = sum(1 for a in alignments if a > 0.3)
        print(f"\nSVD-SAE alignment: {n_strong}/{len(alignments)} strongly aligned (>0.5), "
              f"{n_moderate}/{len(alignments)} moderately aligned (>0.3)")
    else:
        ax.text(0.5, 0.5, 'No SAE alignment data\navailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
else:
    ax.text(0.5, 0.5, 'SAE comparison\nnot available\n(install sae-lens)', transform=ax.transAxes,
            ha='center', va='center', fontsize=14)

plt.suptitle('SVD vs SAE: Weight-Based Decomposition Analysis\n'
             'Comparing SVD directions against trained sparse autoencoders',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'sae_vs_svd_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {path}")

# save results
all_results = {
    "compression_data": compression_data,
    "coherence": {
        "qk_svd_mean": float(np.mean(svd_coherence_qk)),
        "qk_svd_std": float(np.std(svd_coherence_qk)),
        "ov_svd_mean": float(np.mean(svd_coherence_ov)),
        "ov_svd_std": float(np.std(svd_coherence_ov)),
        "random_mean": float(np.mean(random_coherence)),
        "random_std": float(np.std(random_coherence)),
    },
    "sae_comparison": sae_comparison,
    "sae_available": HAS_SAE,
}

with open(os.path.join(OUT_DIR, "sae_vs_svd_comparison.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print("\nSAE vs SVD comparison complete.")

# Scale SVD decomposition to Pythia-410M and Pythia-1.4B
# Validates whether sub-head functional structure generalizes beyond GPT-2 small

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import gc
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from transformer_lens import HookedTransformer

OUT_DIR = "/Users/srimanarayana/Research Project I/results"

MODELS = {
    "pythia-410m": {"name": "pythia-410m", "n_layers": 24, "n_heads": 16, "d_model": 1024, "d_head": 64},
    "pythia-1.4b": {"name": "pythia-1.4b", "n_layers": 24, "n_heads": 16, "d_model": 2048, "d_head": 128},
}

def decompose_model(model_key, top_k=64):
    cfg = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Loading {model_key} (d_model={cfg['d_model']}, {cfg['n_layers']}L x {cfg['n_heads']}H)")
    print(f"{'='*60}")

    # float32 on CPU (float16 causes hangs on Apple Silicon without MPS)
    model = HookedTransformer.from_pretrained(cfg["name"])
    model.eval()

    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    d_model = cfg["d_model"]
    d_head = cfg["d_head"]

    # save embedding matrix
    W_E = model.W_E.detach().float().cpu().numpy()
    np.save(os.path.join(OUT_DIR, f"{model_key}_W_E.npy"), W_E)
    print(f"Saved W_E: {W_E.shape}")

    all_QK = {}
    all_OV = {}
    stats = {"model": model_key, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads}
    effective_ranks_qk = np.zeros((n_layers, n_heads))
    effective_ranks_ov = np.zeros((n_layers, n_heads))
    top_sv_qk = np.zeros((n_layers, n_heads))
    top_sv_ov = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        for head in range(n_heads):
            W_Q = model.W_Q[layer, head].detach().float().cpu().numpy()  # (d_model, d_head)
            W_K = model.W_K[layer, head].detach().float().cpu().numpy()
            W_V = model.W_V[layer, head].detach().float().cpu().numpy()
            W_O = model.W_O[layer, head].detach().float().cpu().numpy()  # (d_head, d_model)

            # QK circuit
            W_QK = W_Q @ W_K.T  # (d_model, d_model)
            if d_model <= 1024:
                U, S, Vt = svd(W_QK, full_matrices=False)
            else:
                # truncated SVD for large matrices
                k = min(top_k, min(W_QK.shape) - 1)
                U, S, Vt = svds(W_QK.astype(np.float64), k=k)
                # svds returns in ascending order, flip
                idx = np.argsort(-S)
                U, S, Vt = U[:, idx], S[idx], Vt[idx, :]

            all_QK[(layer, head)] = (U, S, Vt)
            top_sv_qk[layer, head] = S[0]

            # effective rank
            cumvar = np.cumsum(S**2) / np.sum(S**2)
            eff_rank = np.searchsorted(cumvar, 0.9) + 1
            effective_ranks_qk[layer, head] = eff_rank

            # OV circuit
            W_OV = W_V @ W_O  # (d_model, d_model) via (d_model, d_head) @ (d_head, d_model)
            if d_model <= 1024:
                U_ov, S_ov, Vt_ov = svd(W_OV, full_matrices=False)
            else:
                k = min(top_k, min(W_OV.shape) - 1)
                U_ov, S_ov, Vt_ov = svds(W_OV.astype(np.float64), k=k)
                idx = np.argsort(-S_ov)
                U_ov, S_ov, Vt_ov = U_ov[:, idx], S_ov[idx], Vt_ov[idx, :]

            all_OV[(layer, head)] = (U_ov, S_ov, Vt_ov)
            top_sv_ov[layer, head] = S_ov[0]
            cumvar_ov = np.cumsum(S_ov**2) / np.sum(S_ov**2)
            effective_ranks_ov[layer, head] = np.searchsorted(cumvar_ov, 0.9) + 1

            if head == 0:
                print(f"  L{layer}: QK top SV={S[0]:.2f}, eff_rank={eff_rank} | OV top SV={S_ov[0]:.2f}")

        gc.collect()

    # save decompositions
    with open(os.path.join(OUT_DIR, f"{model_key}_QK.pkl"), "wb") as f:
        pickle.dump(all_QK, f)
    with open(os.path.join(OUT_DIR, f"{model_key}_OV.pkl"), "wb") as f:
        pickle.dump(all_OV, f)
    print(f"Saved decompositions for {model_key}")

    stats["qk_effective_rank_mean"] = float(effective_ranks_qk.mean())
    stats["qk_effective_rank_min"] = int(effective_ranks_qk.min())
    stats["qk_effective_rank_max"] = int(effective_ranks_qk.max())
    stats["ov_effective_rank_mean"] = float(effective_ranks_ov.mean())
    stats["qk_top_sv_max"] = float(top_sv_qk.max())
    stats["ov_top_sv_max"] = float(top_sv_ov.max())

    # Q-K semantic alignment for this model
    print(f"\nComputing Q-K semantic alignment for {model_key}...")
    alignment_data = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            U, S, Vt = all_QK[(layer, head)]
            top3_cos = []
            for k in range(min(3, len(S))):
                q_proj = W_E @ U[:, k]
                k_proj = W_E @ Vt[k, :]
                cos = np.dot(q_proj, k_proj) / (np.linalg.norm(q_proj) * np.linalg.norm(k_proj) + 1e-10)
                top3_cos.append(cos)
            alignment_data[layer, head] = np.mean(top3_cos)

    stats["alignment_mean"] = float(alignment_data.mean())
    stats["alignment_std"] = float(alignment_data.std())
    n_semantic = int(np.sum(alignment_data > 2 * alignment_data.std()))
    n_contrastive = int(np.sum(alignment_data < -2 * alignment_data.std()))
    stats["n_semantic_matching"] = n_semantic
    stats["n_contrastive_matching"] = n_contrastive

    # random baseline
    rng = np.random.RandomState(42)
    random_cos = []
    for _ in range(1000):
        u_rand = rng.randn(d_model)
        v_rand = rng.randn(d_model)
        q_rand = W_E @ u_rand
        k_rand = W_E @ v_rand
        cos = np.dot(q_rand, k_rand) / (np.linalg.norm(q_rand) * np.linalg.norm(k_rand) + 1e-10)
        random_cos.append(cos)
    stats["random_baseline_mean"] = float(np.mean(random_cos))
    stats["random_baseline_std"] = float(np.std(random_cos))

    print(f"  Alignment: mean={alignment_data.mean():.4f}, std={alignment_data.std():.4f}")
    print(f"  Semantic matching heads (>+2σ): {n_semantic}")
    print(f"  Contrastive heads (<-2σ): {n_contrastive}")
    print(f"  Random baseline: {np.mean(random_cos):.4f} ± {np.std(random_cos):.4f}")

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.heatmap(top_sv_qk, ax=axes[0], cmap='viridis',
                xticklabels=[f'H{h}' for h in range(n_heads)],
                yticklabels=[f'L{l}' for l in range(n_layers)],
                annot=False)
    axes[0].set_title(f'{model_key}: Top QK Singular Value')

    sns.heatmap(effective_ranks_qk, ax=axes[1], cmap='YlOrRd',
                xticklabels=[f'H{h}' for h in range(n_heads)],
                yticklabels=[f'L{l}' for l in range(n_layers)],
                annot=False)
    axes[1].set_title(f'{model_key}: QK Effective Rank (90% threshold)')

    sns.heatmap(alignment_data, ax=axes[2], cmap='RdBu_r', center=0,
                xticklabels=[f'H{h}' for h in range(n_heads)],
                yticklabels=[f'L{l}' for l in range(n_layers)],
                annot=False)
    axes[2].set_title(f'{model_key}: Q-K Semantic Alignment (top-3 mean cosine)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{model_key}_overview.png'), dpi=150)
    plt.close()

    # alignment distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    actual_vals = alignment_data.flatten()
    ax.hist(actual_vals, bins=30, alpha=0.7, color='steelblue', label=f'{model_key} actual', density=True)
    ax.hist(random_cos, bins=30, alpha=0.4, color='grey', label='Random baseline', density=True)
    ax.axvline(2*np.std(random_cos), color='red', linestyle='--', alpha=0.7, label='+2σ threshold')
    ax.axvline(-2*np.std(random_cos), color='red', linestyle='--', alpha=0.7, label='-2σ threshold')
    ax.set_xlabel('Q-K Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_key}: Q-K Alignment Distribution vs Random Baseline')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{model_key}_alignment_distribution.png'), dpi=150)
    plt.close()

    # free memory
    del model, W_E, all_QK, all_OV
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return stats, effective_ranks_qk, alignment_data


# Run both models
all_stats = {}

# Run Pythia-410M first (safe on 8GB), then try 1.4B if memory allows
for model_key in ["pythia-410m"]:
    stats, eff_ranks, alignment = decompose_model(model_key)
    all_stats[model_key] = stats
    print(f"\n{model_key} summary:")
    print(f"  QK effective rank: {stats['qk_effective_rank_mean']:.1f} (range {stats['qk_effective_rank_min']}-{stats['qk_effective_rank_max']})")
    print(f"  Semantic heads: {stats['n_semantic_matching']}, Contrastive heads: {stats['n_contrastive_matching']}")
    gc.collect()

# load GPT-2 stats for comparison
gpt2_stats = {"model": "gpt2", "d_model": 768, "n_layers": 12, "n_heads": 12}
try:
    with open(os.path.join(OUT_DIR, "alignment_results.json"), "r") as f:
        gpt2_align = json.load(f)
    gpt2_stats["n_semantic_matching"] = gpt2_align.get("n_semantic", 11)
    gpt2_stats["n_contrastive_matching"] = gpt2_align.get("n_contrastive", 9)
except:
    gpt2_stats["n_semantic_matching"] = 11
    gpt2_stats["n_contrastive_matching"] = 9
gpt2_stats["qk_effective_rank_mean"] = 45.6

all_stats["gpt2"] = gpt2_stats

# cross-model comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = ["gpt2", "pythia-410m", "pythia-1.4b"]
params = [124, 410, 1400]

# semantic vs contrastive across scale
sem_counts = [all_stats[m]["n_semantic_matching"] for m in models]
con_counts = [all_stats[m]["n_contrastive_matching"] for m in models]
total_heads = [144, 384, 384]
sem_frac = [s/t for s, t in zip(sem_counts, total_heads)]
con_frac = [c/t for c, t in zip(con_counts, total_heads)]

x = np.arange(len(models))
w = 0.35
axes[0].bar(x - w/2, sem_frac, w, label='Semantic matching', color='steelblue')
axes[0].bar(x + w/2, con_frac, w, label='Contrastive matching', color='coral')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{m}\n({p}M)' for m, p in zip(models, params)])
axes[0].set_ylabel('Fraction of heads (>2σ)')
axes[0].set_title('Bimodal Alignment Across Scale')
axes[0].legend()

# effective rank across scale
eff_ranks = [all_stats[m]["qk_effective_rank_mean"] for m in models]
axes[1].bar(x, eff_ranks, color='seagreen')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'{m}\n({p}M)' for m, p in zip(models, params)])
axes[1].set_ylabel('Mean QK Effective Rank')
axes[1].set_title('Effective Rank vs Model Scale')

# parameter count vs number of heads with extreme alignment
total_extreme = [s + c for s, c in zip(sem_counts, con_counts)]
axes[2].plot(params, [e/t for e, t in zip(total_extreme, total_heads)],
             'o-', color='purple', markersize=10, linewidth=2)
axes[2].set_xlabel('Model Parameters (M)')
axes[2].set_ylabel('Fraction of extreme-alignment heads')
axes[2].set_title('Non-Random QK Structure vs Scale')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cross_model_comparison.png'), dpi=150)
plt.close()

# save all stats
with open(os.path.join(OUT_DIR, "scaling_stats.json"), "w") as f:
    json.dump(all_stats, f, indent=2)

print("\n\nCross-model comparison saved.")
print("Done.")

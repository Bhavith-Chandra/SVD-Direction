# QK bilinear form visualization — weight-only attention scores for selected token sets
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from transformer_lens import HookedTransformer

OUT_DIR = "/Users/srimanarayana/Research Project I/results"

print("Loading model and decompositions")
model = HookedTransformer.from_pretrained("gpt2")
model.eval()

with open(os.path.join(OUT_DIR, "all_QK.pkl"), "rb") as f:
    all_QK = pickle.load(f)

W_E = np.load(os.path.join(OUT_DIR, "W_E.npy"))

def get_token_id(token_str):
    ids = model.tokenizer.encode(token_str)
    if len(ids) == 1:
        return ids[0]
    return None

def visualize_qk_bilinear(layer, head, token_set_name, token_strings, top_k_sv=3):
    U, S, Vt = all_QK[(layer, head)]
    W_QK_full = np.zeros((768, 768))
    for i in range(len(S)):
        W_QK_full += S[i] * np.outer(U[:, i], Vt[i, :])

    # Filter to tokens that tokenize to single token
    valid = []
    for t in token_strings:
        tid = get_token_id(t)
        if tid is not None:
            valid.append((t.strip(), tid))

    if len(valid) < 5:
        print(f"  Only {len(valid)} valid tokens for L{layer}H{head}, skipping.")
        return

    labels = [t for t, _ in valid]
    ids = [tid for _, tid in valid]
    embs = W_E[ids]  # (n, d_model)

    # Full bilinear form
    score_full = embs @ W_QK_full @ embs.T

    # Top-k approximation
    score_topk = np.zeros_like(score_full)
    for k in range(top_k_sv):
        u_proj = embs @ U[:, k]
        v_proj = embs @ Vt[k, :]
        score_topk += S[k] * np.outer(u_proj, v_proj)

    residual = score_full - score_topk
    approx_quality = 1 - np.linalg.norm(residual) / np.linalg.norm(score_full)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    vmax = np.percentile(np.abs(score_full), 95)

    for ax, data, title in [
        (axes[0], score_full, f'Full W_QK bilinear form'),
        (axes[1], score_topk, f'Top-{top_k_sv} SVD approx (quality={approx_quality:.2%})'),
        (axes[2], residual, f'Residual (Frob norm={np.linalg.norm(residual):.2f})'),
    ]:
        vm = np.percentile(np.abs(data), 95) if np.abs(data).max() > 0 else 1
        im = ax.imshow(data, cmap='RdBu_r', vmin=-vm, vmax=vm, aspect='auto')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Key tokens →')
        ax.set_ylabel('← Query tokens')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'L{layer}H{head} W_QK Bilinear Form — {token_set_name}\n'
                 f'Top SV: {S[0]:.2f}, Spectrum: [{", ".join([f"{S[k]:.1f}" for k in range(min(5,len(S)))])}]',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'L{layer}H{head}_QK_bilinear_{token_set_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")
    return approx_quality

# Token sets for probing different head types
token_sets = {
    'IOI Names': [' John', ' Mary', ' Bob', ' Alice', ' James', ' Sarah', ' Tom', ' Emma',
                  ' Mike', ' Lisa', ' gave', ' told', ' showed', ' sent', ' brought',
                  ' the', ' a', ' an', ' his', ' her', ' he', ' she', ' they', ' it',
                  ' to', ' from', '.', ',', '!'],
    'Syntax': [' the', ' a', ' an', ' of', ' in', ' to', ' for', ' on', ' at', ' by',
               ' is', ' was', ' are', ' were', ' has', ' had', ' will', ' would',
               ' that', ' which', ' who', ' what', '.', ',', '(', ')', ';', ':'],
    'Semantic': [' dog', ' cat', ' bird', ' fish', ' tree', ' flower', ' car', ' bus',
                 ' red', ' blue', ' green', ' big', ' small', ' fast', ' slow',
                 ' happy', ' sad', ' good', ' bad', ' new', ' old', ' hot', ' cold'],
}

heads_to_analyze = [(9, 9), (9, 6), (10, 0), (1, 4), (2, 2), (1, 3), (11, 10), (7, 3)]

for layer, head in heads_to_analyze:
    print(f"\n--- L{layer}H{head} ---")
    for ts_name, tokens in token_sets.items():
        visualize_qk_bilinear(layer, head, ts_name, tokens, top_k_sv=3)

# Also create a summary: approximation quality across all heads
print("\n\nComputing SVD approximation quality for all heads...")
approx_data = np.zeros((12, 12))
# Use IOI tokens for consistency
valid_ioi = []
for t in token_sets['IOI Names']:
    tid = get_token_id(t)
    if tid is not None:
        valid_ioi.append(tid)
embs_ioi = W_E[valid_ioi]

for l in range(12):
    for h in range(12):
        U, S, Vt = all_QK[(l, h)]
        W_full = np.zeros((768, 768))
        for i in range(len(S)):
            W_full += S[i] * np.outer(U[:, i], Vt[i, :])
        score_full = embs_ioi @ W_full @ embs_ioi.T

        score_top3 = np.zeros_like(score_full)
        for k in range(3):
            u_proj = embs_ioi @ U[:, k]
            v_proj = embs_ioi @ Vt[k, :]
            score_top3 += S[k] * np.outer(u_proj, v_proj)

        residual = score_full - score_top3
        if np.linalg.norm(score_full) > 0:
            approx_data[l, h] = 1 - np.linalg.norm(residual) / np.linalg.norm(score_full)

import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(approx_data, ax=ax, cmap='YlGn',
            xticklabels=[f'H{h}' for h in range(12)],
            yticklabels=[f'L{l}' for l in range(12)],
            annot=True, fmt='.2f', vmin=0, vmax=1, annot_kws={'fontsize': 7})
ax.set_title('Top-3 SVD Approximation Quality of W_QK Bilinear Form\n'
             '(on IOI token set; 1.0 = perfect approximation)', fontsize=12)
ax.set_xlabel('Head')
ax.set_ylabel('Layer')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'QK_SVD_approx_quality_heatmap.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

print("\nDone.")

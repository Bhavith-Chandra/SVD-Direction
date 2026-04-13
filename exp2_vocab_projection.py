# Project singular vectors into vocab space to see what tokens each head cares about
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
with open(os.path.join(OUT_DIR, "all_OV.pkl"), "rb") as f:
    all_OV = pickle.load(f)

W_E = np.load(os.path.join(OUT_DIR, "W_E.npy"))
W_U = np.load(os.path.join(OUT_DIR, "W_U.npy"))

def top_tokens_for_direction(direction, W_proj, top_n=15, mode='cosine'):
    direction_norm = direction / (np.linalg.norm(direction) + 1e-10)
    if mode == 'cosine':
        norms = np.linalg.norm(W_proj, axis=1, keepdims=True) + 1e-10
        W_norm = W_proj / norms
        scores = W_norm @ direction_norm
    else:
        scores = W_proj @ direction_norm
    top_idx = np.argsort(scores)[::-1][:top_n]
    bot_idx = np.argsort(scores)[:top_n]
    top_tokens = [(model.tokenizer.decode([i]).strip(), float(scores[i])) for i in top_idx]
    bot_tokens = [(model.tokenizer.decode([i]).strip(), float(scores[i])) for i in bot_idx]
    return top_tokens, bot_tokens

def plot_head_interpretation(layer, head, circuit='QK', top_k=5):
    if circuit == 'QK':
        U, S, Vt = all_QK[(layer, head)]
        dir_labels = ('Query direction', 'Key direction')
    else:
        U, S, Vt = all_OV[(layer, head)]
        dir_labels = ('Input direction', 'Output direction')

    fig, axes = plt.subplots(top_k, 2, figsize=(16, top_k * 3))
    if top_k == 1:
        axes = axes.reshape(1, 2)

    interpretation_data = []

    for k in range(top_k):
        sv_norm = S[k] / S[0]
        u_k = U[:, k]
        v_k = Vt[k, :]

        if circuit == 'QK':
            q_top, q_bot = top_tokens_for_direction(u_k, W_E, top_n=10)
            k_top, k_bot = top_tokens_for_direction(v_k, W_E, top_n=10)
        else:
            q_top, q_bot = top_tokens_for_direction(u_k, W_E, top_n=10)
            k_top, k_bot = top_tokens_for_direction(v_k, W_U.T, top_n=10)

        interpretation_data.append({
            'rank': k, 'sigma': float(S[k]), 'sigma_norm': float(sv_norm),
            'query_top': q_top, 'query_bot': q_bot,
            'key_top': k_top, 'key_bot': k_bot
        })

        ax_q = axes[k, 0]
        labels = [t for t, _ in q_top[::-1]]
        values = [v for _, v in q_top[::-1]]
        colors = ['#4C72B0' if v > 0 else '#DD8452' for v in values]
        ax_q.barh(range(len(labels)), values, color=colors)
        ax_q.set_yticks(range(len(labels)))
        ax_q.set_yticklabels(labels, fontsize=8)
        ax_q.set_title(f'SV {k} (σ={S[k]:.2f}, {sv_norm:.2f}×) — {dir_labels[0]}', fontsize=9)
        ax_q.axvline(0, color='black', linewidth=0.5)

        ax_k = axes[k, 1]
        labels = [t for t, _ in k_top[::-1]]
        values = [v for _, v in k_top[::-1]]
        colors = ['#55A868' if v > 0 else '#C44E52' for v in values]
        ax_k.barh(range(len(labels)), values, color=colors)
        ax_k.set_yticks(range(len(labels)))
        ax_k.set_yticklabels(labels, fontsize=8)
        ax_k.set_title(f'SV {k} — {dir_labels[1]}', fontsize=9)
        ax_k.axvline(0, color='black', linewidth=0.5)

    fig.suptitle(f'L{layer}H{head} {circuit} Circuit — Singular Vector Token Clusters\n'
                 f'Spectrum (top {top_k}): [{", ".join([f"{S[k]:.2f}" for k in range(top_k)])}]',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'L{layer}H{head}_{circuit}_token_clusters.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return interpretation_data

heads_to_analyze = {
    # Known IOI circuit heads
    'Name Mover L9H9': (9, 9),
    'Name Mover L10H0': (10, 0),
    'S-Inhibition L9H6': (9, 6),
    'S-Inhibition L7H3': (7, 3),
    'S-Inhibition L7H9': (7, 9),
    'Duplicate Token L0H1': (0, 1),
    'Induction L1H4': (1, 4),
    'Pos Backup L11H10': (11, 10),
    # Low-rank discoveries from Exp 1
    'Low-rank QK L1H3': (1, 3),
    'Low-rank QK L1H10': (1, 10),
    'Low-rank QK L2H2': (2, 2),
    'Low-rank QK L2H9': (2, 9),
    # Extreme OV head
    'Rank-1 OV L11H8': (11, 8),
}

all_interpretations = {}
for name, (l, h) in heads_to_analyze.items():
    print(f"\n{name}")
    qk_data = plot_head_interpretation(l, h, circuit='QK', top_k=5)
    ov_data = plot_head_interpretation(l, h, circuit='OV', top_k=5)
    all_interpretations[name] = {'QK': qk_data, 'OV': ov_data}

    for k in range(min(3, len(qk_data))):
        d = qk_data[k]
        q_str = ', '.join([f'"{t}"' for t, _ in d['query_top'][:5]])
        k_str = ', '.join([f'"{t}"' for t, _ in d['key_top'][:5]])
        print(f"  SV{k} (σ={d['sigma']:.2f}): Query=[{q_str}] → Key=[{k_str}]")

with open(os.path.join(OUT_DIR, "interpretations.json"), "w") as f:
    json.dump(all_interpretations, f, indent=2)

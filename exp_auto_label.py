# Automated taxonomy of SVD directions via unsupervised clustering
# Groups all singular directions by their token-space signatures
# No API key needed — uses k-means + POS tagging for labels

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

OUT_DIR = "/Users/srimanarayana/Research Project I/results"

# only load tokenizer (not full model) — we just need cached decompositions
tokenizer = AutoTokenizer.from_pretrained("gpt2")

with open(os.path.join(OUT_DIR, "all_QK.pkl"), "rb") as f:
    all_QK = pickle.load(f)

W_E = np.load(os.path.join(OUT_DIR, "W_E.npy"))
vocab_size = W_E.shape[0]

# build token metadata for labeling clusters
print("Building token metadata...")
token_strings = []
token_types_list = []  # rough POS categories based on heuristics

for i in range(vocab_size):
    try:
        s = tokenizer.decode([i])
    except:
        s = f"<{i}>"
    token_strings.append(s)

    # heuristic classification
    stripped = s.strip()
    if not stripped:
        token_types_list.append("whitespace")
    elif stripped in '.,;:!?()[]{}"\'-/\\@#$%^&*':
        token_types_list.append("punctuation")
    elif stripped.isdigit():
        token_types_list.append("number")
    elif stripped.lower() in {'the', 'a', 'an', 'this', 'that', 'these', 'those',
                               'is', 'are', 'was', 'were', 'be', 'been', 'being',
                               'have', 'has', 'had', 'do', 'does', 'did',
                               'will', 'would', 'could', 'should', 'may', 'might',
                               'shall', 'can', 'must', 'need',
                               'not', 'no', 'nor', 'neither',
                               'and', 'or', 'but', 'if', 'then', 'else',
                               'of', 'in', 'to', 'for', 'on', 'at', 'by',
                               'with', 'from', 'up', 'out', 'off', 'over',
                               'into', 'through', 'about', 'after', 'before',
                               'between', 'under', 'above', 'below'}:
        token_types_list.append("function_word")
    elif stripped[0].isupper() and len(stripped) > 1:
        token_types_list.append("proper_noun")
    elif any(c.isalpha() for c in stripped):
        token_types_list.append("content_word")
    else:
        token_types_list.append("other")

type_indices = {}
token_types = token_types_list
for t in set(token_types):
    type_indices[t] = [i for i, tt in enumerate(token_types) if tt == t]
print(f"Token type distribution: {Counter(token_types).most_common()}")

# Extract token-space signatures for all top SVD directions
print("\nExtracting token signatures for all SVD directions...")

n_top = 5  # top-k SVDs per head
all_signatures = []  # (n_heads * n_top, signature_dim)
all_labels = []  # (layer, head, k, circuit_type)

for layer in range(12):
    for head in range(12):
        U, S, Vt = all_QK[(layer, head)]
        for k in range(min(n_top, U.shape[1])):
            # query direction signature: token-space projection
            q_scores = W_E @ U[:, k]  # (vocab_size,)
            k_scores = W_E @ Vt[k, :]

            # feature vector: type-averaged scores (compact representation)
            q_type_means = []
            k_type_means = []
            for t in sorted(type_indices.keys()):
                idx = type_indices[t]
                q_type_means.append(np.mean(q_scores[idx]))
                k_type_means.append(np.mean(k_scores[idx]))
                q_type_means.append(np.std(q_scores[idx]))
                k_type_means.append(np.std(k_scores[idx]))

            # also add top-token statistics
            q_top50_idx = np.argsort(-np.abs(q_scores))[:50]
            k_top50_idx = np.argsort(-np.abs(k_scores))[:50]

            # what types dominate the top-50?
            q_type_dist = Counter(token_types[i] for i in q_top50_idx)
            k_type_dist = Counter(token_types[i] for i in k_top50_idx)

            for t in sorted(type_indices.keys()):
                q_type_means.append(q_type_dist.get(t, 0) / 50.0)
                k_type_means.append(k_type_dist.get(t, 0) / 50.0)

            signature = np.array(q_type_means + k_type_means + [float(S[k]), float(S[k] / S[0])])
            all_signatures.append(signature)
            all_labels.append((layer, head, k, "QK"))

signatures = np.array(all_signatures)
print(f"Signature matrix: {signatures.shape}")

# normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
signatures_scaled = scaler.fit_transform(signatures)

# Determine optimal k via elbow method
print("\nFinding optimal cluster count...")
inertias = []
k_range = range(5, 25)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(signatures_scaled)
    inertias.append(km.inertia_)

# use k=12 as a reasonable default (one per head type is a natural grouping)
n_clusters = 12
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
cluster_labels = km.fit_predict(signatures_scaled)

# Characterize each cluster
print(f"\nCluster characterization (k={n_clusters}):")
cluster_info = {}

for c in range(n_clusters):
    mask = cluster_labels == c
    members = [all_labels[i] for i in range(len(all_labels)) if mask[i]]

    # layer distribution
    layer_dist = Counter(m[0] for m in members)
    # which heads
    head_dist = Counter(f"L{m[0]}H{m[1]}" for m in members)
    # which SVD rank
    rank_dist = Counter(m[2] for m in members)

    # dominant token types in this cluster (average of cluster signatures)
    cluster_sigs = signatures[mask]
    type_names = sorted(type_indices.keys())

    # the first len(type_names)*2 features are query type means/stds
    # the next len(type_names)*2 are key type means/stds
    # then len(type_names) for query top-50 type fractions
    # then len(type_names) for key top-50 type fractions
    n_types = len(type_names)
    q_frac_start = n_types * 4
    q_fracs = cluster_sigs[:, q_frac_start:q_frac_start + n_types].mean(axis=0)
    k_frac_start = q_frac_start + n_types
    k_fracs = cluster_sigs[:, k_frac_start:k_frac_start + n_types].mean(axis=0)

    # auto-label based on dominant token type
    q_dominant = type_names[np.argmax(q_fracs)]
    k_dominant = type_names[np.argmax(k_fracs)]

    if q_dominant == k_dominant:
        auto_label = f"{q_dominant}_matcher"
    else:
        auto_label = f"{q_dominant}_to_{k_dominant}"

    # layer tendency
    early_frac = sum(layer_dist.get(l, 0) for l in range(4)) / len(members)
    late_frac = sum(layer_dist.get(l, 0) for l in range(8, 12)) / len(members)
    if early_frac > 0.6:
        auto_label += "_early"
    elif late_frac > 0.6:
        auto_label += "_late"

    print(f"  Cluster {c} ({len(members)} dirs): {auto_label}")
    print(f"    Layers: {dict(sorted(layer_dist.items()))}")
    print(f"    Query dominant: {q_dominant} ({q_fracs[np.argmax(q_fracs)]:.2f})")
    print(f"    Key dominant: {k_dominant} ({k_fracs[np.argmax(k_fracs)]:.2f})")
    top_heads = head_dist.most_common(5)
    print(f"    Top heads: {top_heads}")

    cluster_info[c] = {
        "label": auto_label,
        "size": len(members),
        "layer_distribution": dict(layer_dist),
        "rank_distribution": dict(rank_dist),
        "top_heads": [(h, c) for h, c in top_heads],
        "query_dominant_type": q_dominant,
        "key_dominant_type": k_dominant,
    }

# Dimensionality reduction for visualization
print("\nComputing t-SNE embedding...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embedded = tsne.fit_transform(signatures_scaled)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# 1. t-SNE colored by cluster
ax = axes[0, 0]
scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=cluster_labels,
                     cmap='tab20', s=8, alpha=0.6)
for c in range(n_clusters):
    mask = cluster_labels == c
    cx, cy = embedded[mask, 0].mean(), embedded[mask, 1].mean()
    ax.annotate(cluster_info[c]["label"], (cx, cy), fontsize=6,
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
ax.set_title(f't-SNE of SVD Direction Token Signatures\n'
             f'{len(all_labels)} directions, {n_clusters} clusters', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# 2. t-SNE colored by layer
ax = axes[0, 1]
layers = [l[0] for l in all_labels]
scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=layers,
                     cmap='viridis', s=8, alpha=0.6)
plt.colorbar(scatter, ax=ax, label='Layer')
ax.set_title('Same t-SNE colored by Layer\n'
             'Do direction types segregate by depth?', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# 3. Cluster size and layer composition
ax = axes[1, 0]
cluster_ids = list(range(n_clusters))
sizes = [cluster_info[c]["size"] for c in cluster_ids]
labels_plot = [cluster_info[c]["label"] for c in cluster_ids]
ax.barh(range(n_clusters), sizes, color=plt.cm.tab20(np.linspace(0, 1, n_clusters)))
ax.set_yticks(range(n_clusters))
ax.set_yticklabels(labels_plot, fontsize=8)
ax.set_xlabel('Number of SVD Directions')
ax.set_title('Cluster Sizes with Auto-Labels', fontsize=12)

# 4. Elbow plot
ax = axes[1, 1]
ax.plot(list(k_range), inertias, 'o-', color='steelblue', linewidth=2)
ax.axvline(n_clusters, color='red', linestyle='--', label=f'k={n_clusters} (selected)')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Cluster Count', fontsize=12)
ax.legend()

plt.suptitle('Automated Taxonomy of SVD Directions\n'
             'Unsupervised clustering reveals recurring direction types across heads',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'direction_taxonomy.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {path}")

# Per-cluster detailed heatmap
fig, ax = plt.subplots(figsize=(14, 8))
cluster_layer_matrix = np.zeros((n_clusters, 12))
for i, (layer, head, k, _) in enumerate(all_labels):
    cluster_layer_matrix[cluster_labels[i], layer] += 1

# normalize by cluster size
for c in range(n_clusters):
    total = cluster_layer_matrix[c].sum()
    if total > 0:
        cluster_layer_matrix[c] /= total

sns.heatmap(cluster_layer_matrix, ax=ax, cmap='Blues',
            xticklabels=[f'L{l}' for l in range(12)],
            yticklabels=[cluster_info[c]["label"] for c in range(n_clusters)],
            annot=True, fmt='.2f', annot_kws={'fontsize': 7})
ax.set_xlabel('Layer')
ax.set_ylabel('Direction Type (cluster)')
ax.set_title('Distribution of Direction Types Across Layers\n'
             '(normalized by cluster size)')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'direction_type_by_layer.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

# Save taxonomy
taxonomy = {
    "n_clusters": n_clusters,
    "n_directions": len(all_labels),
    "cluster_info": {str(k): v for k, v in cluster_info.items()},
    "direction_assignments": [
        {"layer": l, "head": h, "sv_rank": k, "cluster": int(cluster_labels[i]),
         "cluster_label": cluster_info[int(cluster_labels[i])]["label"]}
        for i, (l, h, k, _) in enumerate(all_labels)
    ],
}

with open(os.path.join(OUT_DIR, "direction_taxonomy.json"), "w") as f:
    json.dump(taxonomy, f, indent=2)

print("\nAuto-labeling complete.")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from scipy.stats import ttest_ind, mannwhitneyu
from transformer_lens import HookedTransformer

OUT_DIR = "/Users/srimanarayana/Research Project I/results"

print("Loading model and decompositions")
model = HookedTransformer.from_pretrained("gpt2")
model.eval()

with open(os.path.join(OUT_DIR, "all_QK.pkl"), "rb") as f:
    all_QK = pickle.load(f)

name_mover_positive = [
    "When Mary and John went to the store, John gave a drink to",
    "When Alice and Bob went to the park, Bob gave a flower to",
    "When Sarah and James walked in, James said hello to",
    "When Emma and Tom arrived at the party, Tom handed a gift to",
    "When Lisa and Mike entered the room, Mike waved at",
    "When Kate and Dave went to dinner, Dave offered a seat to",
    "When Jane and Mark went shopping, Mark bought a present for",
    "After Mary and John finished dinner, John passed the salt to",
    "After Alice and Bob left the party, Bob drove home with",
    "After Sarah and James finished work, James called",
]

name_mover_negative = [
    "The cat sat on the mat and looked at the",
    "In the beginning there was nothing but empty space and",
    "The weather today is sunny with a high of about",
    "Python is a programming language used for data science and",
    "The quick brown fox jumps over the lazy dog and",
    "Mountains are formed by tectonic plate movements over",
    "The stock market has been volatile this year due to",
    "Water boils at one hundred degrees Celsius under standard",
    "The library was closed on Sundays but opened on",
    "Photosynthesis converts sunlight into chemical energy in",
]

s_inhibition_positive = name_mover_positive
s_inhibition_negative = name_mover_negative

induction_positive = [
    "The dog chased the cat. The dog chased the",
    "Alice went to Paris. Bob went to London. Alice went to",
    "One two three four five. One two three four",
    "The red ball rolled down the hill. The red ball rolled down the",
    "He said hello to her. She said goodbye to him. He said hello to",
    "First comes A then B then C. First comes A then B then",
    "The king sat on the throne. The queen sat next to the king. The king sat on the",
    "Open the door and close the window. Open the door and close the",
    "Rain falls in spring. Snow falls in winter. Rain falls in",
    "She read a book about history. He read a book about science. She read a book about",
]

induction_negative = [
    "The capital of France is known for its",
    "Machine learning models require large amounts of",
    "The industrial revolution changed the course of human",
    "Climate change affects global temperatures and weather",
    "DNA contains the genetic instructions for biological",
    "The solar system has eight planets orbiting the",
    "Quantum mechanics describes the behavior of particles at",
    "Democracy is a form of government where citizens",
    "The internet has revolutionized communication and",
    "Evolution is driven by natural selection and genetic",
]

def get_residual_projections(model, prompts, layer, direction):
    projections_by_pos = {'last': [], 'mean': [], 'max': []}
    direction_t = torch.tensor(direction, dtype=torch.float32)

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens,
                names_filter=lambda n: f'blocks.{layer}.hook_resid_pre' in n)
        resid = cache[f'blocks.{layer}.hook_resid_pre'][0]
        proj = (resid @ direction_t).cpu().numpy()
        projections_by_pos['last'].append(float(proj[-1]))
        projections_by_pos['mean'].append(float(proj.mean()))
        projections_by_pos['max'].append(float(proj.max()))

    return projections_by_pos

def verify_direction_causally(model, layer, head, k, positive_prompts, negative_prompts,
                               direction_type='both', label=''):
    U, S, Vt = all_QK[(layer, head)]
    u_k = U[:, k]
    v_k = Vt[k, :]

    results = {}
    directions = {'query (u_k)': u_k, 'key (v_k)': v_k}

    for dir_name, direction in directions.items():
        pos_proj = get_residual_projections(model, positive_prompts, layer, direction)
        neg_proj = get_residual_projections(model, negative_prompts, layer, direction)

        for pos_type in ['last', 'mean']:
            pos_vals = np.array(pos_proj[pos_type])
            neg_vals = np.array(neg_proj[pos_type])

            t_stat, p_value = ttest_ind(pos_vals, neg_vals)
            u_stat, p_mw = mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')

            effect_size = (pos_vals.mean() - neg_vals.mean()) / (
                np.sqrt((pos_vals.std()**2 + neg_vals.std()**2) / 2) + 1e-10
            )

            results[f'{dir_name}_{pos_type}'] = {
                'pos_mean': float(pos_vals.mean()),
                'neg_mean': float(neg_vals.mean()),
                'pos_std': float(pos_vals.std()),
                'neg_std': float(neg_vals.std()),
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'p_mannwhitney': float(p_mw),
                'effect_size_d': float(effect_size),
                'significant_005': p_value < 0.05,
                'significant_001': p_value < 0.01,
            }

    return results

test_configs = [
    {
        'name': 'Name Mover L9H9',
        'layer': 9, 'head': 9,
        'positive': name_mover_positive,
        'negative': name_mover_negative,
        'max_k': 5,
    },
    {
        'name': 'Name Mover L10H0',
        'layer': 10, 'head': 0,
        'positive': name_mover_positive,
        'negative': name_mover_negative,
        'max_k': 5,
    },
    {
        'name': 'S-Inhibition L9H6',
        'layer': 9, 'head': 6,
        'positive': s_inhibition_positive,
        'negative': s_inhibition_negative,
        'max_k': 5,
    },
    {
        'name': 'Induction L1H4',
        'layer': 1, 'head': 4,
        'positive': induction_positive,
        'negative': induction_negative,
        'max_k': 5,
    },
]

all_verification_results = {}

for config in test_configs:
    name = config['name']
    print(f"\nVerifying {name}")
    head_results = {}

    for k in range(config['max_k']):
        res = verify_direction_causally(
            model, config['layer'], config['head'], k,
            config['positive'], config['negative'],
            label=name
        )
        head_results[f'SV{k}'] = res

        for key, val in res.items():
            if val['significant_005']:
                sig = '***' if val['significant_001'] else '**'
            else:
                sig = 'n.s.'
            print(f"  SV{k} {key}: d={val['effect_size_d']:.2f}, p={val['p_value']:.4f} {sig}")

    all_verification_results[name] = head_results

fig, axes = plt.subplots(len(test_configs), 2, figsize=(14, len(test_configs) * 4))

for row, config in enumerate(test_configs):
    l, h = config['layer'], config['head']
    U, S, Vt = all_QK[(l, h)]

    for col, (dir_name, dir_fn) in enumerate([
        ('Query direction u_0', lambda: U[:, 0]),
        ('Key direction v_0', lambda: Vt[0, :]),
    ]):
        ax = axes[row, col]
        direction = dir_fn()

        pos_proj = get_residual_projections(model, config['positive'], l, direction)
        neg_proj = get_residual_projections(model, config['negative'], l, direction)

        pos_last = pos_proj['last']
        neg_last = neg_proj['last']

        parts = ax.violinplot([pos_last, neg_last], positions=[0, 1], showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)

        ax.scatter([0]*len(pos_last), pos_last, c='steelblue', alpha=0.5, s=20)
        ax.scatter([1]*len(neg_last), neg_last, c='coral', alpha=0.5, s=20)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Positive', 'Negative'])
        ax.set_ylabel('Projection (last token)')

        t, p = ttest_ind(pos_last, neg_last)
        ax.set_title(f'{config["name"]}: {dir_name}\nt={t:.2f}, p={p:.4f}', fontsize=9)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

plt.suptitle('Causal Verification: Residual Stream Projections onto Top SVD Directions\n'
             'Positive prompts (blue) vs. Negative prompts (red) — last token position',
             fontsize=12, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'causal_verification_violins.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"\nSaved {path}")

fig, axes = plt.subplots(1, len(test_configs), figsize=(5*len(test_configs), 5))
if len(test_configs) == 1:
    axes = [axes]

for idx, config in enumerate(test_configs):
    name = config['name']
    ax = axes[idx]
    head_res = all_verification_results[name]
    n_sv = len(head_res)
    n_metrics = 4  # query_last, query_mean, key_last, key_mean

    data = np.zeros((n_sv, n_metrics))
    metric_labels = ['q_last', 'q_mean', 'k_last', 'k_mean']
    metric_keys = ['query (u_k)_last', 'query (u_k)_mean', 'key (v_k)_last', 'key (v_k)_mean']

    for sv_idx in range(n_sv):
        sv_res = head_res[f'SV{sv_idx}']
        for m_idx, m_key in enumerate(metric_keys):
            data[sv_idx, m_idx] = sv_res[m_key]['effect_size_d']

    sns.heatmap(data, ax=ax, cmap='RdBu_r', center=0,
                xticklabels=metric_labels,
                yticklabels=[f'SV{k}' for k in range(n_sv)],
                annot=True, fmt='.2f', linewidths=0.5)
    ax.set_title(f'{name}\nEffect sizes (Cohen\'s d)', fontsize=9)

plt.suptitle('Effect Sizes: SVD Direction Selectivity for Known Head Functions',
             fontsize=12, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'causal_effect_sizes_heatmap.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

with open(os.path.join(OUT_DIR, "causal_verification_results.json"), "w") as f:
    json.dump(all_verification_results, f, indent=2)

print("\nDone.")

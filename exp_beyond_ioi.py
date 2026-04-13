# Beyond IOI: test sub-head opposition on greater-than and factual recall tasks
# Proves the phenomenon isn't IOI-specific

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

# Task 1: Greater-than (numerical reasoning)
# "The war started in 1492 and ended in 14" -> model should upweight digits > 92

def create_greater_than_dataset(n=50):
    templates = [
        "The war started in {year} and ended in {prefix}",
        "The building was constructed in {year} and demolished in {prefix}",
        "The king was born in {year} and died in {prefix}",
        "The treaty was signed in {year} and expired in {prefix}",
        "The company was founded in {year} and closed in {prefix}",
    ]
    prompts = []
    for i in range(n):
        century = 14 + (i % 5)
        decade = np.random.RandomState(i).randint(10, 90)
        year = century * 100 + decade
        prefix = str(century)
        tmpl = templates[i % len(templates)]
        prompt = tmpl.format(year=year, prefix=prefix)
        # correct: digits that would make the end year > start year
        # the model should upweight tokens for digits > decade
        prompts.append({
            'prompt': prompt,
            'start_year': year,
            'century_prefix': prefix,
            'decade': decade,
        })
    return prompts

def measure_greater_than(hooks=None, dataset=None):
    if dataset is None:
        dataset = gt_data
    scores = []
    for item in dataset:
        tokens = model.to_tokens(item['prompt'])
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        last_logits = logits[0, -1]
        # measure: sum of logits for digits > decade vs <= decade
        decade = item['decade']
        greater_logits = []
        lesser_logits = []
        for d in range(10):
            two_digit = decade // 10 * 10 + d  # same tens digit
            token_str = str(d)
            tid = model.to_single_token(token_str)
            if two_digit > decade % 10:  # simplified: just compare last digit
                greater_logits.append(last_logits[tid].item())
            else:
                lesser_logits.append(last_logits[tid].item())
        if greater_logits and lesser_logits:
            scores.append(np.mean(greater_logits) - np.mean(lesser_logits))
    return np.mean(scores) if scores else 0.0

gt_data = create_greater_than_dataset(50)

# Task 2: Factual recall
def create_factual_dataset():
    facts = [
        ("The capital of France is", " Paris", " Berlin"),
        ("The capital of Germany is", " Berlin", " Paris"),
        ("The capital of Japan is", " Tokyo", " Beijing"),
        ("The capital of Italy is", " Rome", " Madrid"),
        ("The capital of Spain is", " Madrid", " Rome"),
        ("The largest planet in our solar system is", " Jupiter", " Mars"),
        ("Water freezes at zero degrees", " Celsius", " Fahrenheit"),
        ("The Earth orbits the", " Sun", " Moon"),
        ("Shakespeare wrote", " Hamlet", " Odyssey"),
        ("Einstein developed the theory of", " relativity", " evolution"),
        ("The chemical symbol for gold is", " Au", " Ag"),
        ("The speed of light is approximately 300", ",000", ".000"),
        ("The first president of the United States was George", " Washington", " Lincoln"),
        ("The Mona Lisa was painted by Leonardo da", " Vinci", " Monet"),
        ("The Great Wall is located in", " China", " Japan"),
        ("The currency of the United Kingdom is the", " pound", " euro"),
        ("Oxygen has the chemical symbol", " O", " N"),
        ("The Amazon is the longest", " river", " mountain"),
        ("Neil Armstrong was the first person on the", " Moon", " Mars"),
        ("DNA stands for deoxyribonucle", "ic", "ar"),
        ("The boiling point of water is 100 degrees", " Celsius", " Fahrenheit"),
        ("Photosynthesis occurs in", " plants", " animals"),
        ("The atomic number of hydrogen is", " 1", " 2"),
        ("The Pythagorean theorem involves", " triangles", " circles"),
        ("The human body has 206", " bones", " muscles"),
    ]
    return [{'prompt': p, 'correct': c, 'incorrect': ic} for p, c, ic in facts]

fact_data = create_factual_dataset()

def measure_factual(hooks=None):
    logit_diffs = []
    for item in fact_data:
        tokens = model.to_tokens(item['prompt'])
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        try:
            correct_id = model.to_single_token(item['correct'])
            incorrect_id = model.to_single_token(item['incorrect'])
            ld = (logits[0, -1, correct_id] - logits[0, -1, incorrect_id]).item()
            logit_diffs.append(ld)
        except:
            continue
    return np.mean(logit_diffs) if logit_diffs else 0.0

# Baselines
gt_baseline = measure_greater_than()
fact_baseline = measure_factual()
print(f"Greater-than baseline score: {gt_baseline:.3f}")
print(f"Factual recall baseline logit diff: {fact_baseline:.3f}")

# Scan all heads for opposition on each task
def scan_all_heads_for_opposition(task_measure, task_name, max_k=5):
    print(f"\n{'='*60}")
    print(f"Scanning all 144 heads for sub-head opposition on {task_name}")
    print(f"{'='*60}")

    baseline = task_measure()
    head_results = {}

    for layer in range(12):
        for head in range(12):
            U, S, Vt = all_QK[(layer, head)]
            individual_effects = []

            for k in range(max_k):
                u_k = torch.tensor(U[:, k], dtype=torch.float32)
                def make_hook(u_vec):
                    u = u_vec.clone()
                    def hook_fn(resid, hook):
                        proj = torch.einsum('bsd,d->bs', resid, u)
                        return resid - torch.einsum('bs,d->bsd', proj, u)
                    return hook_fn

                hook = make_hook(u_k)
                hook_name = f'blocks.{layer}.hook_resid_pre'
                score = task_measure([(hook_name, hook)])
                effect = baseline - score
                individual_effects.append(float(effect))

            # full head ablation
            def make_head_hook(h):
                def hook_fn(q, hook):
                    q[:, :, h, :] = 0
                    return q
                return hook_fn

            full_score = task_measure([(f'blocks.{layer}.attn.hook_q', make_head_hook(head))])
            full_effect = baseline - full_score

            max_individual = max(abs(e) for e in individual_effects)
            opposition_ratio = max_individual / abs(full_effect) if abs(full_effect) > 0.01 else 0

            has_opposing = any(e > 0 for e in individual_effects) and any(e < 0 for e in individual_effects)

            head_results[f"L{layer}H{head}"] = {
                "individual_effects": individual_effects,
                "full_head_effect": float(full_effect),
                "max_individual": float(max_individual),
                "opposition_ratio": float(opposition_ratio),
                "has_opposing_directions": has_opposing,
                "singular_values": [float(S[k]) for k in range(max_k)],
            }

            if opposition_ratio > 3 or max_individual > 0.5:
                print(f"  L{layer}H{head}: ratio={opposition_ratio:.1f}x, "
                      f"max_indiv={max_individual:.3f}, full={full_effect:.3f}, "
                      f"opposing={has_opposing}")

        print(f"  Layer {layer} done")

    return baseline, head_results

# Run scans
gt_baseline_val, gt_results = scan_all_heads_for_opposition(measure_greater_than, "Greater-Than")
fact_baseline_val, fact_results = scan_all_heads_for_opposition(measure_factual, "Factual Recall")

# Aggregate statistics
def compute_opposition_stats(head_results, task_name):
    ratios = [r["opposition_ratio"] for r in head_results.values() if r["opposition_ratio"] > 0]
    n_opposing = sum(1 for r in head_results.values() if r["has_opposing_directions"])
    n_high_ratio = sum(1 for r in head_results.values() if r["opposition_ratio"] > 3)
    n_very_high = sum(1 for r in head_results.values() if r["opposition_ratio"] > 5)

    print(f"\n{task_name} opposition statistics:")
    print(f"  Heads with opposing directions: {n_opposing}/144 ({n_opposing/144*100:.1f}%)")
    print(f"  Heads with >3x opposition ratio: {n_high_ratio}/144 ({n_high_ratio/144*100:.1f}%)")
    print(f"  Heads with >5x opposition ratio: {n_very_high}/144 ({n_very_high/144*100:.1f}%)")
    if ratios:
        print(f"  Mean opposition ratio: {np.mean(ratios):.2f}")
        print(f"  Median opposition ratio: {np.median(ratios):.2f}")

    return {
        "n_opposing": n_opposing,
        "n_high_ratio_3x": n_high_ratio,
        "n_high_ratio_5x": n_very_high,
        "mean_ratio": float(np.mean(ratios)) if ratios else 0,
        "median_ratio": float(np.median(ratios)) if ratios else 0,
    }

gt_stats = compute_opposition_stats(gt_results, "Greater-Than")
fact_stats = compute_opposition_stats(fact_results, "Factual Recall")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for row, (task_name, head_results, stats) in enumerate([
    ("Greater-Than", gt_results, gt_stats),
    ("Factual Recall", fact_results, fact_stats),
]):
    # opposition ratio heatmap
    ratio_grid = np.zeros((12, 12))
    for l in range(12):
        for h in range(12):
            ratio_grid[l, h] = head_results[f"L{l}H{h}"]["opposition_ratio"]

    import seaborn as sns
    sns.heatmap(ratio_grid, ax=axes[row, 0], cmap='YlOrRd',
                xticklabels=[f'H{h}' for h in range(12)],
                yticklabels=[f'L{l}' for l in range(12)],
                annot=False, vmin=0, vmax=10)
    axes[row, 0].set_title(f'{task_name}: Opposition Ratio\n(max |individual| / |full head|)')

    # full head effect heatmap
    full_grid = np.zeros((12, 12))
    for l in range(12):
        for h in range(12):
            full_grid[l, h] = head_results[f"L{l}H{h}"]["full_head_effect"]

    vm = np.percentile(np.abs(full_grid), 95)
    sns.heatmap(full_grid, ax=axes[row, 1], cmap='RdBu_r', center=0,
                xticklabels=[f'H{h}' for h in range(12)],
                yticklabels=[f'L{l}' for l in range(12)],
                annot=False, vmin=-vm, vmax=vm)
    axes[row, 1].set_title(f'{task_name}: Full Head Effect')

    # histogram of opposition ratios
    ratios = [head_results[k]["opposition_ratio"] for k in head_results
              if head_results[k]["opposition_ratio"] > 0]
    axes[row, 2].hist(ratios, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[row, 2].axvline(3, color='red', linestyle='--', label='>3x threshold')
    axes[row, 2].axvline(5, color='darkred', linestyle='--', label='>5x threshold')
    axes[row, 2].set_xlabel('Opposition Ratio')
    axes[row, 2].set_ylabel('Number of Heads')
    axes[row, 2].set_title(f'{task_name}: Opposition Ratio Distribution\n'
                           f'{stats["n_high_ratio_3x"]}/144 heads >3x')
    axes[row, 2].legend()

plt.suptitle('Sub-Head Functional Opposition Beyond IOI\n'
             'Testing universality on Greater-Than and Factual Recall tasks',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(OUT_DIR, 'beyond_ioi_opposition.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {path}")

# Cross-task comparison: which heads are consistently opposing across tasks?
print(f"\n{'='*60}")
print("Cross-task comparison: consistently opposing heads")
print(f"{'='*60}")

consistent_heads = []
for l in range(12):
    for h in range(12):
        key = f"L{l}H{h}"
        gt_ratio = gt_results[key]["opposition_ratio"]
        fact_ratio = fact_results[key]["opposition_ratio"]
        if gt_ratio > 3 and fact_ratio > 3:
            consistent_heads.append({
                "head": key, "gt_ratio": gt_ratio, "fact_ratio": fact_ratio,
                "gt_full": gt_results[key]["full_head_effect"],
                "fact_full": fact_results[key]["full_head_effect"],
            })
            print(f"  {key}: GT ratio={gt_ratio:.1f}x, Fact ratio={fact_ratio:.1f}x")

print(f"\nHeads with >3x opposition on BOTH tasks: {len(consistent_heads)}/144 "
      f"({len(consistent_heads)/144*100:.1f}%)")

# save everything
all_results = {
    "greater_than": {
        "baseline": float(gt_baseline_val),
        "head_results": gt_results,
        "stats": gt_stats,
    },
    "factual_recall": {
        "baseline": float(fact_baseline_val),
        "head_results": fact_results,
        "stats": fact_stats,
    },
    "consistent_opposing_heads": consistent_heads,
    "n_consistent": len(consistent_heads),
}

with open(os.path.join(OUT_DIR, "beyond_ioi_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\nBeyond-IOI experiment complete.")

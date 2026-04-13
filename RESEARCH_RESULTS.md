# SVD of QK/OV Circuits in GPT-2 — Experimental Results

Results from decomposing all 144 attention heads in GPT-2 small.

Model: GPT-2 small (124M params), 12 layers × 12 heads, d_model=768, d_head=64, vocab=50257.

---

## Singular value spectra

**W_QK:** Top SV ranges from 0.53 to 22.95 (L2H2). Effective rank (90% Frobenius) ranges 7–54, mean 45.6.

Most low-rank QK heads: L1H3 and L1H10 (rank 7), L1H2 (rank 10), L2H2 (rank 15). These are the cleanest candidates for interpretation — few matching rules dominate.

**W_OV:** L11H8 is effectively rank-1 (top SV = 334.87). Almost a pure projection.

L2H2 and L2H9 have strikingly similar decompositions. Redundancy?

## Q-K semantic alignment

Random baseline: cosine mean = 0.005, std = 0.200.

**Bimodal distribution** — 11 heads above +2σ (semantic matching), 9 heads below -2σ (contrastive matching).

Semantic matchers: L0H1 (0.904), L0H5 (0.880), L0H10 (0.843), L5H1 (0.592), L5H0 (0.592), L6H10 (0.586). Mostly layer 0.

Contrastive matchers: L0H9 (-0.849), L10H7 (-0.740), L0H6 (-0.655), L11H1 (-0.539), L11H8 (-0.479). Mostly layers 10-11.

Rules out "QK is just positional." Content-based matching is baked into the weights.

## Causal ablation (IOI task)

Baseline IOI logit diff: 4.469 ± 0.836.

**L9H6 (S-Inhibition) — the big finding:**
- Full head ablation: -0.046 (looks irrelevant)
- SV0 alone: +0.386 (8.4x the net effect)
- SV1 alone: +0.800 (17.4x the net effect)
- SV2 alone: -0.169 (opposing)
- Top-3 cumulative: +0.872

The head is doing massive work that cancels internally. You miss this completely at the head level.

**L9H9 (Name Mover):**
- Full head: +0.273
- SV0: +0.369 (135% of full — more than the head itself because SV1 opposes at -0.135)

**L1H4 (Induction):**
- SV2: 1.742, SV4: 2.940 — cascading downstream effects from early-layer ablation

## Activation verification

Effect sizes (Cohen's d) for residual stream projections, IOI prompts vs. controls:

- L9H9 SV0 query: d = 3.46, p < 0.0001
- L9H6 SV0 query: d = -3.22, p < 0.0001
- L9H6 SV1 query: d = -3.25, p < 0.0001
- L1H4 SV0 query: d = 2.95, p < 0.0001
- L1H4 SV0 key: d = 2.80, p < 0.0001

All well above the "huge effect" threshold. These aren't artifacts.

## Token projections

QK token projections are noisier than OV — lots of rare tokens and control characters in the top lists. But there are clear patterns:

- L11H8 OV SV2 key: discourse markers (Following, According, While, Specifically)
- L2H2 QK: encoding-specific tokens (framework, capped, equilibrium) — structural/formatting head

## Files

65 plots in `results/`. Scripts: exp1 through exp8. Data: `all_QK.pkl`, `all_OV.pkl`, various JSON results.

## What's left

- Automated labeling (needs API key)
- Pythia-1.4B replication
- Attention-score-level ablation (more targeted than residual stream projection)

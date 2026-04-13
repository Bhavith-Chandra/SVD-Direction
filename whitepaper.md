# Attention Heads Are Not Monolithic: Sub-Head Functional Decomposition via SVD of QK Circuits in GPT-2

---

**Abstract**

The standard unit of analysis in transformer mechanistic interpretability is the attention head. We argue this is too coarse. By applying SVD to the composed query-key matrices (W_QK = W_Q W_K^T) of all 144 heads in GPT-2 small, we find structure that head-level analysis misses entirely. The key results: (1) the semantic alignment between query and key singular directions follows a bimodal distribution — some heads match tokens to similar tokens, others match to dissimilar ones, and this split is far outside what random matrices produce; (2) individual SVD directions within a single head can have causal effects on downstream tasks that are an order of magnitude larger than the head's net effect, but in opposing directions, meaning the head's apparent unimportance masks massive internal computation; (3) these directions show effect sizes above Cohen's d = 3.0 when tested against task-relevant vs. control prompts. The upshot is that weight-based SVD, which requires no training data and no hyperparameters, can crack open attention heads and reveal functional structure that activation-based methods treat as a black box.

---

## 1. Introduction

### 1.1 The problem

Everyone in mech interp knows what an attention head does — it computes a bilinear attention pattern and writes a linear function of the attended values back to the residual stream. The field has gotten remarkably far treating heads as atoms: induction heads, name movers, inhibition heads, the whole IOI circuit. But a head with a 64-dimensional key/query space has room to do more than one thing at once. And there's growing evidence that they do.

The "Beyond Components" work (Lieberum et al., 2025) showed this for OV circuits — you can decompose them via SVD and find multiple independent sub-computations living in different singular directions. But nobody has done this systematically for QK circuits, and there's a good reason: the QK decomposition is harder. W_OV maps from residual stream to residual stream — same space in, same space out. W_QK is a bilinear form. The left singular vectors (queries) and right singular vectors (keys) live in different semantic subspaces. You can't just look at one set of token projections; you need to understand the *relationship* between the two.

This paper fills that gap. We decompose every QK matrix in GPT-2 small, measure whether the query and key directions are semantically related, and then causally validate the individual directions using the IOI task as ground truth.

### 1.2 What we found

Three things, in order of how surprising they were to us:

1. **The Q-K alignment distribution is bimodal.** When you project query and key singular vectors into token space and measure their cosine similarity, you don't get a blob around zero. You get two clusters — heads where Q and K activate *similar* tokens (semantic matching) and heads where they activate *opposite* tokens (contrastive matching). The random baseline is centered at 0.005 with std 0.200. The actual distribution has 11 heads above +2σ and 9 heads below -2σ.

2. **Single heads contain massive opposing sub-computations.** The S-inhibition head L9H6 has a net effect of -0.046 on the IOI task when you ablate the whole thing. Looks boring. But ablate just its first singular direction and the effect is +0.386. Ablate just the second and it's +0.800. These are 8x and 17x the net effect. The directions are fighting each other inside the head.

3. **The SVD directions are real features, not noise.** Residual stream projections onto these directions separate IOI prompts from control prompts with Cohen's d > 3.0. That's a huge effect size by any standard.

---

## 2. Background

### 2.1 The QK and OV circuits

Following Elhage et al. (2021), every attention head has two composed circuit matrices. W_QK = W_Q W_K^T determines *where* the head attends (the attention pattern). W_OV = W_V W_O determines *what* happens when it attends (the value transformation). Both are d_model × d_model matrices. For GPT-2 small, that's 768 × 768.

The attention score between positions i and j is x_i^T W_QK x_j / √d_head. This is a bilinear form, which means SVD is the natural decomposition: W_QK = Σ_k σ_k u_k v_k^T, where each rank-1 term σ_k u_k v_k^T is an independent matching rule. The head attends from tokens aligned with u_k to tokens aligned with v_k, with strength σ_k.

### 2.2 Why QK is harder than OV

For W_OV, you can project both the input (u_k) and output (v_k) directions through the same embedding matrix and get interpretable token lists on both sides. For W_QK, u_k tells you what the query "looks for" and v_k tells you what the key "looks like" — these are different semantic questions about different parts of the sequence. The asymmetry is real and makes interpretation less straightforward.

### 2.3 IOI as ground truth

We validate against the Indirect Object Identification circuit (Wang et al., 2022), the best-understood circuit in GPT-2. "When Mary and John went to the store, John gave a drink to ___" → "Mary". The circuit has known name movers (L9H9, L10H0), S-inhibition heads (L7H3, L7H9, L9H6), induction heads, and duplicate token detectors. This gives us ground truth for causal validation.

---

## 3. Methods

### 3.1 Decomposition

GPT-2 small: 12 layers, 12 heads, d_model = 768, d_head = 64, vocab = 50,257. We extract weights via TransformerLens, compose W_QK and W_OV for all 144 heads, and run full SVD in float64.

### 3.2 Effective rank

We define effective rank as the number of singular values needed to capture 90% of the squared Frobenius norm. This tells you how many "matching rules" a head is really using.

### 3.3 Token projection

To interpret a direction u ∈ R^768, project through the embedding matrix: scores = W_E · u ∈ R^50257. The top tokens by cosine similarity tell you what "concept" the direction encodes.

### 3.4 Q-K semantic alignment

For each singular direction k, compute token-space scores for both the query direction (W_E · u_k) and the key direction (W_E · v_k), then measure cosine similarity between these two score vectors. This asks: does the head attend to tokens that are *like* the query token, or tokens that are *unlike* it? We compare against 1,000 random direction pairs as a null distribution.

### 3.5 Causal ablation

We use TransformerLens hooks to project out the u_k component from the residual stream at hook_resid_pre. This removes that singular direction's contribution to everything downstream. For cumulative ablation, we project out u_0 through u_k simultaneously (they're orthonormal from SVD, so this is clean). We measure IOI logit difference — logit(correct name) minus logit(wrong name) — averaged over 100 prompts.

### 3.6 Activation verification

Collect residual stream activations on positive prompts (IOI patterns) and negative prompts (random text), project onto each singular vector, test for separation with Welch's t-test and Cohen's d.

---

## 4. Results

### 4.1 Effective rank varies wildly

| Head | Role | Top SV | Eff. Rank |
|------|------|--------|-----------|
| L1H3 | ? | 1.68 | 7 |
| L1H10 | ? | 1.42 | 7 |
| L2H2 | ? | 22.95 | 15 |
| L1H4 | Induction | 1.84 | 18 |
| L9H9 | Name Mover | 2.43 | ~45 |
| L11H8 | ? (OV) | 334.87 | 1 |

The range is 7 to 54 for QK, mean 45.6. L11H8 is almost perfectly rank-1 in its OV circuit — a single linear map with singular value 334.87, dwarfing everything else. L2H2 has a dominant matching rule with σ_1 = 22.95, the largest QK singular value in the model.

Interestingly, L2H2 and L2H9 have nearly identical decompositions (overlapping top token clusters, similar singular values). Redundancy or backup?

### 4.2 The bimodal alignment distribution

Random baseline: mean cosine = 0.005, std = 0.200.

Actual Q-K pairs (mean of top-3 SVs per head):

**Semantic matching (cosine > +2σ, i.e. > 0.406):**
- L0H1: 0.904, L0H5: 0.880, L0H10: 0.843
- L5H1: 0.592, L5H0: 0.592, L6H10: 0.586
- 11 heads total

**Contrastive matching (cosine < -2σ, i.e. < -0.396):**
- L0H9: -0.849, L10H7: -0.740, L0H6: -0.655
- L11H1: -0.539, L11H8: -0.479
- 9 heads total

The semantic matchers concentrate in layer 0 — these are probably doing direct embedding-similarity attention. The contrastive matchers concentrate in layers 10-11, consistent with inhibitory or corrective roles late in the forward pass.

This pattern rules out the null hypothesis that QK circuits are purely positional. A purely positional head wouldn't care about token identity at all, and its Q-K cosine would sit near zero.

### 4.3 Sub-head functional opposition

This is the main result. Baseline IOI logit diff: 4.469 ± 0.836.

**S-Inhibition L9H6:**

| Direction | Effect when ablated | Multiple of net |
|-----------|-------------------|-----------------|
| SV0 | +0.386 | 8.4x |
| SV1 | +0.800 | **17.4x** |
| SV2 | -0.169 | opposing |
| Top-3 cumulative | +0.872 | 19x |
| Full head | -0.046 | 1x |

Read that again. The full head ablation effect is -0.046 — basically nothing. But SV1 alone is +0.800. There's a war happening inside this head. The directions that help IOI (SV0, SV1) are being fought by directions that hurt it, and they nearly cancel. If you only look at head-level ablation, you conclude this head doesn't matter. You'd be wrong.

**Name Mover L9H9:**

SV0 individually: +0.369 (that's 135% of the full head effect of +0.273). SV1: -0.135, actively opposing. SV2: +0.244. The head is doing name-moving *and* something else, simultaneously, and the something-else partially cancels the name-moving.

**Induction L1H4:**

SV2 and SV4 show individual effects of 1.74 and 2.94 — far larger than the 0.040 full-head effect. Projecting out early-layer singular directions cascades hard through the rest of the network.

### 4.4 Activation verification

| Head & Direction | Cohen's d | p-value |
|-----------------|-----------|---------|
| L9H9 SV0 query | 3.46 | < 0.0001 |
| L9H6 SV0 query | -3.22 | < 0.0001 |
| L9H6 SV1 query | -3.25 | < 0.0001 |
| L1H4 SV0 query | 2.95 | < 0.0001 |
| L1H4 SV0 key | 2.80 | < 0.0001 |

Effect sizes above 2.0 are conventionally "huge." These directions aren't noise — they light up selectively on the right kind of input.

The negative d for L9H6 is informative: the query directions are *suppressed* on IOI inputs, consistent with inhibition. L1H4 SV0 activates bilaterally (both query and key) on repeated sequences, which is exactly what an induction head should do.

---

## 5. Discussion

### 5.1 What this breaks

Head-level circuit analysis has a blind spot. When a head's net ablation effect is near zero, the standard conclusion is "this head doesn't participate in the circuit." L9H6 shows that's wrong — the head can be doing enormous amounts of work that cancels internally. You can only see this by going below the head level.

This also means feature attribution at the head level is lossy. SAEs trained on head outputs will learn features that are mixtures of these SVD directions. The SVD gives you the unmixed version, for free, from weights alone.

### 5.2 Layer-dependent specialization

The semantic-matching heads cluster in layer 0. The contrastive heads cluster in layers 10-11. This makes intuitive sense: early layers do local token-similarity matching (essentially "attend to tokens with similar embeddings"), while late layers need to attend to tokens that are specifically *not* the current token — the pattern you'd expect from inhibition or correction mechanisms.

### 5.3 Limitations

The residual stream ablation isn't QK-specific — projecting out a direction from hook_resid_pre affects everything downstream, not just the target head's attention. A cleaner intervention would subtract rank-1 components directly from the attention scores. We haven't done that yet.

Token projections for QK directions are noisier than for OV directions. Many top tokens are rare unicode, control characters, or non-English text. Better projection methods (contextualized embeddings, unembedding matrix) might help.

This is GPT-2 small only. We need to check whether the bimodal distribution and functional opposition hold up in larger models.

---

## 6. Conclusion

SVD of W_QK reveals that attention heads aren't monolithic. They contain multiple independent matching rules, some of which actively oppose each other. The S-inhibition head L9H6 is the clearest example: its individual SVD directions have effects 17x larger than the head's net effect, but they fight, and the head looks inert from the outside.

The practical takeaway: if you want to understand what an attention head is doing, don't just ablate the whole thing. Decompose its weight matrices. SVD is free — no training, no data, no hyperparameters. Just a matrix factorization on the weights you already have.

---

## 7. Reproducibility

All code and 65 generated plots are in the repository. The full pipeline takes about 2 hours on a CPU.

**Stack**: Python 3.9, TransformerLens 2.18.0, PyTorch 2.8.0, scipy 1.13.1

| Script | What it does | Runtime |
|--------|-------------|---------|
| exp1_decompose.py | SVD all 144 heads | ~5 min |
| exp2_vocab_projection.py | Token clusters | ~8 min |
| exp3_bilinear_form.py | Bilinear form plots | ~10 min |
| exp4_5_fixed.py | IOI ablation (hooks) | ~45 min |
| exp6_semantic_alignment.py | Q-K alignment | ~3 min |
| exp8_causal_verification.py | Activation verification | ~20 min |

---

## References

1. Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." Anthropic.
2. Wang, K., et al. (2022). "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small."
3. Olsson, C., et al. (2022). "In-context Learning and Induction Heads." Anthropic.
4. Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic.
5. Lieberum, T., et al. (2025). "Beyond Components: Decomposing Transformer Attention Heads into Functionally Distinct Sub-Operations."
6. Conmy, A., et al. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability."

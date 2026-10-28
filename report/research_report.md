# Research Project Report: Sparse Autoencoders on GPT-2 Small

**Author:** Research Intern & Antigravity  
**Model:** Pretrained GPT-2 Small (`gpt2-small`, $d_{model} = 768$)  
**Site:** Layer 6 Residual Stream (`blocks.6.hook_resid_post`)  
**Corpus:** WikiText-2 (`wikitext-2-raw-v1`)  

---

## 1. Research Question and Motivation

Transformer representations often exhibit **feature superposition**, packing more concepts into activations than residual dimensions through non-orthogonal linear combinations, causing polysemanticity. **Sparse Autoencoders (SAEs)** untangle these representations into an overcomplete dictionary under an $L_1$ sparsity penalty. This project investigates:

> **How does changing the sparsity penalty ($\lambda$) of a sparse autoencoder affect its ability to reconstruct activations from GPT-2 Small, the number and frequency of active latents, and the contexts that activate learned features?**

We hypothesize that an order-of-magnitude increase in $\lambda$ will suppress active latents per token ($L_0$), modestly increase reconstruction error, and isolate context-specific feature directions.

---

## 2. Architecture and Methodology

We implemented a ReLU Sparse Autoencoder with $d_{in} = 768$ and $d_{sae} = 1024$ latents:
- **Decoder Bias / Centering:** $b_{dec} \in \mathbb{R}^{768}$ initialized to training mean $\bar{x}_{train}$. Input activations are centered: $x_{c} = x - b_{dec}$.
- **Encoder / Decoder:** $h = \text{ReLU}(x_{c} W_{enc} + b_{enc})$ and $\hat{x} = h W_{dec} + b_{dec}$. Row vectors are normalized ($\|W_{dec}[j, :]\|_2 = 1.0$) after gradient updates.
- **Objective:** $\mathcal{L}(x) = \frac{1}{d_{in}} \|x - \hat{x}\|_2^2 + \lambda \sum_{j=1}^{d_{sae}} |h_j|$.

Key evaluation metrics:
1. **Reconstruction MSE:** Coordinate-wise mean squared error $\frac{1}{d_{in}} \|x - \hat{x}\|_2^2$.
2. **Fraction of Variance Explained (FVE):** Error reduction relative to predicting test mean: $\text{FVE} = 1 - \frac{\sum \|x - \hat{x}\|_2^2}{\sum \|x - \bar{x}_{test}\|_2^2}$.
3. **$L_0$ Sparsity:** Mean positive latents per token.

---

## 3. Activation Collection and Dataset

Activations were extracted from frozen GPT-2 Small (`model.eval()`) under `torch.inference_mode()` at `blocks.6.hook_resid_post`. WikiText-2 was split at document level before tokenization to eliminate data leakage:
- **Train Split:** 8,312 token activations (128 document chunks).
- **Validation Split:** 2,049 token activations (37 document chunks).
- **Held-Out Test Split:** 2,112 token activations (32 document chunks).

Padding tokens were excluded; context metadata was preserved per token.

---

## 4. Quantitative Results

Models were trained under identical conditions (Adam, $\text{lr} = 5 \times 10^{-4}$, batch size 128, seed 42, 25 epochs) on the 8,312 training vectors and evaluated on the 2,112 held-out test vectors:

| Metric | Run A ($\lambda = 0.001$) | Run B ($\lambda = 0.01$) | Impact of $10\times$ Sparsity Penalty |
| :--- | :---: | :---: | :---: |
| **Reconstruction MSE** | 0.8865 – 0.9383 | 2.9090 – 2.9148 | Controlled $+210\%$ increase |
| **Variance Accounted For (VAF)** | 99.50% – 99.53% | 98.48% | Remains high ($-1.05\%$ drop) |
| **Fraction of Variance Explained (FVE)** | 99.49% – 99.52% | 98.42% – 98.43% | Explains $>98.4\%$ of residual variance |
| **Mean Active Latents ($L_0$)** | 481.28 – 484.54 | 41.12 – 44.37 | **$-91.5\%$ ($11.7\times$ sparser)** |
| **Median Active Latents ($L_0$)** | 491.0 – 495.0 | 35.0 | Strong leftward shift |
| **Dead Latent Rate (0 test fires)** | 0.0% (0 / 1024) | 0.0% (0 / 1024) | Zero feature collapse |

### Extreme Variance Resilience
Scaling $\lambda$ by $10\times$ pruned active latents by over $90.8\%$ (from 481 to 41 per token). Despite using only $4.0\%$ of dictionary capacity per token, Run B explained **$98.48\%$** of total activation variance without inducing dead features.

---

## 5. Qualitative Analysis and Counterexample Testing

We inspected candidate features from Run B, evaluating top contexts against contrastive examples in the 2,112 test tokens:

1. **Feature #841 (Position-0 Attention Sink | Max Act: 696.74 | Rate: 1.52%):**
   - *Supporting Evidence:* Maximum activation occurs on `<|endoftext|>` at chunk starts (act = 696.74).
   - *Contrastive Testing:* Fires on all 32 document chunks at position 0, including 8 chunks starting with non-delimiter tokens (e.g., `' Ridley'`, `' W'`, `' .'`, act = 684.45 to 694.56). Conversely, it exhibits 0.0000 activation across all 71 sentence-ending periods (`' .'`) and position-1 tokens.
   - *Conclusion:* Feature #841 is not a semantic delimiter. It isolates an attention sink where position 0 absorbs excess residual norm.

2. **Feature #4 (Clausal Coordinator & Delimiter | Max Act: 19.94 | Rate: 2.84%):**
   - *Supporting Evidence:* Fires on coordinating conjunctions (`' and'`, act = 12.23 to 19.94) joining parallel clauses (e.g., *"... civil servant : study **and**"*).
   - *Contrastive Testing:* Inactive on 17 of 37 surface `' and'` tokens (45.9% inactive rate), suppressing in serial name lists. However, it activates on alternative disjunctive coordinators (`' or'`, act = 3.99 and 6.38) and title separators (`' /'`, act = 9.23 in *"... titled Burn /"*).
   - *Conclusion:* Rather than a lexical detector, Feature #4 acts as a clausal continuation coordinator sensitive to broader syntactic boundaries.

3. **Feature #766 (Polysemantic Noise / Sampling Artifact | Max Act: 4.11 | Rate: 0.52%):**
   - *Supporting Evidence:* Fires on `' allowed'` (act = 4.11) and `' Shen'` (act = 0.97) in historical commentary.
   - *Contrastive Testing:* In Document 47 (Du Fu biography, 67 tokens), it fires solely on `' allowed'`, remaining inactive (0.0000) on core terms (`' 740'`, `' Du'`, `' Fu'`, `' father'`, `' died'`). It also fires on theatrical tokens (`' confidence'`, act = 0.04 in doc 17).
   - *Conclusion:* Feature #766 does not represent a "biographical" concept; it exhibits weak alignment and polysemantic dispersion.

---

## 6. Scientific Conclusions

1. **Sparsity vs. Fidelity Trade-off:** Increasing $\lambda$ by $10\times$ yields a non-linear $11.7\times$ reduction in active latents while preserving $>98.4\%$ explained variance.
2. **Reconstruction vs. Interpretability:** High reconstruction fidelity does not ensure monosemantic latents. Even under high sparsity, latents may capture architectural artifacts (Feature #841) or polysemantic noise (Feature #766). Rigorous contrastive testing is essential before assigning semantic labels.

---

## 7. Limitations and Future Directions

1. **Corpus Scale:** 8,312 training vectors from WikiText-2 are sufficient for coarse dictionaries but insufficient for rare concepts. Larger corpora ($>10^6$ tokens) are needed.
2. **Expansion Factor:** Evaluating $4\times$ to $16\times$ expansions will reveal finer dictionary directions.
3. **Dead Latent Mitigation:** Extended training at higher sparsity will require feature resampling or ghost gradients.
